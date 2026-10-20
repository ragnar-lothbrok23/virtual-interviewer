"""Phase 8: facial emotion analysis (DeepFace) of recorded answers.

The aggregation, preprocessing helpers, frame sampling and per-frame logic run for real; DeepFace/TensorFlow is
replaced by fakes except in the opt-in `--deepface` test, which runs the real models on a real face."""
import io
import math
from pathlib import Path

import numpy as np
import pytest

from app.extensions import db
from app.models import AuditLog, Interview, InterviewAnswer, InterviewEvent
from app.services import emotion
from tests.conftest import emotion_frames
from tests.test_interview_room import api, room, started, upload  # noqa: F401  (fixtures)
from tests.test_scheduling import booked, get, job_id  # noqa: F401  (fixtures)
from tests.test_scoring import answer_question, answer_row

av = pytest.importorskip("av")


@pytest.fixture(autouse=True)
def ai(interview_ai):
    return interview_ai


def agg(app, pattern, **kwargs):
    with app.app_context():
        return emotion.aggregate(emotion_frames(pattern, **kwargs), app.config)


# ---------- aggregation (the scoring logic) ----------

def test_composed_attentive_answer(app):
    s = agg(app, "nnnhhnnnhn")
    assert s["status"] == "done" and s["dominant"] == "neutral"
    assert s["attentiveness"] == 100 and s["confidence"] > 85 and s["nervousness"] < 10
    assert s["score"] == pytest.approx(0.5 * s["confidence"] + 0.5 * s["attentiveness"], abs=0.1)
    assert sum(s["distribution"].values()) == pytest.approx(100, abs=0.5)
    assert s["frames"] == s["face_frames"] == s["facing_frames"] == 10


def test_nervous_answer_scores_lower(app):
    calm, nervous = agg(app, "nnnnnnnnnn"), agg(app, "ffsffsffsf")
    assert nervous["nervousness"] > 80 and nervous["confidence"] < 15 and nervous["score"] < calm["score"]


def test_temporal_smoothing_ignores_a_single_odd_frame(app):
    s = agg(app, "nnnnnannnnn")
    assert [point[1] for point in s["timeline"]] == ["neutral"] * 11  # the one "angry" frame doesn't flip it


def test_looking_away_lowers_attentiveness_not_confidence(app):
    s = agg(app, "nnnnnnnnnn", yaw=0.6)
    assert s["attentiveness"] == 0 and s["confidence"] > 85


def test_too_little_face_data_gives_no_score_instead_of_a_low_one(app):
    s = agg(app, "nn--------")
    assert s["status"] == "insufficient" and "face visible in only 20% of frames" in s["reason"]
    assert "score" not in s and "confidence" not in s
    assert agg(app, "nnn")["reason"] == "too few frames"


def test_reliability_reflects_how_stable_the_readings_are(app):
    assert agg(app, "nnnnnnnnnnnn")["reliability"] == "high"
    assert agg(app, "nnnhnnnhnnnhnn")["reliability"] == "medium"
    assert agg(app, "nhnhnhnhnhnh")["reliability"] == "low"
    assert agg(app, "nnnnnnn---")["reliability"] == "low"  # face missing in 30% of frames


def test_absence_and_second_face_are_measured(app):
    s = agg(app, "nnn" + "-" * 12 + "nnnnnnnnnnnnnnnnnnnn22n")
    assert s["longest_absence_seconds"] == 6.0 and s["multi_face_frames"] == 2


def test_non_finite_values_never_become_a_score(app):
    frames = emotion_frames("nnnnn")
    frames[2][1]["probs"]["happy"] = float("nan")
    with app.app_context(), pytest.raises(emotion.EmotionError):
        emotion.aggregate(frames, app.config)


def test_long_timelines_are_downsampled(app):
    s = agg(app, "n" * 500)
    assert len(s["timeline"]) == 60 and s["frames"] == 500


# ---------- helpers used by the real pipeline ----------

def test_yaw_ratio():
    facing = [0, 0, 100, 100, 30, 40, 70, 40, 50, 60, 35, 80, 65, 80, 0.9]
    turned = [0, 0, 100, 100, 30, 40, 70, 40, 75, 60, 35, 80, 65, 80, 0.9]
    assert emotion.yaw_ratio(facing) == pytest.approx(0.0)
    assert emotion.yaw_ratio(turned) == pytest.approx(25 / 40)
    assert emotion.yaw_ratio([0, 0, 1, 1, 30, 40, 30, 40, 30, 50, 0, 0, 0, 0, 0.9]) is None  # eyes coincide


def test_equalise_keeps_shape_and_spreads_brightness():
    rng = np.random.default_rng(0)
    dim = rng.integers(40, 70, (60, 80, 3), dtype=np.uint8)
    out = emotion.equalise(dim)
    assert out.shape == dim.shape and out.dtype == np.uint8 and np.ptp(out) > np.ptp(dim)


def make_video(seconds=2.0, fps=10, audio_only=False):
    from tests.test_media import make_recording

    return make_recording(seconds=seconds, video=not audio_only)


def test_sample_frames_rate_and_cap():
    video = make_video(seconds=3)
    frames = emotion.sample_frames(video, fps=2, max_frames=100)
    assert 5 <= len(frames) <= 7 and frames[0][1].shape == (48, 64, 3)
    assert all(b[0] - a[0] >= 0.49 for a, b in zip(frames, frames[1:]))
    assert len(emotion.sample_frames(video, fps=2, max_frames=3)) == 3


@pytest.mark.parametrize("data, message", [(b"not a video at all", "couldn't be decoded"),
                                           (None, "no video track")])
def test_sample_frames_rejects_unusable_recordings(data, message):
    if data is None:
        data = make_video(audio_only=True)
    with pytest.raises(emotion.EmotionError, match=message):
        emotion.sample_frames(data, fps=2, max_frames=10)


class StubDetector:
    def __init__(self, rows):
        self.rows = rows

    def setInputSize(self, size):  # noqa: N802 (OpenCV API)
        pass

    def setScoreThreshold(self, threshold):  # noqa: N802
        self.threshold = threshold

    def detect(self, image):
        return 1, None if self.rows is None else np.array(self.rows, dtype=np.float32)


class StubDeepFace:
    def __init__(self, faces, fail_first=False):
        self.faces, self.fail_first, self.calls = faces, fail_first, 0

    def analyze(self, image, **kwargs):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise ValueError("Face could not be detected")
        return self.faces


def face_row(x, w, confidence=0.9):
    return [x, 50, w, w, x + 0.3 * w, 80, x + 0.7 * w, 80, x + 0.5 * w, 100, x + 0.35 * w, 120, x + 0.65 * w, 120,
            confidence]


def df_face(x, w, top="neutral"):
    scores = {e: 1.0 for e in emotion.EMOTIONS}
    scores[top] = 94.0
    return {"region": {"x": x, "y": 50, "w": w, "h": w}, "emotion": scores}


def test_analyse_frame_reads_the_main_face(app):
    models = {"detector": StubDetector([face_row(100, 120)]), "deepface": StubDeepFace([df_face(100, 120, "happy")])}
    with app.app_context():
        r = emotion.analyse_frame(np.zeros((300, 400, 3), np.uint8), models, app.config)
    assert r["faces"] == 1 and max(r["probs"], key=r["probs"].get) == "happy"
    assert sum(r["probs"].values()) == pytest.approx(1.0) and r["yaw"] == pytest.approx(0.0)


def test_second_face_counts_only_when_clear_and_comparable(app):
    image = np.zeros((300, 600, 3), np.uint8)
    cases = [([face_row(100, 120), face_row(350, 110, 0.95)], 2),  # another person
             ([face_row(100, 120), face_row(350, 30, 0.95)], 1),  # tiny face (e.g. a photo on the wall)
             ([face_row(100, 120), face_row(350, 110, 0.6)], 1)]  # weak detection
    with app.app_context():
        for rows, expected in cases:
            models = {"detector": StubDetector(rows), "deepface": StubDeepFace([df_face(100, 120)])}
            assert emotion.analyse_frame(image, models, app.config)["faces"] == expected


def test_emotion_is_read_from_the_face_the_detector_chose(app):
    deepface = StubDeepFace([df_face(380, 40, "angry"), df_face(100, 120, "happy")])
    models = {"detector": StubDetector([face_row(100, 120)]), "deepface": deepface}
    with app.app_context():
        r = emotion.analyse_frame(np.zeros((300, 600, 3), np.uint8), models, app.config)
    assert max(r["probs"], key=r["probs"].get) == "happy"


def test_falls_back_to_the_normalised_frame(app):
    deepface = StubDeepFace([df_face(100, 120)], fail_first=True)
    models = {"detector": StubDetector([face_row(100, 120)]), "deepface": deepface}
    with app.app_context():
        r = emotion.analyse_frame(np.zeros((300, 400, 3), np.uint8), models, app.config)
    assert deepface.calls == 2 and r["probs"] is not None


def test_no_face_in_frame(app):
    models = {"detector": StubDetector(None), "deepface": StubDeepFace([])}
    with app.app_context():
        assert emotion.analyse_frame(np.zeros((300, 400, 3), np.uint8), models, app.config) == {
            "faces": 0, "probs": None, "yaw": None}
    assert models["deepface"].calls == 0


# ---------- in the interview ----------

def test_answers_are_analysed_and_hr_sees_the_results(app, room, ai, fake_emotion, login):
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.emotion_status == "done" and fake_emotion.calls == 1
    assert a.emotion_score == pytest.approx(0.5 * a.emotion_confidence + 0.5 * a.emotion_attentiveness, abs=0.1)
    assert a.emotion["dominant"] == "neutral" and a.emotion_at is not None
    application_id = get(app, Interview, room.id).application_id
    page = login("hr@test.com").get(f"/hr/applications/{application_id}").get_data(as_text=True)
    assert "Expressions:" in page and f"Confidence {round(a.emotion_confidence)}" in page
    assert "Emotion score (1 answer)" in page and "indicative only" in page
    candidate_page = room.client.get(f"/candidate/applications/{application_id}").get_data(as_text=True)
    assert "Expressions:" not in candidate_page and "Nervousness" not in candidate_page


def test_insufficient_face_data_is_shown_as_not_assessed(app, room, ai, fake_emotion, login):
    fake_emotion.pattern = "n---------"
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.emotion_status == "insufficient" and a.emotion_score is None
    page = login("hr@test.com").get(f"/hr/applications/{get(app, Interview, room.id).application_id}")
    text = page.get_data(as_text=True)
    assert "not assessed: face visible in only 10% of frames" in text and "couldn't be assessed reliably" in text


def test_integrity_flags_from_the_video(app, room, ai, fake_emotion, login):
    fake_emotion.pattern = "nnnn" + "-" * 12 + "nnnnnnnnnnnnnnnnnnnnnnnnnnnnn222nnnnnnnn"
    started(room)
    answer_question(room, 0)
    with app.app_context():
        kinds = {e.kind: e for e in InterviewEvent.query.all()}
        assert kinds["multiple_faces"].flagged and kinds["multiple_faces"].details == {"seconds": 1.5}
        assert kinds["face_not_visible"].flagged and kinds["face_not_visible"].details == {"seconds": 6.0}
    page = login("hr@test.com").get(f"/hr/applications/{get(app, Interview, room.id).application_id}")
    text = " ".join(page.get_data(as_text=True).split())
    assert "Another face was visible on camera (Q1) (about 2 s)" in text
    assert "Face not visible on camera for a while (Q1) (about 6 s)" in text



def test_analysis_failure_never_affects_the_interview(app, room, ai, fake_emotion):
    fake_emotion.error = RuntimeError("model crashed")
    started(room)
    state = answer_question(room, 0)
    assert state["position"] == 1  # the interview carried on
    a = answer_row(app, 0)
    assert a.emotion_status == "failed" and a.emotion_error == "Unexpected error: model crashed" and a.is_scored


def test_queue_failure_never_affects_the_interview(app, room, ai, monkeypatch):
    from app import tasks

    def broken(answer_id):
        raise ConnectionError("emotion queue down")

    monkeypatch.setattr(tasks, "analyse_emotion", broken)
    started(room)
    assert answer_question(room, 0)["position"] == 1
    assert answer_row(app, 0).emotion_status == "pending"


def test_can_be_switched_off(app, room, ai, fake_emotion):
    app.config["EMOTION_ENABLED"] = False
    try:
        started(room)
        answer_question(room, 0)
    finally:
        app.config["EMOTION_ENABLED"] = True
    assert fake_emotion.calls == 0 and answer_row(app, 0).emotion_status is None


def test_hr_can_reanalyse(app, room, ai, fake_emotion, login):
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    response = login("hr@test.com").post(f"/hr/answers/{a.id}/emotion", follow_redirects=True)
    assert b"being analysed again" in response.data and fake_emotion.calls == 2
    with app.app_context():
        assert AuditLog.query.filter_by(action="answer.emotion_requested").count() == 1


def test_interview_summary(app, room, ai, fake_emotion):
    started(room)
    answer_question(room, 0)
    fake_emotion.pattern = "ffffssffff"
    answer_question(room, 1)
    fake_emotion.pattern = "n---------"
    answer_question(room, 2)
    with app.app_context():
        summary = emotion.interview_emotion_summary(db.session.get(Interview, room.id))
        first, second = (db.session.get(Interview, room.id).answers[i] for i in (0, 1))
        assert summary["answers"] == 2 and summary["analysed"] == 3
        assert summary["score"] == pytest.approx((first.emotion_score + second.emotion_score) / 2, abs=0.1)


def test_consent_explains_the_video_analysis(app, room):
    page = " ".join(room.client.get(f"{room.base}/join").get_data(as_text=True).split())
    assert "analysed on our own server for facial expressions and attentiveness" in page and "only you in view" in page


# ---------- the real models (opt-in: venv\Scripts\pytest --deepface) ----------

@pytest.mark.deepface
def test_real_deepface_on_a_real_face(app):
    """A short video made from a webcam frame of a real face: the real detector and emotion model must find the face
    in every frame and produce a complete, finite summary."""
    import cv2
    from fractions import Fraction

    frame = cv2.imread(str(Path(__file__).resolve().parents[1] / "smoke_test_frame.jpg"))
    assert frame is not None
    out = io.BytesIO()
    with av.open(out, "w", format="webm") as f:
        stream = f.add_stream("libvpx", rate=10)
        stream.width, stream.height, stream.pix_fmt = frame.shape[1], frame.shape[0], "yuv420p"
        for i in range(30):
            vf = av.VideoFrame.from_ndarray(frame[:, :, ::-1].copy(), format="rgb24")
            vf.pts, vf.time_base = i, Fraction(1, 10)
            for packet in stream.encode(vf):
                f.mux(packet)
        for packet in stream.encode():
            f.mux(packet)
    with app.app_context():
        summary = emotion.analyse_recording(out.getvalue())
    assert summary["status"] == "done" and summary["frames"] == summary["face_frames"] == 6
    assert summary["attentiveness"] == 100 and summary["multi_face_frames"] == 0
    assert all(math.isfinite(summary[k]) for k in ("score", "confidence", "nervousness", "attentiveness"))
