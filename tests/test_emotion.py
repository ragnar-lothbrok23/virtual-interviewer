"""Phase 8: facial emotion analysis (DeepFace) of recorded answers.

The aggregation, preprocessing helpers, frame sampling and per-frame logic run for real; DeepFace/TensorFlow is
replaced by fakes except in the opt-in `--deepface` test, which runs the real models on a real face."""
import io
import math
from datetime import timedelta
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


class StubClassifier:
    """Stands in for HSEmotion/DeepFace: returns fixed probabilities and records what it was asked."""

    label = "Stub model"

    def __init__(self, top="neutral", result="default"):
        self.top, self.result, self.calls, self.rows = top, result, 0, []

    def predict(self, image, prepared, row, cfg):
        self.calls += 1
        self.rows.append(list(row))
        if self.result != "default":
            return self.result
        p = {e: 0.01 for e in emotion.EMOTIONS}
        p[self.top] = 1 - 0.01 * (len(emotion.EMOTIONS) - 1)
        return p


def models_with(rows, classifier=None):
    import threading

    return {"detector": StubDetector(rows), "classifier": classifier or StubClassifier(),
            "detect_lock": threading.Lock()}


def face_row(x, w, confidence=0.9):
    return [x, 50, w, w, x + 0.3 * w, 80, x + 0.7 * w, 80, x + 0.5 * w, 100, x + 0.35 * w, 120, x + 0.65 * w, 120,
            confidence]


def test_analyse_frame_reads_the_main_face(app):
    classifier = StubClassifier("happy")
    models = models_with([face_row(380, 40), face_row(100, 120)], classifier)
    with app.app_context():
        r = emotion.analyse_frame(np.zeros((300, 600, 3), np.uint8), models, app.config)
    assert r["faces"] == 1 and max(r["probs"], key=r["probs"].get) == "happy"
    assert r["yaw"] == pytest.approx(0.0) and classifier.rows[0][2] == 120  # the largest face is read


def test_second_face_counts_only_when_clear_and_comparable(app):
    image = np.zeros((300, 600, 3), np.uint8)
    cases = [([face_row(100, 120), face_row(350, 110, 0.95)], 2),  # another person
             ([face_row(100, 120), face_row(350, 30, 0.95)], 1),  # tiny face (e.g. a photo on the wall)
             ([face_row(100, 120), face_row(350, 110, 0.6)], 1)]  # weak detection
    with app.app_context():
        for rows, expected in cases:
            assert emotion.analyse_frame(image, models_with(rows), app.config)["faces"] == expected


@pytest.mark.parametrize("result", [None, {e: float("nan") for e in emotion.EMOTIONS}])
def test_unreadable_expression_counts_as_no_reading(app, result):
    with app.app_context():
        r = emotion.analyse_frame(np.zeros((300, 400, 3), np.uint8),
                                  models_with([face_row(100, 120)], StubClassifier(result=result)), app.config)
    assert r["faces"] == 1 and r["probs"] is None


def test_no_face_in_frame(app):
    classifier = StubClassifier()
    with app.app_context():
        assert emotion.analyse_frame(np.zeros((300, 400, 3), np.uint8), models_with(None, classifier), app.config) == {
            "faces": 0, "probs": None, "yaw": None}
    assert classifier.calls == 0


# ---------- the expression models ----------

def test_aligned_square_levels_the_eyes_and_adds_the_margin():
    from app.services.emotion_models import _aligned_square

    image = np.zeros((400, 400, 3), np.uint8)
    image[150:250, 150:250] = 255  # a white "face" box
    row = [150, 150, 100, 100, 175, 190, 225, 210, 200, 220]  # eyes tilted by about 22 degrees
    crop = _aligned_square(image, row, 0.1)
    assert crop.shape[:2] == (120, 120)  # 100 px face + 10% on each side
    assert crop[60, 60].tolist() == [255, 255, 255]  # centred on the face


def test_aligned_square_at_the_frame_edge_is_filled_not_cut():
    from app.services.emotion_models import _aligned_square

    image = np.full((200, 200, 3), 90, np.uint8)
    crop = _aligned_square(image, [0, 0, 80, 80, 20, 30, 60, 30, 40, 50], 0.2)
    assert crop.shape[:2] == (112, 112) and crop.min() == 90  # edges replicated, no black borders


class FakeSession:
    def __init__(self, logits):
        self.logits, self.inputs = logits, []

    def run(self, outputs, feed):
        self.inputs.append(feed["input"])
        return [np.array([self.logits], dtype=np.float32)]


def hsemotion_with(logits, model_name="enet_b0_8_best_afew"):
    from app.services.emotion_models import HSEmotionClassifier

    clf = HSEmotionClassifier.__new__(HSEmotionClassifier)
    clf.session, clf.size, clf.label = FakeSession(logits), 224 if "_b0_" in model_name else 260, "HSEmotion test"
    return clf


def test_hsemotion_turns_logits_into_probabilities(app):
    from app.services.emotion_models import HSEMOTION_CLASSES

    logits = [0, 0, 0, 0, 4, 1, 0, 0]  # "happy" (index 4) clearly highest
    clf = hsemotion_with(logits)
    with app.app_context():
        p = clf.predict(np.full((300, 400, 3), 128, np.uint8), None, face_row(100, 120), app.config)
    assert list(p) == list(HSEMOTION_CLASSES) and max(p, key=p.get) == "happy"
    assert sum(p.values()) == pytest.approx(1.0)
    x = clf.session.inputs[0]
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32  # RGB, ImageNet-normalised, channels first


def test_hsemotion_rejects_non_finite_output(app):
    clf = hsemotion_with([0, 0, float("nan"), 0, 0, 0, 0, 0])
    with app.app_context():
        assert clf.predict(np.zeros((300, 400, 3), np.uint8), None, face_row(100, 120), app.config) is None


def test_model_files_are_checked_before_use(tmp_path, monkeypatch):
    from app.services import emotion_models

    source = tmp_path / "source.onnx"
    source.write_bytes(b"model bytes")
    good = emotion_models._sha256(source)
    monkeypatch.setitem(emotion_models.MODEL_FILES, "test_model", (source.as_uri(), good))
    target = tmp_path / "models"
    with pytest.raises(emotion_models.ModelError, match="missing"):
        emotion_models.model_path("test_model", target, download=False)
    path = emotion_models.model_path("test_model", target)  # downloaded, then verified
    assert path.read_bytes() == b"model bytes" and not path.with_suffix(".part").exists()
    path.write_bytes(b"tampered")
    with pytest.raises(emotion_models.ModelError, match="checksum"):
        emotion_models.model_path("test_model", target)
    with pytest.raises(emotion_models.ModelError, match="Unknown model"):
        emotion_models.model_path("nope", target)


def test_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    from app.services import emotion_models

    monkeypatch.setitem(emotion_models.MODEL_FILES, "test_model", ((tmp_path / "absent.onnx").as_uri(), "0" * 64))
    with pytest.raises(emotion_models.ModelError, match="Couldn't download"):
        emotion_models.model_path("test_model", tmp_path / "models")
    assert not list((tmp_path / "models").glob("*"))


def test_unknown_model_setting_is_reported(app, monkeypatch):
    from app.services import emotion_models

    monkeypatch.setattr(emotion_models, "model_path", lambda name, directory, download=True: "unused")
    monkeypatch.setitem(emotion_models._cache, "dummy", None)

    class FakeCV2:
        class FaceDetectorYN:
            @staticmethod
            def create(*args):
                return object()

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCV2)
    cfg = dict(app.config, EMOTION_MODEL="something-else")
    with pytest.raises(emotion_models.ModelError, match="Unknown EMOTION_MODEL"):
        emotion_models.load(cfg)


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
    assert "Expressions:" in page and f"Composure {round(a.emotion_confidence)}" in page
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



def test_video_flags_are_dated_when_they_happened(app, room, ai, fake_emotion, login):
    # absence starts 2 s into the recording, the second face at 10 s (frames every 0.5 s)
    fake_emotion.pattern = "nnnn" + "-" * 12 + "nnnn22nnnn"
    started(room)
    answer_question(room, 0)
    room.frozen.tick(3600)  # analysed an hour later (e.g. re-analysed by HR)
    a = answer_row(app, 0)
    login("hr@test.com").post(f"/hr/answers/{a.id}/emotion")
    with app.app_context():
        from app.models import as_utc

        start = as_utc(a.submitted_at) - timedelta(seconds=a.duration_seconds)
        events = {e.kind: e for e in db.session.get(Interview, room.id).events.all()}
        assert as_utc(events["face_not_visible"].created_at) == start + timedelta(seconds=2)
        assert as_utc(events["multiple_faces"].created_at) == start + timedelta(seconds=10)
        kinds = [e.kind for e in db.session.get(Interview, room.id).events.all()]
        assert kinds.index("face_not_visible") < kinds.index("multiple_faces")  # timeline sorted by time
        assert as_utc(events["multiple_faces"].created_at) < as_utc(a.submitted_at)


def test_reanalysis_replaces_the_video_flags(app, room, ai, fake_emotion, login):
    fake_emotion.pattern = "nnnn" + "-" * 12 + "nnnn22nnnn"
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    hr = login("hr@test.com")
    hr.post(f"/hr/answers/{a.id}/emotion")
    with app.app_context():
        assert InterviewEvent.query.filter_by(kind="multiple_faces").count() == 1
        assert InterviewEvent.query.filter_by(kind="face_not_visible").count() == 1
    fake_emotion.pattern = "nnnnnnnnnnnn"  # e.g. after a fix: nothing to flag any more
    hr.post(f"/hr/answers/{a.id}/emotion")
    with app.app_context():
        assert InterviewEvent.query.filter(InterviewEvent.kind.in_(emotion.VIDEO_FLAGS)).count() == 0
        assert InterviewEvent.query.filter_by(kind="started").count() == 1  # other events untouched


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


# ---------- the real models ----------

def _face_video():
    """A short WebM made from a webcam frame of a real face."""
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
    return out.getvalue()


def _check_real_summary(summary):
    assert summary["status"] == "done" and summary["frames"] == summary["face_frames"] == 6
    assert summary["attentiveness"] == 100 and summary["multi_face_frames"] == 0
    assert all(math.isfinite(summary[k]) for k in ("score", "confidence", "nervousness", "attentiveness"))
    assert sum(summary["distribution"].values()) == pytest.approx(100, abs=0.5)


@pytest.mark.real_emotion_model
def test_real_hsemotion_on_a_real_face(app):
    """The model in use, with the real files (skipped when they haven't been downloaded)."""
    from app.services.emotion_models import model_path, ModelError

    try:
        for name in ("face_detection_yunet_2023mar", app.config["HSEMOTION_MODEL"]):
            model_path(name, app.config["EMOTION_MODEL_DIR"], download=False)
    except ModelError as e:
        pytest.skip(f"model files not downloaded: {e}")
    with app.app_context():
        summary = emotion.analyse_recording(_face_video())
    _check_real_summary(summary)
    assert summary["model"] == "HSEmotion enet_b0_8_best_afew"


@pytest.mark.deepface
def test_real_deepface_baseline_on_a_real_face(app):
    """The paper's DeepFace, still selectable as a baseline (opt-in: venv\\Scripts\\pytest --deepface)."""
    app.config["EMOTION_MODEL"] = "deepface"
    try:
        with app.app_context():
            summary = emotion.analyse_recording(_face_video())
    finally:
        app.config["EMOTION_MODEL"] = "hsemotion"
    _check_real_summary(summary)
    assert summary["model"] == "DeepFace (paper baseline)" and summary["distribution"]["contempt"] == 0
