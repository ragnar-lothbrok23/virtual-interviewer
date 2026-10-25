"""Voice emotion (emotion2vec+ on the answer's sound) and how it joins the face reading.

Audio decoding, speech detection, chunking, label mapping, fusion and the HR display run for real; the ONNX model is
replaced by fakes except in the real-model test, which runs only when the exported model file is present."""
import json

import numpy as np
import pytest

from app.models import Interview
from app.services import emotion, emotion_models
from app.services.media import decode_audio
from tests.conftest import emotion_frames
from tests.test_interview_room import api, room, started, upload  # noqa: F401  (fixtures)
from tests.test_media import make_recording
from tests.test_scheduling import booked, get, job_id  # noqa: F401  (fixtures)
from tests.test_scorecard import card, run_interview
from tests.test_scoring import answer_question, answer_row

av = pytest.importorskip("av")
RATE = 16000
E2V_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "<unk>"]


@pytest.fixture(autouse=True)
def ai(interview_ai):
    return interview_ai


def speech(seconds, level=0.1, noise=0.0005, seed=0):
    """A 200 Hz tone (stands in for a voice) over quiet background noise."""
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    return (level * np.sin(2 * np.pi * 200 * t) + rng.normal(0, noise, t.size)).astype(np.float32)


def silence(seconds, noise=0.0005):
    return np.random.default_rng(1).normal(0, noise, int(seconds * RATE)).astype(np.float32)


class FakeVoice:
    """Stands in for the voice model: returns `probs` (with `unclear` given to no emotion) and records each piece."""

    label = "emotion2vec+ test"

    def __init__(self, probs=None, unclear=0.0):
        self.probs = probs or {"neutral": 1.0}
        self.unclear = unclear
        self.pieces = []

    def predict(self, samples):
        self.pieces.append(samples.size / RATE)
        p = {e: self.probs.get(e, 0.0) for e in emotion.EMOTIONS}
        total = sum(p.values())
        return {e: v / total for e, v in p.items()}, self.unclear


def reading(app, samples, clf=None):
    with app.app_context():
        return emotion.voice_reading(samples, clf or FakeVoice(), app.config)


# ---------- speech detection and chunking ----------

def test_speech_is_measured_against_the_background_noise(app):
    cfg = app.config
    assert emotion.speech_seconds(silence(5), RATE, cfg) == 0
    assert emotion.speech_seconds(np.concatenate([silence(3), speech(4)]), RATE, cfg) == pytest.approx(4, abs=0.1)
    # A quiet laptop microphone (speech peaks ~3%) still counts; a hum below VOICE_SPEECH_RMS does not
    assert emotion.speech_seconds(np.concatenate([silence(2), speech(3, level=0.03)]), RATE, cfg) == pytest.approx(3, abs=0.1)
    assert emotion.speech_seconds(speech(5, level=0.002), RATE, cfg) == 0
    assert emotion.speech_seconds(np.zeros(100, np.float32), RATE, cfg) == 0


def test_no_speech_gives_no_voice_reading(app):
    clf = FakeVoice()
    v = reading(app, np.concatenate([silence(10), speech(0.5)]), clf)
    assert v["status"] == "no_speech" and v["speech_seconds"] < 1 and clf.pieces == []


def test_long_answers_are_read_in_pieces_weighted_by_speech(app):
    clf = FakeVoice()
    reading(app, speech(33), clf)
    assert clf.pieces == [15, 18]  # a short 3 s tail joins the piece before it, never read on its own
    clf = FakeVoice()
    reading(app, np.concatenate([speech(15), silence(15), speech(10)]), clf)
    assert clf.pieces == [15, 10]  # the silent piece is skipped

    class ByPiece(FakeVoice):
        def predict(self, samples):
            self.probs = {"sad": 1.0} if len(self.pieces) == 0 else {"neutral": 1.0}
            return super().predict(samples)

    v = reading(app, np.concatenate([speech(5), silence(10), speech(15)]), ByPiece())
    # 5 s of sad speech, 15 s of neutral: averaged by speech, not by piece
    assert v["status"] == "done" and v["distribution"]["sad"] == pytest.approx(25, abs=1)
    assert v["dominant"] == "neutral" and v["nervousness"] == pytest.approx(25, abs=1)


def test_unclear_voice_is_not_used(app):
    v = reading(app, speech(6), FakeVoice({"sad": 1.0}, unclear=0.8))
    assert v["status"] == "unclear" and v["unclear"] == 80
    with app.app_context():
        summary = emotion.fuse(emotion.aggregate(emotion_frames("nnnnnnnnnn"), app.config), v, app.config)
    assert "face" not in summary and summary["nervousness"] < 5 and summary["voice"]["status"] == "unclear"


def test_unusable_model_output_is_a_failed_reading(app):
    class Broken(FakeVoice):
        def predict(self, samples):
            return None

    v = reading(app, speech(6), Broken())
    assert v["status"] == "failed"


# ---------- fusion with the face ----------

def test_voice_joins_the_face_with_its_weight(app):
    with app.app_context():
        face = emotion.aggregate(emotion_frames("nnnnnnnnnn"), app.config)
        before = dict(face)
        voice = emotion.voice_reading(speech(6), FakeVoice({"fear": 1.0}), app.config)
        fused = emotion.fuse(face, voice, app.config)
    w = app.config["VOICE_EMOTION_WEIGHT"]
    assert fused["voice_weight"] == w == 0.1
    assert fused["face"]["score"] == before["score"] and fused["face"]["nervousness"] == before["nervousness"]
    expected_fear = (1 - w) * before["distribution"]["fear"] + w * 100
    assert fused["distribution"]["fear"] == pytest.approx(expected_fear, abs=0.2)
    assert fused["nervousness"] == pytest.approx(before["nervousness"] * (1 - w) + w * 100, abs=0.3)
    assert fused["attentiveness"] == before["attentiveness"]  # attentiveness is the face's alone
    assert fused["score"] == pytest.approx(
        (fused["confidence"] + fused["attentiveness"] + (100 - fused["nervousness"])) / 3, abs=0.1)
    assert fused["score"] < before["score"]
    assert sum(fused["distribution"].values()) == pytest.approx(100, abs=0.5)


@pytest.mark.parametrize("voice", [None, {"status": "no_speech"}, {"status": "unavailable", "reason": "missing"},
                                   {"status": "failed", "reason": "x"}])
def test_without_a_voice_reading_the_face_reading_stands(app, voice):
    with app.app_context():
        face = emotion.aggregate(emotion_frames("nnnhhnnnhn"), app.config)
        score = face["score"]
        fused = emotion.fuse(face, voice, app.config)
    assert fused["score"] == score and "face" not in fused and fused["voice"] == voice


def test_voice_never_creates_a_score_without_a_visible_face(app):
    with app.app_context():
        face = emotion.aggregate(emotion_frames("n---------"), app.config)
        voice = emotion.voice_reading(speech(6), FakeVoice({"happy": 1.0}), app.config)
        fused = emotion.fuse(face, voice, app.config)
    assert fused["status"] == "insufficient" and "score" not in fused and fused["voice"]["status"] == "done"


# ---------- the model wrapper ----------

class FakeSession:
    def __init__(self, logits):
        self.logits, self.inputs = logits, []

    def run(self, outputs, feed):
        self.inputs.append(feed["wav"])
        return [np.array([self.logits], dtype=np.float32)]


def classifier_with(logits):
    clf = emotion_models.VoiceEmotionClassifier.__new__(emotion_models.VoiceEmotionClassifier)
    clf.session, clf.labels, clf.label = FakeSession(logits), E2V_LABELS, "emotion2vec+ base"
    return clf


def test_model_labels_map_to_the_face_classes():
    logits = [0, 0, 3, 0, 1, 2, 0, 0, -9]  # fearful highest, then "other"
    probs, unclear = classifier_with(logits).predict(speech(2))
    assert list(probs) == list(emotion_models.HSEMOTION_CLASSES) and max(probs, key=probs.get) == "fear"
    assert sum(probs.values()) == pytest.approx(1) and probs["contempt"] == 0
    e = np.exp(np.array(logits, float))
    assert unclear == pytest.approx((e[5] + e[8]) / e.sum(), abs=1e-6)  # "other" and "<unk>" carry no emotion


def test_non_finite_model_output_is_rejected():
    assert classifier_with([0, 0, float("nan"), 0, 0, 0, 0, 0, 0]).predict(speech(2)) is None


def test_voice_model_files_are_checked_before_use(tmp_path):
    name = "emotion2vec_plus_base"
    with pytest.raises(emotion_models.ModelError, match="export_voice_model.py"):
        emotion_models.VoiceEmotionClassifier(tmp_path, name)
    model = tmp_path / f"{name}.onnx"
    model.write_bytes(b"model bytes")
    info = tmp_path / f"{name}.json"
    info.write_text("not json", encoding="utf-8")
    with pytest.raises(emotion_models.ModelError, match="not a valid model description"):
        emotion_models.VoiceEmotionClassifier(tmp_path, name)
    info.write_text(json.dumps({"labels": E2V_LABELS, "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(emotion_models.ModelError, match="checksum"):
        emotion_models.VoiceEmotionClassifier(tmp_path, name)
    info.write_text(json.dumps({"labels": ["a", "b"], "sha256": emotion_models._sha256(model)}), encoding="utf-8")
    with pytest.raises(emotion_models.ModelError, match="labels"):
        emotion_models.VoiceEmotionClassifier(tmp_path, name)


# ---------- decoding the recording's sound ----------

def test_recordings_are_decoded_to_16khz_mono(app):
    samples = decode_audio(make_recording(seconds=2))
    assert samples.dtype == np.float32 and abs(samples.size / RATE - 2) < 0.15
    assert np.sqrt(np.mean(samples ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.15)  # the 0.3 tone survives
    mp4 = decode_audio(make_recording(container="mp4", audio_codec="aac"))  # Safari
    assert abs(mp4.size / RATE - 1) < 0.15
    assert decode_audio(make_recording(seconds=3), max_seconds=1).size == RATE
    assert decode_audio(make_recording(audio=False)) is None
    with pytest.raises(ValueError, match="couldn't be decoded"):
        decode_audio(b"not a recording")


def use_voice(monkeypatch, clf):
    monkeypatch.setattr(emotion_models, "load_voice", lambda cfg: clf)


def test_analyse_voice_on_a_real_recording(app, monkeypatch):
    clf = FakeVoice({"happy": 1.0})
    use_voice(monkeypatch, clf)
    with app.app_context():
        v = emotion.analyse_voice(make_recording(seconds=3), app.config)
        assert v["status"] == "done" and v["dominant"] == "happy" and v["model"] == "emotion2vec+ test"
        assert emotion.analyse_voice(make_recording(audio=False), app.config)["status"] == "no_speech"
        assert emotion.analyse_voice(b"garbage", app.config)["status"] == "failed"
        app.config["VOICE_EMOTION_ENABLED"] = False
        try:
            assert emotion.analyse_voice(make_recording(), app.config) is None
        finally:
            app.config["VOICE_EMOTION_ENABLED"] = True


def test_missing_voice_model_leaves_the_face_analysis_alone(app):
    with app.app_context():  # conftest makes load_voice fail, as a missing model file would
        v = emotion.analyse_voice(make_recording(), app.config)
    assert v["status"] == "unavailable" and "disabled in tests" in v["reason"]


def test_voice_is_not_read_when_the_transcript_has_no_speech(app, monkeypatch):
    clf = FakeVoice({"sad": 1.0})
    use_voice(monkeypatch, clf)
    monkeypatch.setattr(emotion, "_load_models", lambda: {"classifier": type("C", (), {"label": "face test"})()})
    monkeypatch.setattr(emotion, "sample_frames", lambda media, fps, n: [(i / 2, None) for i in range(10)])
    frames = iter(emotion_frames("nnnnnnnnnn"))
    monkeypatch.setattr(emotion, "analyse_frame", lambda image, models, cfg: next(frames)[1])
    with app.app_context():
        summary = emotion.analyse_recording(make_recording(seconds=3), speech=False)
    assert summary["voice"]["status"] == "no_speech" and clf.pieces == [] and "face" not in summary


# ---------- in the interview ----------

def test_hr_sees_the_voice_reading(app, room, ai, fake_emotion, login):
    with app.app_context():
        fake_emotion.voice = emotion.voice_reading(speech(6), FakeVoice({"fear": 0.6, "neutral": 0.4}), app.config)
    run_interview(room, ai)
    a = answer_row(app, 0)
    assert a.emotion["voice"]["status"] == "done" and a.emotion["face"]["score"] > a.emotion_score
    application_id = get(app, Interview, room.id).application_id
    client = login("hr@test.com")
    page = client.get(f"/hr/applications/{application_id}").get_data(as_text=True)
    assert "Voice:" in page and "mostly fear" in page and "included at 10%" in page
    assert f"face alone: emotion {round(a.emotion['face']['score'])}" in page
    info = card(app, room).details["emotion_info"]
    used = info["counts"]["used"]
    assert used >= 1 and info["voice"] == {"answers": used, "weight": 0.1, "nervousness": 60.0,
                                           "models": ["emotion2vec+ test"]}
    page = " ".join(client.get(f"/hr/interviews/{room.id}/scorecard").get_data(as_text=True).split())
    assert f"Voice (emotion2vec+ test) included at 10% in {used} of {used} answer(s); vocal nervousness 60" in page
    pdf = client.get(f"/hr/interviews/{room.id}/scorecard.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")


def test_hr_sees_why_there_is_no_voice_reading(app, room, ai, fake_emotion, login):
    fake_emotion.voice = {"status": "unavailable", "reason": "Voice model is missing"}
    run_interview(room, ai)
    client = login("hr@test.com")
    text = client.get(f"/hr/applications/{get(app, Interview, room.id).application_id}").get_data(as_text=True)
    assert "Voice:" in text and "not available; face only" in text
    page = " ".join(client.get(f"/hr/interviews/{room.id}/scorecard").get_data(as_text=True).split())
    assert "Voice not included (no usable voice reading): face only." in page
    assert card(app, room).details["emotion_info"]["voice"]["answers"] == 0


def test_silent_answer_never_shows_a_voice_emotion(app, room, ai, fake_emotion, login):
    ai.transcript = ""  # the transcription finds no speech
    with app.app_context():
        fake_emotion.voice = emotion.voice_reading(speech(6), FakeVoice({"sad": 1.0}), app.config)
    started(room)
    answer_question(room, 0)
    a = answer_row(app, 0)
    assert a.speech_detected is False
    text = login("hr@test.com").get(f"/hr/applications/{get(app, Interview, room.id).application_id}").get_data(
        as_text=True)
    assert "mostly sad" not in text and "no speech in the transcript" in text


def test_download_command_checks_the_voice_model(app, monkeypatch, tmp_path):
    monkeypatch.setitem(app.config, "EMOTION_MODEL_DIR", tmp_path)  # no voice model exported here
    monkeypatch.setattr(emotion_models, "model_path", lambda name, directory, download=True: f"{directory}/{name}")
    result = app.test_cli_runner().invoke(args=["download-emotion-models"])
    assert result.exit_code != 0 and "export_voice_model.py" in result.output and "VOICE_EMOTION_ENABLED=0" in result.output
    monkeypatch.setattr(emotion_models.VoiceEmotionClassifier, "__init__", lambda self, directory, name: None)
    result = app.test_cli_runner().invoke(args=["download-emotion-models"])
    assert result.exit_code == 0 and "emotion2vec_plus_base: checksum OK" in result.output


@pytest.mark.real_emotion_model
def test_real_voice_model(app):
    """The exported emotion2vec+ model (skipped when it hasn't been exported)."""
    try:
        clf = emotion_models.VoiceEmotionClassifier(app.config["EMOTION_MODEL_DIR"], app.config["VOICE_EMOTION_MODEL"])
    except emotion_models.ModelError as e:
        pytest.skip(f"voice model not exported: {e}")
    with app.app_context():
        v = emotion.voice_reading(np.concatenate([speech(20), silence(3)]), clf, app.config)
    assert v["status"] in ("done", "unclear") and v["model"] == "emotion2vec+ base"
    assert sum(v["distribution"].values()) == pytest.approx(100, abs=0.5)
