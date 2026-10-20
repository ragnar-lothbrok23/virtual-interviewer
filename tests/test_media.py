"""Audio extraction for transcription: real WebM/MP4 files are generated with PyAV, so the actual demuxing runs."""
import io
from fractions import Fraction

import numpy as np
import pytest

from app.extensions import db
from app.models import InterviewAnswer
from app.services import media as media_service, storage
from tests.test_interview_room import api, room, started, upload  # noqa: F401  (fixtures)
from tests.test_scheduling import booked, get, job_id  # noqa: F401  (fixtures)

av = pytest.importorskip("av")


def make_recording(container="webm", audio_codec="libopus", video=True, audio=True, seconds=1.0):
    """A small real recording: a 440 Hz tone and/or plain video frames."""
    out = io.BytesIO()
    with av.open(out, "w", format=container) as f:
        v_stream = a_stream = None
        if video:
            v_stream = f.add_stream("libvpx" if container == "webm" else "libx264", rate=10)
            v_stream.width, v_stream.height, v_stream.pix_fmt = 64, 48, "yuv420p"
        if audio:
            a_stream = f.add_stream(audio_codec, rate=48000)
            a_stream.layout = "mono"
        if v_stream:
            for i in range(int(seconds * 10)):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8), format="rgb24")
                for packet in v_stream.encode(frame):
                    f.mux(packet)
            for packet in v_stream.encode():
                f.mux(packet)
        if a_stream:
            samples = a_stream.codec_context.frame_size or 960
            t = 0
            while t < seconds * 48000:
                wave = (np.sin(2 * np.pi * 440 * (np.arange(samples) + t) / 48000) * 0.3).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(wave.reshape(1, -1), format="flt", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = 48000, t, Fraction(1, 48000)
                for packet in a_stream.encode(frame):
                    f.mux(packet)
                t += samples
            for packet in a_stream.encode():
                f.mux(packet)
    return out.getvalue()


def audio_info(data):
    with av.open(io.BytesIO(data)) as f:
        stream = f.streams.audio[0]
        return f.format.name, stream.codec_context.name, len(f.streams.video), float(f.duration or 0) / 1e6


def test_opus_track_is_copied_out_of_webm(app):
    recording = make_recording(seconds=2)
    with app.app_context():
        audio, mime = media_service.extract_audio(recording)
    assert mime == "audio/ogg" and len(audio) < len(recording)
    container, codec, videos, duration = audio_info(audio)
    assert container == "ogg" and codec == "opus" and videos == 0 and 1.5 < duration < 2.5


def test_aac_track_is_copied_out_of_mp4(app):
    """Safari records MP4 with AAC audio."""
    recording = make_recording(container="mp4", audio_codec="aac")
    with app.app_context():
        audio, mime = media_service.extract_audio(recording)
    assert mime == "audio/aac" and audio_info(audio)[1] == "aac"


@pytest.mark.parametrize("data", [make_recording(audio=False), b"\x1a\x45\xdf\xa3" + b"\x00" * 200, b""])
def test_unusable_files_fall_back_to_the_full_recording(app, data):
    with app.app_context():
        assert media_service.extract_audio(data) is None


def test_missing_pyav_falls_back(app, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_av(name, *args, **kwargs):
        if name == "av":
            raise ImportError("no av")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_av)
    with app.app_context():
        assert media_service.extract_audio(make_recording()) is None


def test_transcription_sends_only_the_audio(app, room, interview_ai, monkeypatch):
    seen = []
    original = interview_ai.transcribe

    def spy(content, mime_type):
        seen.append((len(content), mime_type))
        return original(content, mime_type)

    from app.services import gemini

    monkeypatch.setattr(gemini, "transcribe", spy)
    started(room)
    recording = make_recording(seconds=2)
    assert api(room, "ask", {"position": 0}).status_code == 200
    room.frozen.tick(40)
    assert upload(room, 0, data=recording, duration="40").status_code == 200
    assert seen and seen[0][1] == "audio/ogg" and seen[0][0] < len(recording)
    with app.app_context():
        answer = InterviewAnswer.query.filter_by(position=0).one()
        assert answer.media_type == "video/webm"  # the stored recording is still the full video
        assert storage.read_resume(answer.media_path) == recording


def test_transcription_falls_back_to_the_video(app, room, interview_ai, monkeypatch):
    seen = []
    original = interview_ai.transcribe
    from app.services import gemini

    monkeypatch.setattr(gemini, "transcribe", lambda content, mime: seen.append(mime) or original(content, mime))
    started(room)
    assert api(room, "ask", {"position": 0}).status_code == 200
    room.frozen.tick(40)
    assert upload(room, 0, duration="40").status_code == 200  # a minimal fake WebM without a real audio track
    assert seen == ["video/webm"]
