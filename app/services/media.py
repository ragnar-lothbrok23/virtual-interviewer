"""Audio extraction from recorded answers.

Transcription only needs the sound. Sending Gemini the audio track instead of the whole video is 3-5x faster and much
more consistent (measured on real answers: a 57 s answer took 7-12 s instead of 24-67 s), which keeps the follow-up
decision inside its time limit. The track is copied as-is (no re-encoding, a few milliseconds); the full video is still
stored for HR and for emotion analysis. If anything goes wrong the caller falls back to the full recording.
Voice-emotion analysis decodes the sound to 16 kHz mono samples instead (decode_audio).
"""
import io

from flask import current_app

# codec -> (container, MIME type Gemini accepts)
AUDIO_CONTAINERS = {"opus": ("ogg", "audio/ogg"), "vorbis": ("ogg", "audio/ogg"), "aac": ("adts", "audio/aac")}


def extract_audio(media):
    """(audio bytes, mime type) copied out of a WebM/MP4 recording, or None if it can't be done."""
    try:
        import av
    except ImportError:  # optional speed-up; transcription still works on the full recording
        current_app.logger.warning("PyAV is not installed; transcribing full recordings (slower)")
        return None
    try:
        with av.open(io.BytesIO(media)) as source:
            if not source.streams.audio:
                return None
            audio = source.streams.audio[0]
            target = AUDIO_CONTAINERS.get(audio.codec_context.name)
            if target is None:
                return None
            out = io.BytesIO()
            packets = 0
            with av.open(out, "w", format=target[0]) as dest:
                stream = dest.add_stream_from_template(audio)
                for packet in source.demux(audio):
                    if packet.dts is None:  # the demuxer's final flush packet
                        continue
                    packet.stream = stream
                    dest.mux(packet)
                    packets += 1
            data = out.getvalue()
    except Exception as e:  # corrupt or unusual files: fall back to the full recording
        current_app.logger.warning("Couldn't extract audio (%s); transcribing the full recording", e)
        return None
    if not packets or len(data) < 64:
        return None
    return data, target[1]


def decode_audio(media, rate=16000, max_seconds=None):
    """The recording's sound as mono float32 samples at `rate` Hz (for voice-emotion analysis), or None if it has no
    audio track. Raises ValueError if the audio can't be decoded."""
    import av
    import numpy as np

    chunks, total = [], 0
    limit = int(max_seconds * rate) if max_seconds else None
    try:
        with av.open(io.BytesIO(media)) as source:
            if not source.streams.audio:
                return None
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            for frame in source.decode(source.streams.audio[0]):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
                    total += chunks[-1].size
                if limit and total >= limit:
                    break
            else:
                for out in resampler.resample(None):  # flush what the resampler still holds
                    chunks.append(out.to_ndarray().reshape(-1))
    except Exception as e:
        raise ValueError(f"The recording's audio couldn't be decoded: {e}") from e
    samples = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    return samples[:limit] if limit else samples
