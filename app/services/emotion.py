"""Facial and vocal emotion analysis of recorded answers (paper Section 3.2).

Per answer, in a separate background worker (so it never slows down transcription or follow-up decisions):

1. Frames are sampled from the recording (EMOTION_SAMPLE_FPS) with PyAV.
2. Faces are found with YuNet on a histogram-normalised copy of the frame (CLAHE on brightness, paper preprocessing).
   YuNet's default 0.9 threshold missed faces in a typical dim home webcam (12/60 frames); at 0.7 it found 60/60 with
   no false faces on face-masked, blank or noise frames. Its landmarks give face presence, a second person in view,
   and head turn (attentiveness).
3. The expression model reads the face from the original frame, aligned (eyes level, paper preprocessing) with a 10%
   margin (EMOTION_FACE_EXPAND). The model is HSEmotion (AffectNet-trained, ONNX Runtime): it replaced the paper's
   DeepFace after a labelled test on the candidate's own webcam (86% vs 57% of frames read correctly, macro-F1 0.74 vs
   0.26; see services/emotion_models.py). DeepFace remains selectable as the paper baseline (EMOTION_MODEL=deepface).
4. Temporal smoothing (exponential moving average across frames) and whole-answer averages remove frame-to-frame
   flicker; a reliability label tells HR when the readings were unstable.
5. Voice (the paper's audio modality): emotion2vec+ reads the emotion in the answer's sound, in 15-second pieces
   (VOICE_CHUNK_SECONDS) that contain speech. Its reading joins the face's expressions with a small weight
   (VOICE_EMOTION_WEIGHT, 10%): in the labelled test the voice models recognised little emotion in natural speech
   and more voice made the combined reading worse. Without usable speech, or if the voice model isn't available, the
   reading is the face's alone; the face-only numbers are kept in the summary ("face") for comparison.

Signals the paper names (confidence, nervousness, attentiveness; 0-100):
- confidence   = share of composed expressions (neutral + happy), shown to HR as "composure"
- nervousness  = share of fear + sad
- attentiveness = share of sampled frames with the face visible and turned towards the camera
The emotion score combines the paper's three indicators (§3.4): confidence, engagement (attentiveness) and stability
(100 - nervousness), a third each (EMOTION_WEIGHTS). The mapping from the model's expressions
to these signals is an implementation decision (the paper doesn't define it). Expression models are sensitive to
lighting, camera quality and individual/cultural differences, so: answers where the face is visible in too few frames
get no emotion score ("insufficient") instead of a low one, the emotion part is only 15% of the final score (paper
Eq. 1), and HR is shown these caveats.
"""
import io
import math
from datetime import timedelta

import numpy as np
from flask import current_app

from ..extensions import db
from ..models import InterviewAnswer, InterviewEvent, as_utc, utcnow

from .emotion_models import HSEMOTION_CLASSES as EMOTIONS  # angry, contempt, disgust, fear, happy, neutral, ...
COMPOSED = ("neutral", "happy")
VIDEO_FLAGS = ("multiple_faces", "face_not_visible")  # integrity events derived from this analysis
NERVOUS = ("fear", "sad")


class EmotionError(RuntimeError):
    """The recording can't be analysed (unreadable video, model unavailable)."""


# ---------- models (loaded once per worker process) ----------

def _load_models():
    from .emotion_models import ModelError, load

    try:
        return load(current_app.config)
    except ModelError as e:
        raise EmotionError(str(e)) from e
    except ImportError as e:
        raise EmotionError(f"The emotion model's library is not installed: {e}") from e


# ---------- per frame ----------

def sample_frames(media, fps, max_frames):
    """[(seconds, BGR image)] sampled at `fps` from a WebM/MP4 recording."""
    import av

    every = 1.0 / fps
    out = []
    try:
        with av.open(io.BytesIO(media)) as container:
            if not container.streams.video:
                raise EmotionError("The recording has no video track.")
            next_t = 0.0
            for frame in container.decode(container.streams.video[0]):
                t = float(frame.time or 0.0)
                if t + 1e-6 < next_t:
                    continue
                out.append((round(t, 2), frame.to_ndarray(format="bgr24")))
                next_t = t + every
                if len(out) >= max_frames:
                    break
    except EmotionError:
        raise
    except Exception as e:
        raise EmotionError(f"The recording couldn't be decoded: {e}") from e
    return out


def equalise(image):
    """Histogram normalisation of brightness (CLAHE), as in the paper's preprocessing."""
    import cv2

    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def yaw_ratio(row):
    """Horizontal offset of the nose tip from the midpoint of the eyes, in eye distances: about 0 when facing the
    camera, growing as the head turns away."""
    (rx, ry), (lx, ly), (nx, _) = row[4:6], row[6:8], row[8:10]
    eye_distance = math.hypot(lx - rx, ly - ry)
    if eye_distance < 1:
        return None
    return abs(nx - (rx + lx) / 2) / eye_distance


def analyse_frame(image, models, cfg):
    """{"faces": n, "probs": {emotion: 0-1} or None, "yaw": float or None} for one frame."""
    detector = models["detector"]
    prepared = equalise(image)
    height, width = prepared.shape[:2]
    with models["detect_lock"]:  # one OpenCV detector object per process
        detector.setInputSize((width, height))
        detector.setScoreThreshold(cfg["EMOTION_DETECTOR_THRESHOLD"])
        _, rows = detector.detect(prepared)
    rows = [] if rows is None else [r for r in rows if r[2] > 0 and r[3] > 0]
    if not rows:
        return {"faces": 0, "probs": None, "yaw": None}
    rows.sort(key=lambda r: r[2] * r[3], reverse=True)
    main = rows[0]
    # A second person: a clearly detected face of comparable size (small or weak detections are ignored)
    others = [r for r in rows[1:] if r[-1] >= cfg["EMOTION_SECOND_FACE_CONFIDENCE"] and r[2] >= 0.4 * main[2]]
    probs = models["classifier"].predict(image, prepared, main, cfg)
    if probs is not None and not all(math.isfinite(v) for v in probs.values()):
        probs = None
    return {"faces": 1 + len(others), "probs": probs, "yaw": yaw_ratio(main)}


# ---------- per answer (pure: testable without TensorFlow) ----------

def aggregate(frames, cfg):
    """Summary of one answer from per-frame results [(seconds, frame result)]."""
    n = len(frames)
    with_face = [(t, f) for t, f in frames if f["faces"] >= 1 and f["probs"]]
    facing = [1 for _, f in with_face if f["yaw"] is not None and f["yaw"] <= cfg["EMOTION_FACING_YAW"]]
    multi = sum(1 for _, f in frames if f["faces"] >= 2)

    # How stable the raw readings were: share of consecutive face frames whose top emotion changed
    tops = [max(f["probs"], key=f["probs"].get) for _, f in with_face]
    flip_rate = (sum(a != b for a, b in zip(tops, tops[1:])) / (len(tops) - 1)) if len(tops) > 1 else 0.0

    # Temporal smoothing (paper 3.2): exponential moving average of the probabilities over consecutive face frames
    alpha, smoothed, timeline, current = cfg["EMOTION_SMOOTHING"], [], [], None
    for t, f in with_face:
        current = dict(f["probs"]) if current is None else {
            e: alpha * f["probs"][e] + (1 - alpha) * current[e] for e in EMOTIONS}
        smoothed.append(current)
        timeline.append([t, max(current, key=current.get)])

    # Longest stretch without a visible face
    longest, start, absence_at = 0.0, None, None
    step = (frames[1][0] - frames[0][0]) if n > 1 else 0.0
    for t, f in frames:
        if f["faces"] == 0 or not f["probs"]:
            start = t if start is None else start
            if t - start + step > longest:
                longest, absence_at = t - start + step, start
        else:
            start = None
    multi_at = next((t for t, f in frames if f["faces"] >= 2), None)

    summary = {"frames": n, "face_frames": len(with_face), "facing_frames": len(facing), "multi_face_frames": multi,
               "longest_absence_seconds": round(longest, 1), "flip_rate": round(flip_rate, 2),
               "multi_face_at": multi_at, "absence_at": absence_at,  # seconds into the recording
               "timeline": _downsample(timeline, 60)}
    face_ratio = len(with_face) / n if n else 0.0
    if n < cfg["EMOTION_MIN_FRAMES"] or face_ratio < cfg["EMOTION_MIN_FACE_RATIO"]:
        summary.update(status="insufficient", reason=(
            "too few frames" if n < cfg["EMOTION_MIN_FRAMES"] else
            f"face visible in only {round(face_ratio * 100)}% of frames"))
        return summary

    mean = {e: float(np.mean([s[e] for s in smoothed])) for e in EMOTIONS}
    reliability = ("low" if flip_rate > cfg["EMOTION_UNSTABLE_FLIP_RATE"] or face_ratio < 0.8 else
                   "medium" if flip_rate > cfg["EMOTION_UNSTABLE_FLIP_RATE"] / 2 else "high")
    summary.update(status="done", reliability=reliability, **signals(mean, round(100 * len(facing) / n, 1), cfg))
    return summary


def signals(mean, attentiveness, cfg):
    """The paper's emotion indicators and the emotion score from mean expression probabilities {emotion: 0-1}."""
    confidence = round(100 * sum(mean[e] for e in COMPOSED), 1)
    nervousness = round(100 * sum(mean[e] for e in NERVOUS), 1)
    stability = round(100 - nervousness, 1)  # emotional stability: calm rather than fearful or sad
    w = cfg["EMOTION_WEIGHTS"]
    score = round(w["confidence"] * confidence + w["attentiveness"] * attentiveness + w["stability"] * stability, 1)
    for value in (confidence, nervousness, attentiveness, stability, score):
        if not math.isfinite(value):
            raise EmotionError("The emotion signals couldn't be computed.")
    return {"distribution": {e: round(mean[e] * 100, 1) for e in EMOTIONS}, "dominant": max(mean, key=mean.get),
            "confidence": confidence, "nervousness": nervousness, "attentiveness": attentiveness,
            "stability": stability, "score": score}


# ---------- voice ----------

def _window_levels(samples, rate):
    window = int(0.03 * rate)
    if samples.size < window:
        return np.zeros(0)
    frames = samples[:samples.size // window * window].reshape(-1, window)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def noise_floor(samples, rate, cfg):
    """Background noise level: the quietest moments of the answer, capped (someone talking without pauses would
    otherwise raise the "noise" to the level of their own voice)."""
    levels = _window_levels(samples, rate)
    return min(float(np.percentile(levels, 10)), cfg["VOICE_NOISE_FLOOR_MAX"]) if levels.size else 0.0


def speech_seconds(samples, rate, cfg, floor=None):
    """Seconds of speech in a piece of audio: 30 ms windows clearly louder than the background noise."""
    levels = _window_levels(samples, rate)
    if not levels.size:
        return 0.0
    floor = noise_floor(samples, rate, cfg) if floor is None else floor
    return float(np.sum(levels > max(cfg["VOICE_SPEECH_RMS"], 3 * floor))) * 0.03


def voice_reading(samples, classifier, cfg, rate=16000):
    """The voice's emotion over an answer: the model reads each VOICE_CHUNK_SECONDS piece that has speech, and the
    pieces are averaged by how much speech they contain."""
    floor = noise_floor(samples, rate, cfg)  # measured over the whole answer, used for every piece
    total_speech = speech_seconds(samples, rate, cfg, floor)
    if total_speech < cfg["VOICE_MIN_SPEECH_SECONDS"]:
        return {"status": "no_speech", "speech_seconds": round(total_speech, 1)}
    size = int(cfg["VOICE_CHUNK_SECONDS"] * rate)
    starts = list(range(0, samples.size, size))
    if len(starts) > 1 and samples.size - starts[-1] < size // 3:  # a short tail joins the piece before it
        starts.pop()
    readings = []
    for i, start in enumerate(starts):
        piece = samples[start:starts[i + 1] if i + 1 < len(starts) else samples.size]
        weight = speech_seconds(piece, rate, cfg, floor)
        if weight < 0.5:
            continue
        result = classifier.predict(piece)
        if result is not None:
            readings.append((weight, *result))
    if not readings:
        return {"status": "failed", "reason": "the voice model gave no usable reading"}
    total = sum(w for w, _, _ in readings)
    mean = {e: sum(w * p[e] for w, p, _ in readings) / total for e in EMOTIONS}
    unclear = sum(w * u for w, _, u in readings) / total
    result = {"speech_seconds": round(total_speech, 1), "unclear": round(unclear * 100, 1),
              "distribution": {e: round(v * 100, 1) for e, v in mean.items()}, "dominant": max(mean, key=mean.get),
              "nervousness": round(100 * sum(mean[e] for e in NERVOUS), 1), "model": classifier.label}
    if unclear > cfg["VOICE_MAX_UNCLEAR"]:
        result.update(status="unclear", reason="the voice model couldn't tell an emotion")
    else:
        result["status"] = "done"
    return result


def analyse_voice(media, cfg):
    """Voice reading of one recording; never raises (the face analysis goes ahead without it)."""
    from .emotion_models import ModelError, load_voice
    from .media import decode_audio

    if not cfg["VOICE_EMOTION_ENABLED"]:
        return None
    try:
        classifier = load_voice(cfg)
    except (ModelError, ImportError) as e:
        current_app.logger.warning("Voice emotion is unavailable: %s", e)
        return {"status": "unavailable", "reason": str(e)}
    try:
        samples = decode_audio(media, 16000, cfg["VOICE_MAX_SECONDS"])
        if samples is None:
            return {"status": "no_speech", "reason": "the recording has no sound track", "speech_seconds": 0.0}
        return voice_reading(samples, classifier, cfg)
    except Exception as e:  # noqa: BLE001 - voice is a supporting signal: record the problem, keep the face reading
        current_app.logger.error("Voice emotion analysis failed: %s", e)
        return {"status": "failed", "reason": str(e)}


def fuse(summary, voice, cfg):
    """Add the voice's reading to the face's: expression probabilities are combined (VOICE_EMOTION_WEIGHT for the
    voice) and the indicators recomputed. The face-only numbers stay under "face"."""
    summary["voice"] = voice
    if summary.get("status") != "done" or not voice or voice.get("status") != "done":
        return summary
    w = cfg["VOICE_EMOTION_WEIGHT"]
    face = {k: summary[k] for k in ("distribution", "dominant", "confidence", "nervousness", "stability", "score")}
    mean = {e: (1 - w) * face["distribution"][e] / 100 + w * voice["distribution"][e] / 100 for e in EMOTIONS}
    summary.update(signals(mean, summary["attentiveness"], cfg), face=face, voice_weight=w)
    return summary


def _downsample(points, limit):
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)]


def analyse_recording(media, speech=None):
    """Run the full pipeline on one recording. `speech` is the transcription's verdict when known: if it found no
    speech, the voice isn't read (loud noise without words would otherwise be read as an emotion)."""
    cfg = current_app.config
    frames = sample_frames(media, cfg["EMOTION_SAMPLE_FPS"], cfg["EMOTION_MAX_FRAMES"])
    if not frames:
        raise EmotionError("The recording has no video frames.")
    models = _load_models()
    summary = aggregate([(t, analyse_frame(image, models, cfg)) for t, image in frames], cfg)
    summary["model"] = models["classifier"].label  # which model produced these numbers
    if speech is False:
        voice = {"status": "no_speech", "reason": "no speech in the transcript", "speech_seconds": 0.0}
    else:
        voice = analyse_voice(media, cfg)
    return fuse(summary, voice, cfg)


# ---------- task body ----------

def run_emotion(answer_id):
    """Analyse one answer and store the result; never raises (emotion analysis is best-effort)."""
    from . import storage
    from .interview_room import log_event

    answer = db.session.get(InterviewAnswer, answer_id)
    if answer is None or not answer.media_path:
        return None
    try:
        summary = analyse_recording(storage.read_resume(answer.media_path), speech=answer.speech_detected)
    except Exception as e:  # best-effort: any failure is recorded on the answer, never raised
        db.session.rollback()
        answer = db.session.get(InterviewAnswer, answer_id)
        message = str(e) if isinstance(e, (EmotionError, storage.StorageError)) else f"Unexpected error: {e}"
        answer.emotion_status, answer.emotion_error = InterviewAnswer.EMOTION_FAILED, message
        answer.emotion_at = utcnow()
        db.session.commit()
        current_app.logger.error("Emotion analysis of answer %s failed: %s", answer_id, e)
        from .scorecard import refresh

        refresh(answer.interview)
        return answer.emotion_status
    answer.emotion = summary
    answer.emotion_status = summary["status"]
    answer.emotion_error = summary.get("reason")
    answer.emotion_score = summary.get("score")
    answer.emotion_confidence = summary.get("confidence")
    answer.emotion_nervousness = summary.get("nervousness")
    answer.emotion_attentiveness = summary.get("attentiveness")
    answer.emotion_at = utcnow()

    cfg = current_app.config
    interview = answer.interview
    # Re-analysis replaces this answer's earlier video flags instead of adding duplicates
    InterviewEvent.query.filter(InterviewEvent.interview_id == interview.id,
                                InterviewEvent.position == answer.position,
                                InterviewEvent.kind.in_(VIDEO_FLAGS)).delete(synchronize_session=False)
    if summary["multi_face_frames"] >= cfg["EMOTION_MULTI_FACE_MIN_FRAMES"]:
        event = log_event(interview, "multiple_faces", answer.position,
                          seconds=round(summary["multi_face_frames"] / cfg["EMOTION_SAMPLE_FPS"], 1))
        _place_in_interview(event, answer, summary.get("multi_face_at"))
    if summary["longest_absence_seconds"] >= cfg["EMOTION_ABSENCE_FLAG_SECONDS"]:
        event = log_event(interview, "face_not_visible", answer.position, seconds=summary["longest_absence_seconds"])
        _place_in_interview(event, answer, summary.get("absence_at"))
    db.session.commit()
    from .scorecard import refresh

    refresh(interview)
    return answer.emotion_status


def _place_in_interview(event, answer, offset_seconds):
    """Video flags are found after the answer was recorded: date them when they happened in the interview (recording
    start + offset), not when the analysis ran, so the timeline shows them in order."""
    if event is None or answer.submitted_at is None:
        return
    started = as_utc(answer.submitted_at) - timedelta(seconds=answer.duration_seconds or 0)
    event.created_at = started + timedelta(seconds=offset_seconds or 0)


def interview_emotion_summary(interview):
    """Averages over the answers with an emotion score (Phase 9 fuses this into the final score)."""
    done = [a for a in interview.answers if a.emotion_status == InterviewAnswer.EMOTION_DONE]
    analysed = [a for a in interview.answers if a.emotion_status in (InterviewAnswer.EMOTION_DONE,
                                                                     InterviewAnswer.EMOTION_INSUFFICIENT)]
    if not done:
        return {"answers": 0, "analysed": len(analysed)} if analysed else None
    avg = lambda attr: round(sum(getattr(a, attr) for a in done) / len(done), 1)  # noqa: E731
    counts = {}
    for a in done:
        counts[a.emotion["dominant"]] = counts.get(a.emotion["dominant"], 0) + 1
    models = {a.emotion.get("model") for a in done}
    return {"answers": len(done), "analysed": len(analysed), "score": avg("emotion_score"),
            "model": models.pop() if len(models) == 1 else None,
            "confidence": avg("emotion_confidence"), "nervousness": avg("emotion_nervousness"),
            "attentiveness": avg("emotion_attentiveness"), "dominant": max(counts, key=counts.get)}
