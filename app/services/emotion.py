"""Facial emotion analysis of recorded answers with DeepFace (paper Section 3.2).

Per answer, in a separate background worker (so it never slows down transcription or follow-up decisions):

1. Frames are sampled from the recording (EMOTION_SAMPLE_FPS) with PyAV.
2. Faces are found with YuNet (one of DeepFace's detectors) on a histogram-normalised copy of the frame (CLAHE on
   brightness, paper preprocessing). YuNet's default 0.9 threshold missed faces in a typical dim home webcam (12/60
   frames); at 0.7 it found 60/60 with no false faces on face-masked, blank or noise frames. Its landmarks give face
   presence, a second person in view, and head turn (attentiveness).
3. DeepFace's emotion model reads the face through DeepFace's own crop and face alignment (paper preprocessing) on the
   original frame, with a 10% margin around the face (EMOTION_FACE_EXPAND). Normalising the brightness first made the
   emotion readings worse (it helps detection only). On a labelled test of posed expressions the tight crop read 44%
   of frames correctly and missed most smiles; the 10% margin read 57% correctly (smiles 82%).
4. Temporal smoothing (exponential moving average across frames) and whole-answer averages remove frame-to-frame
   flicker; a reliability label tells HR when the readings were unstable.

Measured on a real interview (neutral face, glasses, backlit room): the emotion model labelled near-identical frames
anywhere from "angry" to "happy" and changed its top label between consecutive frames about half the time. Hence the
averaging, the reliability label and the caveats shown to HR.

Signals the paper names (confidence, nervousness, attentiveness; 0-100):
- confidence   = share of composed expressions (neutral + happy)
- nervousness  = share of fear + sad
- attentiveness = share of sampled frames with the face visible and turned towards the camera
The emotion score combines confidence and attentiveness (EMOTION_WEIGHTS). The mapping from DeepFace's 7 emotions to
these signals is an implementation decision (the paper doesn't define it). Facial expression models are sensitive to
lighting, camera quality and individual/cultural differences, so: answers where the face is visible in too few frames
get no emotion score ("insufficient") instead of a low one, the emotion part is only 15% of the final score (paper
Eq. 1), and HR is shown these caveats.
"""
import io
import math
import threading
from datetime import timedelta

import numpy as np
from flask import current_app

from ..extensions import db
from ..models import InterviewAnswer, InterviewEvent, as_utc, utcnow

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
COMPOSED = ("neutral", "happy")
VIDEO_FLAGS = ("multiple_faces", "face_not_visible")  # integrity events derived from this analysis
NERVOUS = ("fear", "sad")


class EmotionError(RuntimeError):
    """The recording can't be analysed (unreadable video, model unavailable)."""


# ---------- models (loaded once per worker process) ----------

_models = None
_models_lock = threading.Lock()
_predict_lock = threading.Lock()  # TensorFlow models are not guaranteed thread-safe


def _load_models():
    global _models
    with _models_lock:
        if _models is None:
            import os

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            try:
                from deepface import DeepFace
                from deepface.models.face_detection.YuNet import YuNetClient
            except ImportError as e:
                raise EmotionError(f"DeepFace is not installed: {e}") from e
            # DeepFace reads YuNet's threshold from this variable (the worker process is dedicated to this)
            os.environ["yunet_score_threshold"] = str(current_app.config["EMOTION_DETECTOR_THRESHOLD"])
            detector = YuNetClient().model  # OpenCV's FaceDetectorYN with DeepFace's YuNet weights
            blank = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(blank, actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)
            _models = {"deepface": DeepFace, "detector": detector}
    return _models


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
    with _predict_lock:
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
    probs = None
    for source in (image, prepared):  # the original frame first; the normalised one if DeepFace finds no face in it
        try:
            with _predict_lock:
                faces = models["deepface"].analyze(source, actions=["emotion"], detector_backend="yunet", align=True,
                                                   enforce_detection=True, silent=True,
                                                   expand_percentage=cfg["EMOTION_FACE_EXPAND"])
        except ValueError:
            continue
        face = max(faces, key=lambda f: _overlap(f["region"], main))  # the same face the detector chose
        raw = {e: float(face["emotion"].get(e, 0.0)) for e in EMOTIONS}
        total = sum(raw.values())
        if total > 0 and all(math.isfinite(v) for v in raw.values()):
            probs = {e: raw[e] / total for e in EMOTIONS}
        break
    return {"faces": 1 + len(others), "probs": probs, "yaw": yaw_ratio(main)}


def _overlap(region, row):
    x, y, w, h = (float(v) for v in row[:4])
    ix = max(0.0, min(region["x"] + region["w"], x + w) - max(region["x"], x))
    iy = max(0.0, min(region["y"] + region["h"], y + h) - max(region["y"], y))
    return ix * iy


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
    confidence = round(100 * sum(mean[e] for e in COMPOSED), 1)
    nervousness = round(100 * sum(mean[e] for e in NERVOUS), 1)
    attentiveness = round(100 * len(facing) / n, 1)
    w = cfg["EMOTION_WEIGHTS"]
    score = round(w["confidence"] * confidence + w["attentiveness"] * attentiveness, 1)
    for value in (confidence, nervousness, attentiveness, score):
        if not math.isfinite(value):
            raise EmotionError("The emotion signals couldn't be computed.")
    reliability = ("low" if flip_rate > cfg["EMOTION_UNSTABLE_FLIP_RATE"] or face_ratio < 0.8 else
                   "medium" if flip_rate > cfg["EMOTION_UNSTABLE_FLIP_RATE"] / 2 else "high")
    summary.update(status="done", reliability=reliability, distribution={e: round(mean[e] * 100, 1) for e in EMOTIONS},
                   dominant=max(mean, key=mean.get), confidence=confidence, nervousness=nervousness,
                   attentiveness=attentiveness, score=score)
    return summary


def _downsample(points, limit):
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)]


def analyse_recording(media):
    """Run the full pipeline on one recording."""
    cfg = current_app.config
    frames = sample_frames(media, cfg["EMOTION_SAMPLE_FPS"], cfg["EMOTION_MAX_FRAMES"])
    if not frames:
        raise EmotionError("The recording has no video frames.")
    models = _load_models()
    return aggregate([(t, analyse_frame(image, models, cfg)) for t, image in frames], cfg)


# ---------- task body ----------

def run_emotion(answer_id):
    """Analyse one answer and store the result; never raises (emotion analysis is best-effort)."""
    from . import storage
    from .interview_room import log_event

    answer = db.session.get(InterviewAnswer, answer_id)
    if answer is None or not answer.media_path:
        return None
    try:
        summary = analyse_recording(storage.read_resume(answer.media_path))
    except Exception as e:  # best-effort: any failure is recorded on the answer, never raised
        db.session.rollback()
        answer = db.session.get(InterviewAnswer, answer_id)
        message = str(e) if isinstance(e, (EmotionError, storage.StorageError)) else f"Unexpected error: {e}"
        answer.emotion_status, answer.emotion_error = InterviewAnswer.EMOTION_FAILED, message
        answer.emotion_at = utcnow()
        db.session.commit()
        current_app.logger.error("Emotion analysis of answer %s failed: %s", answer_id, e)
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
    return {"answers": len(done), "analysed": len(analysed), "score": avg("emotion_score"),
            "confidence": avg("emotion_confidence"), "nervousness": avg("emotion_nervousness"),
            "attentiveness": avg("emotion_attentiveness"), "dominant": max(counts, key=counts.get)}
