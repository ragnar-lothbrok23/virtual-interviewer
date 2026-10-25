"""Model files and classifiers for facial emotion analysis.

- Face detection: YuNet (OpenCV's FaceDetectorYN), as before.
- Expressions: HSEmotion (EfficientNet trained on AffectNet, run with ONNX Runtime) by default. It replaced the paper's
  DeepFace after a labelled test on the candidate's own webcam: 86% of frames read correctly vs 57% for DeepFace,
  macro-F1 0.74 vs 0.26, every expression recognised at least sometimes. DeepFace can still be selected
  (EMOTION_MODEL=deepface) as the paper baseline for evaluation; it is only imported then.

- Voice: emotion2vec+ base (speech emotion model, ONNX Runtime), chosen over two wav2vec2 models and Gemini in a
  labelled test (see VoiceEmotionClassifier). It is published only for PyTorch, so it is converted once with
  export_voice_model.py, which writes the .onnx file and a .json file with its labels and SHA-256.

Model files are downloaded once into EMOTION_MODEL_DIR and checked against a SHA-256 before use, so a corrupted or
altered file is never loaded. `flask --app run download-emotion-models` fetches them ahead of time (do this when
deploying, so the first interview doesn't wait for a download).
"""
import hashlib
import math
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

HSEMOTION_URL = "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/{}.onnx"
MODEL_FILES = {
    "face_detection_yunet_2023mar": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "enet_b0_8_best_afew": (HSEMOTION_URL.format("enet_b0_8_best_afew"),
                            "7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f"),
    "enet_b0_8_best_vgaf": (HSEMOTION_URL.format("enet_b0_8_best_vgaf"),
                            "fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898"),
    "enet_b2_8": (HSEMOTION_URL.format("enet_b2_8"),
                  "180a9d4845b59393de4511598a0d1d34b705034691ea32959ce5009db7cf52b7"),
}
HSEMOTION_CLASSES = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelError(RuntimeError):
    """A model file is missing, can't be downloaded, or doesn't match its checksum."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(name, directory, download=True):
    """Path of a verified model file, downloading it first if allowed."""
    if name not in MODEL_FILES:
        raise ModelError(f"Unknown model: {name}")
    url, expected = MODEL_FILES[name]
    path = Path(directory) / f"{name}.onnx"
    if not path.is_file():
        if not download:
            raise ModelError(f"Model file {path} is missing (run: flask --app run download-emotion-models)")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:  # noqa: S310
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        except Exception as e:
            partial.unlink(missing_ok=True)
            raise ModelError(f"Couldn't download {name}: {e}") from e
        os.replace(partial, path)
    actual = _sha256(path)
    if actual != expected:
        raise ModelError(f"Model file {path} doesn't match its checksum (got {actual[:12]}…); delete it and download again")
    return path


# ---------- classifiers ----------

def _aligned_square(image, row, margin):
    """Square crop around the face with `margin` on each side, rotated so the eyes are level (face alignment)."""
    import cv2

    x, y, w, h = (float(v) for v in row[:4])
    (rx, ry), (lx, ly) = row[4:6], row[6:8]
    cx, cy = x + w / 2, y + h / 2
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    side = max(w, h) * (1 + 2 * margin)
    # rotate around the face centre, then cut a square of `side` centred on it (edges replicated if it leaves the frame)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    m[0, 2] += side / 2 - cx
    m[1, 2] += side / 2 - cy
    size = max(1, int(round(side)))
    return cv2.warpAffine(image, m, (size, size), borderMode=cv2.BORDER_REPLICATE)


class HSEmotionClassifier:
    name = "hsemotion"

    def __init__(self, path, model_name):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2  # leave CPU for the web server and the main worker
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.size = 224 if "_b0_" in model_name else 260
        self.label = f"HSEmotion {model_name}"

    def predict(self, image, prepared, row, cfg):
        """Probabilities over HSEMOTION_CLASSES for the face in `row`, read from the original frame."""
        import cv2

        face = _aligned_square(image, row, cfg["EMOTION_FACE_EXPAND"] / 100)
        rgb = cv2.cvtColor(cv2.resize(face, (self.size, self.size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        logits = self.session.run(None, {"input": x})[0][0][:len(HSEMOTION_CLASSES)].astype(np.float64)
        if not np.all(np.isfinite(logits)):
            return None
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return {c: float(v) for c, v in zip(HSEMOTION_CLASSES, p)}


class DeepFaceClassifier:
    """The paper's DeepFace emotion model (baseline). Reads the face through DeepFace's own crop and alignment."""

    name = "deepface"
    label = "DeepFace (paper baseline)"

    def __init__(self, threshold):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ["yunet_score_threshold"] = str(threshold)  # DeepFace reads its YuNet threshold from here
        from deepface import DeepFace

        DeepFace.analyze(np.zeros((48, 48, 3), dtype=np.uint8), actions=["emotion"], detector_backend="skip",
                         enforce_detection=False, silent=True)
        self.deepface = DeepFace
        self.lock = threading.Lock()  # TensorFlow models are not guaranteed thread-safe

    def predict(self, image, prepared, row, cfg):
        for source in (image, prepared):  # the original frame first; the normalised one if no face is found in it
            try:
                with self.lock:
                    faces = self.deepface.analyze(source, actions=["emotion"], detector_backend="yunet", align=True,
                                                  enforce_detection=True, silent=True,
                                                  expand_percentage=cfg["EMOTION_FACE_EXPAND"])
            except ValueError:
                continue
            face = max(faces, key=lambda f: _overlap(f["region"], row))  # the face the detector chose
            raw = {e: float(face["emotion"].get(e, 0.0)) for e in HSEMOTION_CLASSES}  # DeepFace has no contempt
            total = sum(raw.values())
            if total > 0 and all(math.isfinite(v) for v in raw.values()):
                return {e: raw[e] / total for e in HSEMOTION_CLASSES}
            return None
        return None


def _overlap(region, row):
    x, y, w, h = (float(v) for v in row[:4])
    ix = max(0.0, min(region["x"] + region["w"], x + w) - max(region["x"], x))
    iy = max(0.0, min(region["y"] + region["h"], y + h) - max(region["y"], y))
    return ix * iy


class VoiceEmotionClassifier:
    """Speech emotion from the candidate's voice (the paper's audio modality), emotion2vec+ base on ONNX Runtime.

    Chosen on a labelled recording of the candidate (16 clips: 14 acted emotions + 2 natural interview answers):
    emotion2vec+ 25% correct, wav2vec2 (acted-speech model) 38% and Gemini 25%. wav2vec2 scored higher only by
    calling the calm natural answers "sad", which would mark calm candidates as nervous; emotion2vec+ read both as
    neutral. Voice models picked up little of the emotion in natural, quiet speech, which is why voice is only a
    small part of the emotion reading (VOICE_EMOTION_WEIGHT) next to the face (81% on the same clips).
    """

    # emotion2vec+ labels -> the face model's classes; "other" and "<unk>" carry no emotion and are left out
    LABELS = {"angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy", "neutral": "neutral",
              "sad": "sad", "surprised": "surprise"}

    def __init__(self, directory, name):
        import json

        import onnxruntime as ort

        path, info_path = Path(directory) / f"{name}.onnx", Path(directory) / f"{name}.json"
        if not path.is_file() or not info_path.is_file():
            raise ModelError(f"Voice model {path} is missing (create it with: python export_voice_model.py {directory})")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            labels, expected = list(info["labels"]), info["sha256"]
        except (ValueError, KeyError, TypeError) as e:
            raise ModelError(f"{info_path} is not a valid model description: {e}") from e
        actual = _sha256(path)
        if actual != expected:
            raise ModelError(f"Voice model {path} doesn't match its checksum (got {actual[:12]}…); export it again")
        if not set(self.LABELS) <= set(labels):
            raise ModelError(f"Voice model labels {labels} are not the expected emotion2vec+ labels")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2  # leave CPU for the web server and the main worker
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.labels = labels
        self.label = f"emotion2vec+ {name.removeprefix('emotion2vec_plus_')}"

    def predict(self, samples):
        """(probabilities over HSEMOTION_CLASSES, share of the model's belief that went to no emotion) for 16 kHz
        mono float32 samples, or None if the output isn't usable."""
        logits = self.session.run(None, {"wav": samples[np.newaxis].astype(np.float32)})[0][0].astype(np.float64)
        if not np.all(np.isfinite(logits)):
            return None
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        probs = {c: 0.0 for c in HSEMOTION_CLASSES}
        for label, value in zip(self.labels, p):
            if label in self.LABELS:
                probs[self.LABELS[label]] += float(value)
        total = sum(probs.values())
        if total <= 0:
            return None
        return {c: v / total for c, v in probs.items()}, float(1 - total)


# ---------- loading (once per worker process) ----------

_cache = {}
_lock = threading.Lock()


def load(cfg):
    """{"detector": cv2.FaceDetectorYN, "classifier": ..., "detect_lock": Lock} for the configured model."""
    key = (cfg["EMOTION_MODEL"], cfg["HSEMOTION_MODEL"], str(cfg["EMOTION_MODEL_DIR"]))
    with _lock:
        if key not in _cache:
            import cv2

            directory = cfg["EMOTION_MODEL_DIR"]
            download = cfg["EMOTION_MODEL_DOWNLOAD"]
            detector = cv2.FaceDetectorYN.create(str(model_path("face_detection_yunet_2023mar", directory, download)),
                                                 "", (320, 320), cfg["EMOTION_DETECTOR_THRESHOLD"])
            if cfg["EMOTION_MODEL"] == "deepface":
                classifier = DeepFaceClassifier(cfg["EMOTION_DETECTOR_THRESHOLD"])
            elif cfg["EMOTION_MODEL"] == "hsemotion":
                name = cfg["HSEMOTION_MODEL"]
                classifier = HSEmotionClassifier(model_path(name, directory, download), name)
            else:
                raise ModelError(f"Unknown EMOTION_MODEL {cfg['EMOTION_MODEL']!r} (use hsemotion or deepface)")
            _cache[key] = {"detector": detector, "classifier": classifier, "detect_lock": threading.Lock()}
        return _cache[key]


def load_voice(cfg):
    """The configured voice-emotion classifier (loaded once per process)."""
    key = ("voice", cfg["VOICE_EMOTION_MODEL"], str(cfg["EMOTION_MODEL_DIR"]))
    with _lock:
        if key not in _cache:
            _cache[key] = VoiceEmotionClassifier(cfg["EMOTION_MODEL_DIR"], cfg["VOICE_EMOTION_MODEL"])
        return _cache[key]
