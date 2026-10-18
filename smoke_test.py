"""Phase 0 smoke test: checks that Gemini (text + embeddings) and DeepFace (webcam emotion) work.

Usage (from C:\\dev\\virtual-interviewer):
    venv\\Scripts\\python smoke_test.py                 # Gemini + webcam
    venv\\Scripts\\python smoke_test.py --skip-webcam   # Gemini only
    venv\\Scripts\\python smoke_test.py --image me.jpg  # use a photo instead of the webcam
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow startup noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# DeepFace logs emoji; the default Windows console encoding (cp1252) crashes on them mid-download
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()


def test_gemini():
    print("\n=== Gemini API ===")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("FAIL: GEMINI_API_KEY is empty. Put your key in .env and run again.")
        return False

    from google import genai

    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
    try:
        reply = client.models.generate_content(
            model=model,
            contents="Ask one short technical interview question about Python.",
        )
        print(f"OK  text ({model}): {reply.text.strip()}")

        import numpy as np

        texts = ["Built REST APIs with Flask", "Backend web development in Python", "Baking sourdough bread"]
        # One call per text: gemini-embedding-2 merges a list of contents into a single embedding
        vecs = [np.array(client.models.embed_content(model=embed_model, contents=t).embeddings[0].values) for t in texts]
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"OK  embeddings ({embed_model}, dim {len(vecs[0])})")
        print(f"    similarity Flask vs backend: {cos(vecs[0], vecs[1]):.3f}  (should be higher)")
        print(f"    similarity Flask vs baking:  {cos(vecs[0], vecs[2]):.3f}")
        return True
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        if "NOT_FOUND" not in str(e):
            return False
        print("Models available to your key:")
        try:
            for m in client.models.list():
                print("   ", m.name)
        except Exception:
            pass
        print("If the model above is not listed, change GEMINI_MODEL / GEMINI_EMBED_MODEL in .env.")
        return False


def grab_webcam_frame():
    import cv2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    for _ in range(15):  # let auto-exposure settle
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def test_deepface(image_path=None):
    print("\n=== DeepFace emotion ===")
    import cv2
    from deepface import DeepFace

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"FAIL: could not read image {image_path}")
            return False
    else:
        print("Look at the webcam...")
        frame = grab_webcam_frame()
        if frame is None:
            print("FAIL: could not open the webcam (is another app using it?). Try --image <photo>.")
            return False
        cv2.imwrite("smoke_test_frame.jpg", frame)
        print("Saved the captured frame to smoke_test_frame.jpg")

    # Histogram equalisation on brightness (paper 3.2) helps in dim rooms
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycc[:, :, 0])
    frame = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    print("Analysing (first run downloads model weights, ~1 min)...")
    # yunet is fast but misses dim/backlit faces; retinaface is slower but much more robust
    faces = None
    for detector in ("yunet", "retinaface"):
        try:
            faces = DeepFace.analyze(frame, actions=["emotion"], detector_backend=detector, enforce_detection=True, silent=True)
            print(f"Face found by {detector}")
            break
        except ValueError as e:
            if "Face could not be detected" not in str(e):
                print(f"FAIL: {e}")
                return False
    if faces is None:
        print("FAIL: no face detected. Sit facing the camera with light in front of you and try again.")
        return False

    face = faces[0]
    top = sorted(face["emotion"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"OK  dominant emotion: {face['dominant_emotion']}  (face confidence {face.get('face_confidence', 0):.2f})")
    print("    top 3: " + ", ".join(f"{k} {v:.1f}%" for k, v in top))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-webcam", action="store_true")
    parser.add_argument("--image")
    args = parser.parse_args()

    results = {"Gemini": test_gemini()}
    if not args.skip_webcam or args.image:
        results["DeepFace"] = test_deepface(args.image)

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"{name:9s} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
