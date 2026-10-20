"""Live webcam emotion preview, with the same face detection and expression model as the interview analysis
(YuNet + HSEmotion, app/services/emotion.py).

Usage: venv\\Scripts\\python live_emotion.py
Keys:  S = save snapshot (to expression_snapshots/)   Q / Esc / close window = quit
"""
import csv
import os
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import cv2
from PIL import Image, ImageTk

EMOTIONS = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust", "contempt"]
COLORS = {"happy": "#2e9d5b", "neutral": "#6c757d", "surprise": "#e0a800", "sad": "#3b6fd1",
          "angry": "#d63b3b", "fear": "#8a4fc7", "disgust": "#7a8a2e", "contempt": "#a0522d"}
SMOOTHING = 0.3  # weight of the newest reading in the moving average (paper 3.2: temporal smoothing)
SNAP_DIR = Path(__file__).parent / "expression_snapshots"


class Analyzer(threading.Thread):
    """Runs the interview pipeline on the newest frame in the background so the video stays smooth."""

    def __init__(self):
        super().__init__(daemon=True)
        self.frame = None
        self.result = None  # (region, raw scores, smoothed scores, ms, model)
        self.smoothed = None
        self.lock = threading.Lock()
        self.running = True
        self.ready = False
        self.stage = "Starting…"  # shown in the window while loading
        self.error = None  # any unexpected failure is shown instead of hanging on "Loading"
        self.started = time.time()

    def run(self):
        try:
            self.stage = "Loading the face and expression models…"
            try:
                from app import create_app
                from app.services import emotion
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    f"The project's packages aren't installed for this Python ({sys.executable}, "
                    f"{sys.version.split()[0]}): {e}. Run it with:  venv\\Scripts\\python live_emotion.py") from None
            app = create_app()
            with app.app_context():
                cfg = dict(app.config)
                models = emotion._load_models()
            detector, classifier = models["detector"], models["classifier"]
            self.stage = "Waiting for camera…"
            while self.running:
                with self.lock:
                    frame = None if self.frame is None else self.frame.copy()
                if frame is None:
                    time.sleep(0.05)
                    continue
                t = time.time()
                prepared = emotion.equalise(frame)
                with models["detect_lock"]:
                    detector.setInputSize((prepared.shape[1], prepared.shape[0]))
                    _, rows = detector.detect(prepared)
                probs = None
                if rows is not None and len(rows):
                    row = max(rows, key=lambda r: r[2] * r[3])
                    probs = classifier.predict(frame, prepared, row, cfg)
                if probs is None:
                    self.result = None  # no face this frame
                else:
                    raw = {e: 100 * probs.get(e, 0.0) for e in EMOTIONS}
                    if self.smoothed is None:
                        self.smoothed = raw
                    else:
                        self.smoothed = {e: SMOOTHING * raw[e] + (1 - SMOOTHING) * self.smoothed[e] for e in EMOTIONS}
                    x, y, w, h = (int(v) for v in row[:4])
                    region = {"x": x, "y": y, "w": w, "h": h}
                    self.result = (region, raw, dict(self.smoothed), (time.time() - t) * 1000, classifier.label)
                self.ready = True
        except Exception as e:  # never die silently
            import traceback

            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Live emotion preview  ·  S = snapshot   Q = quit")
        root.configure(bg="#1e1f24")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit("Could not open the webcam (is another app using it?).")

        self.video = tk.Label(root, bg="#1e1f24")
        self.video.grid(row=0, column=0, padx=10, pady=10)

        panel = tk.Frame(root, bg="#1e1f24")
        panel.grid(row=0, column=1, sticky="n", padx=(0, 14), pady=10)
        self.headline = tk.Label(panel, text="Loading model…", font=("Segoe UI", 20, "bold"), fg="white", bg="#1e1f24", width=14, anchor="w")
        self.headline.pack(anchor="w")
        self.sub = tk.Label(panel, text="", font=("Segoe UI", 10), fg="#aab", bg="#1e1f24", anchor="w", justify="left")
        self.sub.pack(anchor="w", pady=(0, 10))

        tk.Label(panel, text="Smoothed (bars) vs this frame (number)", font=("Segoe UI", 9), fg="#889", bg="#1e1f24").pack(anchor="w")
        self.bars = {}
        for e in EMOTIONS:
            row = tk.Frame(panel, bg="#1e1f24")
            row.pack(anchor="w", pady=3)
            tk.Label(row, text=e, width=8, anchor="w", font=("Segoe UI", 11), fg="white", bg="#1e1f24").pack(side="left")
            canvas = tk.Canvas(row, width=200, height=18, bg="#2c2e35", highlightthickness=0)
            canvas.pack(side="left")
            value = tk.Label(row, text="", width=6, anchor="e", font=("Consolas", 10), fg="#ccd", bg="#1e1f24")
            value.pack(side="left")
            self.bars[e] = (canvas, value)

        self.status = tk.Label(panel, text="", font=("Segoe UI", 9), fg="#7c7", bg="#1e1f24", anchor="w")
        self.status.pack(anchor="w", pady=(12, 0))

        self.analyzer = Analyzer()
        self.analyzer.start()
        self.last_frame = None

        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.tick()

    def tick(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)  # mirror, like a selfie camera
            self.last_frame = frame
            with self.analyzer.lock:
                self.analyzer.frame = frame
            self.draw(frame.copy())
        self.root.after(30, self.tick)

    def draw(self, frame):
        res = self.analyzer.result
        if self.analyzer.error:
            self.headline.config(text="Error", fg="#e66")
            self.sub.config(text=self.analyzer.error[:300] + "\n(details in the terminal)")
        elif not self.analyzer.ready:
            waited = time.time() - self.analyzer.started
            self.headline.config(text="Loading…", fg="white")
            self.sub.config(text=f"{self.analyzer.stage}  ({waited:.0f}s)\nThe first start can take up to a minute.")
        elif res is None:
            self.headline.config(text="No face", fg="#e66")
            self.sub.config(text="Face the camera with light in front of you")
        else:
            region, raw, smooth, ms, detector = res
            top = max(smooth, key=smooth.get)
            color = COLORS[top]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            bgr = tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(frame, f"{top} {smooth[top]:.0f}%", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            self.headline.config(text=top.capitalize(), fg=color)
            self.sub.config(text=f"This frame: {max(raw, key=raw.get)}  ·  {ms:.0f} ms  ·  {detector}")
            for e in EMOTIONS:
                canvas, value = self.bars[e]
                canvas.delete("all")
                canvas.create_rectangle(0, 0, 2 * smooth[e], 18, fill=COLORS[e], width=0)
                value.config(text=f"{raw[e]:5.1f}")
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.config(image=img)
        self.video.image = img

    def on_key(self, event):
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.quit()
        elif key == "s":
            self.snapshot()

    def snapshot(self):
        res = self.analyzer.result
        if self.last_frame is None or res is None:
            self.status.config(text="No face to snapshot", fg="#e66")
            return
        SNAP_DIR.mkdir(exist_ok=True)
        _, raw, smooth, _, _ = res
        top = max(raw, key=raw.get)
        stamp = datetime.now().strftime("%H%M%S")
        cv2.imwrite(str(SNAP_DIR / f"{stamp}_{top}.jpg"), self.last_frame)
        log = SNAP_DIR / "log.csv"
        new = not log.exists()
        with log.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "frame_top", "smoothed_top"] + EMOTIONS)
            w.writerow([stamp, top, max(smooth, key=smooth.get)] + [f"{raw[e]:.1f}" for e in EMOTIONS])
        self.status.config(text=f"Saved snapshot: {top}", fg="#7c7")

    def quit(self):
        self.analyzer.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
