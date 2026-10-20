"""Record a labelled expression test clip for measuring the emotion model's real accuracy.

Usage: venv\\Scripts\\python expression_test.py
The window shows your camera and tells you which expression to make; hold it until the next one. Frames are saved
with the expression you were asked for (instance/expression_test/<timestamp>/, never committed). Takes ~2 minutes.
Keys: Esc / close window = stop early (frames recorded so far are kept).
"""
import csv
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageTk

# (label, what to do). Each is held for HOLD seconds after a short GET_READY; frames from the first SETTLE seconds of
# each hold are not used (the face is still changing).
SEQUENCE = [
    ("neutral", "Relaxed, neutral face, looking at the camera"),
    ("happy", "Smile naturally"),
    ("neutral", "Back to a neutral face"),
    ("surprise", "Look surprised: eyebrows up, mouth a little open"),
    ("sad", "Look sad: corners of the mouth down"),
    ("neutral", "Neutral again"),
    ("angry", "Frown, look annoyed"),
    ("fear", "Look worried or scared"),
    ("happy", "Big smile"),
    ("disgust", "Look disgusted: wrinkle your nose"),
    ("neutral", "Neutral, as if listening to an interview question"),
    ("neutral", "Neutral, glance down at the screen as if reading"),
]
GET_READY, HOLD, SETTLE, FPS = 3.0, 6.0, 1.0, 5
OUT_ROOT = Path(__file__).parent / "instance" / "expression_test"


class Recorder:
    def __init__(self, root):
        self.root = root
        root.title("Expression test  ·  Esc = stop")
        root.configure(bg="#1e1f24")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video = tk.Label(root, bg="#1e1f24")
        self.video.pack(side="left", padx=10, pady=10)
        panel = tk.Frame(root, bg="#1e1f24")
        panel.pack(side="left", fill="y", padx=10, pady=10)
        self.step_label = tk.Label(panel, text="", font=("Segoe UI", 11), fg="#aab", bg="#1e1f24", anchor="w")
        self.step_label.pack(anchor="w")
        self.headline = tk.Label(panel, text="Starting…", font=("Segoe UI", 28, "bold"), fg="white", bg="#1e1f24",
                                 anchor="w", width=12)
        self.headline.pack(anchor="w", pady=(6, 2))
        self.instruction = tk.Label(panel, text="", font=("Segoe UI", 12), fg="#dde", bg="#1e1f24", anchor="w",
                                    justify="left", wraplength=320)
        self.instruction.pack(anchor="w")
        self.countdown = tk.Label(panel, text="", font=("Consolas", 20), fg="#7c7", bg="#1e1f24", anchor="w")
        self.countdown.pack(anchor="w", pady=10)
        self.saved = tk.Label(panel, text="", font=("Segoe UI", 9), fg="#889", bg="#1e1f24", anchor="w")
        self.saved.pack(anchor="w")
        root.bind("<Escape>", lambda e: self.finish())
        root.protocol("WM_DELETE_WINDOW", self.finish)

        self.out = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "labels.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.log)
        self.writer.writerow(["file", "label", "step", "seconds_into_hold"])
        self.count = 0
        self.step = 0
        self.phase_start = time.time() + 2.0  # a moment to look at the window first
        self.phase = "ready"
        self.last_saved = 0.0
        self.done = False
        self.tick()

    def tick(self):
        if self.done:
            return
        ok, frame = self.cap.read()
        now = time.time()
        if ok:
            frame = cv2.flip(frame, 1)  # mirror view (saved frames are mirrored too; expressions are unaffected)
            label, text = SEQUENCE[self.step]
            elapsed = now - self.phase_start
            self.step_label.config(text=f"Expression {self.step + 1} of {len(SEQUENCE)}")
            if self.phase == "ready":
                self.headline.config(text=f"Next: {label.upper()}", fg="#e0a800")
                self.instruction.config(text=text)
                self.countdown.config(text=f"in {max(0.0, GET_READY - elapsed):.0f}…" if elapsed >= 0 else "")
                if elapsed >= GET_READY:
                    self.phase, self.phase_start = "hold", now
            else:
                self.headline.config(text=label.upper(), fg="#2e9d5b")
                self.instruction.config(text=text + "  (hold it)")
                self.countdown.config(text=f"{max(0.0, HOLD - elapsed):.1f} s")
                if elapsed >= SETTLE and now - self.last_saved >= 1.0 / FPS:
                    name = f"{self.count:04d}_{self.step:02d}_{label}.jpg"
                    cv2.imwrite(str(self.out / name), frame)
                    self.writer.writerow([name, label, self.step, round(elapsed, 2)])
                    self.count += 1
                    self.last_saved = now
                    self.saved.config(text=f"{self.count} frames saved")
                if elapsed >= HOLD:
                    self.step += 1
                    if self.step >= len(SEQUENCE):
                        self.finish()
                        return
                    self.phase, self.phase_start = "ready", now
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.configure(image=img)
            self.video.image = img
        self.root.after(30, self.tick)

    def finish(self):
        if self.done:
            return
        self.done = True
        self.cap.release()
        self.log.close()
        print(f"Saved {self.count} labelled frames to {self.out}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    Recorder(root)
    root.mainloop()
