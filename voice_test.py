"""Record a labelled voice + face test for measuring emotion models' real accuracy (like expression_test.py).

Usage: venv\\Scripts\\python voice_test.py
The window shows your camera, which emotion to use and what to say. After a short countdown it records your voice
(16 kHz WAV) and camera frames for a few seconds, so the same clips can test the voice model, the face model and
both combined. Saved with the emotion you were asked for in instance/voice_test/<timestamp>/ (never committed).
Takes about 4 minutes. Keys: Esc / close window = stop early (clips recorded so far are kept).
"""
import csv
import time
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
from PIL import Image, ImageTk

RATE = 16000
FPS = 5  # camera frames saved per second while recording
SENTENCES = ["I worked on that project for six months and we delivered it on time.",
             "The meeting has been moved to tomorrow morning at ten o'clock."]
EMOTIONS = [("happy", "Happy, cheerful"), ("sad", "Sad, disappointed"), ("angry", "Angry, irritated"),
            ("fear", "Nervous or scared, unsure of yourself"), ("surprise", "Surprised, astonished"),
            ("disgust", "Disgusted, as if something is gross")]
# (label, how to say it, sentence or None for a free answer, seconds)
SEQUENCE = []
for sentence in SENTENCES:
    SEQUENCE.append(("neutral", "Calm, normal voice, as if answering an interviewer", sentence, 6))
    SEQUENCE += [(label, how, sentence, 6) for label, how in EMOTIONS]
SEQUENCE += [("neutral", "Answer naturally, as in a real interview: tell us about a project you built", None, 15),
             ("neutral", "Answer naturally: what do you do when you get stuck on a problem?", None, 15)]
GET_READY = 4.0
MIN_VOICE_PEAK = 0.05   # the check: your voice must reach at least 5% of full scale
MIN_BRIGHTNESS = 50     # the check: average camera brightness (0-255); a closed shutter gives ~20


def pick_microphone():
    """The input device to use: the system default if it isn't silent, else the first one that picks up sound."""
    candidates = [sd.default.device[0]] + [i for i, d in enumerate(sd.query_devices())
                                           if d["max_input_channels"] > 0 and d["hostapi"] == 0]
    seen = set()
    for index in candidates:
        if index in seen or index is None or index < 0:
            continue
        seen.add(index)
        try:
            r = sd.rec(int(0.6 * RATE), samplerate=RATE, channels=1, dtype="int16", device=index)
            sd.wait()
        except Exception:  # noqa: BLE001 - some devices refuse the format; skip them
            continue
        if np.abs(r).max() > 20:  # a live microphone always has some background noise; a dead input is ~0
            return index
    return sd.default.device[0]
OUT_ROOT = Path(__file__).parent / "instance" / "voice_test"
BG, FG, MUTED = "#1e1f24", "white", "#aab"


class Recorder:
    def __init__(self, root):
        self.root = root
        root.title("Voice + face test  ·  Esc = stop")
        root.configure(bg=BG)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video = tk.Label(root, bg=BG)
        self.video.pack(side="left", padx=10, pady=10)
        panel = tk.Frame(root, bg=BG)
        panel.pack(side="left", fill="y", padx=10, pady=10)
        self.step_label = tk.Label(panel, text="", font=("Segoe UI", 11), fg=MUTED, bg=BG, anchor="w")
        self.step_label.pack(anchor="w")
        self.headline = tk.Label(panel, text="Starting…", font=("Segoe UI", 28, "bold"), fg=FG, bg=BG, anchor="w",
                                 width=13)
        self.headline.pack(anchor="w", pady=(6, 2))
        self.how = tk.Label(panel, text="", font=("Segoe UI", 12), fg="#dde", bg=BG, anchor="w", justify="left",
                            wraplength=340)
        self.how.pack(anchor="w")
        self.sentence = tk.Label(panel, text="", font=("Segoe UI", 14, "italic"), fg="#7cc", bg=BG, anchor="w",
                                 justify="left", wraplength=340)
        self.sentence.pack(anchor="w", pady=(10, 0))
        self.countdown = tk.Label(panel, text="", font=("Consolas", 20), fg="#7c7", bg=BG, anchor="w")
        self.countdown.pack(anchor="w", pady=10)
        tk.Label(panel, text="Microphone", font=("Segoe UI", 9), fg=MUTED, bg=BG).pack(anchor="w")
        self.meter = tk.Canvas(panel, width=340, height=12, bg="#2c2e35", highlightthickness=0)
        self.meter.pack(anchor="w")
        self.saved = tk.Label(panel, text="", font=("Segoe UI", 9), fg="#889", bg=BG, anchor="w")
        self.saved.pack(anchor="w", pady=(8, 0))
        root.bind("<Escape>", lambda e: self.finish())
        root.protocol("WM_DELETE_WINDOW", self.finish)

        self.out = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "labels.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.log)
        self.writer.writerow(["file", "label", "step", "sentence", "frames"])
        self.step, self.done = 0, False
        self.phase, self.phase_start = "check", time.time()
        self.check_voice, self.check_bright = 0.0, 0.0
        self.chunks, self.level, self.frames_saved, self.last_frame_at = [], 0.0, 0, 0.0
        # The microphone stays open; audio is kept only while a clip is recording
        self.device = pick_microphone()
        self.saved.config(text=f"Microphone: {sd.query_devices(self.device)['name']}")
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=self._audio,
                                     device=self.device)
        self.stream.start()
        self.tick()

    def _audio(self, indata, frames, time_info, status):
        self.level = float(np.abs(indata).max()) / 32768.0
        if self.phase == "recording":
            self.chunks.append(indata.copy())

    def tick(self):
        if self.done:
            return
        label, how, sentence, seconds = SEQUENCE[self.step]
        now = time.time()
        elapsed = now - self.phase_start
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)  # mirror view
        self.step_label.config(text=f"Clip {self.step + 1} of {len(SEQUENCE)}")
        self.how.config(text=how)
        self.sentence.config(text=f"Say: “{sentence}”" if sentence else "(answer in your own words)")
        if self.phase == "check":
            # Before anything is recorded: the voice must register and the camera must show a lit picture
            self.check_voice = max(self.check_voice * 0.97, self.level)
            self.check_bright = float(frame.mean()) if ok else 0.0
            voice_ok, camera_ok = self.check_voice >= MIN_VOICE_PEAK, self.check_bright >= MIN_BRIGHTNESS
            self.step_label.config(text="Setup check")
            self.headline.config(text="Check", fg="#e0a800")
            self.how.config(text=("✔ Microphone hears you" if voice_ok else "✘ Say something: the microphone bar must "
                                  "move well past the line (raise the input volume in Windows Sound settings if not)")
                            + "\n" + ("✔ Camera picture is bright enough" if camera_ok else
                                      f"✘ Camera is too dark ({self.check_bright:.0f}): open the camera shutter / "
                                      "press the camera key, turn on a light"))
            self.sentence.config(text="Count out loud from one to five.")
            self.countdown.config(text="starting…" if voice_ok and camera_ok else "")
            if voice_ok and camera_ok:
                if now - self.phase_start > 3:  # both fine for a moment: begin
                    self.phase, self.phase_start = "ready", now
            else:
                self.phase_start = now
        elif self.phase == "ready":
            self.headline.config(text=f"Next: {label.upper()}", fg="#e0a800")
            self.countdown.config(text=f"recording in {max(0.0, GET_READY - elapsed):.0f}…" if elapsed >= 0 else "")
            if elapsed >= GET_READY:
                self.chunks, self.frames_saved, self.last_frame_at = [], 0, 0.0
                self.phase, self.phase_start = "recording", now
        else:
            self.headline.config(text=f"● {label.upper()}", fg="#e05555")
            self.countdown.config(text=f"speak now… {max(0.0, seconds - elapsed):.1f} s")
            if ok and now - self.last_frame_at >= 1.0 / FPS:
                cv2.imwrite(str(self.out / f"{self.step:02d}_{label}_f{self.frames_saved:03d}.jpg"), frame)
                self.frames_saved += 1
                self.last_frame_at = now
            if elapsed >= seconds:
                self._save_clip(label, sentence)
                self.step += 1
                if self.step >= len(SEQUENCE):
                    self.finish()
                    return
                self.phase, self.phase_start = "ready", time.time()
        self.meter.delete("all")
        width = int(340 * min(1.0, self.level * 3))
        self.meter.create_rectangle(0, 0, width, 12, fill="#e05555" if self.phase == "recording" else "#5a8", width=0)
        line = int(340 * MIN_VOICE_PEAK * 3)
        self.meter.create_line(line, 0, line, 12, fill="#e0a800", width=2)  # the level the check asks for
        if ok:
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.configure(image=img)
            self.video.image = img
        self.root.after(30, self.tick)

    def _save_clip(self, label, sentence):
        self.phase = "saving"
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros((0, 1), dtype=np.int16)
        name = f"{self.step:02d}_{label}.wav"
        with wave.open(str(self.out / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(audio.tobytes())
        self.writer.writerow([name, label, self.step, sentence or "", self.frames_saved])
        self.log.flush()
        peak = int(np.abs(audio).max()) if len(audio) else 0
        self.saved.config(text=f"Saved clip {self.step + 1}: {len(audio) / RATE:.1f} s audio, {self.frames_saved} frames"
                               + ("  (very quiet: move closer to the microphone?)" if peak < 1500 else ""))

    def finish(self):
        if self.done:
            return
        self.done = True
        self.stream.stop()
        self.stream.close()
        self.cap.release()
        self.log.close()
        print(f"Saved {self.step} labelled clips to {self.out}")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    Recorder(root)
    root.mainloop()
