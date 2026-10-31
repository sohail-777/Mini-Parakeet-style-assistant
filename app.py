"""
Mini Parakeet-style assistant (fully local).

  Mic -> faster-whisper (local STT) -> transcript
  Transcript -> Ollama (local LLM) -> answer streamed token by token

Run:  python app.py
"""

import json
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# config
WHISPER_MODEL = "base.en"          # tiny.en = faster, small.en = more accurate
OLLAMA_MODEL = "llama3.2:3b"       # any model you've pulled with `ollama pull`
OLLAMA_URL = "http://localhost:11434/api/chat"

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
SILENCE_RMS = 0.01                 # raise if background noise triggers speech
END_OF_SPEECH_BLOCKS = 8           # 0.8 s of silence ends an utterance
MAX_SEGMENT_SECONDS = 15           # force a flush on long monologues
MIN_SPEECH_SECONDS = 0.4           # ignore clicks/pops shorter than this
PREROLL_BLOCKS = 3                 # keep a little audio before speech starts
MAX_CONTEXT_CHARS = 3000           # how much transcript the LLM sees

SYSTEM_PROMPT = (
    "You are a real-time interview and meeting assistant. The user gives you a "
    "live transcript (it may contain recognition errors). Identify the most "
    "recent question in it and answer it clearly and concisely, in the first "
    "person, as a strong answer the user could say out loud."
)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Local Answer Assistant")
        root.geometry("720x620")

        self.ui_q: queue.Queue = queue.Queue()      # worker threads -> UI
        self.audio_q: queue.Queue = queue.Queue()   # mic callback -> STT worker
        self.model: WhisperModel | None = None
        self.stream: sd.InputStream | None = None
        self.stt_thread: threading.Thread | None = None
        self.generating = False

        self._build_ui()
        self.root.after(50, self._drain_ui_queue)
        threading.Thread(target=self._load_whisper, daemon=True).start()

    # UI
    def _build_ui(self):
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=10, pady=8)

        self.start_btn = tk.Button(bar, text="Start", width=10,
                                   command=self.start, state="disabled")
        self.stop_btn = tk.Button(bar, text="Stop", width=10,
                                  command=self.stop, state="disabled")
        self.suggest_btn = tk.Button(bar, text="Suggest Answer", width=16,
                                     command=self.suggest, state="disabled")
        for b in (self.start_btn, self.stop_btn, self.suggest_btn):
            b.pack(side="left", padx=4)

        self.status = tk.Label(bar, text="Loading speech model...", anchor="e")
        self.status.pack(side="right")

        tk.Label(self.root, text="Transcription", anchor="w",
                 font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=10)
        self.transcript_box = scrolledtext.ScrolledText(
            self.root, height=12, wrap="word")
        self.transcript_box.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        tk.Label(self.root, text="Suggested answer", anchor="w",
                 font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=10)
        self.answer_box = scrolledtext.ScrolledText(
            self.root, height=12, wrap="word", state="disabled")
        self.answer_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _set_status(self, text: str):
        self.status.config(text=text)

    def _append(self, box: scrolledtext.ScrolledText, text: str):
        box.config(state="normal")
        box.insert("end", text)
        box.see("end")
        if box is self.answer_box:
            box.config(state="disabled")

    def _drain_ui_queue(self):
        """All widget updates happen here, on the Tk thread."""
        try:
            while True:
                kind, payload = self.ui_q.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "ready":
                    self.start_btn.config(state="normal")
                    self._set_status("Ready")
                elif kind == "transcript":
                    self._append(self.transcript_box, payload + " ")
                elif kind == "token":
                    self._append(self.answer_box, payload)
                elif kind == "answer_done":
                    self.generating = False
                    self._refresh_suggest_btn()
                    self._set_status("Listening..." if self.stream else "Ready")
                elif kind == "error":
                    self.generating = False
                    self._refresh_suggest_btn()
                    self._set_status(payload)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_ui_queue)

    def _refresh_suggest_btn(self):
        has_text = bool(self.transcript_box.get("1.0", "end").strip())
        state = "normal" if has_text and not self.generating else "disabled"
        self.suggest_btn.config(state=state)

    # speech to text
    def _load_whisper(self):
        try:
            # First run downloads the model once; afterwards it is fully offline.
            self.model = WhisperModel(WHISPER_MODEL, device="cpu",
                                      compute_type="int8")
            self.ui_q.put(("ready", None))
        except Exception as e:  # noqa: BLE001
            self.ui_q.put(("error", f"Whisper load failed: {e}"))

    def start(self):
        if self.stream is not None:
            return
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
                callback=self._on_audio)
            self.stream.start()
        except Exception as e:  # noqa: BLE001
            self.stream = None
            self._set_status(f"Mic error: {e}")
            return

        self.stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        self.stt_thread.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_status("Listening...")

    def stop(self):
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.audio_q.put(None)  # tells the STT loop to flush and exit
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._set_status("Stopped")
        self.root.after(1500, self._refresh_suggest_btn)

    def _on_audio(self, indata, frames, time_info, status):
        self.audio_q.put(indata[:, 0].copy())

    def _stt_loop(self):
        """Group mic blocks into utterances (split on silence), transcribe each."""
        preroll: list[np.ndarray] = []
        buf: list[np.ndarray] = []
        speaking = False
        silent_blocks = 0
        max_blocks = int(MAX_SEGMENT_SECONDS / BLOCK_SECONDS)

        while True:
            block = self.audio_q.get()
            if block is None:
                if buf:
                    self._transcribe(buf)
                return

            loud = float(np.sqrt(np.mean(block ** 2))) > SILENCE_RMS
            if not speaking:
                if loud:
                    speaking = True
                    silent_blocks = 0
                    buf = preroll[:] + [block]
                else:
                    preroll.append(block)
                    preroll = preroll[-PREROLL_BLOCKS:]
                continue

            buf.append(block)
            silent_blocks = 0 if loud else silent_blocks + 1
            if silent_blocks >= END_OF_SPEECH_BLOCKS or len(buf) >= max_blocks:
                self._transcribe(buf)
                buf, preroll, speaking, silent_blocks = [], [], False, 0

    def _transcribe(self, blocks: list[np.ndarray]):
        audio = np.concatenate(blocks)
        if len(audio) < SAMPLE_RATE * MIN_SPEECH_SECONDS:
            return
        try:
            segments, _ = self.model.transcribe(
                audio, language="en", beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            self.ui_q.put(("error", f"Transcription error: {e}"))
            return
        if text:
            self.ui_q.put(("transcript", text))

    # LLM
    def suggest(self):
        transcript = self.transcript_box.get("1.0", "end").strip()
        if not transcript or self.generating:
            return
        self.generating = True
        self.suggest_btn.config(state="disabled")
        self.answer_box.config(state="normal")
        self.answer_box.delete("1.0", "end")
        self.answer_box.config(state="disabled")
        self._set_status("Generating answer...")
        threading.Thread(target=self._stream_answer,
                         args=(transcript[-MAX_CONTEXT_CHARS:],),
                         daemon=True).start()

    def _stream_answer(self, transcript: str):
        payload = {
            "model": OLLAMA_MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
        }
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True,
                               timeout=(5, 120)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        self.ui_q.put(("token", token))
                    if chunk.get("done"):
                        break
            self.ui_q.put(("answer_done", None))
        except requests.exceptions.ConnectionError:
            self.ui_q.put(("error", "Can't reach Ollama. Is it running?"))
        except Exception as e:  # noqa: BLE001
            self.ui_q.put(("error", f"LLM error: {e}"))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
