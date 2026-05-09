"""
modules/speech.py
Microphone listening (Whisper) and Text-to-Speech (Kokoro).
Separated cleanly so you can swap either component later.
"""

import queue
import threading
import re

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline

from config import (
    SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION,
    WHISPER_MODEL, GPU_DEVICE, COMPUTE_TYPE,
    TTS_VOICE, TTS_SPEED, TTS_LANG
)


# ── Text-to-Speech ─────────────────────────────────────────────────────────────

class Speaker:
    """
    Converts text to voice using Kokoro TTS.
    Thread-safe — only one speaker call at a time.
    """

    def __init__(self):
        print("[JARVIS] Loading TTS engine...")
        self.pipeline = KPipeline(lang_code=TTS_LANG, device='cpu')
        self._lock = threading.Lock()
        print("[JARVIS] TTS ready.")

    def speak(self, text: str):
        """Speak text. Blocks until audio finishes playing."""
        with self._lock:
            clean = self._clean(text)
            if not clean.strip():
                return
            generator = self.pipeline(clean, voice=TTS_VOICE, speed=TTS_SPEED)
            for _, _, audio in generator:
                sd.play(audio, 24000)
                sd.wait()

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown and symbols that sound bad when spoken."""
        text = re.sub(r'[*_`#~]', '', text)        # markdown
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # links
        text = re.sub(r'\n+', '. ', text)           # newlines → pauses
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ── Microphone / Speech-to-Text ───────────────────────────────────────────────

class Listener:
    """
    Listens on the microphone using VAD (volume-based silence detection).
    When speech ends, passes audio to Whisper on GPU.
    Calls callback(text) with the transcription.

    Thread model:
      - sounddevice audio callback → runs in C thread at ~60Hz
      - _transcribe_worker → background Python thread
    """

    def __init__(self, on_speech_callback):
        print(f"[JARVIS] Loading Whisper ({WHISPER_MODEL}, device={GPU_DEVICE})...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=GPU_DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("[JARVIS] Whisper ready.")

        self.callback     = on_speech_callback
        self.audio_queue  = queue.Queue()
        self.is_listening = False

        self._buffer        = []
        self._silence_frames = 0
        self._speaking      = False
        self._silence_limit = int(SILENCE_DURATION * (SAMPLE_RATE / 512))

    # ── Public control ─────────────────────────────────────────────────────────

    def start(self):
        """Start listening. Non-blocking — runs in background threads."""
        self.is_listening = True

        # Worker thread: takes audio from queue, transcribes, calls callback
        self._worker = threading.Thread(
            target=self._transcribe_worker,
            daemon=True,
            name="Whisper-Worker"
        )
        self._worker.start()

        # Audio stream: calls _audio_callback at 60Hz
        self._stream = sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = 1,
            blocksize  = 512,
            callback   = self._audio_callback,
            dtype      = "float32"
        )
        self._stream.start()
        print("[JARVIS] Microphone open — listening.")

    def pause(self):
        """Pause listening (e.g. while JARVIS is speaking)."""
        self.is_listening = False

    def resume(self):
        """Resume listening."""
        self._buffer         = []
        self._silence_frames = 0
        self._speaking       = False
        self.is_listening    = True

    def stop(self):
        """Shut down cleanly."""
        self.is_listening = False
        self.audio_queue.put(None)  # sentinel to stop worker
        self._stream.stop()
        self._stream.close()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time, status):
        """
        Called by sounddevice ~60 times per second.
        Detects speech start/end using volume threshold.
        """
        if not self.is_listening:
            return

        chunk  = indata[:, 0].copy()
        volume = np.abs(chunk).mean()

        # Zero-crossing rate — helps filter background noise
        zcr = np.sum(np.diff(np.sign(chunk)) != 0) / len(chunk)

        # Speech = loud enough AND voice-like frequency
        is_speech = volume > SILENCE_THRESHOLD and zcr > 0.04

        if is_speech:
            self._speaking       = True
            self._silence_frames = 0
            self._buffer.append(chunk)

        elif self._speaking:
            self._buffer.append(chunk)
            self._silence_frames += 1

            if self._silence_frames >= self._silence_limit:
                # End of utterance — send to transcription
                audio = np.concatenate(self._buffer)
                self.audio_queue.put(audio)
                self._buffer         = []
                self._speaking       = False
                self._silence_frames = 0

    def _transcribe_worker(self):
        """
        Background thread: takes audio chunks from queue and transcribes.
        Blocks on queue.get() when idle — uses no CPU while waiting.
        """
        while True:
            audio = self.audio_queue.get()
            if audio is None:
                break   # shutdown signal

            segments, info = self.model.transcribe(
                audio,
                language   = "en",
                beam_size  = 1,         # fastest, still accurate
                vad_filter = True,      # Whisper's own VAD as second pass
            )

            text = " ".join(s.text for s in segments).strip()

            # Filter out noise artefacts
            if text and len(text) > 2 and not self._is_noise(text):
                self.callback(text)

    @staticmethod
    def _is_noise(text: str) -> bool:
        """Filter common Whisper hallucinations on silence/noise."""
        noise_phrases = {
            "thank you", "thanks for watching", "you", "uh", "um",
            ".", "..", "...", "bye", "okay", "ok"
        }
        return text.lower().strip().strip(".").strip() in noise_phrases
