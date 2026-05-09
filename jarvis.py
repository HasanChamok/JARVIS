"""
jarvis.py — Main brain. Orchestrates all modules.
"""

import threading
import ollama
import numpy as np

from config import OLLAMA_MODEL, OLLAMA_HOST, USER_NAME
from modules.speech  import Listener, Speaker
from modules.memory  import Memory
from modules.todo    import TodoManager
from modules import files, apps, realtime, personality
from modules.realtime import get_location
from voice_auth import is_my_voice

import soundfile as sf
import os


class JARVIS:

    def __init__(self, status_callback=None):
        print("\n[JARVIS] Booting up...\n")

        self.memory      = Memory()
        self.todos       = TodoManager()
        self.speaker     = Speaker()
        self.joke_engine = personality.JokeEngine()

        # ← Pass interrupt callback so listener can trigger it directly
        self.listener = Listener(
            on_speech_callback    = self.on_speech,
            on_interrupt_callback = self.on_interrupt  # ← NEW
        )

        self.running       = False
        self.status_cb     = status_callback
        self.is_speaking   = False
        self._speak_thread = None

        self._cached_weather = realtime.get_weather()
        self.location = get_location()
        print(f"[JARVIS] Location: {self.location['city']}, {self.location['country']}")
        print("[JARVIS] All systems ready.\n")

    def _status(self, state: str, text: str = ""):
        if self.status_cb:
            self.status_cb(state, text)

    # ── NEW: called directly by Listener when "jarvis" heard mid-speech ────────
    def on_interrupt(self):
        """Triggered when wake word is heard while Jarvis is speaking."""
        if self.is_speaking:
            print("[JARVIS] ⚡ Interrupted by wake word!")
            self.speaker.stop()
            self.is_speaking = False
            self._status("listening", "")
            self.listener.resume()

    def _verify_audio(self, audio: np.ndarray) -> bool:
        temp_path = "temp_cmd_verify.wav"
        try:
            audio_int16 = (audio * 32767).astype(np.int16)
            sf.write(temp_path, audio_int16, 16000)
            return is_my_voice(temp_path)
        except Exception as e:
            print(f"[JARVIS] Voice check error: {e}")
            return False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def on_speech(self, text: str, audio: np.ndarray):

        # Gate 1 — skip very short text
        if len(text.strip()) < 5:
            return

        # Gate 2 — verify it's Hasan's voice
        if not self._verify_audio(audio):
            print("[JARVIS] 🚫 Unknown voice — ignoring command.")
            return

        print(f"\n[YOU]    {text}")
        self._status("thinking", text)
        self.listener.pause()  # switches to interrupt-only mode

        response = self._route(text)
        response = self.joke_engine.maybe_append_joke(response)

        print(f"[JARVIS] {response}\n")
        self._status("speaking", response)

        self.memory.add("user",      text)
        self.memory.add("assistant", response)

        self.is_speaking   = True
        self._speak_thread = threading.Thread(
            target=self._speak_and_resume,
            args=(response,),
            daemon=True
        )
        self._speak_thread.start()

    def _speak_and_resume(self, response: str):
        try:
            self.speaker.speak(response)
        finally:
            self.is_speaking = False
            self._status("listening", "")
            self.listener.resume()

    def _route(self, text: str) -> str:
        text_lower = text.lower().strip()

        if any(w in text_lower for w in ["goodbye", "shut down", "exit jarvis", "stop jarvis", "go offline"]):
            self.running = False
            return f"Going offline. Take care of yourself, {USER_NAME}."

        handlers = [
            personality.parse_and_handle,
            self.todos.parse_and_handle,
            self.memory.parse_and_handle,
            files.parse_and_handle,
            apps.parse_and_handle,
            realtime.parse_and_handle,
        ]

        for handler in handlers:
            result = handler(text)
            if result is not None:
                return result

        return self._llm(text)

    def _llm(self, user_text: str, tool_result: str = None) -> str:
        system_prompt = personality.build_system_prompt(
            weather      = self._cached_weather,
            facts_string = self.memory.get_facts_string(),
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages += self.memory.get_context_messages()

        content = user_text
        if tool_result:
            content += f"\n\n[Context: {tool_result}]"
        messages.append({"role": "user", "content": content})

        try:
            client   = ollama.Client(host=OLLAMA_HOST)
            response = client.chat(model=OLLAMA_MODEL, messages=messages)
            return response["message"]["content"].strip()
        except Exception as e:
            return f"I'm having trouble reaching my brain — is Ollama running? Error: {e}"

    def morning_greeting(self) -> str:
        greeting = personality.get_greeting()
        time_str = realtime.get_time_response()
        weather  = realtime.get_weather_response()
        todos    = self.todos.morning_summary()
        return f"{greeting} {time_str} {weather} {todos}"

    def start(self):
        self.running = True

        greeting = self.morning_greeting()
        print(f"[JARVIS] {greeting}\n")
        self._status("speaking", greeting)
        self.is_speaking = True
        self.speaker.speak(greeting)
        self.is_speaking = False

        self._status("listening", "")
        self.listener.start()

        try:
            while self.running:
                threading.Event().wait(0.5)
        except KeyboardInterrupt:
            print("\n[JARVIS] Interrupted.")

        self.listener.stop()
        self.memory.save()
        print("[JARVIS] Memory saved. Offline.")