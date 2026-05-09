"""
jarvis.py — Main brain. Orchestrates all modules.

Architecture:
  Listener (mic) → on_speech() → intent router → module or LLM → Speaker (TTS)

Module pipeline (in order):
  1. personality.py  — greetings, jokes, hello
  2. todo.py         — add/delete/update/list tasks
  3. memory.py       — remember facts, search history
  4. files.py        — open files and folders
  5. apps.py         — launch applications
  6. realtime.py     — weather, time, search, system stats
  7. LLM fallback    — everything else → Ollama
"""

import threading
import ollama

from config import OLLAMA_MODEL, OLLAMA_HOST, USER_NAME
from modules.speech      import Listener, Speaker
from modules.memory      import Memory
from modules.todo        import TodoManager
from modules import files, apps, realtime, personality
from modules.realtime import get_location


class JARVIS:

    def __init__(self, status_callback=None):
        print("\n[JARVIS] Booting up...\n")

        # ── Core modules ───────────────────────────────────────────────────────
        self.memory      = Memory()
        self.todos       = TodoManager()
        self.speaker     = Speaker()
        self.joke_engine = personality.JokeEngine()

        # ── Speech ─────────────────────────────────────────────────────────────
        self.listener = Listener(on_speech_callback=self.on_speech)

        # ── State ──────────────────────────────────────────────────────────────
        self.running     = False
        self.status_cb   = status_callback   # hook for GUI

        # ── Weather cache (fetch once per session) ─────────────────────────────
        self._cached_weather = realtime.get_weather()
        # Auto-detect location
        self.location = get_location()
        print(f"[JARVIS] Location: {self.location['city']}, {self.location['country']}")
        print("[JARVIS] All systems ready.\n")

    # ── Status hook for GUI ────────────────────────────────────────────────────

    def _status(self, state: str, text: str = ""):
        if self.status_cb:
            self.status_cb(state, text)

    # ── Main entry: called when user finishes speaking ─────────────────────────

    def on_speech(self, text: str):
        """
        Called by Listener thread when a complete utterance is transcribed.
        Full pipeline: route → respond → speak.
        """
        print(f"\n[YOU]    {text}")
        self._status("thinking", text)
        self.listener.pause()   # don't pick up JARVIS's own voice

        response = self._route(text)

        # Optionally append a joke
        response = self.joke_engine.maybe_append_joke(response)

        print(f"[JARVIS] {response}\n")
        self._status("speaking", response)

        # Save to permanent memory
        self.memory.add("user",      text)
        self.memory.add("assistant", response)

        # Speak, then resume listening
        self.speaker.speak(response)
        self._status("listening", "")
        self.listener.resume()

    # ── Intent router ──────────────────────────────────────────────────────────

    def _route(self, text: str) -> str:
        """
        Try each module in order. First match wins.
        If nothing matches, fall through to LLM.
        """
        text_lower = text.lower().strip()

        # Exit
        if any(w in text_lower for w in ["goodbye", "shut down", "exit jarvis", "stop jarvis", "go offline"]):
            self.running = False
            return f"Going offline. Take care of yourself, {USER_NAME}."

        # Module pipeline
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

        # Nothing matched — send to LLM
        return self._llm(text)

    # ── LLM call ───────────────────────────────────────────────────────────────

    def _llm(self, user_text: str, tool_result: str = None) -> str:
        """Send to Ollama with full system prompt + conversation history."""

        system_prompt = personality.build_system_prompt(
            weather      = self._cached_weather,
            facts_string = self.memory.get_facts_string(),
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages += self.memory.get_context_messages()

        # If a tool gave us extra data, include it
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

    # ── Morning greeting ───────────────────────────────────────────────────────

    def morning_greeting(self) -> str:
        """
        Full greeting spoken on startup:
        hi + time + weather + todo summary.
        """
        greeting = personality.get_greeting()
        time_str = realtime.get_time_response()
        weather  = realtime.get_weather_response()
        todos    = self.todos.morning_summary()

        return f"{greeting} {time_str} {weather} {todos}"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Start JARVIS. Blocks until shutdown."""
        self.running = True

        # Greet on startup
        greeting = self.morning_greeting()
        print(f"[JARVIS] {greeting}\n")
        self._status("speaking", greeting)
        self.speaker.speak(greeting)

        # Start listening
        self._status("listening", "")
        self.listener.start()

        # Keep alive (main thread waits here)
        try:
            while self.running:
                threading.Event().wait(0.5)
        except KeyboardInterrupt:
            print("\n[JARVIS] Interrupted.")

        # Clean shutdown
        self.listener.stop()
        self.memory.save()
        print("[JARVIS] Memory saved. Offline.")
