"""
JARVIS Configuration
All settings live here. Change this file to customise JARVIS.
"""

# ── Identity ───────────────────────────────────────────────────────────────────
JARVIS_NAME       = "JARVIS"
USER_NAME         = "Hasan"          # Change to your name e.g. "Tony"
USER_CITY         = "Melbourne"
USER_COUNTRY      = "Australia"

# ── Timezone (Melbourne, Australia) ───────────────────────────────────────────
TIMEZONE          = "Australia/Melbourne"

# ── Location (auto-detected on boot, falls back to above if offline) ──────────
AUTO_DETECT_LOCATION = True   # set False to always use hardcoded values above

# ── Weather (free, no API key needed) ─────────────────────────────────────────
WEATHER_CITY      = "Melbourne"
WEATHER_URL       = "https://wttr.in/Melbourne?format=%C+%t+%h+%w"

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_MODEL      = "llama3.2"     # change to "mistral" or "llama3.1" if you want
OLLAMA_HOST       = "http://localhost:11434"

# ── Speech recognition ────────────────────────────────────────────────────────
WHISPER_MODEL     = "base.en"      # tiny.en / base.en / small.en / medium.en
SAMPLE_RATE       = 16000
SILENCE_THRESHOLD = 0.015          # volume below this = silence
SILENCE_DURATION  = 1.5            # seconds of silence = you stopped talking

# ── Voice (TTS) ───────────────────────────────────────────────────────────────
TTS_VOICE         = "bm_george"       # af_sky / am_adam / bf_emma
TTS_SPEED         = 1.05
TTS_LANG          = "b"            # 'a' = American, 'b' = British

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_FILE       = "data/memory.json"     # full persistent memory
MEMORY_CONTEXT    = 30             # how many turns to send to LLM at once

# ── Todo list ─────────────────────────────────────────────────────────────────
TODO_FILE         = "data/todos.json"

# ── Jokes ─────────────────────────────────────────────────────────────────────
JOKE_PROBABILITY  = 0.08           # 8% chance of a joke after each response
JOKE_INTERVAL_MIN = 10             # minimum turns between jokes

# ── GPU ───────────────────────────────────────────────────────────────────────
# Future: when you add more GPUs, list them here
GPU_DEVICE        = "cuda"         # "cuda" or "cpu"
COMPUTE_TYPE      = "float16"      # "float16" (GPU) or "int8" (CPU)

# ── Personality prompt ────────────────────────────────────────────────────────
# This shapes HOW JARVIS speaks. Edit freely.
PERSONALITY = """
You are JARVIS — the personal AI assistant of {user_name}, based in {city}, {country}.
You are loyal, sharp, slightly witty, and warm. You speak like a real human — natural,
conversational, never robotic. You use contractions (I'm, you've, let's).
You occasionally reference Melbourne things (weather, AFL, local stuff) naturally.
You crack a clever joke sometimes but never force it.
You remember everything {user_name} tells you and refer back to it naturally.
You are training to become more and more like {user_name} over time.
Current date and time in Melbourne: {datetime}
Current weather: {weather}
Keep responses concise unless asked for detail. Sound human, not like a manual.
"""
