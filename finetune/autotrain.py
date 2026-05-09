"""
auto_train.py
Runs in the background and automatically retrains JARVIS.

Two triggers:
  1. Every night at midnight
  2. Whenever 20+ new turns accumulate (checked every 30 mins)

Incremental learning:
  Each run builds on the PREVIOUS trained model — not base llama.
  Permanent core examples always included — model never forgets basics.
"""

import json
import time
import subprocess
import schedule
from pathlib import Path
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
MEMORY_FILE       = "../data/memory.json"
LAST_TRAINED_FILE = "last_trained.json"
LOG_FILE          = "auto_train.log"
MIN_NEW_TURNS     = 20     # minimum new turns before retraining
CHECK_INTERVAL    = 30     # check every 30 minutes


# ── Logging ────────────────────────────────────────────────────────────────────

def log(message: str):
    """Write to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line      = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Turn counting ──────────────────────────────────────────────────────────────

def get_current_turn_count() -> int:
    """Count turns currently in memory.json."""
    try:
        data = json.loads(Path(MEMORY_FILE).read_text(encoding="utf-8"))
        return len(data.get("turns", []))
    except:
        return 0

def get_last_trained_count() -> int:
    """How many turns existed when we last trained."""
    try:
        data = json.loads(Path(LAST_TRAINED_FILE).read_text())
        return data.get("turn_count", 0)
    except:
        return 0

def get_new_turn_count() -> int:
    return get_current_turn_count() - get_last_trained_count()


# ── Training pipeline ──────────────────────────────────────────────────────────

def run_step(command: list, step_name: str) -> bool:
    """
    Run a subprocess step. Returns True if succeeded.
    Streams output live so you can watch progress.
    """
    log(f"Starting: {step_name}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"FAILED: {step_name}")
        log(f"Error: {result.stderr[-500:]}")   # last 500 chars of error
        return False

    log(f"Completed: {step_name}")
    return True


def run_full_pipeline():
    """
    Execute the complete training pipeline:
      1. Clean memory → dataset.json
      2. Train model  → jarvis-finetuned/
      3. Export GGUF  → handled by train.py
      4. Reload Ollama model
    """
    new_turns = get_new_turn_count()
    current   = get_current_turn_count()

    log("="*40)
    log(f"STARTING TRAINING PIPELINE")
    log(f"New turns: {new_turns}")
    log(f"Total turns: {current}")
    log("="*40)

    start_time = time.time()

    # Step 1 — clean memory into dataset
    # Pass save_checkpoint=False here — train.py will save it after success
    ok = run_step(
        ["python", "clean_memory.py"],
        "Cleaning memory.json into dataset"
    )
    if not ok:
        log("Pipeline aborted at step 1.")
        return False

    # Step 2 — train
    # This automatically loads previous jarvis model if it exists
    ok = run_step(
        ["python", "train.py"],
        "Training model (incremental)"
    )
    if not ok:
        log("Pipeline aborted at step 2.")
        return False

    # Step 3 — reload into Ollama
    ok = run_step(
        ["ollama", "create", "jarvis", "-f", "Modelfile"],
        "Loading updated model into Ollama"
    )
    if not ok:
        log("Pipeline aborted at step 3.")
        return False

    elapsed = round((time.time() - start_time) / 60, 1)
    log(f"PIPELINE COMPLETE in {elapsed} minutes")
    log(f"Jarvis updated with {new_turns} new conversations")
    log("="*40)
    return True


# ── Scheduled triggers ─────────────────────────────────────────────────────────

def check_and_train_if_ready():
    """
    Called every 30 minutes.
    Only trains if MIN_NEW_TURNS threshold is reached.
    """
    new_turns = get_new_turn_count()

    if new_turns == 0:
        return   # nothing new at all

    if new_turns < MIN_NEW_TURNS:
        log(f"Checked: {new_turns} new turns (need {MIN_NEW_TURNS}). Waiting.")
        return

    log(f"Threshold reached: {new_turns} new turns. Training now.")
    run_full_pipeline()


def midnight_training():
    """
    Called every midnight.
    Trains if ANY new turns exist — even less than MIN_NEW_TURNS.
    Because at midnight we want to capture everything from the day.
    """
    new_turns = get_new_turn_count()

    if new_turns == 0:
        log("Midnight check: no new turns. Skipping.")
        return

    log(f"Midnight training: {new_turns} new turns found.")
    run_full_pipeline()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("AUTO-TRAIN started")
    log(f"Watching: {MEMORY_FILE}")
    log(f"Threshold: {MIN_NEW_TURNS} new turns")
    log(f"Checking every: {CHECK_INTERVAL} minutes")
    log(f"Midnight training: enabled")

    # Run once immediately on startup
    # This handles the case where you restart auto_train
    # and there is already enough data waiting
    initial_new = get_new_turn_count()
    if initial_new >= MIN_NEW_TURNS:
        log(f"Startup check: {initial_new} new turns found. Training now.")
        run_full_pipeline()
    else:
        log(f"Startup check: {initial_new} new turns. Waiting for {MIN_NEW_TURNS}.")

    # Schedule midnight training
    schedule.every().day.at("00:00").do(midnight_training)

    # Schedule threshold check every 30 minutes
    schedule.every(CHECK_INTERVAL).minutes.do(check_and_train_if_ready)

    log("Scheduler running. Press Ctrl+C to stop.\n")

    while True:
        schedule.run_pending()
        time.sleep(60)