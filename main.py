"""
main.py — Entry point.
Run this file to start JARVIS.

    python main.py           # headless (terminal only)
    python main.py --gui     # with GUI dashboard
"""

import sys
import argparse
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from voice_auth import is_my_voice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_voice():
    """Listen and verify if it's Hasan's voice before starting JARVIS."""
    print("🎤 Say something to verify your voice...")

    sample_rate = 16000
    duration = 5  # seconds

    try:
        # Record using sounddevice (no PyAudio needed!)
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()  # Wait until recording is done

        # Save temporarily for resemblyzer
        temp_path = "temp_verify.wav"
        sf.write(temp_path, audio, sample_rate)

        result = is_my_voice(temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:
        print(f"❌ Recording error: {e}")
        return False


def run_headless():
    from jarvis import JARVIS
    j = JARVIS()
    j.start()


def run_gui():
    from PyQt6.QtWidgets import QApplication
    from gui import JARVISDashboard
    app = QApplication(sys.argv)
    app.setApplicationName("JARVIS")
    window = JARVISDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS Personal AI Assistant")
    parser.add_argument("--gui", action="store_true", help="Launch with GUI dashboard")
    args = parser.parse_args()

    # ✅ Voice verification before launching
    print("🔐 JARVIS Voice Authentication")
    print("================================")

    attempts = 0
    max_attempts = 3
    authenticated = False

    while attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts} of {3}...")

        if verify_voice():
            print("✅ Voice verified! Welcome back, Hasan. Starting JARVIS...")
            authenticated = True
            break
        else:
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"❌ Voice not recognized. {remaining} attempt(s) left.")
            else:
                print("🚫 Access denied. Too many failed attempts.")

    if authenticated:
        if args.gui:
            run_gui()
        else:
            run_headless()
    else:
        print("🔒 JARVIS will not start. Goodbye.")
        sys.exit(1)