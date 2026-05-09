"""
main.py — Entry point.
Run this file to start JARVIS.

    python main.py           # headless (terminal only)
    python main.py --gui     # with GUI dashboard
"""

import sys
import argparse
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    if args.gui:
        run_gui()
    else:
        run_headless()
