"""
autostart.py
Run this once to install JARVIS as a Windows startup program.
After running, JARVIS will launch automatically every time you log in.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_startup():
    """Add JARVIS to Windows startup via Task Scheduler."""

    python_exe  = sys.executable
    jarvis_dir  = Path(__file__).parent.resolve()
    main_script = jarvis_dir / "main.py"

    task_name = "JARVIS_Autostart"

    # Build the scheduled task command
    cmd = (
        f'schtasks /create /tn "{task_name}" '
        f'/tr "\\"{python_exe}\\" \\"{main_script}\\"" '
        f'/sc onlogon '
        f'/rl limited '
        f'/f'   # force overwrite if exists
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ JARVIS will now start automatically when you log in.")
        print(f"   Script: {main_script}")
        print(f"   Python: {python_exe}")
        print(f"\n   To remove: schtasks /delete /tn \"{task_name}\" /f")
    else:
        print("❌ Task Scheduler failed. Trying startup folder method...")
        _install_via_startup_folder(python_exe, main_script)


def _install_via_startup_folder(python_exe: str, main_script: Path):
    """Fallback: add a .bat file to Windows startup folder."""
    startup = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
    bat_path = startup / "JARVIS.bat"

    # The bat file changes to the right directory first
    bat_content = f"""@echo off
cd /d "{main_script.parent}"
start /min "" "{python_exe}" "{main_script}"
"""
    bat_path.write_text(bat_content)
    print(f"✅ Added JARVIS to startup folder: {bat_path}")
    print("   JARVIS will launch minimised every time you log in.")


def remove_startup():
    """Remove JARVIS from startup."""
    subprocess.run('schtasks /delete /tn "JARVIS_Autostart" /f', shell=True)
    # Also remove bat if it exists
    startup = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/JARVIS.bat"
    if startup.exists():
        startup.unlink()
    print("✅ JARVIS removed from startup.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true", help="Remove from startup")
    args = parser.parse_args()

    if args.remove:
        remove_startup()
    else:
        install_startup()
