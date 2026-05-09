g@echo off
title JARVIS v2 Setup
echo.
echo  ============================================
echo   J.A.R.V.I.S  v2  Setup  (Windows + RTX)
echo  ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)
echo [OK] Python found

echo.
echo [1/6] Installing PyTorch with CUDA 12.1 for RTX 4050...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
echo [OK] PyTorch installed

echo.
echo [2/6] Installing core dependencies...
pip install faster-whisper kokoro sounddevice soundfile pytz -q
echo [OK] Core deps done

echo.
echo [3/6] Installing real-time and memory deps...
pip install ollama requests -q
echo [OK] Real-time deps done

echo.
echo [4/6] Installing GUI...
pip install PyQt6 -q
echo [OK] PyQt6 installed

echo.
echo [5/6] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama not installed.
    echo        Download from: https://ollama.com/download
    echo        Then run: ollama pull llama3.2
) else (
    echo [OK] Ollama found — pulling llama3.2...
    ollama pull llama3.2
)

echo.
echo [6/6] Verifying GPU...
python -c "import torch; g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'; m=torch.cuda.get_device_properties(0).total_memory//1024**3 if torch.cuda.is_available() else 0; print(f'  GPU: {g} ({m}GB VRAM)')"

echo.
echo  ============================================
echo  JARVIS v2 is ready!
echo.
echo  Commands:
echo    Start (headless):  python main.py
echo    Start (with GUI):  python main.py --gui
echo    Auto-start setup:  python autostart.py
echo    Remove auto-start: python autostart.py --remove
echo.
echo  Before starting, run in a separate terminal:
echo    ollama serve
echo  ============================================
echo.
pause
