@echo off
title JARVIS Fine-tuning Pipeline
echo.
echo  ============================================
echo   JARVIS Level 3 Fine-tuning — RTX 4050
echo  ============================================
echo.

echo [1/4] Installing Unsloth and dependencies...
pip install unsloth trl datasets transformers accelerate bitsandbytes -q
echo [OK] Dependencies installed

echo.
echo [2/4] Cleaning memory.json into training dataset...
python clean_memory.py
if errorlevel 1 (
    echo [ERROR] Dataset generation failed
    pause & exit /b 1
)
echo [OK] Dataset ready

echo.
echo [3/4] Running fine-tuning (this will take 10-30 mins)...
echo       Watch your GPU temperature — keep it under 85C
python train.py
if errorlevel 1 (
    echo [ERROR] Training failed
    pause & exit /b 1
)
echo [OK] Training complete

echo.
echo [4/4] Loading fine-tuned model into Ollama...
ollama create jarvis -f Modelfile
if errorlevel 1 (
    echo [ERROR] Ollama model creation failed
    pause & exit /b 1
)
echo [OK] JARVIS model loaded into Ollama

echo.
echo  ============================================
echo  Done! Update config.py:
echo    OLLAMA_MODEL = "jarvis"
echo  Then run: python main.py
echo  ============================================
echo.
pause