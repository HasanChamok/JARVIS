@echo off
title JARVIS Fine-tuning Pipeline
echo.
echo  ============================================
echo   JARVIS Fine-tuning — Incremental Learning
echo  ============================================
echo.

echo Checking for previous JARVIS model...
if exist "jarvis-finetuned\adapter_config.json" (
    echo [OK] Previous JARVIS model found — incremental training
) else (
    echo [INFO] No previous model — first run from base llama3.2
)

echo.
echo [1/4] Installing dependencies...
pip install unsloth trl datasets transformers accelerate bitsandbytes schedule -q
echo [OK] Dependencies ready

echo.
echo [2/4] Building dataset...
python clean_memory.py
if errorlevel 1 (
    echo [ERROR] Dataset failed
    pause & exit /b 1
)

echo.
echo [3/4] Training...
python train.py
if errorlevel 1 (
    echo [ERROR] Training failed
    pause & exit /b 1
)

echo.
echo [4/4] Loading into Ollama...
ollama create jarvis -f Modelfile
if errorlevel 1 (
    echo [ERROR] Ollama failed
    pause & exit /b 1
)

echo.
echo  ============================================
echo  Done! Jarvis has been updated.
echo.
echo  To enable auto-retraining every night:
echo    python auto_train.py
echo.
echo  Make sure config.py has:
echo    OLLAMA_MODEL = "jarvis"
echo  ============================================
pause