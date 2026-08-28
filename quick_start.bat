@echo off
REM Quick Start Script for HunyuanVideo-I2V Video Generation
REM This script helps download models and generate videos

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo HunyuanVideo-I2V Quick Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if huggingface-cli is available
huggingface-cli --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: HuggingFace CLI not found
    echo Installing HuggingFace Hub...
    pip install huggingface-hub
    if errorlevel 1 (
        echo ERROR: Failed to install huggingface-hub
        pause
        exit /b 1
    )
)

echo.
echo Available options:
echo 1. Download models
echo 2. Generate videos from images
echo 3. Check model status
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo ============================================================
    echo Downloading Models (~100GB total)
    echo ============================================================
    echo.
    echo This will take considerable time. Models will be saved to .\ckpts\
    echo.
    echo Starting model download...
    echo.
    
    echo [1/3] Downloading HunyuanVideo-I2V model...
    huggingface-cli download tencent/HunyuanVideo-I2V --local-dir .\ckpts
    if errorlevel 1 (
        echo ERROR: Failed to download main model
        pause
        exit /b 1
    )
    
    echo.
    echo [2/3] Downloading Text Encoder (LLaVA)...
    cd ckpts
    huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir .\text_encoder_i2v
    if errorlevel 1 (
        echo ERROR: Failed to download LLaVA model
        cd ..
        pause
        exit /b 1
    )
    
    echo.
    echo [3/3] Downloading CLIP model...
    huggingface-cli download openai/clip-vit-large-patch14 --local-dir .\text_encoder_2
    if errorlevel 1 (
        echo ERROR: Failed to download CLIP model
        cd ..
        pause
        exit /b 1
    )
    cd ..
    
    echo.
    echo ============================================================
    echo All models downloaded successfully!
    echo ============================================================
    echo.
    
    goto :menu

) else if "%choice%"=="2" (
    echo.
    echo ============================================================
    echo Generating videos from images
    echo ============================================================
    echo.
    
    python generate_videos_i2v.py
    
    if errorlevel 1 (
        echo.
        echo ERROR: Video generation failed
        echo Please check the error messages above
        pause
        exit /b 1
    )
    
    echo.
    echo ============================================================
    echo Video generation completed!
    echo Check output_videos\ directory for results
    echo ============================================================
    echo.
    pause
    
    goto :menu

) else if "%choice%"=="3" (
    echo.
    echo ============================================================
    echo Checking model status...
    echo ============================================================
    echo.
    
    python -c "
import os
models = {
    'HunyuanVideo-I2V': './ckpts/hunyuan-video-i2v-720p',
    'Text Encoder I2V': './ckpts/text_encoder_i2v',
    'Text Encoder 2': './ckpts/text_encoder_2',
    'VAE': './ckpts/vae'
}
print()
for name, path in models.items():
    status = '✓ Available' if os.path.exists(path) else '✗ Missing'
    print(f'  {status} - {name}')
print()
"
    
    pause
    goto :menu

) else if "%choice%"=="4" (
    echo Exiting...
    exit /b 0

) else (
    echo Invalid choice
    goto :menu
)

:menu
cls
goto :eof
