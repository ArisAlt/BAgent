@echo off
SETLOCAL EnableDelayedExpansion

SET "TESSERACT_DIR=C:\Program Files\Tesseract-OCR"

REM Check if directory exists
IF NOT EXIST "%TESSERACT_DIR%\tesseract.exe" (
    echo ❌ Tesseract not found at: %TESSERACT_DIR%
    echo Please install it or update the script with the correct path.
    pause
    exit /b 1
)

REM Get current PATH
FOR /F "tokens=2*" %%A IN ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') DO (
    SET "current_path=%%B"
)

echo 🔍 Checking current PATH...

echo !current_path! | find /I "%TESSERACT_DIR%" >nul
IF !ERRORLEVEL! EQU 0 (
    echo ✅ Tesseract is already in PATH.
    pause
    exit /b 0
)

REM Append to system PATH
SET "new_path=!current_path!;%TESSERACT_DIR%"

echo 🔧 Adding Tesseract to system PATH...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path /t REG_EXPAND_SZ /d "!new_path!" /f >nul

echo ✅ Tesseract added to PATH.
echo 🔁 You must RESTART your computer or log out/in for changes to take effect.
pause
