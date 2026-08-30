@echo off
setlocal
rem Build the PDF Translate app with PyInstaller (onedir) and drop models.json
rem next to the executable so it can be edited/configured without rebuilding.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (echo Python not found in PATH & exit /b 1)

echo [1/2] Running PyInstaller (onedir)...
python -m PyInstaller --noconfirm --clean PDFTranslate.spec
if errorlevel 1 (echo Build failed. & exit /b 1)

echo [2/2] Copying models.json next to the executable...
copy /Y "models.json" "dist\PDFTranslate\models.json" >nul
if errorlevel 1 (echo Failed to copy models.json & exit /b 1)

echo.
echo Done. Executable: dist\PDFTranslate\PDFTranslate.exe
echo models.json lives next to it - edit to configure your AI models.
endlocal
