@echo off
setlocal
rem Build the PDF Translate app with PyInstaller (onedir) and drop config assets
rem (models.json + AI config manual) next to the executable so they can be edited
rem without rebuilding.  The copies are done by copy_assets.py, which keeps the
rem Chinese filename out of this batch file for code-page safety.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (echo Python not found in PATH & exit /b 1)

echo [1/2] Running PyInstaller (onedir)...
python -m PyInstaller --noconfirm --clean PDFTranslate.spec
if errorlevel 1 (echo Build failed. & exit /b 1)

echo [2/2] Copying config assets next to the executable...
python copy_assets.py
if errorlevel 1 (echo Failed to copy assets & exit /b 1)

echo.
echo Done. Executable: dist\PDFTranslate\PDFTranslate.exe
echo models.json and the AI config manual live next to it - edit to configure.
endlocal
