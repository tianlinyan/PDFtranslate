@echo off
rem Launcher for PDF Translate on Windows.
rem DocLayout semantic structure (B-4) is enabled below; it degrades to the
rem geometric backend automatically when doclayout-yolo / the model is absent.
set "PDFTRANSLATE_STRUCTURE_PARSER=doclayout"
set "PDFTRANSLATE_DOCLAYOUT_DEVICE=cpu"
rem Cross-page DocLayout parallelism (worker processes; default 1 = sequential).
rem Set to your CPU core count to make a many-page PDF use multiple cores for the
rem structure extraction.  Uncomment to override (1 = sequential, no multiprocessing).
set "PDFTRANSLATE_STRUCTURE_CONCURRENCY=8"
rem Prefer the short-path venv (C:\pv, which carries DocLayout-YOLO and inherits the
rem system site-packages).  Falls back to the system python when it is absent.
set "PY=python"
if exist "C:\pv\Scripts\python.exe" set "PY=C:\pv\Scripts\python.exe"
rem Optional PDF to open on startup:  run.bat "path\to\doc.pdf"
cd /d "%~dp0"
"%PY%" main.py %*