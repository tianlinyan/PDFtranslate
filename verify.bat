@echo off
rem verify_doclayout.bat - run verify_doclayout.py to confirm DocLayout is active.
rem Usage: verify.bat [doc.pdf] [--probe]
rem   structure_parser == "doclayout"  => DocLayout really running (model produced regions)
rem   otherwise ("geo"/"")              => degraded to the geometric backend
set "PY=python"
if exist "C:\pv\Scripts\python.exe" set "PY=C:\pv\Scripts\python.exe"
cd /d "%~dp0"
"%PY%" verify_doclayout.py %*