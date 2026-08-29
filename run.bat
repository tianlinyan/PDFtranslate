@echo off
rem Launcher for PDF Translate on Windows.
rem Optional argument: a PDF file to open on startup.
cd /d "%~dp0"
python main.py %*
