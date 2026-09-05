@echo off
rem Launcher for PDF Translate on Windows.
rem Recommended defaults (0.3.8): semantic structure ON, IR pipeline OFF.
rem
rem These are FORCED here (whether or not the parent environment already has a
rem PDFTRANSLATE_* variable) so every launch behaves the same and you never get
rem surprised by a stray environment variable.  To opt into the headless IR
rem pipeline, edit the next line to `set "PDFTRANSLATE_IR_MODE=1"` (or launch
rem `python main.py` directly with your own environment).
set "PDFTRANSLATE_STRUCTURE_MODE=1"
set "PDFTRANSLATE_IR_MODE="
rem Optional argument: a PDF file to open on startup.
cd /d "%~dp0"
python main.py %*
