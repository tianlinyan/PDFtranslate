# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building the PDF Translate app (Windows, onedir).

Build (from the project root):
    python -m PyInstaller --noconfirm --clean PDFTranslate.spec

Output goes to ``dist/PDFTranslate/``:
    PDFTranslate.exe      <- the executable
    _internal/            <- bundled Python runtime + dependencies
    models.json           <- copied next to the executable by ``copy_models.py``
                            (edit this file to declare your AI models)

The app resolves ``models.json`` / ``glossary.json`` next to the executable when
frozen (see ``translate_app/settings.py`` -> ``resource_dir``), so the user can
edit them without rebuilding.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# RapidOCR ships its ONNX detection/recognition/classification models as package
# data (``models/*.onnx``) plus per-component ``config.yaml`` files.  Collect all
# of it so OCR works in the frozen app.  onnxruntime / opencv / numpy / PyQt6 /
# pymupdf are pulled in by PyInstaller's own hooks.
datas = collect_data_files("rapidocr_onnxruntime")

# Optional safety net: pull in submodules that are imported dynamically.  The
# built-in hooks already cover onnxruntime, cv2, numpy, pymupdf and PyQt6, so we
# keep the list minimal.
hiddenimports = collect_submodules("rapidocr_onnxruntime")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PDFTranslate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PDFTranslate",
)
