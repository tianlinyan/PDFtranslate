"""Copy runtime-editable assets next to the executable after a PyInstaller
onedir build.

Invoked by ``build.bat``.  Keeping the Chinese manual filename here (rather than
in the batch file) makes the copy robust on any Windows code page — Python
handles Unicode paths natively.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "PDFTranslate"

#: source (project root) -> destination directory (output bundle).
ASSETS: dict[Path, Path] = {
    ROOT / "models.json": DIST,
    ROOT / "AI配置手册.md": DIST,
}


def main() -> int:
    failures = 0
    for src, dst_dir in ASSETS.items():
        if not src.exists():
            print(f"[skip] missing source: {src.name}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        target = dst_dir / src.name
        shutil.copy2(src, target)
        print(f"[ok] copied {src.name} -> {target}")
        if not target.exists():
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
