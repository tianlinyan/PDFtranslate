"""Test package: makes the project root importable and quarantines user state.

Preferences are read **implicitly** when ``MainWindow`` is constructed, so a test
that merely builds the GUI would otherwise pick up (and could overwrite) the
developer's real ``~/.pdftranslate/prefs.json`` — a saved preference then changes
a widget default on one machine only, i.e. a red suite that is not the code's
fault (that is how ``image_text`` broke ``test_qt_wiring``).  Redirect the file
to a private temp directory here, before any test imports ``translate_app``;
``PDFTRANSLATE_PREFS_PATH`` is honoured (see ``settings.prefs_path``), so an
outer runner can still point it elsewhere.

Cache directories stay opt-in per test module (``PDFTRANSLATE_CACHE_DIR`` /
``PDFTRANSLATE_OCR_CACHE_DIR``) — those tests set them deliberately because the
env var is also what *activates* disk caching; prefs have no such switch, they
only ever need to be moved out of the way.
"""
import atexit
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_PREFS_DIR = tempfile.mkdtemp(prefix="pdftranslate_test_prefs_")
os.environ.setdefault(
    "PDFTRANSLATE_PREFS_PATH", os.path.join(_PREFS_DIR, "prefs.json")
)
atexit.register(shutil.rmtree, _PREFS_DIR, True)
