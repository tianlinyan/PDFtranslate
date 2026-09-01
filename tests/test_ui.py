"""Tests for the pure (Qt-free) helpers of the main window.

Only functions that need no ``QApplication`` are exercised here, so the suite
stays runnable on a headless machine.
"""

from __future__ import annotations

import unittest

from translate_app.main_window import LANGUAGES, resolve_language


class ResolveLanguageTest(unittest.TestCase):
    def test_every_listed_label_maps_to_its_model_name(self):
        for label, code in LANGUAGES:
            self.assertEqual(code, resolve_language(label))

    def test_language_name_maps_to_itself(self):
        self.assertEqual("Simplified Chinese", resolve_language("Simplified Chinese"))
        self.assertEqual("German", resolve_language("  german  "))

    def test_typed_language_is_honoured(self):
        """Regression: the combo is editable, but the code read
        ``currentData()`` — which still points at the previously selected item
        when the typed text matches nothing.  A typed language was silently
        replaced by the last selected one, in both the request and the output
        file name."""
        self.assertEqual("Português", resolve_language("Português"))
        self.assertEqual("Japanese", resolve_language("Japanese"))

    def test_empty_input_falls_back_to_the_first_language(self):
        self.assertEqual(LANGUAGES[0][1], resolve_language(""))
        self.assertEqual(LANGUAGES[0][1], resolve_language("   "))


if __name__ == "__main__":
    unittest.main()
