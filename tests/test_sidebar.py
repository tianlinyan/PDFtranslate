"""AnswerBridge: a user decision waits — it never silently times out / skips.

Regression: the old ``timeout=600`` made an unanswered question return ``None``; the
flow then proceeded as "未选择" (keep) and stacked a second question row.  The default
is now indefinite wait, with an optional ``cancel`` probe so a worker cancellation
interrupts the wait instead of the flow hanging or skipping.
"""
from __future__ import annotations

import threading
import time
import unittest

from translate_app.sidebar import AnswerBridge


class AnswerBridgeTest(unittest.TestCase):
    def test_ask_blocks_until_answer(self):
        bridge = AnswerBridge()
        result: dict = {}

        def asker():
            result["v"] = bridge.ask("是否保留原文？", ["保留", "翻译"], "blk:1")

        t = threading.Thread(target=asker, daemon=True)
        t.start()
        time.sleep(0.15)
        # Still waiting — the unanswered question is NOT silently skipped.
        self.assertIsNone(result.get("v"))
        bridge.answer("保留", "blk:1")
        t.join(timeout=2)
        self.assertEqual({"value": "保留", "target": "blk:1"}, result["v"])

    def test_ask_interrupted_by_cancel(self):
        cancel = threading.Event()
        bridge = AnswerBridge(cancel=cancel.is_set)
        result: dict = {}

        def asker():
            result["v"] = bridge.ask("q", ["a"])

        t = threading.Thread(target=asker, daemon=True)
        t.start()
        time.sleep(0.15)
        cancel.set()
        t.join(timeout=2)
        self.assertIsNone(result["v"])   # cancel returns None, not a fake "keep"

    def test_set_cancel_retargets(self):
        bridge = AnswerBridge()
        cancel = threading.Event()
        bridge.set_cancel(cancel.is_set)
        self.assertIsNotNone(bridge._cancel)

    def test_is_pending_tracks_question_lifecycle(self):
        bridge = AnswerBridge()
        self.assertFalse(bridge.is_pending())
        result: dict = {}

        def asker():
            result["v"] = bridge.ask("q", ["a"], "x")

        t = threading.Thread(target=asker, daemon=True)
        t.start()
        try:
            time.sleep(0.1)
            self.assertTrue(bridge.is_pending())        # question is up, awaiting answer
            self.assertEqual("x", bridge.pending_target)
            bridge.answer("keep", "x")
            t.join(timeout=2)
            self.assertFalse(bridge.is_pending())       # answered → no longer pending
            self.assertEqual({"value": "keep", "target": "x"}, result["v"])
        finally:
            t.join(timeout=1)

    def test_cancel_clears_pending(self):
        # A cancelled ask returns None AND clears the pending flag, so the next user
        # message is not wrongly routed as an answer.
        cancel = threading.Event()
        bridge = AnswerBridge(cancel=cancel.is_set)
        result: dict = {}

        def asker():
            result["v"] = bridge.ask("q", ["a"])

        t = threading.Thread(target=asker, daemon=True)
        t.start()
        try:
            time.sleep(0.1)
            self.assertTrue(bridge.is_pending())
            cancel.set()
            t.join(timeout=2)
            self.assertIsNone(result["v"])
            self.assertFalse(bridge.is_pending())
        finally:
            t.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
