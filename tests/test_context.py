import json
import unittest

from xarencode.context import TRIM_NOTE, ContextManager


def tool_exchange(i, size=200):
    return [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                                                              "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * size},
        {"role": "assistant", "content": f"answer {i}"},
    ]


class ContextTests(unittest.TestCase):
    def test_estimate_formula(self):
        cm = ContextManager(4096)
        self.assertEqual(cm.raw_tokens([{"role": "user", "content": "a" * 30}]), 10 + 6)
        self.assertEqual(cm.budget, 4096 - 768 - 64)

    def test_no_trim_under_budget(self):
        cm = ContextManager(4096)
        hist = tool_exchange(1)
        msgs, est = cm.fit("sys", [], hist)
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertLess(est, cm.budget)

    def test_elides_old_tool_results_first(self):
        cm = ContextManager(2000)
        hist = sum((tool_exchange(i, 1000) for i in range(4)), [])
        cm.trim("sys", [], hist, 0.9)
        tools = [m for m in hist if m["role"] == "tool"]
        self.assertTrue(tools[0]["content"].startswith("[elided: read_file f0.py"))
        self.assertEqual(len([m for m in hist if m["role"] == "user"]), 4)  # nothing dropped yet

    def test_drops_oldest_exchanges_keeping_pairs(self):
        cm = ContextManager(1200)
        hist = sum((tool_exchange(i, 600) for i in range(6)), [])
        cm.fit("sys", [], hist)
        self.assertEqual(hist[0]["content"], TRIM_NOTE)
        self.assertEqual(hist[0]["role"], "user")
        ids_calls = {tc["id"] for m in hist for tc in m.get("tool_calls") or []}
        ids_results = {m["tool_call_id"] for m in hist if m["role"] == "tool"}
        self.assertEqual(ids_calls, ids_results)
        self.assertIn("question 5", [m["content"] for m in hist])
        self.assertEqual(sum(1 for m in hist if m.get("content") == TRIM_NOTE), 1)

    def test_trims_to_sixty_percent(self):
        cm = ContextManager(4096)
        hist = sum((tool_exchange(i, 0) for i in range(60)), [])
        for h in hist:
            if h["role"] == "assistant" and h["content"]:
                h["content"] = "w" * 200
        before = cm.estimate("s", [], hist)
        self.assertGreater(before, cm.budget)
        cm.fit("s", [], hist)
        self.assertLessEqual(cm.estimate("s", [], hist), int(cm.budget * 0.6))

    def test_huge_single_message_is_shortened(self):
        cm = ContextManager(2048)
        hist = [{"role": "user", "content": "z" * 20000}]
        cm.fit("s", [], hist)
        self.assertIn("[... trimmed ...]", hist[0]["content"])
        self.assertEqual(len(hist), 1)

    def test_calibration_clamped(self):
        cm = ContextManager(4096)
        cm.estimate("s", [], [{"role": "user", "content": "a" * 300}])
        for _ in range(20):
            cm.observe(100000)
        self.assertEqual(cm.ratio, 1.6)
        for _ in range(40):
            cm.observe(1)
        self.assertEqual(cm.ratio, 0.6)

    def test_force_trim_is_harder(self):
        cm = ContextManager(4096)
        hist = sum((tool_exchange(i, 500) for i in range(8)), [])
        cm.fit("s", [], hist)
        before = cm.estimate("s", [], hist)
        cm.force_trim("s", [], hist)
        self.assertLess(cm.estimate("s", [], hist), before)
        self.assertGreater(cm.ratio, 1.0)

    def test_reply_tokens(self):
        cm = ContextManager(4096)
        self.assertEqual(cm.reply_tokens(1024, 1000), 1024)
        self.assertEqual(cm.reply_tokens(1024, 3500), 4096 - 3500 - 32)
        self.assertEqual(cm.reply_tokens(1024, 4090), 64)


if __name__ == "__main__":
    unittest.main()
