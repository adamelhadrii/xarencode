import json
import unittest
from types import SimpleNamespace as NS

from xarencode.llm import ToolCallDelta
from xarencode.toolcall import CallAccumulator, repair_json


class FakeRegistry:
    def __init__(self):
        self.tools = {
            "read_file": NS(name="read_file", read_only=True, parameters={
                "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
                "required": ["path"]}),
            "run_shell": NS(name="run_shell", read_only=False, parameters={
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}),
            "write_file": NS(name="write_file", read_only=False, parameters={
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
            "vm": NS(name="vm", read_only=False, parameters={
                "properties": {"action": {"type": "string", "enum": ["start", "stop"]}, "force": {"type": "boolean"}},
                "required": ["action"]}),
        }

    def get(self, n):
        return self.tools.get(n)

    def names(self):
        return list(self.tools)


def run(parts, content="", finish="tool_calls"):
    acc = CallAccumulator()
    for p in parts:
        acc.feed(ToolCallDelta(*p))
    return acc.finish(content, finish, FakeRegistry())


class RepairTests(unittest.TestCase):
    def test_valid_and_double_encoded(self):
        self.assertEqual(repair_json('{"a":1}'), ({"a": 1}, False))
        self.assertEqual(repair_json(json.dumps('{"a":1}')), ({"a": 1}, False))
        self.assertEqual(repair_json(""), ({}, False))

    def test_lossless(self):
        self.assertEqual(repair_json('{"a": 1,}'), ({"a": 1}, False))
        self.assertEqual(repair_json("```json\n{\"a\": 1}\n```"), ({"a": 1}, False))
        self.assertEqual(repair_json("{'a': True}"), ({"a": True}, False))

    def test_lossy(self):
        obj, lossy = repair_json('{"path": "a.py", "limit": 5')
        self.assertEqual(obj, {"path": "a.py", "limit": 5})
        self.assertTrue(lossy)
        obj, lossy = repair_json('{"path": "a.p')
        self.assertTrue(lossy)
        self.assertEqual(obj, {"path": "a.p"})
        self.assertEqual(repair_json("nonsense"), (None, True))


class AccumulatorTests(unittest.TestCase):
    def test_fragments_joined(self):
        _, calls = run([(0, "c1", "read_file", '{"pa'), (0, None, None, 'th": "x"}')])
        self.assertEqual(calls[0].args, {"path": "x"})
        self.assertIsNone(calls[0].error)
        self.assertEqual(calls[0].id, "c1")

    def test_missing_id(self):
        _, calls = run([(0, None, "read_file", '{"path":"x"}')])
        self.assertEqual(calls[0].id, "call_1")

    def test_aliases(self):
        _, calls = run([(0, "a", "bash", '{"cmd": "ls"}')])
        self.assertEqual((calls[0].name, calls[0].args, calls[0].error), ("run_shell", {"command": "ls"}, None))
        _, calls = run([(0, "a", "cat", '{"file_path": "x", "limit": "12"}')])
        self.assertEqual(calls[0].args, {"path": "x", "limit": 12})

    def test_unknown_tool_suggestion(self):
        _, calls = run([(0, "a", "read_fil", "{}")])
        self.assertIn("read_file", calls[0].error)

    def test_unknown_args_dropped_and_required(self):
        _, calls = run([(0, "a", "read_file", '{"path": "x", "bogus": 1}')])
        self.assertEqual(calls[0].args, {"path": "x"})
        _, calls = run([(0, "a", "read_file", "{}")])
        self.assertIn("missing", calls[0].error)

    def test_enum_and_bool(self):
        _, calls = run([(0, "a", "vm", '{"action": "reboot"}')])
        self.assertIn("one of", calls[0].error)
        _, calls = run([(0, "a", "vm", '{"action": "stop", "force": "true"}')])
        self.assertEqual(calls[0].args["force"], True)

    def test_trailing_comma_repaired(self):
        _, calls = run([(0, "a", "run_shell", '{"command": "ls",}')])
        self.assertIsNone(calls[0].error)

    def test_lossy_only_for_read_only(self):
        _, calls = run([(0, "a", "read_file", '{"path": "x"')])
        self.assertIsNone(calls[0].error)
        _, calls = run([(0, "a", "run_shell", '{"command": "rm x"')])
        self.assertIn("malformed", calls[0].error)

    def test_length_cuts_everything(self):
        _, calls = run([(0, "a", "write_file", '{"path": "x", "content": "ab')], finish="length")
        self.assertIn("cut off", calls[0].error)
        self.assertEqual(calls[0].history_args, json.dumps(calls[0].args) if calls[0].args else "{}")

    def test_text_fallback(self):
        text = 'Sure.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a"}}\n</tool_call>'
        content, calls = run([], text, "stop")
        self.assertEqual(content, "Sure.")
        self.assertEqual((calls[0].name, calls[0].args, calls[0].source), ("read_file", {"path": "a"}, "text"))

    def test_text_fallback_bare_json_and_fence(self):
        content, calls = run([], '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```', "stop")
        self.assertEqual(content, "")
        self.assertEqual(calls[0].name, "run_shell")

    def test_plain_text_not_a_call(self):
        content, calls = run([], "just {braces} here", "stop")
        self.assertEqual((content, calls), ("just {braces} here", []))

    def test_think_removed(self):
        acc = CallAccumulator()
        content, _ = acc.finish("<think>hmm</think>Answer", "stop", FakeRegistry())
        self.assertEqual(content, "Answer")
        self.assertEqual(acc.reasoning, "hmm")

    def test_unparseable_history_args_valid_json(self):
        _, calls = run([(0, "a", "run_shell", "garbage")])
        self.assertTrue(calls[0].error)
        self.assertEqual(json.loads(calls[0].history_args), {})


if __name__ == "__main__":
    unittest.main()
