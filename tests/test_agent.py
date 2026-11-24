import json
import os
import threading
import unittest
import unittest.mock

from tests.helpers import IsolatedTestCase, ScriptedFrontend
from tests.mock_llama_server import MockServer
from xarencode.agent import Agent, Approval, cap_output
from xarencode.config import default_config
from xarencode.llm import LlamaClient
from xarencode.tools import build_registry


def call(name, args):
    return {"tool_call": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class AgentTestCase(IsolatedTestCase):
    def make(self, script, approvals=None, **cfg_over):
        self.mock = MockServer(script).start()
        self.addCleanup(self.mock.stop)
        self.cfg = default_config()
        self.cfg["agent"].update(cfg_over)
        self.fe = ScriptedFrontend(approvals)
        self.agent = Agent(self.cfg, LlamaClient(self.mock.url), build_registry(self.cfg), self.fe, "system prompt", self.work)
        return self.agent

    def tool_msgs(self):
        return [m for m in self.agent.history if m["role"] == "tool"]


class YoloTests(AgentTestCase):
    """Normal mode asks about downloads and deletes every time; yolo answers yes to ASK and nothing else."""

    def setUp(self):
        super().setUp()
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name in ("curl", "pip", "npm"):  # stand-ins: no test may touch the network
            (bin_dir / name).write_text("#!/bin/sh\necho fake $0\n")
            (bin_dir / name).chmod(0o755)
        saved = os.environ["PATH"]
        os.environ["PATH"] = f"{bin_dir}:{saved}"
        self.addCleanup(os.environ.__setitem__, "PATH", saved)

    RISKY = ["curl -O https://x", "curl -O https://y", "rm a.txt", "rm b.txt", "pip install x", "pip install y",
             "npm install", "npm install left-pad"]

    def script(self, cmds):
        return [[call("run_shell", {"command": c})] for c in cmds] + [[{"text": "done"}]]

    def test_normal_mode_asks_every_time_even_after_always(self):
        for n in "ab":
            (self.work / f"{n}.txt").write_text("x")
        a = self.make(self.script(self.RISKY), approvals=[Approval("always")] * len(self.RISKY))
        a.run_turn("go")
        self.assertEqual(len(self.fe.names("approve")), len(self.RISKY))
        self.assertEqual(a.approvals, set())
        self.assertFalse((self.work / "a.txt").exists())  # approved, so it ran

    def test_yolo_approves_downloads_and_deletes_without_asking(self):
        for n in "ab":
            (self.work / f"{n}.txt").write_text("x")
        a = self.make(self.script(self.RISKY))
        a.yolo = True
        a.run_turn("go")
        self.assertEqual(self.fe.names("approve"), [])
        self.assertEqual(len([m for m in self.tool_msgs() if m["content"].startswith("exit=0")]), len(self.RISKY))
        self.assertEqual(len([e for e in self.fe.names("done")]), len(self.RISKY))  # the tool cards still show
        self.assertFalse((self.work / "a.txt").exists())

    def test_yolo_is_only_the_constructor_argument_and_off_again(self):
        cfg = default_config()
        make = lambda **kw: Agent(cfg, LlamaClient("http://127.0.0.1:1"), build_registry(cfg), ScriptedFrontend(), "s", self.work, **kw)  # noqa: E731
        self.assertFalse(make().yolo)
        self.assertTrue(make(yolo=True).yolo)
        cfg["safety"]["yolo"] = True  # a stray config value, or an environment variable, changes nothing
        with unittest.mock.patch.dict(os.environ, {"XARENCODE_YOLO": "1", "YOLO": "1"}):
            self.assertFalse(make().yolo)
        (self.work / "a.txt").write_text("x")
        b = self.make(self.script(["rm a.txt"]), approvals=[Approval("no")])
        b.yolo = True
        b.yolo = False  # what /yolo off does
        b.run_turn("go")
        self.assertEqual(len(self.fe.names("approve")), 1)
        self.assertTrue((self.work / "a.txt").exists())

    def test_yolo_never_runs_hard_blocks(self):
        (self.work / "x").mkdir()
        cmds = ["dd if=a of=/dev/sda", "rm -rf x", "mkfs.ext4 x", "cat /dev/sda", "rm -r -f x"]
        a = self.make(self.script(cmds))
        a.yolo = True
        a.run_turn("go")
        self.assertEqual(self.fe.names("approve"), [])
        self.assertEqual([m["content"][:8] for m in self.tool_msgs()], ["BLOCKED:"] * len(cmds))
        self.assertTrue((self.work / "x").is_dir())

    def test_yolo_still_requires_typed_sudo_and_never_disables_sudo_never(self):
        a = self.make(self.script(["sudo true"]), approvals=[Approval("no")])
        a.yolo = True
        a.run_turn("go")
        self.assertEqual([e[2] for e in self.fe.names("approve")], ["ELEVATED"])
        self.assertEqual(self.tool_msgs()[0]["content"], "User declined")
        b = self.make(self.script(["sudo true"]))
        b.cfg["safety"]["sudo"] = "never"
        b.yolo = True
        b.run_turn("go")
        self.assertTrue(self.tool_msgs()[0]["content"].startswith("BLOCKED:"))

    def test_a_model_supplied_argument_cannot_turn_yolo_on(self):
        extra = {"command": "touch made.txt", "yolo": True, "approve": True, "auto_approve": True, "dangerously_skip": True}
        a = self.make([[call("run_shell", extra)], [{"text": "ok"}]], approvals=[Approval("no")])
        a.run_turn("go")
        self.assertFalse(a.yolo)
        self.assertEqual(len(self.fe.names("approve")), 1)
        self.assertFalse((self.work / "made.txt").exists())
        for name, args in [("write_file", {"path": "w.txt", "content": "x", "yolo": True}),
                           ("dev_server", {"action": "start", "command": "touch d.txt", "yolo": True})]:
            fe = ScriptedFrontend([Approval("no")])
            b = Agent(a.cfg, LlamaClient(self.mock.url), build_registry(a.cfg), fe, "s", self.work)
            self.mock.state.script[:] = [[call(name, args)], [{"text": "ok"}]]
            b.run_turn("go")
            self.assertEqual(len(fe.names("approve")), 1, name)

    def test_yolo_keeps_the_workdir_sandbox(self):
        outside = self.tmp / "outside.txt"
        a = self.make([[call("write_file", {"path": str(outside), "content": "x"})], [{"text": "ok"}]])
        a.yolo = True
        a.run_turn("go")
        self.assertTrue(self.tool_msgs()[0]["content"].startswith("BLOCKED:"))  # default config: blocked outright
        self.assertEqual(self.fe.names("approve"), [])
        self.assertFalse(outside.exists())
        b = self.make([[call("write_file", {"path": str(outside), "content": "x"})], [{"text": "ok"}]],
                      approvals=[Approval("no")])
        b.cfg["safety"]["write_outside_workdir"] = True
        b.yolo = True
        b.auto_edits = True
        b.run_turn("go")
        self.assertEqual(len(self.fe.names("approve")), 1)  # asked despite yolo and auto-edits
        self.assertFalse(outside.exists())
        c = self.make([[call("write_file", {"path": "in.txt", "content": "x"})], [{"text": "ok"}]])
        c.yolo = True
        c.run_turn("go")
        self.assertEqual(self.fe.names("approve"), [])
        self.assertEqual((self.work / "in.txt").read_text(), "x")

    def test_dev_server_is_no_way_around_the_rule(self):
        a = self.make([[{"text": "x"}]])
        a.cfg["dev"] = {"fetch": {"dir": ".", "cmd": "pip install x && python3 -m http.server 8999"}}
        tctx, tool = a.tool_context(), build_registry(a.cfg).get("dev_server")
        for args in ({"action": "start", "command": "curl -O https://x"}, {"action": "start", "command": "rm -r build"},
                     {"action": "start", "name": "fetch"}):
            plan = tool.plan(args, tctx)
            self.assertEqual((plan.verdict.name, plan.approval_key), ("ASK", None), args)
            self.assertTrue(plan.tag, args)
        plain = tool.plan({"action": "start", "command": "npm run dev"}, tctx)
        self.assertEqual((plan.tag != "", plain.tag, plain.approval_key), (True, "", "dev_server npm run dev"))


class PlainApprovalTests(unittest.TestCase):
    def ask(self, verdict_name, answer):
        import io
        from types import SimpleNamespace
        from unittest import mock
        from xarencode.plain import PlainFrontend
        fe = PlainFrontend(out=io.StringIO(), err=io.StringIO(), interactive=True)
        plan = SimpleNamespace(verdict=SimpleNamespace(name=verdict_name), reason="r", preview="p", tag="")
        with mock.patch("builtins.input", return_value=answer):
            return fe.approve(SimpleNamespace(name="run_shell"), plan).kind

    def test_elevated_needs_the_word_sudo(self):
        self.assertEqual(self.ask("ELEVATED", "y"), "no")
        self.assertEqual(self.ask("ELEVATED", "yes"), "no")
        self.assertEqual(self.ask("ELEVATED", "sudo"), "yes")
        self.assertEqual(self.ask("ASK", "y"), "yes")


class AgentTests(AgentTestCase):
    def test_tool_then_answer(self):
        (self.work / "f.txt").write_text("x")
        a = self.make([[call("list_dir", {"path": "."})], [{"text": "There is one file."}]])
        a.run_turn("what is here?")
        second = self.mock.requests[1]["messages"]
        tool = [m for m in second if m["role"] == "tool"][0]
        assistant = [m for m in second if m.get("tool_calls")][0]
        self.assertEqual(tool["tool_call_id"], assistant["tool_calls"][0]["id"])
        self.assertIn("f.txt", tool["content"])
        self.assertEqual(self.fe.text_buf, "There is one file.")
        self.assertEqual(a.history[-1]["content"], "There is one file.")
        self.assertEqual(second[0], {"role": "system", "content": "system prompt"})
        self.assertTrue(self.fe.stats_seen)
        self.assertEqual(self.fe.stats_seen[0].tok_per_s, 12.5)

    def test_trailing_comma_repaired(self):
        a = self.make([[call("run_shell", '{"command": "echo hi",}')], [{"text": "done"}]])
        a.run_turn("go")
        self.assertIn("hi", self.tool_msgs()[0]["content"])

    def test_length_cut_write_does_not_run(self):
        a = self.make([[call("write_file", '{"path": "big.txt", "content": "abc'), {"finish_reason": "length"}], [{"text": "ok"}]])
        a.run_turn("write it")
        self.assertFalse((self.work / "big.txt").exists())
        self.assertIn("cut off", self.tool_msgs()[0]["content"])
        self.assertEqual(json.loads(a.history[1]["tool_calls"][0]["function"]["arguments"]) is not None, True)

    def test_text_tool_call_fallback(self):
        (self.work / "f.txt").write_text("x")
        a = self.make([[{"text_tool_call": {"name": "list_dir", "arguments": {"path": "."}}}], [{"text": "ok"}]])
        a.run_turn("look")
        self.assertIn("f.txt", self.tool_msgs()[0]["content"])
        self.assertNotIn("tool_call", a.history[1]["content"])

    def test_bash_alias(self):
        a = self.make([[call("bash", {"command": "echo aliased"})], [{"text": "ok"}]])
        a.run_turn("go")
        self.assertIn("aliased", self.tool_msgs()[0]["content"])

    def test_dd_blocked_and_never_started(self):
        a = self.make([[call("run_shell", {"command": "dd if=/dev/zero of=/dev/sda"})], [{"text": "ok"}]])
        a.run_turn("wipe")
        self.assertTrue(self.tool_msgs()[0]["content"].startswith("BLOCKED:"))
        self.assertEqual(self.fe.names("approve"), [])
        self.assertEqual([e for e in self.fe.events if e[0] == "output"], [])

    def test_context_overflow_trims_and_retries_once(self):
        over = {"error": {"status": 400, "type": "exceed_context_size_error", "message": "x"}}
        a = self.make([[over], [{"text": "recovered"}]])
        a.history.extend([{"role": "user", "content": "old"}, {"role": "assistant", "content": "a" * 3000}])
        a.run_turn("again")
        self.assertEqual(self.fe.text_buf, "recovered")
        self.assertEqual(len(self.mock.requests), 2)
        self.assertTrue(self.fe.names("notice"))

    def test_second_overflow_gives_up(self):
        over = {"error": {"status": 400, "type": "exceed_context_size_error", "message": "x"}}
        a = self.make([[over], [over]])
        a.run_turn("hi")
        self.assertIn("error", [e[1] for e in self.fe.names("notice")])

    def test_decline_ends_turn_and_skips_rest(self):
        a = self.make([[call("write_file", {"path": "a.txt", "content": "1"})], [{"text": "never"}]], approvals=[Approval("no")])
        a.run_turn("write")
        self.assertFalse((self.work / "a.txt").exists())
        self.assertEqual(self.tool_msgs()[0]["content"], "User declined")
        self.assertEqual(len(self.mock.requests), 1)

    def test_feedback_continues(self):
        a = self.make([[call("write_file", {"path": "a.txt", "content": "1"})], [{"text": "ok, different"}]],
                      approvals=[Approval("feedback", "use b.txt")])
        a.run_turn("write")
        self.assertIn("use b.txt", self.tool_msgs()[0]["content"])
        self.assertEqual(self.fe.text_buf, "ok, different")

    def test_always_remembers_key(self):
        w = lambda n: call("run_shell", {"command": f"npm run build -- --n{n}"})  # noqa: E731
        a = self.make([[w(1)], [w(2)], [{"text": "done"}]], approvals=[Approval("always")])
        a.run_turn("build")
        self.assertEqual(len(self.fe.names("approve")), 1)  # second "npm run" was auto-approved
        self.assertEqual(a.approvals, {"npm run build"})

    def test_dev_server_always_does_not_cover_other_commands(self):
        ds = lambda c: call("dev_server", {"action": "start", "command": c})  # noqa: E731
        cmds = ["npm run build", "rm -r src", "git push --force origin main", "curl x -o a.py && python3 a.py"]
        a = self.make([[ds(c)] for c in cmds] + [[{"text": "done"}]],
                      approvals=[Approval("always"), Approval("no")])
        a.run_turn("go")
        self.assertEqual(a.approvals, {"dev_server npm run build"})
        # the second command (rm -r src) had to be asked about; declining ended the turn
        self.assertEqual(len(self.fe.names("approve")), 2)

    def test_dev_server_compound_has_no_always_key(self):
        tctx = self.make([[{"text": "x"}]]).tool_context()
        tool = build_registry(self.cfg).get("dev_server")
        for c in ["rm -r src", "git push --force origin main", "a && b", "sudo true"]:
            plan = tool.plan({"action": "start", "command": c}, tctx)
            self.assertIsNone(plan.approval_key, c)
            self.assertIn(plan.verdict.name, ("ASK", "BLOCK"), c)
        self.assertEqual(tool.plan({"action": "start", "command": "sudo true"}, tctx).verdict.name, "BLOCK")

    def test_auto_edits_skips_prompt(self):
        a = self.make([[call("write_file", {"path": "a.txt", "content": "1"})], [{"text": "ok"}]])
        a.auto_edits = True
        a.run_turn("write")
        self.assertEqual(self.fe.names("approve"), [])
        self.assertEqual((self.work / "a.txt").read_text(), "1")
        self.assertEqual(a.session_stats.edits, 1)
        self.assertTrue(any(e[1] == "celebrate" for e in self.fe.names("notice")))

    def test_elevated_requires_approval(self):
        a = self.make([[call("run_shell", {"command": "sudo true"})], [{"text": "ok"}]], approvals=[Approval("no")])
        a.run_turn("go")
        self.assertEqual(self.fe.names("approve")[0][2], "ELEVATED")

    def test_keyboard_interrupt_during_approval_leaves_history_consistent(self):
        a = self.make([[call("write_file", {"path": "a.txt", "content": "1"}), call("list_dir", {"path": "."})], [{"text": "after"}]])
        sunk = []
        a.sink = lambda msg, ui: sunk.append(msg)

        def boom(call, plan):
            raise KeyboardInterrupt
        self.fe.approve = boom
        a.run_turn("go")
        asst = [m for m in a.history if m.get("tool_calls")][0]
        ids = [c["id"] for c in asst["tool_calls"]]
        self.assertEqual([m["tool_call_id"] for m in self.tool_msgs()], ids)
        self.assertTrue(all("interrupted" in m["content"] for m in self.tool_msgs()))
        self.assertEqual([m["tool_call_id"] for m in sunk if m["role"] == "tool"], ids)
        self.assertFalse((self.work / "a.txt").exists())
        self.fe.approve = lambda call, plan: Approval("yes")
        a.run_turn("again")  # the next turn works and the history is valid
        self.assertEqual(a.history[-1]["content"], "after")

    def test_interrupt_with_reused_call_ids_still_answers_the_new_call(self):
        a = self.make([[call("list_dir", {"path": "."})], [{"text": "first done"}],
                       [call("write_file", {"path": "a.txt", "content": "1"})], [{"text": "unreached"}]])
        a.run_turn("one")  # the mock numbers every call "call_1"
        self.fe.approve = lambda call, plan: (_ for _ in ()).throw(KeyboardInterrupt)
        a.run_turn("two")
        issued = [tc["id"] for m in a.history for tc in m.get("tool_calls") or []]
        self.assertEqual(len(issued), 2)
        self.assertEqual(len(set(issued)), 2, issued)  # ids never repeat within a session
        last_asst = max(i for i, m in enumerate(a.history) if m.get("tool_calls"))
        replies = [m["tool_call_id"] for m in a.history[last_asst + 1:] if m["role"] == "tool"]
        self.assertEqual(replies, [issued[1]])
        self.assertIn("interrupted", a.history[-1]["content"])
        self.assertFalse((self.work / "a.txt").exists())

    def test_text_fallback_ids_are_unique_across_turns(self):
        block = lambda p: {"text": f'<tool_call>{{"name": "list_dir", "arguments": {{"path": "{p}"}}}}</tool_call>'}  # noqa: E731
        a = self.make([[block(".")], [{"text": "ok"}], [block(".")], [{"text": "ok"}]])
        a.run_turn("one")
        a.run_turn("two")
        issued = [tc["id"] for m in a.history for tc in m.get("tool_calls") or []]
        self.assertEqual(len(issued), len(set(issued)), issued)

    def test_repeat_guard(self):
        same = [call("list_dir", {"path": "."})]
        a = self.make([same, same, same, [{"text": "stop"}]])
        a.run_turn("loop")
        self.assertIn("repeated call", self.tool_msgs()[2]["content"])

    def test_three_errors_stop_turn(self):
        bad = lambda n: [call("read_file", {"path": f"nope{n}"})]  # noqa: E731
        a = self.make([bad(1), bad(2), bad(3), [{"text": "unreached"}]])
        a.run_turn("go")
        self.assertEqual(len(self.mock.requests), 3)
        self.assertEqual(len(self.tool_msgs()), 3)

    def test_step_limit(self):
        script = [[call("list_dir", {"path": f"d{i}"})] for i in range(5)]
        a = self.make(script, max_steps=2)
        a.run_turn("go")
        self.assertIn("step limit", self.tool_msgs()[-1]["content"])
        # every assistant tool call has a matching tool message
        ids = {tc["id"] for m in a.history for tc in m.get("tool_calls", [])}
        self.assertEqual(ids, {m["tool_call_id"] for m in self.tool_msgs()})

    def test_interrupt_keeps_partial(self):
        a = self.make([[{"text": "partial words"}, {"delay": 5}, {"text": "never"}]])
        threading.Timer(0.5, a.interrupt).start()
        a.run_turn("talk")
        self.assertEqual(a.history[-1]["content"], "partial words [interrupted]")

    def test_server_down_reports_error(self):
        a = self.make([])
        a.client = LlamaClient("http://127.0.0.1:1")
        a.run_turn("hi")
        self.assertEqual(self.fe.names("notice")[0][1], "error")
        self.assertEqual(a.history[0]["role"], "user")

    def test_sink_receives_messages(self):
        got = []
        a = self.make([[{"text": "hi"}]])
        a.sink = lambda m, ui: got.append(m["role"])
        a.run_turn("x")
        self.assertEqual(got, ["user", "assistant"])


class CapTests(unittest.TestCase):
    def test_cap_lines_and_chars(self):
        text = "\n".join(f"line{i}" for i in range(200))
        out = cap_output(text, 100000)
        self.assertIn("line0", out)
        self.assertIn("line199", out)
        self.assertIn("150 lines omitted", out)
        self.assertEqual(len(out.splitlines()), 51)
        long = cap_output("x" * 5000, 2000)
        self.assertLess(len(long), 2100)
        self.assertEqual(cap_output("short", 2000), "short")


if __name__ == "__main__":
    unittest.main()
