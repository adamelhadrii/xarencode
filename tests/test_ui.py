"""UI and slash-command tests. Command tests are stdlib only; the rest skip when rich is missing."""
import importlib.util
import io
import json
import unittest
from pathlib import Path

from tests.helpers import IsolatedTestCase, ScriptedFrontend
from tests.mock_llama_server import MockServer
from xarencode import persona, sessions
from xarencode.agent import Agent, Approval
from xarencode.commands import App, at_completions, command_list, dispatch, expand_mentions, run_bang
from xarencode.config import default_config
from xarencode.llm import LlamaClient
from xarencode.server import LlamaServer
from xarencode.tools import build_registry

HAS_RICH = importlib.util.find_spec("rich") is not None
HAS_PT = HAS_RICH and importlib.util.find_spec("prompt_toolkit") is not None


class CommandTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.mock = MockServer([[{"text": "short summary"}]]).start()
        self.addCleanup(self.mock.stop)
        self.cfg = default_config()
        self.cfg["ui"]["share_dir"] = str(self.tmp / "share")
        self.fe = ScriptedFrontend()
        self.agent = Agent(self.cfg, LlamaClient(self.mock.url), build_registry(self.cfg), self.fe, persona.system_prompt(self.cfg, self.work), self.work)
        self.session = sessions.Session.new(self.cfg, self.work)
        self.agent.sink = self.session.append
        self.lines = []
        self.app = App(self.cfg, self.agent, LlamaServer(self.cfg), self.session, self.lines.append, self.work)

    def notices(self):
        return [e[2] for e in self.fe.names("notice")]

    def test_unknown_and_help(self):
        self.assertTrue(dispatch(self.app, "/nope"))
        self.assertIn("unknown command", self.notices()[0])
        dispatch(self.app, "/help")
        self.assertIn("/share", self.lines[0])
        self.assertFalse(dispatch(self.app, "hello"))

    def test_all_commands_registered(self):
        names = {n for n, _ in command_list()}
        want = {"help", "clear", "compact", "context", "sessions", "resume", "exit", "model", "server", "config", "doctor", "cd",
                "permissions", "dev", "vm", "share", "vibe", "yolo", "copy", "pet", "mascot", "header",
                "theme", "timestamps", "sidebar"}
        self.assertEqual(names, {f"/{n}" for n in want})

    def test_theme_timestamps_sidebar_commands(self):
        dispatch(self.app, "/theme light")
        self.assertEqual(self.cfg["ui"]["theme"], "light")
        dispatch(self.app, "/theme invalid")
        self.assertIn("usage", self.notices()[-1])
        dispatch(self.app, "/timestamps on")
        self.assertTrue(self.cfg["ui"]["timestamps"])
        dispatch(self.app, "/sidebar on")
        self.assertTrue(self.cfg["ui"]["sidebar"])

    def test_vibe_changes_system_prompt(self):
        dispatch(self.app, "/vibe hype")
        self.assertIn(persona.VIBES["hype"], self.agent.system)
        dispatch(self.app, "/vibe bogus")
        self.assertIn("unknown vibe", self.notices()[-1])

    def test_cd(self):
        sub = self.work / "sub"
        sub.mkdir()
        dispatch(self.app, "/cd sub")
        self.assertEqual(self.agent.workdir, sub.resolve())
        self.assertIn(str(sub.resolve()), self.agent.system)
        dispatch(self.app, "/cd /")
        self.assertIn("refusing", self.notices()[-1])
        dispatch(self.app, "/cd missing")
        self.assertIn("not a folder", self.notices()[-1])

    def test_compact_and_clear(self):
        self.agent.history += [{"role": "user", "content": "do things"}, {"role": "assistant", "content": "did"}]
        dispatch(self.app, "/compact")
        self.assertEqual(len(self.agent.history), 1)
        self.assertIn("short summary", self.agent.history[0]["content"])
        dispatch(self.app, "/clear")
        self.assertEqual(self.agent.history, [])

    def test_context_output(self):
        self.agent.history.append({"role": "user", "content": "hello"})
        dispatch(self.app, "/context")
        self.assertIn("tool schemas", self.lines[0])
        self.assertIn("of 4096", self.lines[0])

    def test_permissions(self):
        self.agent.approvals.add("npm run")
        self.agent.auto_edits = True
        dispatch(self.app, "/permissions")
        self.assertIn("npm run", self.lines[0])
        dispatch(self.app, "/permissions reset")
        self.assertEqual((self.agent.approvals, self.agent.auto_edits), (set(), False))

    def test_yolo_command_toggles_for_the_session_only(self):
        self.assertFalse(self.agent.yolo)
        dispatch(self.app, "/yolo on")
        self.assertTrue(self.agent.yolo)
        self.assertIn("hard blocks", self.notices()[-1].lower())
        dispatch(self.app, "/yolo")
        self.assertIn("yolo: on", self.lines[-1])
        dispatch(self.app, "/yolo off")
        self.assertFalse(self.agent.yolo)
        dispatch(self.app, "/yolo maybe")
        self.assertIn("usage", self.notices()[-1])
        self.assertFalse(self.agent.yolo)
        self.assertNotIn("yolo", self.cfg["safety"])  # never written to the config
        dispatch(self.app, "/permissions")
        self.assertIn("yolo: off", self.lines[-1])

    def test_share_md_and_resume(self):
        self.agent.run_turn("hi")
        dispatch(self.app, "/share md")
        self.assertTrue(any("saved" in n for n in self.notices()))
        self.assertEqual(len(list((self.tmp / "share").glob("*.md"))), 1)
        old_id = self.session.meta["id"]
        dispatch(self.app, "/clear")
        self.assertEqual(self.agent.history, [])
        dispatch(self.app, f"/resume {old_id[:15]}")
        self.assertEqual(self.agent.history[0]["content"], "hi")
        self.assertEqual(self.session.meta["id"], old_id)

    def test_share_empty(self):
        dispatch(self.app, "/share md")
        self.assertIn("nothing to share", self.notices()[-1])

    def test_vm_none_configured(self):
        dispatch(self.app, "/vm")
        self.assertIn("no virtual machines", self.lines[0])

    def test_bang_runs_and_blocks(self):
        run_bang(self.app, "echo bang")
        self.assertIn("bang", self.agent.history[-1]["content"])
        n = len(self.agent.history)
        run_bang(self.app, "rm -rf x")
        self.assertEqual(len(self.agent.history), n)
        self.assertIn("blocked", self.notices()[-1])
        run_bang(self.app, "touch made.txt")  # would be ASK for the model, runs for the user
        self.assertTrue((self.work / "made.txt").exists())

    def test_bang_after_interrupt_still_runs(self):
        self.agent.cancelled.set()  # as left behind by Esc or Ctrl+C during a turn
        run_bang(self.app, "echo after-esc")
        self.assertIn("after-esc", self.agent.history[-1]["content"])
        self.assertIn("exit 0", self.agent.history[-1]["content"])

    def test_mentions(self):
        (self.work / "a.py").write_text("print('hi')\n")
        (self.work / "b.bin").write_bytes(b"\0\0")
        out = expand_mentions("fix @a.py and @b.bin and @missing and user@example.com", self.work, self.cfg)
        self.assertIn("[attached a.py]", out)
        self.assertIn("print('hi')", out)
        self.assertEqual(out.count("[attached"), 1)

    def test_at_completions(self):
        (self.work / "src").mkdir()
        (self.work / "src" / "main.py").write_text("")
        (self.work / "node_modules").mkdir()
        self.assertEqual(at_completions("s", self.work), ["src/"])
        self.assertEqual(at_completions("src/m", self.work), ["src/main.py"])
        self.assertEqual(at_completions("n", self.work), [])



@unittest.skipUnless(HAS_RICH, "rich is not installed")
class UiTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        from rich.console import Console
        from xarencode import ui
        self.ui_mod = ui
        self.cfg = default_config()
        self.buf = io.StringIO()
        self.console = Console(file=self.buf, width=100, force_terminal=False, highlight=False)
        self.ui = ui.RichUI(self.cfg, self.console, interactive=False)

    def out(self):
        return self.buf.getvalue()

    def agent_with(self, script):
        mock = MockServer(script).start()
        self.addCleanup(mock.stop)
        agent = Agent(self.cfg, LlamaClient(mock.url), build_registry(self.cfg), self.ui, "sys", self.work)
        self.ui.agent = agent
        return agent, mock

    def test_diff_panel(self):
        diff = "--- a/x.py\n+++ b/x.py\n@@ -3,2 +3,2 @@\n keep\n-old line\n+new line"
        self.console.print(self.ui_mod.render_diff(diff, "x.py"))
        out = self.out()
        self.assertIn("edit · x.py (+1 −1)", out)
        self.assertIn("   4 - old line", out)
        self.assertIn("   4 + new line", out)

    def test_banner_and_status_line(self):
        self.ui.print_banner(self.work)
        out = self.out()
        self.assertIn("XarenCode 0.1.0", out)
        self.assertIn("Qwen3-4B", out)
        agent, _ = self.agent_with([])
        line = self.ui.status_line()
        for part in ("Qwen3-4B", "ctx 0.0k/4.1k", "▱▱▱▱", "● ready", "auto-edits off"):
            self.assertIn(part, line)
        agent.auto_edits = True
        self.assertIn("auto-edits on", self.ui.status_line())

    def test_yolo_badge_in_status_bar_and_banner(self):
        agent, _ = self.agent_with([])
        self.assertNotIn("YOLO", self.ui.status_line())
        self.ui.print_banner(self.work)
        self.assertNotIn("YOLO", self.out())
        agent.yolo = True
        self.assertIn("YOLO", self.ui.status_line())
        frags = self.ui.status_fragments()
        self.assertTrue(any("YOLO" in text and "bold" in style for style, text in frags))
        self.ui.print_banner(self.work)
        self.assertIn("auto-approve is on", self.out())
        self.assertIn("sudo still apply", self.out())
        agent.yolo = False
        self.assertNotIn("YOLO", self.ui.status_line())

    def test_stream_renders_final_markdown_and_footer(self):
        from xarencode.llm import PromptProgress
        self.ui.begin_stream()
        self.ui.prompt_progress(PromptProgress(total=100, cache=40, processed=60))
        self.assertIn("Reading prompt… 60% (60/100 tok · 40 cached)", self.ui.footer().plain)
        self.ui.text("Hello **world**")
        self.assertIn("esc to interrupt", self.ui.footer().plain)
        self.assertIn("tok/s", self.ui.footer().plain)
        self.ui.tool_preparing("write_file", 1300)
        self.assertIn("✎ writing… 1.3 KB", self.ui.footer().plain)
        self.ui.end_stream()
        self.assertIn("Hello", self.out())
        self.ui.end_stream()  # idempotent

    def test_reasoning_collapses(self):
        self.ui.begin_stream()
        self.ui.reasoning("thinking hard")
        self.ui.text("answer")
        self.ui.end_stream()
        self.assertIn("Thought for", self.out())

    def test_end_to_end_list_and_write(self):
        (self.work / "seen.txt").write_text("x")
        agent, _ = self.agent_with([
            [{"text": "Looking. "}, {"tool_call": {"name": "list_dir", "arguments": json.dumps({"path": "."})}}],
            [{"tool_call": {"name": "write_file", "arguments": json.dumps({"path": "n.py", "content": "print(1)\n"})}}],
            [{"text": "All done."}]])
        agent.auto_edits = True
        agent.run_turn("go")
        out = self.out()
        for part in ("⏺ List(.)", "⎿", "⏺ Write(n.py)", "edit · n.py (+1 −0)", "print(1)", "All done.", "✦"):
            self.assertIn(part, out)
        self.assertEqual((self.work / "n.py").read_text(), "print(1)\n")

    def test_shell_card(self):
        agent, _ = self.agent_with([[{"tool_call": {"name": "run_shell", "arguments": json.dumps({"command": "echo hi"})}}], [{"text": "ok"}]])
        agent.run_turn("run")
        self.assertIn("⏺ Bash(echo hi)", self.out())
        self.assertIn("exit 0", self.out())

    def test_approve_options(self):
        from unittest import mock
        from xarencode.safety import ASK
        from xarencode.tools import Plan
        from xarencode.toolcall import ToolCall
        call = ToolCall("c1", "run_shell", {"command": "npm run build"})
        plan = Plan(ASK, "x", "npm run build", "bash", "npm run")
        with mock.patch.object(self.ui_mod, "choose", return_value=0) as ch:
            self.assertEqual(self.ui.approve(call, plan), Approval("yes"))
            options = ch.call_args[0][0]
        self.assertEqual(len(options), 3)
        self.assertIn("don't ask again for npm run this session", options[1])
        self.assertIn("tell Xaren what to do differently", options[2])
        with mock.patch.object(self.ui_mod, "choose", return_value=1):
            self.assertEqual(self.ui.approve(call, plan).kind, "always")
        with mock.patch.object(self.ui_mod, "choose", return_value=None):
            self.assertEqual(self.ui.approve(call, plan).kind, "no")
        with mock.patch.object(self.ui_mod, "choose", return_value=2), mock.patch.object(self.ui_mod, "ask_line", return_value="use make instead"):
            self.assertEqual(self.ui.approve(call, plan), Approval("feedback", "use make instead"))
        self.assertIn("run this?", self.out())

    def test_approve_edit_shows_diff_and_key_label(self):
        from unittest import mock
        from xarencode.tools import ToolContext
        (self.work / "a.py").write_text("x = 1\n")
        reg = build_registry(self.cfg)
        args = {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}
        plan = reg.get("edit_file").plan(args, ToolContext(self.work, self.cfg))
        from xarencode.toolcall import ToolCall
        with mock.patch.object(self.ui_mod, "choose", return_value=0) as ch:
            self.ui.approve(ToolCall("c", "edit_file", args), plan)
        self.assertIn("edits to a.py", ch.call_args[0][0][1])
        self.assertIn("edit · a.py (+1 −1)", self.out())

    def test_render_transcript_and_share(self):
        cfg = self.cfg
        cfg["ui"]["share_dir"] = str(self.tmp / "share")
        s = sessions.Session.new(cfg, self.work)
        home = str(Path.home())
        s.append({"role": "user", "content": f"read {home}/secret.txt token=abcd1234efgh"})
        s.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                  "function": {"name": "edit_file", "arguments": json.dumps({"path": "a.py"})}}]})
        s.append({"role": "tool", "tool_call_id": "c", "content": "Edited a.py"},
                 {"summary": "Edited (+1 -1)", "diff": "@@ -1,1 +1,1 @@\n-a\n+b", "tool": "edit_file"})
        s.append({"role": "assistant", "content": "Done with **bold** text."})
        for fmt in ("svg", "html", "md"):
            p = sessions.export(s, fmt, self.tmp / "share", cfg)
            text = p.read_text()
            self.assertNotIn(home, text)
            self.assertNotIn("abcd1234efgh", text)
        svg = (self.tmp / "share" / f"xarencode-{s.meta['id']}.svg").read_text()
        self.assertIn("<svg", svg)
        self.assertIn("edit", svg)
        html = (self.tmp / "share" / f"xarencode-{s.meta['id']}.html").read_text()
        self.assertIn("<html", html.lower())

    def test_history_trim(self):
        p = self.tmp / "hist"
        p.write_text("".join(f"\n# 2026-01-01 00:00:{i % 60:02d}\n+cmd {i}\n" for i in range(30)))
        self.ui_mod.history_trim(p, keep=5)
        text = p.read_text()
        self.assertEqual(text.count("# 2026"), 5)
        self.assertIn("+cmd 29", text)
        self.assertNotIn("+cmd 24\n", text)

    def test_notice_levels(self):
        for lvl in ("info", "warn", "error", "celebrate"):
            self.ui.notice(lvl, f"msg-{lvl}")
        out = self.out()
        for lvl in ("info", "warn", "error", "celebrate"):
            self.assertIn(f"msg-{lvl}", out)

    def test_tool_label(self):
        self.assertEqual(self.ui_mod.tool_label("run_shell", {"command": "ls"}), "ls")
        self.assertEqual(self.ui_mod.tool_label("vm", {"action": "start", "name": "a"}), "start a")


@unittest.skipUnless(HAS_PT, "prompt_toolkit is not installed")
class PromptToolkitTests(IsolatedTestCase):
    def completions(self, text, workdir):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from xarencode import ui
        comp = ui.make_completer(lambda: workdir)
        return list(comp.get_completions(Document(text, len(text)), CompleteEvent()))

    def test_slash_completion_has_descriptions(self):
        got = self.completions("/sh", self.work)
        self.assertEqual([c.text for c in got], ["/share"])
        self.assertTrue(got[0].display_meta_text)

    def test_at_completion(self):
        (self.work / "main.py").write_text("")
        got = self.completions("fix @ma", self.work)
        self.assertEqual([c.text for c in got], ["@main.py"])
        self.assertEqual(self.completions("plain text", self.work), [])

    def test_prompt_session_builds(self):
        from rich.console import Console
        from xarencode import ui
        rui = ui.RichUI(default_config(), Console(file=io.StringIO()), interactive=False)
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            sess = ui.make_prompt_session(rui, lambda: self.work)
            self.assertTrue(sess.multiline)
            inp.send_text("/he\r")
            self.assertEqual(sess.prompt(), "/he")


if __name__ == "__main__":
    unittest.main()
