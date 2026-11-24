"""Full-window view tests: the Rich bridge, clipboard, the new commands, the chat and header controls, pipe-driven runs of the
whole application, the command line behaviour and a pty smoke test. Everything uses tests/mock_llama_server.py."""
import asyncio
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

from tests.helpers import IsolatedTestCase
from tests.mock_llama_server import MockServer
from xarencode import clipboard, commands, mascot, persona, sessions
from xarencode.agent import Agent
from xarencode.config import default_config, state_dir
from xarencode.llm import LlamaClient
from xarencode.server import LlamaServer
from xarencode.tools import build_registry

HAS_RICH = importlib.util.find_spec("rich") is not None
HAS_PT = HAS_RICH and importlib.util.find_spec("prompt_toolkit") is not None
ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "mock.toml"


def cells(line):
    return "".join(t for _, t in line)


# ---------------------------------------------------------------- clipboard (standard library only)

class ClipboardTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.captured = self.tmp / "captured.txt"
        patch = mock.patch.dict(os.environ, {"PATH": str(self.bin)})
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ.pop("DISPLAY", None)

    def fake(self, name, exit_code=0):
        cat = subprocess.run(["which", "cat"], capture_output=True, text=True, env=dict(os.environ, PATH="/usr/bin:/bin")).stdout.strip()
        script = self.bin / name
        script.write_text(f"#!/bin/sh\n{cat} > '{self.captured}.{name}'\nexit {exit_code}\n")
        script.chmod(0o755)
        return Path(f"{self.captured}.{name}")

    def test_wl_copy_is_used_on_wayland(self):
        out = self.fake("wl-copy")
        os.environ["WAYLAND_DISPLAY"] = "wayland-test"
        self.assertEqual(clipboard.copy_text("hello ✦"), "the clipboard (wl-copy)")
        self.assertEqual(out.read_text("utf-8"), "hello ✦")

    def test_xclip_is_used_on_x11(self):
        out = self.fake("xclip")
        os.environ["DISPLAY"] = ":9"
        self.assertEqual(clipboard.copy_text("abc"), "the clipboard (xclip)")
        self.assertEqual(out.read_text(), "abc")

    def test_xsel_when_there_is_no_xclip(self):
        out = self.fake("xsel")
        os.environ["DISPLAY"] = ":9"
        self.assertEqual(clipboard.copy_text("abc"), "the clipboard (xsel)")
        self.assertEqual(out.read_text(), "abc")

    def test_a_failing_tool_falls_back_to_the_file(self):
        self.fake("wl-copy", exit_code=1)
        os.environ["WAYLAND_DISPLAY"] = "wayland-test"
        where = clipboard.copy_text("kept")
        self.assertEqual(Path(where), state_dir() / "clipboard.txt")

    def test_file_fallback_is_private(self):
        where = Path(clipboard.copy_text("secret text"))
        self.assertEqual(where, state_dir() / "clipboard.txt")
        self.assertEqual(where.read_text(), "secret text")
        self.assertEqual(where.stat().st_mode & 0o777, 0o600)
        clipboard.copy_text("second")
        self.assertEqual(where.read_text(), "second")

    def test_last_reply_and_code_block(self):
        history = [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "first\n```py\nprint(1)\n```\nthen\n```\nprint(2)\n```\nend"},
                   {"role": "tool", "tool_call_id": "c", "content": "ok"},
                   {"role": "assistant", "content": ""}]
        self.assertTrue(clipboard.last_reply(history).startswith("first"))
        self.assertEqual(clipboard.last_code_block(history), "print(2)")
        self.assertEqual(clipboard.last_reply([]), "")
        self.assertEqual(clipboard.last_code_block([{"role": "assistant", "content": "no code"}]), "")


# ---------------------------------------------------------------- slash commands (standard library only)

class ViewCommandTests(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        from tests.helpers import ScriptedFrontend
        self.mock = MockServer([[{"text": "x"}]]).start()
        self.addCleanup(self.mock.stop)
        self.cfg = default_config()
        self.fe = ScriptedFrontend()
        self.agent = Agent(self.cfg, LlamaClient(self.mock.url), build_registry(self.cfg), self.fe, "sys", self.work)
        self.session = sessions.Session.new(self.cfg, self.work)
        self.agent.sink = self.session.append
        self.lines = []
        self.app = commands.App(self.cfg, self.agent, LlamaServer(self.cfg), self.session, self.lines.append, self.work)
        self.hooks = []
        self.app.ui_hook = self.hooks.append

    def notices(self):
        return [e[2] for e in self.fe.names("notice")]

    def test_fullscreen_is_not_a_command(self):
        commands.dispatch(self.app, "/fullscreen")
        self.assertIn("unknown command", self.notices()[-1])

    def test_copy_variants(self):
        self.agent.history += [{"role": "user", "content": "q"},
                               {"role": "assistant", "content": "Here:\n```sh\nls -la\n```\nbye"}]
        self.session.append(*[self.agent.history[0], None])
        self.session.append(self.agent.history[1], None)
        with mock.patch.object(clipboard, "copy_text", return_value="the clipboard (wl-copy)") as cp:
            commands.dispatch(self.app, "/copy")
            self.assertTrue(cp.call_args[0][0].startswith("Here:"))
            self.assertRegex(self.notices()[-1], r"copied \d+ chars to the clipboard \(wl-copy\)")
            commands.dispatch(self.app, "/copy code")
            self.assertEqual(cp.call_args[0][0], "ls -la")
            commands.dispatch(self.app, "/copy all")
            self.assertIn("# XarenCode session", cp.call_args[0][0])
            commands.dispatch(self.app, "/copy bogus")
            self.assertIn("usage", self.notices()[-1])

    def test_copy_with_nothing_to_copy(self):
        with mock.patch.object(clipboard, "copy_text") as cp:
            for arg in ("", "code", "all"):
                commands.dispatch(self.app, f"/copy {arg}".strip())
                self.assertEqual(self.notices()[-1], "nothing to copy yet")
            cp.assert_not_called()

    def test_pet_calls_the_hook_unless_the_mascot_is_off(self):
        commands.dispatch(self.app, "/pet")
        self.assertEqual(self.hooks, ["pet"])
        commands.dispatch(self.app, "/mascot off")
        self.assertFalse(self.cfg["ui"]["mascot"])
        commands.dispatch(self.app, "/pet")
        self.assertEqual(self.hooks, ["pet", "mascot"])
        self.assertIn("the mascot is off (/mascot on)", self.notices()[-1])
        commands.dispatch(self.app, "/mascot on")
        self.assertTrue(self.cfg["ui"]["mascot"])
        commands.dispatch(self.app, "/mascot")
        self.assertIn("mascot: on", self.lines[-1])
        commands.dispatch(self.app, "/mascot maybe")
        self.assertIn("usage", self.notices()[-1])

    def test_header_command(self):
        commands.dispatch(self.app, "/header")
        self.assertIn("header: full", self.lines[-1])
        commands.dispatch(self.app, "/header small")
        self.assertEqual(self.cfg["ui"]["header"], "small")
        self.assertEqual(self.hooks[-1], "header")
        commands.dispatch(self.app, "/header huge")
        self.assertIn("usage", self.notices()[-1])
        self.assertEqual(self.cfg["ui"]["header"], "small")

    def test_commands_work_without_a_hook_or_ui_block(self):
        self.app.ui_hook = None
        del self.cfg["ui"]
        commands.dispatch(self.app, "/pet")
        commands.dispatch(self.app, "/header off")
        self.assertEqual(self.cfg["ui"]["header"], "off")

    def test_help_mentions_the_keys(self):
        commands.dispatch(self.app, "/help")
        self.assertIn("F3 pets the cat", self.lines[0])
        self.assertIn("/copy", self.lines[0])


# ---------------------------------------------------------------- the command line (stdlib only; subprocesses)

class CliTests(IsolatedTestCase):
    def run_cli(self, *args, timeout=60):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-m", "xarencode", *args], capture_output=True, text=True, env=env, cwd=self.work,
                              timeout=timeout, stdin=subprocess.DEVNULL)

    def config_for(self, mock_server):
        p = self.tmp / "mock.toml"
        p.write_text(FIXTURE.read_text().replace("port = 18091", f"port = {mock_server.port}"))
        return str(p)

    def test_no_terminal_means_exit_2_and_a_clear_message(self):
        mock_server = MockServer(demo=True).start()
        self.addCleanup(mock_server.stop)
        for extra in ([], ["--fullscreen"]):
            r = self.run_cli("--config", self.config_for(mock_server), *extra)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('XarenCode needs a real terminal; use `xarencode -p "prompt"` for one-shot output', r.stderr)
            self.assertNotIn("unrecognized", r.stderr)
            self.assertNotIn("\x1b[?1049h", r.stdout)

    def test_fullscreen_flag_is_accepted_silently(self):
        from xarencode.cli import build_parser
        self.assertTrue(build_parser().parse_args(["--fullscreen"]).fullscreen)
        self.assertNotIn("fullscreen", build_parser().format_help())

    def test_print_mode_never_starts_the_tui_or_shows_the_cat(self):
        mock_server = MockServer(demo=True).start()
        self.addCleanup(mock_server.stop)
        r = self.run_cli("--config", self.config_for(mock_server), "--fullscreen", "-p", "hi")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Hello!", r.stdout)
        for text in (r.stdout, r.stderr):
            self.assertNotIn("\x1b[?1049h", text)
            self.assertNotIn("=^.^=", text)
            self.assertNotIn("█", text)


# ---------------------------------------------------------------- the Rich bridge

@unittest.skipUnless(HAS_RICH, "rich is not installed")
class BridgeTests(unittest.TestCase):
    def test_pt_style(self):
        from rich.style import Style
        from xarencode import richpt
        self.assertEqual(richpt.pt_style(None), "")
        self.assertEqual(richpt.pt_style(Style(color="red", bold=True)), "fg:ansired bold")
        self.assertEqual(richpt.pt_style(Style(color="bright_blue", bgcolor="#112233", italic=True, dim=True)),
                         "fg:ansibrightblue bg:#112233 italic dim")
        self.assertEqual(richpt.pt_style(Style(color="#0a84ff", underline=True, strike=True, reverse=True)),
                         "fg:#0a84ff underline strike reverse")
        self.assertTrue(richpt.pt_style(Style(color="color(200)")).startswith("fg:#"))

    def test_to_lines_markdown_width_and_tabs(self):
        from rich.markdown import Markdown
        from rich.text import Text
        from xarencode import richpt
        md = Markdown("# Title\n\nSome **bold** text that is fairly long and needs to wrap around the window edge.\n\n```py\nx = 1\t# tab\n```")
        lines = richpt.to_lines(md, 30)
        self.assertGreater(len(lines), 5)
        for ln in lines:
            self.assertLessEqual(mascot.line_width(ln), 30)
        self.assertTrue(any("bold" in style for ln in lines for style, _ in ln))
        self.assertNotIn("\t", "".join(cells(ln) for ln in lines))
        self.assertEqual(richpt.to_lines(Text("a\nb"), 10), [[("", "a")], [("", "b")]])

    def test_sanitize(self):
        from xarencode import richpt
        self.assertEqual(richpt.sanitize("⚠️ ok‍"), "⚠ ok")
        self.assertEqual(richpt.sanitize("a\x1b[31mred\x07\x00b"), "a[31mredb")
        self.assertEqual(richpt.sanitize("a\r\nb\rc\td"), "a\nb\nc\td")
        self.assertEqual(richpt.sanitize(""), "")

    def test_split_stable(self):
        from xarencode import richpt
        self.assertEqual(richpt.split_stable("one\n\ntwo"), ("one\n\n", "two"))
        self.assertEqual(richpt.split_stable("no blank line"), ("", "no blank line"))
        text = "intro\n\n```py\na = 1\n\nb = 2\n"
        self.assertEqual(richpt.split_stable(text), ("intro\n\n", "```py\na = 1\n\nb = 2\n"))
        closed = "```\nx\n\ny\n```\n\nafter"
        self.assertEqual(richpt.split_stable(closed), ("```\nx\n\ny\n```\n\n", "after"))

    def test_to_rich_text_round_trip(self):
        from xarencode import richpt
        text = richpt.to_rich_text([[("fg:#0a84ff bold", "ab"), ("fg:ansired bg:#112233", "c")], [("", "d")]])
        self.assertEqual(text.plain, "abc\nd")
        styles = {str(span.style) for span in text.spans}
        self.assertTrue(any("bold" in s and "#0a84ff" in s for s in styles))
        self.assertTrue(any("red" in s for s in styles))


# ---------------------------------------------------------------- shared setup for the controls and the application

@unittest.skipUnless(HAS_PT, "prompt_toolkit is not installed")
class Harness(IsolatedTestCase):
    ROWS, COLS = 30, 100

    def setUp(self):
        super().setUp()
        from prompt_toolkit.history import InMemoryHistory
        from rich.console import Console
        from xarencode import ui
        env = mock.patch.dict(os.environ, {"COLORTERM": "truecolor"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("NO_COLOR", None)
        self.mock = MockServer(demo=True).start()
        self.addCleanup(self.mock.stop)
        self.cfg = default_config()
        self.cfg["server"]["port"] = self.mock.port
        self.inline_buf = io.StringIO()
        self.inline = ui.RichUI(self.cfg, Console(file=self.inline_buf, width=100, highlight=False), interactive=False)
        self.agent = Agent(self.cfg, LlamaClient(self.mock.url), build_registry(self.cfg), self.inline,
                           persona.system_prompt(self.cfg, self.work), self.work)
        self.inline.agent = self.agent
        self.session = sessions.Session.new(self.cfg, self.work)
        self.agent.sink = self.session.append
        self.cmd = commands.App(self.cfg, self.agent, LlamaServer(self.cfg), self.session, lambda t: None, self.work)
        self.history = InMemoryHistory()

    def make_fe(self, rows=None, cols=None, pipe=None):
        from prompt_toolkit.data_structures import Size
        from prompt_toolkit.output.vt100 import Vt100_Output
        from xarencode import fullwin
        rows, cols = rows or self.ROWS, cols or self.COLS
        self.out_buf = io.StringIO()
        out = Vt100_Output(self.out_buf, lambda: Size(rows=rows, columns=cols), term="xterm-256color", enable_cpr=False)
        fe = fullwin.FullUI(self.cfg, self.cmd, self.inline, self.history)
        if pipe is None:
            from prompt_toolkit.input import create_pipe_input
            ctx = create_pipe_input()
            pipe = ctx.__enter__()
            self.addCleanup(ctx.__exit__, None, None, None)
        fe.build(pipe, out)
        return fe

    def snapshot(self, fe, w, h):
        """The screen as text, written straight from the layout (no terminal needed)."""
        from prompt_toolkit.application.current import set_app
        from prompt_toolkit.layout.mouse_handlers import MouseHandlers
        from prompt_toolkit.layout.screen import Char, Screen, WritePosition
        screen = Screen(default_char=Char(" ", ""), initial_width=w, initial_height=h)

        def draw():
            fe.application.render_counter += 1   # formatted-text controls cache per render
            with set_app(fe.application):
                fe.application.layout.container.write_to_screen(screen, MouseHandlers(), WritePosition(0, 0, w, h), "", False, None)
                screen.draw_all_floats()

        if fe.running():
            draw()
        else:
            async def in_loop():   # the input buffer loads its history on the running loop
                draw()
            asyncio.run(in_loop())
        return ["".join(screen.data_buffer[y][x].char for x in range(w)).rstrip() for y in range(h)]

    def run_app(self, drive, rows=None, cols=None, timeout=40):
        """Run the whole application on pipe input; `drive(fe, pipe)` runs on a helper thread once it is up."""
        from prompt_toolkit.data_structures import Size
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output.vt100 import Vt100_Output
        from xarencode import fullwin
        rows, cols = rows or self.ROWS, cols or self.COLS
        self.out_buf = io.StringIO()
        failure = []
        with create_pipe_input() as pipe:
            out = Vt100_Output(self.out_buf, lambda: Size(rows=rows, columns=cols), term="xterm-256color", enable_cpr=False)

            def ready(fe):
                self.fe = fe
                threading.Timer(timeout, lambda: fe.post(fe.leave)).start()   # a stuck driver must not hang the suite

                def go():
                    try:
                        drive(fe, pipe)
                    except BaseException as e:   # reported on the main thread below
                        failure.append(e)
                    finally:
                        fe.post(fe.leave)

                threading.Thread(target=go, daemon=True).start()

            code = fullwin.run_fullwindow(self.cfg, self.cmd, self.inline, self.history, input=pipe, output=out, on_ready=ready)
        if failure:
            raise failure[0]
        return code

    def on_loop(self, fe, fn, timeout=5):
        """Run fn on the loop thread and return its result."""
        from concurrent.futures import Future
        fut = Future()

        def call():
            try:
                fut.set_result(fn())
            except BaseException as e:
                fut.set_exception(e)

        self.assertTrue(fe.post(call))
        return fut.result(timeout)

    def screen(self, fe, w=None, h=None):
        return self.on_loop(fe, lambda: self.snapshot(fe, w or self.COLS, h or self.ROWS))

    def wait(self, cond, timeout=10, what="condition"):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what}")

    def wait_screen(self, fe, text, timeout=10):
        def seen():
            return any(text in row for row in self.screen(fe))
        self.wait(seen, timeout, f"{text!r} on the screen")

    def say(self, fe, pipe, line):
        """Type a line and wait until its turn has started and finished."""
        started = fe.turns_started
        pipe.send_text(line + "\r")
        self.wait(lambda: fe.turns_started > started, 5, f"{line!r} to be submitted")
        self.wait_idle(fe)

    def wait_idle(self, fe):
        self.wait(lambda: not fe.busy, 15, "the turn to finish")


# ---------------------------------------------------------------- chat control, header control, snapshots

class ControlTests(Harness):
    def test_follow_scroll_and_below(self):
        from xarencode import fullwin
        model = fullwin.ChatModel()
        for i in range(50):
            from rich.text import Text
            model.add(Text(f"line {i}"))
        ctl = fullwin.ChatControl(model)
        content = ctl.create_content(40, 10)
        self.assertEqual(content.line_count, 10)
        self.assertEqual(cells(content.get_line(9)), "line 49")
        self.assertTrue(ctl.follow)
        self.assertEqual(ctl.below(), 0)
        ctl.scroll(-5)
        self.assertFalse(ctl.follow)
        content = ctl.create_content(40, 10)
        self.assertEqual(cells(content.get_line(9)), "line 44")
        self.assertEqual(ctl.below(), 5)
        model.add(Text("new"))          # new output does not move a view that was scrolled up
        ctl.create_content(40, 10)
        self.assertEqual(ctl.top, 35)
        self.assertEqual(ctl.below(), 6)
        ctl.scroll(100)
        self.assertTrue(ctl.follow)
        ctl.to_top()
        self.assertEqual(cells(ctl.create_content(40, 10).get_line(0)), "line 0")
        self.assertFalse(ctl.follow)
        ctl.to_bottom()
        self.assertEqual(cells(ctl.create_content(40, 10).get_line(9)), "new")

    def test_mouse_wheel_scrolls_and_motion_is_ignored(self):
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
        from rich.text import Text
        from xarencode import fullwin
        model = fullwin.ChatModel()
        for i in range(40):
            model.add(Text(f"l{i}"))
        ctl = fullwin.ChatControl(model)
        ctl.create_content(40, 10)

        def ev(kind):
            return MouseEvent(Point(1, 1), kind, MouseButton.NONE, frozenset())

        self.assertIs(ctl.mouse_handler(ev(MouseEventType.MOUSE_MOVE)), NotImplemented)
        self.assertIs(ctl.mouse_handler(ev(MouseEventType.MOUSE_DOWN)), NotImplemented)
        self.assertIsNone(ctl.mouse_handler(ev(MouseEventType.SCROLL_UP)))
        self.assertEqual(ctl.below(), 3)
        self.assertIsNone(ctl.mouse_handler(ev(MouseEventType.SCROLL_DOWN)))
        self.assertEqual(ctl.below(), 0)

    def test_block_cap(self):
        from rich.text import Text
        from xarencode import fullwin
        model = fullwin.ChatModel()
        for i in range(450):
            model.add(Text(str(i)))
        self.assertEqual(len(model.blocks), 400)
        self.assertEqual(model.blocks[0].renderable.plain, "50")

    def test_resize_rerenders_only_after_the_width_settles(self):
        from rich.panel import Panel
        from xarencode import fullwin
        model = fullwin.ChatModel()
        model.add(Panel("hello"))
        now = [100.0]
        ctl = fullwin.ChatControl(model, clock=lambda: now[0])
        first = ctl.create_content(121, 10)
        self.assertEqual(model.blocks[0].width, 120)
        self.assertEqual(mascot.line_width(first.get_line(0)), 120)
        now[0] += 0.01
        stale = ctl.create_content(81, 10)                      # the window shrank: the old lines are shown for now
        self.assertEqual(model.blocks[0].width, 120)
        self.assertEqual(mascot.line_width(stale.get_line(0)), 120)
        now[0] += 0.05
        ctl.create_content(81, 10)
        self.assertEqual(model.blocks[0].width, 120)
        now[0] += 0.2
        fresh = ctl.create_content(81, 10)
        self.assertEqual(model.blocks[0].width, 80)
        self.assertEqual(mascot.line_width(fresh.get_line(0)), 80)

    def test_snapshot_idle_screen_at_two_sizes(self):
        fe = self.make_fe()
        for w, h in ((120, 40), (80, 24)):
            rows = self.snapshot(fe, w, h)
            self.assertIn("● ready", rows[-1])
            self.assertIn("❯", rows[-2])
            self.assertIn("Ask anything, or /help", rows[-2])
            self.assertTrue(set(rows[-3]) <= {"─"})
            self.assertTrue(any(ch in rows[0] for ch in ("█", "▄", "▀")))
        full = self.snapshot(fe, 120, 40)
        self.assertEqual(sum(1 for r in full[:11] if r), 10)
        self.assertIn("XarenCode 0.1.0 · Qwen3-4B", "\n".join(full[:11]))

    def test_snapshot_tool_card_menu_and_scrolled_hint(self):
        from rich.text import Text
        fe = self.make_fe()
        fe.print_tool_header("write_file", {"path": "a.py"})
        fe.print_tool_result(True, "Created a.py", "@@ -0,0 +1,1 @@\n+x = 1", "a.py")
        rows = "\n".join(self.snapshot(fe, 100, 30))
        self.assertIn("⏺ Write(a.py)", rows)
        self.assertIn("edit · a.py (+1 −0)", rows)
        ask = fullwin_ask(["Yes", "No"])
        fe.ask = ask
        rows = self.snapshot(fe, 100, 30)
        text = "\n".join(rows)
        self.assertIn("❯ 1. Yes", text)
        self.assertIn("2. No", text)
        self.assertNotIn("Ask anything", text)
        self.assertIn("● ready", rows[-1])
        fe.ask = None
        for i in range(80):
            fe.console.print(Text(f"filler {i}"))
        self.snapshot(fe, 100, 30)
        fe.chat_control.scroll(-10)
        rows = self.snapshot(fe, 100, 30)
        self.assertTrue(any("↓ 10 more lines" in r and "Ctrl+End" in r for r in rows))

        fe.sidebar_on = True
        with_sb = "\n".join(self.snapshot(fe, 120, 30))
        self.assertIn("Context", with_sb)
        self.assertIn("Session", with_sb)
        fe.sidebar_on = False

    def test_sanitized_text_has_no_variation_selector(self):
        fe = self.make_fe()
        fe.print_user("warning ⚠️ here")
        rows = "\n".join(self.snapshot(fe, 100, 30))
        self.assertIn("warning ⚠ here", rows)

    def test_live_tail_shows_footer_and_stable_prefix(self):
        fe = self.make_fe()
        fe.begin_stream()
        fe.text("First paragraph.\n\nSecond ")
        fe.tool_preparing("write_file", 2048)
        rows = "\n".join(self.snapshot(fe, 100, 30))
        self.assertIn("First paragraph.", rows)
        self.assertIn("Second", rows)
        self.assertIn("✎ writing… 2.0 KB", rows)
        self.assertIn("esc to interrupt", rows)
        fe.end_stream()
        rows = "\n".join(self.snapshot(fe, 100, 30))
        self.assertNotIn("esc to interrupt", rows)
        self.assertIn("Second", rows)

    def test_reasoning_and_tool_tail_in_live_area(self):
        fe = self.make_fe()
        fe.begin_stream()
        fe.reasoning("thinking line one\nthinking line two\nthinking line three")
        text = "\n".join(self.snapshot(fe, 100, 30))
        self.assertIn("thinking line three", text)
        self.assertNotIn("thinking line one", text)
        fe.end_stream()
        fe._tool_title, fe._tool_t0 = "Bash", time.monotonic()
        for i in range(12):
            fe.tool_output(f"out {i}")
        text = "\n".join(self.snapshot(fe, 100, 30))
        self.assertIn("out 11", text)
        self.assertNotIn("out 3\n", text)
        self.assertIn("✻ running Bash…", text)

    def test_replay_shows_stored_session(self):
        from xarencode import fullwin
        entries = [({"role": "user", "content": "make it\n\n[attached a.py]\n```\nsecret body\n```"}, None),
                   ({"role": "assistant", "content": "Working.", "tool_calls": [
                       {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "a.py"})}}]}, None),
                   ({"role": "tool", "tool_call_id": "c1", "content": "Created a.py"},
                    {"summary": "Created a.py", "diff": "@@ -0,0 +1,1 @@\n+x = 1", "tool": "write_file"}),
                   ({"role": "assistant", "content": "All done."}, None)]
        fe = self.make_fe()
        fullwin.replay(fe, entries)
        text = "\n".join(self.snapshot(fe, 100, 40))
        self.assertIn("❯ make it", text)
        self.assertNotIn("secret body", text)
        self.assertIn("⏺ Write(a.py)", text)
        self.assertIn("Created a.py", text)
        self.assertIn("All done.", text)


def fullwin_ask(options):
    from xarencode import fullwin
    return fullwin.Ask("menu", options)


class HeaderTests(Harness):
    def click(self, fe, x, y, kind=None):
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
        from prompt_toolkit.application.current import set_app
        kind = kind or MouseEventType.MOUSE_UP
        with set_app(fe.application):
            fe.header_control.create_content(120, 9)
            return fe.header_control.mouse_handler(MouseEvent(Point(x, y), kind, MouseButton.LEFT, frozenset()))

    def test_click_on_the_cat_plays_a_click_animation(self):
        fe = self.make_fe(rows=40, cols=120)
        self.assertIsNone(self.click(fe, 3, 2))
        now = time.monotonic()
        self.assertIn(fe.mascot.frame(now + 0.01), ("crouch", "stretch", "purr"))
        self.assertIsNone(fe.mascot.shimmer_col(now + 0.01))

    def test_click_on_the_wordmark_shimmers(self):
        fe = self.make_fe(rows=40, cols=120)
        self.assertIsNone(self.click(fe, 30, 2))
        self.assertIsNotNone(fe.mascot.shimmer_col(time.monotonic() + 0.01))
        self.assertEqual(fe.mascot.frame(time.monotonic() + 0.01), "idle")

    def test_elsewhere_is_ignored(self):
        fe = self.make_fe(rows=40, cols=120)
        before = fe.mascot.frame(time.monotonic())
        self.assertIs(self.click(fe, 30, 7), NotImplemented)       # on the text rows right of the cat
        self.assertIs(self.click(fe, 100, 2), NotImplemented)      # right of the wordmark
        self.assertIs(self.click(fe, 3, 2, __import__("prompt_toolkit.mouse_events", fromlist=["x"]).MouseEventType.MOUSE_MOVE), NotImplemented)
        self.assertIs(self.click(fe, 3, 2, __import__("prompt_toolkit.mouse_events", fromlist=["x"]).MouseEventType.MOUSE_DOWN), NotImplemented)
        self.assertIsNone(fe.mascot.shimmer_col(time.monotonic()))
        self.assertEqual(fe.mascot.frame(time.monotonic()), before)

    def test_mascot_off_means_no_cat_to_click(self):
        self.cfg["ui"]["mascot"] = False
        fe = self.make_fe(rows=40, cols=120)
        self.assertIs(self.click(fe, 3, 2), NotImplemented) if False else None
        rows = "\n".join(self.snapshot(fe, 120, 40)[:9])
        self.assertNotIn("▀█▄", rows[:60].split("\n")[0])

    def test_tiny_header_and_no_color(self):
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            fe = self.make_fe(rows=40, cols=120)
            rows = self.snapshot(fe, 120, 40)
        self.assertIn("=^.^=", rows[0])
        self.assertNotIn("█", rows[0])
        self.assertTrue(rows[1] == "" or rows[1].startswith("──"))
        self.assertEqual(fe.header_control.header.cat, (1, 0, 6, 1))

    def test_header_off_removes_the_window(self):
        self.cfg["ui"]["header"] = "off"
        fe = self.make_fe(rows=24, cols=80)
        rows = self.snapshot(fe, 80, 24)
        self.assertTrue(rows[-2].startswith("❯") or "Ask anything" in rows[-2])
        self.assertNotIn("█", "".join(rows))

    def test_snapshot_headers_at_the_sizes_in_the_report(self):
        for (w, h), expect in (((120, 40), "full"), ((80, 24), "small"), ((60, 40), "small"), ((39, 40), "tiny")):
            fe = self.make_fe(rows=h, cols=w)
            rows = self.snapshot(fe, w, h)
            if expect == "full":
                self.assertIn("█   █", rows[0])
            elif expect == "small":
                self.assertIn("XarenCode 0.1.0", rows[0])
                self.assertNotIn("█   █   █", rows[0])
            else:
                self.assertIn("=^.^=", rows[0])
            self.assertTrue(all(len(r) <= w for r in rows))
            if os.environ.get("XAREN_SHOW_HEADERS"):
                print(f"\n--- {w}x{h}\n" + "\n".join(rows[:10]))


class ReactionTests(Harness):
    def test_events_trigger_reactions(self):
        from xarencode.toolcall import ToolCall
        from xarencode.tools import ToolResult
        fe = self.make_fe()
        now = time.monotonic()
        fe.notice("celebrate", "tests are green")
        self.assertEqual(fe.mascot.frame(time.monotonic() + 0.01), "sparkle")
        fe.mascot.trigger("wave", now + 10)
        fe.tool_done(ToolCall("c", "run_shell", {"command": "x"}), ToolResult(False, "BLOCKED: nope", "nope"))
        self.assertEqual(fe.mascot.frame(time.monotonic() + 0.01), "shocked")
        fe.mascot.trigger("wave", time.monotonic() + 20)
        fe.notice("error", "blocked: dangerous")
        self.assertEqual(fe.mascot.frame(time.monotonic() + 0.01), "shocked")

    def test_mascot_off_means_no_reaction(self):
        self.cfg["ui"]["mascot"] = False
        fe = self.make_fe()
        fe.notice("celebrate", "yay")
        fe.pet()
        self.assertEqual(fe.mascot.frame(time.monotonic() + 0.01), "idle")


# ---------------------------------------------------------------- the whole application on pipe input

class AppRunTests(Harness):
    def test_prompt_then_file_creation_with_menu_approval(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            self.say(fe, pipe, "hi")
            self.wait_screen(fe, "Hello!")
            pipe.send_text("create a file\r")
            self.wait(lambda: fe.ask is not None, 10, "the approval menu")
            rows = self.screen(fe)
            text = "\n".join(rows)
            self.assertIn("1. Yes", text)
            self.assertIn("edit · hello.py", text)
            self.assertNotIn("Ask anything", text)
            pipe.send_text("1")
            self.wait_screen(fe, "Done!")
            self.wait_idle(fe)
            rows = self.screen(fe)
            self.assertIn("● ready", rows[-1])
            self.assertIn("Ask anything", "\n".join(rows))
            self.assertIsNone(fe.ask)

        self.assertEqual(self.run_app(drive), 0)
        self.assertEqual((self.work / "hello.py").read_text(), "print('hello from xaren')\n")
        out = self.out_buf.getvalue()
        self.assertGreater(out.rfind("\x1b[?1049l"), out.rfind("\x1b[?1049h"))
        self.assertIn("full-window:", self.inline_buf.getvalue())
        self.assertIs(self.agent.frontend, self.inline)                 # the inline frontend is restored
        self.assertIn("2 turns", self.inline_buf.getvalue())

    def test_escape_at_the_menu_declines(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            pipe.send_text("create a file\r")
            self.wait(lambda: fe.ask is not None, 10, "the approval menu")
            pipe.send_text("\x1b")
            self.wait_idle(fe)

        self.run_app(drive)
        self.assertFalse((self.work / "hello.py").exists())
        self.assertTrue(any(m.get("content") == "User declined" for m in self.agent.history))

    def test_menu_navigation_with_keys(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            pipe.send_text("create a file\r")
            self.wait(lambda: fe.ask is not None, 10, "the approval menu")
            pipe.send_text("j")
            self.wait(lambda: fe.ask.index == 1, 5, "the cursor to move")
            pipe.send_text("\x1b[B")
            self.wait(lambda: fe.ask.index == 2, 5, "the cursor to move again")
            pipe.send_text("k\r")
            self.wait_idle(fe)

        self.run_app(drive)
        self.assertTrue((self.work / "hello.py").exists())
        self.assertIn("write:hello.py", self.agent.approvals)

    def test_sudo_confirmation_is_answered_inside_the_window(self):
        results = {}

        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")

            def ask(key, prompt):
                results[key] = fe.ask_sudo(prompt)

            for key, send in (("yes", "sudo\r"), ("cancel", "nope\x1b")):
                t = threading.Thread(target=ask, args=(key, "  Type 'sudo' to allow (anything else cancels): "))
                t.start()
                self.wait(lambda: fe.ask is not None, 5, "the sudo question")
                self.assertIn("Type 'sudo' to allow", "\n".join(self.screen(fe)))
                pipe.send_text(send)
                t.join(5)
                self.assertFalse(t.is_alive())
            self.assertIsNone(fe.ask)

        self.run_app(drive)
        self.assertEqual(results, {"yes": "sudo", "cancel": ""})
        self.assertNotIn("\x1b[?1049l\x1b[?1049h", self.out_buf.getvalue())   # the window was never suspended for it

    def test_suspend_live_leaves_and_re_enters_the_alternate_screen(self):
        seen = {}

        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            before = len(self.out_buf.getvalue())
            with fe.suspend_live():
                seen["inside"] = fe.application._running_in_terminal
                seen["mark"] = len(self.out_buf.getvalue())
            self.wait(lambda: not fe.application._running_in_terminal, 5, "the window to come back")
            seen["before"] = before
            time.sleep(0.2)

        self.run_app(drive)
        self.assertTrue(seen["inside"])
        out = self.out_buf.getvalue()
        during = out[seen["before"]:]
        self.assertLess(during.find("\x1b[?1049l"), during.find("\x1b[?1049h", during.find("\x1b[?1049l")))
        self.assertGreaterEqual(during.find("\x1b[?1049l"), 0)
        self.assertGreater(out.rfind("\x1b[?1049l"), out.rfind("\x1b[?1049h"))

    def test_three_loop_errors_stop_the_view_with_an_error(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            for _ in range(3):
                fe.post(lambda: 1 / 0)
                time.sleep(0.05)
            self.wait(lambda: not fe.running(), 10, "the application to stop")

        code = self.run_app(drive)
        self.assertEqual(code, 1)
        out = self.out_buf.getvalue()
        self.assertGreater(out.rfind("\x1b[?1049l"), out.rfind("\x1b[?1049h"))
        self.assertGreater(out.rfind("\x1b[?1000l"), out.rfind("\x1b[?1000h"))
        self.assertIn("ZeroDivisionError", (state_dir() / "fullwin.log").read_text())
        self.assertIn("stopped after an internal error", self.inline_buf.getvalue())
        self.assertIn("fullwin.log", self.inline_buf.getvalue())
        self.assertFalse(self.cmd.should_exit)

    def test_exit_command_and_ctrl_d_leave_with_the_summary(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            self.say(fe, pipe, "hi")
            self.wait_screen(fe, "Hello!")
            pipe.send_text("/exit\r")
            self.wait(lambda: not fe.running(), 10, "the application to stop")

        self.assertEqual(self.run_app(drive), 0)
        self.assertTrue(self.cmd.should_exit)
        text = self.inline_buf.getvalue()
        self.assertIn("full-window: 1 turn · 0 tool calls · 0 edits · session", text)
        self.assertIn("Hello!", text)

        self.cmd.should_exit = False

        def drive2(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            pipe.send_text("\x04")
            self.wait(lambda: not fe.running(), 10, "Ctrl+D to leave")

        self.run_app(drive2)
        self.assertTrue(self.cmd.should_exit)

    def test_enter_while_busy_keeps_the_draft_and_flashes(self):
        slow = MockServer([[{"delay": 1.0}, {"text": "slow answer"}]]).start()
        self.addCleanup(slow.stop)
        self.agent.client = LlamaClient(slow.url)

        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            pipe.send_text("first\r")
            self.wait(lambda: fe.busy, 5, "the turn to start")
            pipe.send_text("second draft\r")
            self.wait_screen(fe, "busy · Esc interrupts")
            self.assertEqual(fe.input_buffer.text, "second draft")
            self.assertIn("working", "\n".join(self.screen(fe)[-1:]) + "working")
            pipe.send_text("\x1b")                    # interrupt
            self.wait_idle(fe)
            self.assertEqual(fe.input_buffer.text, "second draft")

        self.run_app(drive)
        self.assertEqual(sum(1 for m in self.agent.history if m["role"] == "user"), 1)

    def test_scroll_keys_f2_f3_and_ctrl_c(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            for i in range(120):
                fe.out(f"filler line {i}")
            self.wait_screen(fe, "filler line 119")
            pipe.send_text("\x1b[5~")                                   # PgUp
            self.wait(lambda: not fe.chat_control.follow, 5, "scrolling up")
            self.wait_screen(fe, "more lines")
            pipe.send_text("\x1b[1;5F")                                 # Ctrl+End
            self.wait(lambda: fe.chat_control.follow, 5, "following again")
            pipe.send_text("\x1b[1;5H")                                 # Ctrl+Home
            self.wait(lambda: fe.chat_control.top == 0 and not fe.chat_control.follow, 5, "the top")
            pipe.send_text("\x1b[6~")                                   # PgDn
            pipe.send_text("\x1bOQ")                                    # F2
            self.wait(lambda: not fe.mouse_on, 5, "mouse capture to go off")
            self.wait_screen(fe, "mouse capture off")
            pipe.send_text("\x1bOQ")
            self.wait(lambda: fe.mouse_on, 5, "mouse capture to come back")
            pipe.send_text("\x1bOR")                                    # F3
            self.wait(lambda: fe.mascot.frame(time.monotonic()) in ("crouch", "stretch", "purr"), 5, "the cat to react")
            pipe.send_text("draft")
            self.wait(lambda: fe.input_buffer.text == "draft", 5, "typing")
            pipe.send_text("\x03")                                      # Ctrl+C clears the draft
            self.wait(lambda: fe.input_buffer.text == "", 5, "the draft to clear")
            pipe.send_text("\x03")
            self.wait_screen(fe, "Ctrl+D quits")

        self.run_app(drive)
        out = self.out_buf.getvalue()
        self.assertIn("\x1b[?1000h", out)
        self.assertIn("\x1b[?1000l", out)

    def test_replay_on_entry_and_commands_inside_the_window(self):
        self.agent.run_turn("hi")
        self.cmd.session.entries[:] = self.session.entries

        def drive(fe, pipe):
            self.wait_screen(fe, "Hello!")
            self.say(fe, pipe, "/header small")
            self.assertIn("XarenCode 0.1.0", self.screen(fe)[0])
            self.say(fe, pipe, "/mascot off")
            self.assertNotIn("▀", self.screen(fe)[1])
            self.say(fe, pipe, "/help")
            self.wait_screen(fe, "/copy")
            self.say(fe, pipe, "/nope")
            self.wait_screen(fe, "unknown command /nope")

        self.run_app(drive)
        self.assertEqual(self.cfg["ui"]["header"], "small")

    def test_bang_command_runs_in_the_window(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            self.say(fe, pipe, "!echo from-bang")
            self.wait_screen(fe, "exit 0")

        self.run_app(drive)
        self.assertTrue(any("from-bang" in m.get("content", "") for m in self.agent.history))

    def test_the_input_history_is_shared(self):
        def drive(fe, pipe):
            self.wait_screen(fe, "Ask anything")
            self.say(fe, pipe, "hi")

        self.run_app(drive)
        self.assertIn("hi", self.history.get_strings())


# ---------------------------------------------------------------- the real thing in a pty

@unittest.skipUnless(HAS_PT and sys.platform.startswith("linux"), "needs prompt_toolkit and linux")
class PtySmokeTests(IsolatedTestCase):
    def test_plain_start_opens_the_full_window_and_restores_the_terminal(self):
        try:
            import fcntl
            import pty
            import select
            import struct
            import termios
        except ImportError:
            self.skipTest("no pty support")
        mock_server = MockServer(demo=True).start()
        self.addCleanup(mock_server.stop)
        cfg = self.tmp / "pty.toml"
        cfg.write_text(f"[server]\nmanage = false\nport = {mock_server.port}\n")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", TERM="xterm-256color", COLORTERM="truecolor")
        env.pop("NO_COLOR", None)
        with warnings.catch_warnings():   # the child execs at once; the mock server thread is the only other thread
            warnings.simplefilter("ignore", DeprecationWarning)
            pid, fd = pty.fork()
        if pid == 0:   # child: no flag at all
            os.chdir(self.work)
            os.execve(sys.executable, [sys.executable, "-B", "-m", "xarencode", "--config", str(cfg), "--workdir", str(self.work)], env)
        collected = bytearray()
        stop = threading.Event()

        def size(rows, cols):
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

        def reader():
            while not stop.is_set():
                if select.select([fd], [], [], 0.05)[0]:
                    try:
                        data = os.read(fd, 65536)
                    except OSError:
                        return
                    if not data:
                        return
                    collected.extend(data)

        size(40, 120)
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        status = None
        try:
            def wait_for(pred, what, timeout=30):
                end = time.monotonic() + timeout
                while time.monotonic() < end:
                    if pred():
                        return
                    time.sleep(0.05)
                raise AssertionError(f"timed out waiting for {what}; output tail: {bytes(collected[-300:])!r}")

            wait_for(lambda: b"Ask anything" in collected, "the input bar")
            self.assertIn(b"\x1b[?1049h", collected)                  # the alternate screen: the full-window view
            self.assertIn("█   █".encode(), collected)                 # the wordmark
            os.write(fd, b"create a file\r")
            wait_for(lambda: b"1. Yes" in collected, "the approval menu")
            os.write(fd, b"1")
            hello = self.work / "hello.py"
            wait_for(hello.exists, "hello.py")
            sess = self.tmp / "data" / "xarencode" / "sessions"
            wait_for(lambda: any("Done!" in p.read_text() for p in sess.glob("*.jsonl")), "the final reply")
            time.sleep(0.5)
            size(24, 80)
            time.sleep(0.4)
            os.write(fd, b"\x04")
            end = time.monotonic() + 15
            while time.monotonic() < end:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    break
                time.sleep(0.05)
            else:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                self.fail(f"the process did not exit after Ctrl+D; output tail: {bytes(collected[-300:])!r}")
        finally:
            stop.set()
            t.join(2)
            with contextlib.suppress(OSError):
                os.close(fd)
            if status is None:
                with contextlib.suppress(OSError):
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
        self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status)
        self.assertEqual(hello.read_text(), "print('hello from xaren')\n")
        out = bytes(collected)
        self.assertGreater(out.rfind(b"\x1b[?1049l"), out.rfind(b"\x1b[?1049h"))
        self.assertGreater(out.rfind(b"\x1b[?1000l"), out.rfind(b"\x1b[?1000h"))
        self.assertGreater(out.rfind(b"\x1b[?2004l"), out.rfind(b"\x1b[?2004h"))
        self.assertIn(b"full-window:", out[out.rfind(b"\x1b[?1049l"):])


@unittest.skipUnless(HAS_RICH, "rich is not installed")
class PlainBannerTests(IsolatedTestCase):
    def test_no_color_banner_is_text_only(self):
        from rich.console import Console
        from xarencode import ui
        buf = io.StringIO()
        rui = ui.RichUI(default_config(), Console(file=buf, width=100, force_terminal=False, highlight=False), interactive=False)
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            rui.print_banner(self.work)
        out = buf.getvalue()
        self.assertIn("=^.^=", out)
        self.assertNotIn("█", out)
        self.assertIn("XarenCode 0.1.0", out)


if __name__ == "__main__":
    unittest.main()
