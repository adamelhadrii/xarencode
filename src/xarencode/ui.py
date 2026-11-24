"""Inline terminal UI (rich + prompt_toolkit): streaming replies, tool cards, approvals, input and status bar."""
from __future__ import annotations

import atexit
import contextlib
import datetime as _dt
import io
import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.syntax import Syntax
from rich.text import Text

from . import commands, mascot, persona, richpt, sessions
from .agent import Agent, Approval, BaseFrontend, TurnStats
from .config import config_path, data_dir, model_short_name
from .server import LlamaServer, ServerStartError
from .tools import build_registry, devserver

TOOL_TITLES = {"run_shell": "Bash", "read_file": "Read", "list_dir": "List", "search_code": "Search", "write_file": "Write",
               "edit_file": "Update", "dev_server": "Dev", "vm": "VM"}
ADD_STYLE, DEL_STYLE = "on #12361f", "on #451a1f"

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "code_theme": "monokai", "dim": "fg:#7a828e", "rule": "fg:#3a404c",
        "status_bg": "#1b1f27", "status_fg": "#a6b0bf",
        "menu_bg": "#20242c", "menu_fg": "#d8dee9", "meta_bg": "#2b303b", "meta_fg": "#8c96a6",
        "add_bg": "on #12361f", "del_bg": "on #451a1f", "sidebar_border": "#3a404c",
    },
    "light": {
        "code_theme": "friendly", "dim": "fg:#5c6370", "rule": "fg:#c8cdd5",
        "status_bg": "#e8ecf2", "status_fg": "#2c323c",
        "menu_bg": "#f0f2f5", "menu_fg": "#1e222a", "meta_bg": "#e2e6ec", "meta_fg": "#5c6370",
        "add_bg": "on #d4f4dd", "del_bg": "on #fadadd", "sidebar_border": "#c8cdd5",
    },
    "contrast": {
        "code_theme": "monokai", "dim": "fg:#b0b8c4", "rule": "fg:#6e7681",
        "status_bg": "#000000", "status_fg": "#ffffff",
        "menu_bg": "#111111", "menu_fg": "#ffffff", "meta_bg": "#222222", "meta_fg": "#cccccc",
        "add_bg": "on #0d4420", "del_bg": "on #61121a", "sidebar_border": "#8b949e",
    },
    "mono": {
        "code_theme": "bw", "dim": "dim", "rule": "dim",
        "status_bg": "", "status_fg": "",
        "menu_bg": "#262626", "menu_fg": "#d0d0d0", "meta_bg": "#303030", "meta_fg": "#8a8a8a",
        "add_bg": "reverse", "del_bg": "dim", "sidebar_border": "dim",
    },
}


def theme_of(cfg: dict) -> dict[str, str]:
    if os.environ.get("NO_COLOR"):
        return THEMES["mono"]
    name = str(cfg.get("ui", {}).get("theme", "dark")).lower()
    return THEMES.get(name, THEMES["dark"])


# ---------------------------------------------------------------- pure render helpers

def tool_label(name: str, args: dict) -> str:
    key = {"run_shell": "command", "read_file": "path", "list_dir": "path", "search_code": "pattern",
           "write_file": "path", "edit_file": "path"}.get(name)
    if key and key in args:
        return str(args[key])
    if name == "dev_server":
        return f"{args.get('action', '')} {args.get('name') or args.get('command') or ''}".strip()
    if name == "vm":
        return f"{args.get('action', '')} {args.get('name', '')}".strip()
    return next((str(v) for v in args.values()), "")


def diff_counts(diff: str) -> tuple[int, int]:
    add = sum(1 for x in diff.splitlines() if x.startswith("+") and not x.startswith("+++"))
    rem = sum(1 for x in diff.splitlines() if x.startswith("-") and not x.startswith("---"))
    return add, rem


def render_diff(diff: str, path: str, accent: str = "#0a84ff", add_style: str = ADD_STYLE, del_style: str = DEL_STYLE) -> Panel:
    rows: list[Text] = []
    old = new = 0
    for ln in diff.splitlines():
        if ln.startswith(("---", "+++")):
            continue
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", ln)
            if m:
                old, new = int(m.group(1)), int(m.group(2))
            rows.append(Text("  ⋯", style="dim"))
        elif ln.startswith("+"):
            rows.append(Text(f"{new:>4} + {ln[1:]}", style=add_style))
            new += 1
        elif ln.startswith("-"):
            rows.append(Text(f"{old:>4} - {ln[1:]}", style=del_style))
            old += 1
        elif ln.startswith(" "):
            rows.append(Text(f"{new:>4}   {ln[1:]}", style="dim"))
            old += 1
            new += 1
        else:
            rows.append(Text(ln, style="dim"))
    add, rem = diff_counts(diff)
    return Panel(Group(*rows), title=f"edit · {path} (+{add} −{rem})", title_align="left", border_style=accent, box=box.ROUNDED)


def fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


def history_trim(path: Path, keep: int = 1000) -> None:
    """prompt_toolkit FileHistory stores entries as '# timestamp' blocks; keep the newest `keep`."""
    try:
        lines = path.read_text("utf-8").split("\n")
    except OSError:
        return
    starts = [i for i, ln in enumerate(lines) if ln.startswith("# ")]
    if len(starts) > keep:
        path.write_text("\n".join(lines[starts[len(starts) - keep]:]), "utf-8")


# ---------------------------------------------------------------- stream view and key watcher

class StreamView:
    """Live renderable: the tail of the reply as Markdown plus a status footer."""

    def __init__(self, ui: "RichUI"):
        self.ui = ui

    def __rich_console__(self, console, options):
        ui = self.ui
        room = max(3, console.size.height - 4)
        if ui._buf.strip():
            lines = console.render_lines(Markdown(ui._buf, code_theme=ui.code_theme()), options.update(height=None), pad=False)
            for line in lines[-room:]:
                yield from line
                yield Segment.line()
        elif ui._think.strip():
            tail = ui._think.strip().splitlines()[-2:]
            yield Text("\n".join(tail), style="dim italic")
        yield ui.footer()


class EscWatcher:
    """Puts the terminal in cbreak mode and calls on_esc when a lone Esc arrives."""

    def __init__(self, on_esc):
        self.on_esc = on_esc
        self.fd = None
        self.old = None
        self.stop_ev = threading.Event()
        self.thread = None

    def start(self) -> None:
        try:
            import termios
            import tty
            if not sys.stdin.isatty():
                return
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except (ImportError, OSError, ValueError):
            self.fd = None
            return
        atexit.register(self._restore)
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_ev.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.1)
                if ready and os.read(self.fd, 32) == b"\x1b":
                    self.on_esc()
            except (OSError, ValueError):
                return

    def _restore(self) -> None:
        if self.fd is not None and self.old is not None:
            import termios
            with contextlib.suppress(OSError, termios.error):
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def stop(self) -> None:
        self.stop_ev.set()
        if self.thread:
            self.thread.join(1)
            self.thread = None
        self._restore()
        self.fd = None


# ---------------------------------------------------------------- the frontend

class RichUI(BaseFrontend):
    def __init__(self, cfg: dict, console: Console | None = None, interactive: bool = True):
        self.cfg = cfg
        self.console = console or Console(highlight=False)
        self.accent = cfg["ui"]["accent"]
        self.interactive = interactive
        self.agent: Agent | None = None
        self.stats_last = TurnStats(ctx_size=cfg["server"]["ctx_size"])
        self.state = "ready"
        self._buf = self._think = ""
        self._live: Live | None = None
        self._tail: deque[str] = deque(maxlen=8)
        self._tail_live: Live | None = None
        self._watcher = EscWatcher(self._on_esc)
        self._t0 = self._first = 0.0
        self._deltas = 0
        self._verb = "Cooking"
        self._preparing: tuple[str, int] | None = None
        self._pp = None
        self._diff_shown: set[str] = set()
        self._header_shown: set[str] = set()
        self.auto_edit_hint = False

    def code_theme(self) -> str:
        return theme_of(self.cfg)["code_theme"]

    # -- streaming
    def _on_esc(self) -> None:
        if self.agent:
            self.agent.interrupt()

    def footer(self) -> Text:
        now = time.monotonic()
        secs = int(now - self._t0)
        if self._preparing:
            name, n = self._preparing
            body = f"✎ writing… {fmt_kb(n)}" if name in ("write_file", "edit_file") else f"✎ preparing {name}… {fmt_kb(n)}"
        elif self._pp and not self._buf and not self._think:
            pct = min(100, round(100 * self._pp.processed / max(1, self._pp.total)))
            cache = f" · {self._pp.cache} cached" if self._pp.cache else ""
            body = f"✻ Reading prompt… {pct}% ({self._pp.processed}/{self._pp.total} tok{cache})"
        else:
            verb = "Thinking" if self._think and not self._buf else self._verb
            body = f"✻ {verb}… {secs}s"
            if self._deltas:
                tps = self._deltas / max(now - self._first, 1e-3) if self._first else 0
                body += f" · {self._deltas} tok · {tps:.1f} tok/s"
        return Text(f"{body} · esc to interrupt", style=f"dim {self.accent}")

    def begin_stream(self) -> None:
        self._buf = self._think = ""
        self._t0, self._first, self._deltas = time.monotonic(), 0.0, 0
        self._verb = persona.spinner_verb()
        self._preparing = None
        self._pp = None
        self.state = "working"
        if self._live is None:
            self._live = Live(StreamView(self), console=self.console, transient=True, refresh_per_second=8, vertical_overflow="crop")
            self._live.start()
        if self.interactive:
            self._watcher.start()

    def _tick(self) -> None:
        self._deltas += 1
        self._first = self._first or time.monotonic()

    def prompt_progress(self, ev) -> None:
        self._pp = ev

    def text(self, d: str) -> None:
        self._pp = None
        self._buf += d
        self._tick()

    def reasoning(self, d: str) -> None:
        self._pp = None
        self._think += d
        self._first = self._first or time.monotonic()

    def tool_preparing(self, name: str, nbytes: int) -> None:
        self._pp = None
        self._preparing = (name, nbytes)

    def end_stream(self) -> None:
        self._watcher.stop()
        if self._live is not None:
            self._live.stop()
            self._live = None
            if self._think.strip():
                self.console.print(Text(f"✻ Thought for {int(time.monotonic() - self._t0)}s", style="dim"))
            if self._buf.strip():
                self.console.print(Markdown(self._buf.strip(), code_theme=self.code_theme()))
        self._buf = self._think = ""
        self._preparing = None
        self._pp = None
        self.state = "ready"

    # -- tool cards
    def print_tool_header(self, name: str, args: dict, key: str | None = None) -> None:
        if key is not None:
            if key in self._header_shown:
                return
            self._header_shown.add(key)
        t = Text()
        t.append("⏺ ", style=self.accent)
        t.append(f"{TOOL_TITLES.get(name, name)}", style="bold")
        t.append(f"({tool_label(name, args)[:100]})", style="dim")
        self.console.print(t)

    def print_tool_result(self, ok: bool, summary: str, diff: str | None = None, path: str = "") -> None:
        th = theme_of(self.cfg)
        t = Text("  ⎿  ", style="dim")
        t.append("✓ " if ok else "✖ ", style="green" if ok else "red")
        t.append(summary.splitlines()[0][:160] if summary else "done", style="green" if ok else "red")
        self.console.print(t)
        if diff:
            self.console.print(render_diff(diff, path, self.accent, th["add_bg"], th["del_bg"]))

    def tool_start(self, call, plan) -> None:
        self.state = "running " + call.name
        self.print_tool_header(call.name, call.args, call.id)
        self._tail.clear()
        if call.name in ("run_shell", "dev_server") and self.interactive:
            self._tail_live = Live(self._tail_view(), console=self.console, transient=True, refresh_per_second=8)
            self._tail_live.start()

    def _tail_view(self):
        return Text("\n".join("      " + ln[:150] for ln in self._tail), style="dim")

    def tool_output(self, line: str) -> None:
        self._tail.append(line)
        if self._tail_live:
            self._tail_live.update(self._tail_view())

    @contextlib.contextmanager
    def suspend_live(self):
        """Free the terminal (for a password prompt)."""
        had = self._tail_live is not None
        if had:
            self._tail_live.stop()
            self._tail_live = None
        try:
            yield
        finally:
            if had and self.interactive:
                self._tail_live = Live(self._tail_view(), console=self.console, transient=True, refresh_per_second=8)
                self._tail_live.start()

    def tool_done(self, call, result) -> None:
        if self._tail_live:
            self._tail_live.stop()
            self._tail_live = None
        self.state = "ready"
        diff = result.detail.get("diff")
        if call.id in self._diff_shown:
            diff = None
        self._diff_shown.discard(call.id)
        self._header_shown.discard(call.id)
        if call.name in ("write_file", "edit_file") and result.ok and diff:
            self.print_tool_result(True, result.summary, diff, result.detail.get("path", ""))
        else:
            self.print_tool_result(result.ok, result.summary or result.content, None)

    def notice(self, level: str, text: str) -> None:
        style = {"info": "dim", "warn": "yellow", "error": "bold red", "celebrate": f"bold {self.accent}"}[level]
        prefix = {"info": "  ", "warn": "⚠ ", "error": "✖ ", "celebrate": "✦ "}[level]
        self.console.print(Text(prefix + text, style=style))

    def stats(self, s: TurnStats) -> None:
        self.stats_last = s

    # -- approvals (the three ask_* methods are what a full-window frontend overrides)
    def ask_choice(self, options: list[str]) -> int | None:
        return choose(options)

    def ask_text(self, prompt: str) -> str:
        return ask_line(prompt)

    def ask_sudo(self, prompt: str) -> str:
        return input(prompt).strip() if self.interactive else ""

    def approve(self, call, plan) -> Approval:
        self.print_tool_header(call.name, call.args, call.id)
        if plan.preview_kind == "diff" and plan.preview:
            path = call.args.get("path", "")
            th = theme_of(self.cfg)
            self.console.print(render_diff(plan.preview, path, self.accent, th["add_bg"], th["del_bg"]))
            self._diff_shown.add(call.id)
        elif plan.preview_kind == "bash":
            self.console.print(Panel(Syntax(plan.preview, "bash", theme=self.code_theme(), background_color="default", word_wrap=True),
                                     title="run this?", title_align="left", border_style="yellow", box=box.ROUNDED))
        elif plan.reason or plan.preview:
            self.console.print(Text("  " + (plan.preview or plan.reason), style="yellow"))
        if plan.tag:
            self.console.print(Text(f"  ⚠ {plan.tag} (asked every time)", style="bold yellow"))
        if plan.hard_ask:
            self.console.print(Text("  ⚠ outside the working directory (asked even in YOLO mode)", style="bold yellow"))
        if plan.verdict.name == "ELEVATED":
            self.console.print(Text("  This needs administrator rights.", style="bold yellow"))
            answer = self.ask_sudo("  Type 'sudo' to allow (anything else cancels): ")
            return Approval("yes") if answer == "sudo" else Approval("no")
        options = ["Yes"]
        if plan.approval_key:
            what = plan.approval_key.split(":", 1)[-1] if plan.approval_key.split(":")[0] in ("write", "edit") else plan.approval_key
            kind = "edits to" if plan.approval_key.startswith(("write:", "edit:")) else ""
            options.append(f"Yes, and don't ask again for {kind + ' ' if kind else ''}{what} this session")
        options.append(f"No, tell {self.cfg['persona']['name']} what to do differently")
        pick = self.ask_choice(options)
        if pick is None:
            return Approval("no")
        if pick == 0:
            return Approval("yes")
        if pick == len(options) - 1:
            text = self.ask_text(f"  Tell {self.cfg['persona']['name']}: ")
            return Approval("feedback", text) if text else Approval("no")
        return Approval("always")

    # -- status line and banner
    def status_line(self) -> str:
        return "".join(text for _, text in self.status_fragments())

    def status_fragments(self) -> list[tuple[str, str]]:
        """The status bar as (style, text) pairs for prompt_toolkit; status_line() is the plain-text view."""
        a = self.agent
        s = self.stats_last
        size = self.cfg["server"]["ctx_size"]
        used = (a.ctx.last_est if a else 0) or s.ctx_used
        filled = min(4, round(4 * used / size)) if size else 0
        cwd = str(a.workdir if a else Path.cwd()).replace(str(Path.home()), "~")
        auto = "on" if a and a.auto_edits else "off"
        tps = f"{s.tok_per_s:.1f} tok/s · " if s.tok_per_s else ""
        text = (f" {model_short_name(self.cfg)} · {tps}ctx {used / 1000:.1f}k/{size / 1000:.1f}k {'▰' * filled}{'▱' * (4 - filled)}"
                f" · {cwd} · ● {self.state} · ⏵⏵ auto-edits {auto} ")
        return [("", text), (mascot.YOLO_STYLE, " YOLO ")] if a and a.yolo else [("", text)]

    def print_banner(self, workdir: Path) -> None:
        hdr = mascot.banner(self.cfg, workdir, yolo=bool(self.agent and self.agent.yolo), width=self.console.width)
        self.console.print(richpt.to_rich_text(hdr.lines))

    # -- transcript printing (also used by render_transcript)
    def print_user(self, text: str) -> None:
        t = Text()
        if self.cfg.get("ui", {}).get("timestamps"):
            t.append(f"[{_dt.datetime.now().strftime('%H:%M')}] ", style="dim")
        t.append("❯ ", style=f"bold {self.accent}")
        t.append(text, style="bold")
        self.console.print(t)

    def print_assistant(self, text: str) -> None:
        self.console.print(Markdown(text, code_theme=self.code_theme()))


def render_transcript(meta: dict, entries, cfg: dict, width: int = 100) -> Console:
    """Re-render a stored session as a recorded Console, ready for save_svg/save_html."""
    console = Console(record=True, width=width, file=io.StringIO(), force_terminal=True, color_system="truecolor", highlight=False)
    ui = RichUI(cfg, console, interactive=False)
    red = sessions.redact
    head = Text(f"✦ XarenCode · {meta.get('model', '')} · {red(meta.get('workdir', ''))}", style=f"bold {ui.accent}")
    console.print(Panel(head, box=box.ROUNDED, border_style=ui.accent, expand=False))
    calls: dict[str, dict] = {}
    for msg, info in entries:
        role, content = msg.get("role"), red((msg.get("content") or "").strip())
        if role == "user":
            ui.print_user(content)
        elif role == "assistant":
            if content:
                ui.print_assistant(content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls[tc.get("id", "")] = args
                ui.print_tool_header(fn.get("name", "tool"), {k: red(str(v)) for k, v in args.items()})
        elif role == "tool":
            info = info or {}
            ok = not content.startswith(("ERROR", "BLOCKED", "User declined"))
            summary = red(info.get("summary") or (content.splitlines()[0] if content else "done"))
            diff = red(info["diff"]) if info.get("diff") else None
            ui.print_tool_result(ok, summary, diff, red(calls.get(msg.get("tool_call_id", ""), {}).get("path", "")))
    return console


# ---------------------------------------------------------------- prompt_toolkit pieces

def choose(options: list[str]) -> int | None:
    """Small inline menu: arrows, 1-9 and Enter choose; Esc or Ctrl+C cancels."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FormattedTextControl, Layout, Window

    idx = [0]
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(e):
        idx[0] = (idx[0] - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(e):
        idx[0] = (idx[0] + 1) % len(options)

    @kb.add("enter")
    def _enter(e):
        e.app.exit(result=idx[0])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _esc(e):
        e.app.exit(result=None)

    for n in range(1, min(len(options), 9) + 1):
        kb.add(str(n))(lambda e, n=n: e.app.exit(result=n - 1))

    def render():
        return [("bold fg:ansicyan" if i == idx[0] else "", f"  {'❯' if i == idx[0] else ' '} {i + 1}. {o}\n") for i, o in enumerate(options)]

    app = Application(layout=Layout(Window(FormattedTextControl(render), height=len(options))), key_bindings=kb,
                      full_screen=False, erase_when_done=True)
    return app.run()


def ask_line(text: str) -> str:
    from prompt_toolkit import prompt as pt_prompt
    try:
        return pt_prompt(text).strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def make_completer(workdir_fn):
    from prompt_toolkit.completion import Completer, Completion

    class XarenCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if text.startswith("/") and " " not in text:
                for name, desc in commands.command_list():
                    if name.startswith(text.lower()):
                        yield Completion(name, start_position=-len(text), display_meta=desc)
                return
            m = re.search(r"(?<!\S)@([\w./~-]*)$", text)
            if m:
                for c in commands.at_completions(m.group(1), workdir_fn()):
                    yield Completion("@" + c, start_position=-len(m.group(0)))

    return XarenCompleter()


def input_history():
    """The file-backed input history shared by every prompt (trimmed to the newest 1000 entries)."""
    from prompt_toolkit.history import FileHistory
    hist_path = data_dir() / "input_history"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history_trim(hist_path)
    return FileHistory(str(hist_path))


def make_prompt_session(ui: RichUI, workdir_fn):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    history = input_history()
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(e):
        e.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(e):
        e.current_buffer.insert_text("\n")

    @kb.add("s-tab")
    def _auto(e):
        if ui.agent:
            ui.agent.auto_edits = not ui.agent.auto_edits
        e.app.invalidate()

    style = Style.from_dict({"bottom-toolbar": "noreverse fg:#8a8a8a bg:default", "bottom-toolbar.text": "noreverse fg:#8a8a8a bg:default"})
    return PromptSession(history=history, key_bindings=kb, multiline=True, completer=make_completer(workdir_fn),
                         complete_while_typing=True, bottom_toolbar=ui.status_fragments, style=style,
                         prompt_continuation=lambda w, n, wrap: "  ", enable_history_search=False,
                         placeholder=[("fg:#6b6b6b", "Ask anything, or /help")])


# ---------------------------------------------------------------- the chat loop

def _edit_file(path: Path) -> None:
    subprocess.call([os.environ.get("EDITOR", "nano"), str(path)])


def _warm_up(cfg, server: LlamaServer, static_system: str, tools: list[dict], warm_client, cache) -> None:
    """Restore or prime the server's prefix KV cache while the user types."""
    from .prefix import PrefixCache
    try:
        bg = PrefixCache(cfg, warm_client, tools, static_system)
        res = bg.warm()
        if res in ("restored", "primed"):
            cache.dirty = False
    except Exception:
        pass


def run_chat(cfg: dict, workdir: Path, resume: str | None = None, config_file: Path | None = None, yolo: bool = False) -> int:
    from .llm import LlamaClient
    from .prefix import PrefixCache

    console = Console(highlight=False)
    ui = RichUI(cfg, console)
    if workdir == Path("/"):
        console.print(Text("Refusing to run in /. Pick a project folder.", style="bold red"))
        return 1
    server = LlamaServer(cfg)
    try:
        with console.status("[dim]starting llama-server (loading the model can take a minute)…") as st:
            server.ensure(lambda msg: st.update(f"[dim]{msg}…"))
    except ServerStartError as e:
        console.print(Text(f"✖ {e}", style="bold red"))
        return 1
    registry = build_registry(cfg)
    system = persona.system_prompt(cfg, workdir)
    history: list[dict] = []
    session = sessions.Session.new(cfg, workdir)
    if resume is not None:
        info = sessions.find_session(resume)
        if info:
            session = sessions.Session.load(info.path)
            history = session.messages()
        else:
            ui.notice("warn", "no saved session to resume; starting fresh")
    schemas = registry.schemas()
    static = getattr(system, "static", str(system))
    cache = PrefixCache(cfg, server.client, schemas, static)
    agent = Agent(cfg, server.client, registry, ui, system, workdir, history, sink=session.append, yolo=yolo, prefix_cache=cache)
    ui.agent = agent
    app = commands.App(cfg, agent, server, session, lambda text: console.print(Text(text)), workdir,
                       config_file or config_path(), editor=_edit_file)
    warm_client = LlamaClient(f"http://{server.client.host}:{server.client.port}")
    ui._cancel_warm = warm_client.cancel
    threading.Thread(target=_warm_up, args=(cfg, server, static, schemas, warm_client, cache), daemon=True).start()
    from .fullwin import run_fullwindow
    code = 0
    try:
        code = run_fullwindow(cfg, app, ui, input_history())
    finally:
        warm_client.cancel()
        console.print(Text(mascot.farewell_prefix(cfg) + persona.farewell(agent.session_stats, cfg), style="dim"))
        _shutdown(cfg, server, console)
        sessions.prune(cfg["agent"]["keep_sessions"])
    return code


def _shutdown(cfg, server: LlamaServer, console: Console) -> None:
    running = devserver.running()
    if running and sys.stdin.isatty():
        try:
            if input(f"Stop {len(running)} running dev server(s) ({', '.join(running)})? [y/N] ").strip().lower() in ("y", "yes"):
                devserver.stop_all()
        except (EOFError, KeyboardInterrupt):
            pass
    if server.started_by_us and not cfg["server"]["keep_running"]:
        server.stop()
