"""The full-window chat view: a fixed header with the mascot, a scrolling chat area, a fixed input bar and a status bar.

prompt_toolkit owns the screen and the keyboard (an Application with full_screen=True); Rich stays the renderer, and every
chat item is a Rich renderable converted to prompt_toolkit fragments by richpt.

Threads: the main thread runs the Application. Every submitted line (a prompt, a /command or a !command) runs on one worker
thread named "xaren-turn". Only the loop thread touches prompt_toolkit objects; the worker changes FullUI/ChatModel data and
then asks for a redraw or posts a function to the loop with FullUI.post().
"""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import datetime as _dt
import json
import os
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (ConditionalContainer, Dimension, Float, FloatContainer, FormattedTextControl, HSplit, Layout,
                                   UIContent, UIControl, VSplit, Window)
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from . import clipboard, commands, mascot, persona
from .config import model_short_name, state_dir
from .richpt import Line, make_console, sanitize, split_stable, to_lines
from .ui import TOOL_TITLES, RichUI, make_completer, theme_of

MAX_REPLAY = 200
RESET_SEQUENCES = "\x1b[?1000l\x1b[?1003l\x1b[?1006l\x1b[?2004l\x1b[?1049l\x1b[?25h"
DIM = "fg:#6b6b6b"


def log_path() -> Path:
    return state_dir() / "fullwin.log"


def log_error(text: str) -> None:
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"--- {_dt.datetime.now().isoformat(timespec='seconds')}\n{text.rstrip()}\n")
    except OSError:
        pass


# ---------------------------------------------------------------- chat model and controls

@dataclass
class Block:
    renderable: Any
    width: int = 0
    lines: list[Line] | None = None


class ChatModel:
    """The finished chat items. The lock guards `blocks` and `version`; `dirty` tells the ticker a redraw is due."""
    MAX_BLOCKS = 400

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.blocks: list[Block] = []
        self.version = 0
        self.dirty = False

    def add(self, renderable: Any) -> None:
        with self.lock:
            self.blocks.append(Block(renderable))
            if len(self.blocks) > self.MAX_BLOCKS:
                del self.blocks[:len(self.blocks) - self.MAX_BLOCKS]
            self.version += 1
            self.dirty = True


class BlockSink:
    """Stands in for the Rich console: everything RichUI prints becomes a block of the chat."""
    width = 100
    is_terminal = False

    def __init__(self, model: ChatModel, on_add: Callable[[], None] | None = None):
        self.model = model
        self.on_add = on_add

    def print(self, *objs, **kw) -> None:
        objs = tuple(Text(o) if isinstance(o, str) else o for o in objs)
        self.model.add(Text("") if not objs else objs[0] if len(objs) == 1 else Group(*objs))
        if self.on_add:
            self.on_add()


class ChatControl(UIControl):
    """The scrolling chat area. It hands prompt_toolkit only the visible lines, so prompt_toolkit never wraps or scrolls it."""
    RESIZE_SETTLE = 0.12   # seconds the width must stay put before the blocks are rendered again

    def __init__(self, model: ChatModel, live: Callable[[int], list[Line]] | None = None, clock: Callable[[], float] = time.monotonic):
        self.model = model
        self.live = live
        self.clock = clock
        self.follow = True
        self.top = 0
        self._w: int | None = None       # the width the blocks are rendered at
        self._seen: int | None = None    # the newest window width, and when it first appeared
        self._seen_at = 0.0
        self._redraw_due = False
        self._height, self._total = 24, 0
        self._console = make_console()
        self._flat: list[Line] = []
        self._flat_key: tuple | None = None
        self._flat_w: int = 0
        self._built_count: int = 0
        self._first_block: Any = None

    def is_focusable(self) -> bool:
        return False

    def _render_width(self, window_width: int) -> int:
        w = max(10, window_width - 1)
        if self._w is None:
            self._w = self._seen = w
        elif w == self._w:
            self._seen = w
        else:
            now = self.clock()
            if w != self._seen:
                self._seen, self._seen_at = w, now
            if now - self._seen_at >= self.RESIZE_SETTLE:
                self._w = w
            else:
                self._schedule_redraw()
        return self._w

    def _schedule_redraw(self) -> None:
        app = get_app_or_none()
        if self._redraw_due or app is None or app.loop is None:
            return
        self._redraw_due = True

        def fire() -> None:
            self._redraw_due = False
            app.invalidate()

        app.loop.call_later(self.RESIZE_SETTLE + 0.01, fire)

    def _lines(self, w: int) -> list[Line]:
        with self.model.lock:
            key = (self.model.version, w)
            if key != self._flat_key:
                blocks = self.model.blocks
                can_append = (
                    w == self._flat_w
                    and self._built_count <= len(blocks)
                    and (self._built_count == 0 or (blocks and blocks[0] is self._first_block))
                )
                if can_append:
                    for b in blocks[self._built_count:]:
                        if b.lines is None or b.width != w:
                            try:
                                b.lines = to_lines(b.renderable, w, self._console)
                            except Exception:
                                log_error(traceback.format_exc())
                                b.lines = [[("fg:ansired", "⚠ could not render this item")]]
                            b.width = w
                        self._flat.extend(b.lines)
                else:
                    flat: list[Line] = []
                    for b in blocks:
                        if b.lines is None or b.width != w:
                            try:
                                b.lines = to_lines(b.renderable, w, self._console)
                            except Exception:  # one bad item must not blank the whole chat
                                log_error(traceback.format_exc())
                                b.lines = [[("fg:ansired", "⚠ could not render this item")]]
                            b.width = w
                        flat.extend(b.lines)
                    self._flat = flat
                self._flat_w = w
                self._built_count = len(blocks)
                self._first_block = blocks[0] if blocks else None
                self._flat_key = key
            return self._flat

    def create_content(self, width: int, height: int) -> UIContent:
        w = self._render_width(width)
        with self.model.lock:
            flat = self._lines(w)
            live = self.live(w) if self.live else []
        n = len(flat)
        total = n + len(live)
        self._height, self._total = height, total
        max_top = max(0, total - height)
        self.top = max_top if self.follow else max(0, min(self.top, max_top))
        lo, hi = self.top, min(total, self.top + height)
        shown = flat[lo:min(hi, n)] + live[max(0, lo - n):max(0, hi - n)]
        return UIContent(get_line=lambda i: shown[i], line_count=len(shown), show_cursor=False)

    def scroll(self, delta: int) -> None:
        """Move the view by `delta` lines; negative goes up. Reaching the bottom turns follow back on."""
        max_top = max(0, self._total - self._height)
        self.top = max(0, min(max_top, (max_top if self.follow else self.top) + delta))
        self.follow = self.top >= max_top

    def to_top(self) -> None:
        self.top = 0
        self.follow = self._total <= self._height

    def to_bottom(self) -> None:
        self.follow = True

    def below(self) -> int:
        return 0 if self.follow else max(0, self._total - (self.top + self._height))

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)
            return None
        return NotImplemented   # MOUSE_MOVE in particular: any other answer repaints the screen on every mouse move


def _terminal_size() -> tuple[int, int]:
    app = get_app_or_none()
    try:
        size = app.output.get_size() if app else None
    except Exception:
        size = None
    return (size.columns, size.rows) if size else (80, 24)


class HeaderControl(UIControl):
    """The cat, the wordmark and the greeting. A click on the cat pets it; a click on the wordmark makes it shimmer."""

    def __init__(self, fe: "FullUI", clock: Callable[[], float] = time.monotonic):
        self.fe = fe
        self.clock = clock
        self.header: mascot.Header | None = None
        self._cache_key: tuple | None = None

    def is_focusable(self) -> bool:
        return False

    def _build(self, cols: int, rows: int) -> mascot.Header:
        fe = self.fe
        now = self.clock()
        agent = fe.agent
        yolo = bool(agent and agent.yolo)
        frame = fe.mascot.frame(now)
        shimmer = fe.mascot.shimmer_col(now)
        ui_cfg = fe.cfg.get("ui", {})
        key = (cols, rows, frame, shimmer, yolo, str(ui_cfg.get("header", "full")),
               bool(ui_cfg.get("mascot", True)), str(fe.cmd.workdir), model_short_name(fe.cfg))
        if key != self._cache_key or self.header is None:
            self.header = mascot.banner(fe.cfg, fe.cmd.workdir, yolo=yolo, width=cols, rows=rows,
                                        frame=frame, shimmer=shimmer, fullscreen=True)
            self._cache_key = key
        return self.header

    def rows(self) -> int:
        return len(self._build(*_terminal_size()).lines)

    def create_content(self, width: int, height: int) -> UIContent:
        self.header = hdr = self._build(width, _terminal_size()[1])
        return UIContent(get_line=lambda i: hdr.lines[i], line_count=len(hdr.lines), show_cursor=False)

    def mouse_handler(self, mouse_event: MouseEvent):
        hdr = self.header
        if hdr is None or mouse_event.event_type != MouseEventType.MOUSE_UP:
            return NotImplemented
        x, y = mouse_event.position.x, mouse_event.position.y

        def inside(rect) -> bool:
            return rect is not None and rect[0] <= x < rect[0] + rect[2] and rect[1] <= y < rect[1] + rect[3]

        if inside(hdr.cat):
            self.fe.pet()
            return None
        if inside(hdr.word):
            self.fe.shimmer()
            return None
        return NotImplemented


@dataclass
class Ask:
    kind: str                          # "menu" | "text"
    options: list[str]
    prompt: str = ""
    index: int = 0
    done: Future = field(default_factory=Future)


# ---------------------------------------------------------------- the frontend

class FullUI(RichUI):
    """RichUI that draws into the chat model instead of the terminal. Its Frontend methods are called on the worker thread."""

    def __init__(self, cfg: dict, cmd: commands.App, inline: RichUI, history):
        self.chat = ChatModel()
        # interactive=False keeps the parent from starting its inline Live displays and key watcher
        super().__init__(cfg, BlockSink(self.chat, self._added), interactive=False)
        self.cmd = cmd
        self.inline = inline
        self.history = history
        self.agent = cmd.agent
        self.stats_last = inline.stats_last
        self.busy = False
        self.mouse_on = True
        self.sidebar_on = bool(cfg.get("ui", {}).get("sidebar", False))
        self.ask: Ask | None = None
        self.mascot = mascot.Mascot(now=time.monotonic())
        self.application: Application | None = None
        self.chat_control = ChatControl(self.chat, self.live_lines)
        self.header_control = HeaderControl(self)
        self.input_buffer: Buffer | None = None
        self.input_window = self.ask_buffer = self.ask_menu_window = self.ask_text_window = None
        self._streaming = False
        self._stable_src, self._stable_lines, self._stable_w = "", [], 0
        self._tool_t0 = 0.0
        self._tool_title = ""
        self._flash = ""
        self._flash_until = 0.0
        self._pending: list[Ask] = []
        self._worker: threading.Thread | None = None
        self._ticker: asyncio.Task | None = None
        self._mascot_evt: asyncio.Event | None = None
        self._errors = 0
        self.turns_started = 0
        self._con = make_console()
        self.on_ready: Callable[["FullUI"], None] | None = None
        self._cancel_warm = getattr(inline, "_cancel_warm", None)

    # -- plumbing
    def running(self) -> bool:
        app = self.application
        return bool(app and app.is_running and app.loop is not None and app.future is not None and not app.future.done())

    def post(self, fn: Callable[[], None]) -> bool:
        """Run `fn` on the loop thread, inside the application's context (in_terminal() needs that context)."""
        app = self.application
        if not self.running() or app.context is None:
            return False
        try:
            app.loop.call_soon_threadsafe(fn, context=app.context.copy())
        except RuntimeError:
            return False
        return True

    def _added(self) -> None:
        if not self.busy and self.application is not None:
            self.application.invalidate()

    def wake(self) -> None:
        """Make the mascot task look at its state again (callable from any thread)."""
        self.post(lambda: self._mascot_evt is not None and self._mascot_evt.set())
        if self.application is not None:
            self.application.invalidate()

    def flash(self, text: str, seconds: float = 3.0) -> None:
        def show() -> None:
            self._flash, self._flash_until = text, time.monotonic() + seconds
            if self.application is not None:
                self.application.invalidate()
                self.application.loop.call_later(seconds + 0.05, self.application.invalidate)

        self.post(show)

    def out(self, text: str) -> None:
        self.console.print(Text(sanitize(text)))

    # -- streaming
    def begin_stream(self) -> None:
        self._buf = self._think = ""
        self._t0, self._first, self._deltas = time.monotonic(), 0.0, 0
        self._verb = persona.spinner_verb()
        self._preparing = None
        self._pp = None
        self.state = "working"
        self._stable_src, self._stable_lines, self._stable_w = "", [], 0
        self._streaming = True
        self.chat.dirty = True

    def prompt_progress(self, ev) -> None:
        self._pp = ev
        self.chat.dirty = True

    def text(self, d: str) -> None:
        self._pp = None
        self._buf += sanitize(d)
        self._tick()
        self.chat.dirty = True

    def reasoning(self, d: str) -> None:
        self._pp = None
        self._think += sanitize(d)
        self._first = self._first or time.monotonic()
        self.chat.dirty = True

    def tool_preparing(self, name: str, nbytes: int) -> None:
        self._pp = None
        self._preparing = (name, nbytes)
        self.chat.dirty = True

    def end_stream(self) -> None:
        if not self._streaming:
            return
        with self.chat.lock:
            if self._think.strip():
                self.console.print(Text(f"✻ Thought for {int(time.monotonic() - self._t0)}s", style="dim"))
            if self._buf.strip():
                self.console.print(Markdown(self._buf.strip(), code_theme=self.code_theme()))
            self._streaming = False
            self._buf = self._think = ""
            self._preparing = None
            self._pp = None
            self._stable_src, self._stable_lines = "", []
        self.state = "ready"

    def live_lines(self, width: int) -> list[Line]:
        """The unfinished part of the chat: the reply so far, or the reasoning, or a running tool, plus the footer."""
        out: list[Line] = []
        if self._streaming:
            buf = self._buf
            ct = self.code_theme()
            if buf.strip():
                stable, rest = split_stable(buf)
                if stable != self._stable_src or width != self._stable_w:
                    self._stable_lines = to_lines(Markdown(stable, code_theme=ct), width, self._con) if stable.strip() else []
                    self._stable_src, self._stable_w = stable, width
                out += self._stable_lines
                if rest.strip():
                    out += ([[]] if out else []) + to_lines(Markdown(rest, code_theme=ct), width, self._con)
            elif self._think.strip():
                out += to_lines(Text("\n".join(self._think.strip().splitlines()[-2:]), style="dim italic"), width, self._con)
            out += to_lines(self.footer(), width, self._con)
        elif self._tool_t0:
            tail = [ln[:max(1, width - 8)] for ln in self._tail]
            if tail:
                out += to_lines(Text("\n".join("      " + ln for ln in tail), style="dim", no_wrap=True), width, self._con)
            secs = int(time.monotonic() - self._tool_t0)
            out += to_lines(Text(f"✻ running {self._tool_title}… {secs}s · esc to interrupt", style=f"dim {self.accent}"), width, self._con)
        return out

    # -- tool cards, notices, printing
    def print_tool_header(self, name: str, args: dict, key: str | None = None) -> None:
        super().print_tool_header(name, {k: sanitize(v) if isinstance(v, str) else v for k, v in args.items()}, key)

    def print_tool_result(self, ok: bool, summary: str, diff: str | None = None, path: str = "") -> None:
        super().print_tool_result(ok, sanitize(summary), sanitize(diff) if diff else diff, path)

    def print_user(self, text: str) -> None:
        super().print_user(sanitize(text))

    def print_assistant(self, text: str) -> None:
        super().print_assistant(sanitize(text))

    def tool_start(self, call, plan) -> None:
        super().tool_start(call, plan)
        self._tool_title = TOOL_TITLES.get(call.name, call.name)
        self._tool_t0 = time.monotonic()
        self.chat.dirty = True

    def tool_output(self, line: str) -> None:
        self._tail.append(sanitize(line))
        self.chat.dirty = True

    def tool_done(self, call, result) -> None:
        self._tool_t0 = 0.0
        super().tool_done(call, result)
        if (result.content or "").startswith("BLOCKED"):
            self.react("shocked")

    def notice(self, level: str, text: str) -> None:
        super().notice(level, sanitize(text))
        if level == "celebrate":
            self.react("celebrate")
        elif level == "error" and text.lower().startswith("blocked"):
            self.react("shocked")

    def stats(self, s) -> None:
        super().stats(s)
        self.chat.dirty = True

    # -- the mascot
    def react(self, kind: str) -> None:
        if mascot.on(self.cfg):
            self.mascot.trigger(kind, time.monotonic())
            self.wake()

    def pet(self) -> None:
        if not mascot.on(self.cfg):
            return self.flash("the mascot is off (/mascot on)")
        self.mascot.trigger("click", time.monotonic())
        self.wake()

    def shimmer(self) -> None:
        self.mascot.shimmer(time.monotonic())
        self.wake()

    def hook(self, action: str) -> None:
        if action == "pet":
            self.pet()
        elif action == "sidebar":
            self.sidebar_on = bool(self.cfg.get("ui", {}).get("sidebar", False))
            self.wake()
        elif action == "theme":
            self.header_control._cache_key = None
            self.wake()
        else:   # "header", "mascot", "timestamps"
            self.header_control._cache_key = None
            self.wake()

    async def _mascot_loop(self) -> None:
        last = None
        evt = self._mascot_evt
        while True:
            evt.clear()
            now = time.monotonic()
            self.mascot.update(now, busy=self.busy, alert=self.ask is not None, yolo=bool(self.agent and self.agent.yolo))
            sig = (self.mascot.frame(now), self.mascot.shimmer_col(now))
            if sig != last:
                last = sig
                self.application.invalidate()
            nxt = self.mascot.next_change(now)
            timeout = None if nxt == float("inf") else max(0.02, nxt - now)
            try:
                await asyncio.wait_for(evt.wait(), timeout)
            except asyncio.TimeoutError:
                pass

    async def _tick_loop(self) -> None:
        """Runs only while a turn is busy: redraw 20 times a second when something changed, once a second otherwise."""
        quiet = 0
        while self.busy:
            await asyncio.sleep(0.05)
            if self.chat.dirty or quiet >= 19:
                self.chat.dirty = False
                quiet = 0
                self.application.invalidate()
            else:
                quiet += 1
        self.application.invalidate()

    # -- asks: the worker waits on a Future, the loop thread shows the panel
    def _ask(self, ask: Ask):
        if not self.running():
            return None
        self._pending.append(ask)
        self.post(lambda: self._open(ask))
        try:
            return ask.done.result()
        finally:
            with contextlib.suppress(ValueError):
                self._pending.remove(ask)

    def _open(self, ask: Ask) -> None:
        self.ask = ask
        self.chat_control.to_bottom()
        if ask.kind == "text":
            self.ask_buffer.reset()
            self.application.layout.current_window = self.ask_text_window
        else:
            self.application.layout.current_window = self.ask_menu_window
        self.wake()

    def _close(self, value) -> None:
        ask, self.ask = self.ask, None
        if self.input_window is not None and self.application is not None:
            self.application.layout.current_window = self.input_window
        if ask is not None and not ask.done.done():
            ask.done.set_result(value)
        self.wake()

    def ask_choice(self, options: list[str]) -> int | None:
        return self._ask(Ask("menu", options))

    def ask_text(self, prompt: str) -> str:
        return self._ask(Ask("text", [], prompt)) or ""

    def ask_sudo(self, prompt: str) -> str:
        return (self._ask(Ask("text", [], prompt)) or "").strip()

    def approve(self, call, plan):
        self.post(self.chat_control.to_bottom)
        return super().approve(call, plan)

    @contextlib.contextmanager
    def suspend_live(self):
        """Give the real terminal to a program that needs it (the sudo password prompt, $EDITOR); the window comes back after."""
        if not self.running():
            yield
            return
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        async def hold() -> None:
            try:
                async with in_terminal():
                    entered.set()
                    await asyncio.get_running_loop().run_in_executor(None, release.wait)
            finally:
                entered.set()
                finished.set()

        if not self.post(lambda: self.application.create_background_task(hold())):
            yield
            return
        entered.wait(5)
        try:
            yield
        finally:
            release.set()
            finished.wait(5)

    # -- turns
    def submit(self, line: str) -> bool:
        """Start the worker for one line. Called on the loop thread; refuses while a turn runs."""
        if self.busy:
            self.flash("busy · Esc interrupts")
            return False
        if self._cancel_warm:
            self._cancel_warm()
            self._cancel_warm = None
        self.busy = True
        self.turns_started += 1
        self._ticker = self.application.create_background_task(self._tick_loop())
        self.wake()
        self._worker = threading.Thread(target=self._handle, args=(line,), name="xaren-turn", daemon=True)
        self._worker.start()
        return True

    def _handle(self, line: str) -> None:
        cmd = self.cmd
        try:
            self.print_user(line)
            if commands.dispatch(cmd, line):
                pass
            elif line.startswith("!"):
                self.state, self._tool_title, self._tool_t0 = "running shell", "Bash", time.monotonic()
                self._tail.clear()
                commands.run_bang(cmd, line[1:].strip())
            else:
                self.agent.run_turn(commands.expand_mentions(line, cmd.workdir, self.cfg))
        except Exception as e:
            log_error(traceback.format_exc())
            self.end_stream()
            self.notice("error", f"internal error: {e}")
        finally:
            self._tool_t0 = 0.0
            self.end_stream()
            self.state = "ready"
            self.console.print(Text(""))
            self.busy = False
            self.post(self._after_turn)

    def _after_turn(self) -> None:
        self.wake()
        if self.cmd.should_exit:
            self.leave()

    def leave(self) -> None:
        app = self.application
        if self.running() and not app.future.done():
            app.exit()

    def interrupt(self) -> None:
        if self.agent:
            self.agent.interrupt()

    # -- loop-thread helpers
    def _on_loop_error(self, loop, context) -> None:
        exc = context.get("exception")
        text = "".join(traceback.format_exception(exc)) if exc else str(context.get("message", ""))
        log_error(text)
        self._errors += 1
        self.console.print(Text(f"✖ internal error: {exc or context.get('message', '')} (see {log_path()})", style="bold red"))
        if self._errors >= 3 and self.running():
            self.application.exit(exception=exc or RuntimeError(str(context.get("message", "loop error"))))

    def _graceful(self) -> None:
        self.cmd.should_exit = True
        if self.busy:
            self.interrupt()
        self.leave()

    def on_start(self) -> None:
        """Runs on the loop thread right after the application starts."""
        app = self.application
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._on_loop_error)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, self._graceful)
        self._mascot_evt = asyncio.Event()
        app.create_background_task(self._mascot_loop())
        app.key_processor.before_key_press += lambda sender: (self.mascot.poke(time.monotonic()), self._mascot_evt.set())
        if self.agent and self.agent.yolo:
            from .safety import YOLO_NOTICE
            self.notice("warn", YOLO_NOTICE)
        if self.cmd.workdir == Path.home():
            self.notice("warn", "you launched from your home folder; file tools can reach everything here")
        if self.on_ready:
            self.on_ready(self)

    # -- pieces of the screen
    def rule_fragments(self) -> list[tuple[str, str]]:
        cols = _terminal_size()[0] - (31 if self.sidebar_on and _terminal_size()[0] >= 100 else 0)
        rule_style = theme_of(self.cfg)["rule"]
        if self._flash and time.monotonic() < self._flash_until:
            msg, style = f" {self._flash} ", f"bold fg:{self.accent}"
        elif self.chat_control.below():
            msg, style = f" ↓ {self.chat_control.below()} more lines · Ctrl+End to follow ", "fg:ansiyellow"
        else:
            return [(rule_style, "─" * max(1, cols))]
        return [(rule_style, "── "), (style, msg.strip()), (rule_style, " " + "─" * max(0, cols - 4 - len(msg.strip())))]

    def sidebar_fragments(self) -> list[tuple[str, str]]:
        th = theme_of(self.cfg)
        dim = th["dim"]
        bold = f"bold fg:{self.accent}"
        a = self.agent
        s = self.stats_last
        size = self.cfg["server"]["ctx_size"]
        used = (a.ctx.last_est if a else 0) or s.ctx_used
        pct = min(100, round(100 * used / size)) if size else 0
        filled = min(10, round(10 * used / size)) if size else 0
        bar = "▰" * filled + "▱" * (10 - filled)
        st = a.session_stats if a else None
        out: list[tuple[str, str]] = [
            (bold, " XarenCode\n"),
            (dim, f" model   {model_short_name(self.cfg)}\n"),
            (dim, f" vibe    {self.cfg['persona']['vibe']}\n"),
            (dim, f" theme   {self.cfg.get('ui', {}).get('theme', 'dark')}\n\n"),
            (bold, " Context\n"),
            ("", f" {bar} {pct}%\n"),
            (dim, f" {used / 1000:.1f}k / {size / 1000:.1f}k tok\n\n"),
        ]
        if st:
            out += [
                (bold, " Session\n"),
                (dim, f" turns   {st.turns}\n"),
                (dim, f" tools   {st.tool_calls}\n"),
                (dim, f" edits   {st.edits}\n"),
                (dim, f" tokens  {st.tokens}\n\n"),
            ]
        auto = "on" if a and a.auto_edits else "off"
        yolo = "ON" if a and a.yolo else "off"
        out += [
            (bold, " Safety\n"),
            (dim, f" auto-edits  {auto}\n"),
            ("bold fg:ansired" if yolo == "ON" else dim, f" yolo        {yolo}\n"),
            (dim, f" allowed     {len(a.approvals) if a else 0}\n\n"),
            (bold, " Keys\n"),
            (dim, " Esc       interrupt\n"),
            (dim, " Shift+Tab auto-edits\n"),
            (dim, " Ctrl+B    sidebar\n"),
            (dim, " F2 / F3   select / pet\n"),
        ]
        return out

    def ask_fragments(self) -> list[tuple[str, str]]:
        ask = self.ask
        if ask is None or ask.kind != "menu":
            return []
        out: list[tuple[str, str]] = [(DIM, "  ↑↓ choose · Enter confirm · Esc cancel\n")]
        for i, option in enumerate(ask.options):
            cur = i == ask.index
            out.append((f"bold fg:{self.accent}" if cur else "", f"  {'❯' if cur else ' '} {i + 1}. {option}" + ("\n" if i < len(ask.options) - 1 else "")))
        return out

    # -- building the application
    def build(self, input=None, output=None) -> Application:
        cfg_accent = self.accent
        buf = self.input_buffer = Buffer(history=self.history, completer=make_completer(lambda: self.cmd.workdir),
                                         complete_while_typing=True, multiline=True, name="input")
        self.ask_buffer = Buffer(multiline=False, name="ask")

        def prefix(line_number: int, wrap_count: int):
            return [(f"bold fg:{cfg_accent}", "❯ ")] if line_number == 0 and wrap_count == 0 else [("", "  ")]

        placeholder = ConditionalProcessor(AfterInput("Ask anything, or /help", style=DIM), Condition(lambda: not buf.text))
        self.input_window = Window(BufferControl(buf, input_processors=[placeholder]), height=Dimension(min=1, max=8),
                                   dont_extend_height=True, wrap_lines=True, get_line_prefix=prefix)

        # the ask panel: a menu, or a one-line text field, replaces the input bar
        mkb = KeyBindings()

        def move(step: int):
            def handler(event) -> None:
                if self.ask is not None:
                    self.ask.index = (self.ask.index + step) % len(self.ask.options)
            return handler

        for key, step in (("up", -1), ("k", -1), ("down", 1), ("j", 1)):
            mkb.add(key)(move(step))
        for n in range(1, 10):
            mkb.add(str(n))(lambda event, n=n: self._close(n - 1) if self.ask and n <= len(self.ask.options) else None)
        mkb.add("enter")(lambda event: self._close(self.ask.index) if self.ask else None)
        mkb.add("escape", eager=True)(lambda event: self._close(None))
        self.ask_menu_window = Window(FormattedTextControl(self.ask_fragments, focusable=True, key_bindings=mkb, show_cursor=False),
                                      height=lambda: Dimension.exact(len(self.ask.options) + 1 if self.ask else 1))

        tkb = KeyBindings()
        tkb.add("enter")(lambda event: self._close(self.ask_buffer.text) if self.ask else None)
        tkb.add("escape", eager=True)(lambda event: self._close(""))
        self.ask_text_window = Window(BufferControl(self.ask_buffer, key_bindings=tkb), height=1,
                                      get_line_prefix=lambda n, w: [(f"bold fg:{cfg_accent}", self.ask.prompt.lstrip() if self.ask else "")])

        th = theme_of(self.cfg)
        is_menu = Condition(lambda: self.ask is not None and self.ask.kind == "menu")
        is_text = Condition(lambda: self.ask is not None and self.ask.kind == "text")
        no_ask = Condition(lambda: self.ask is None)
        show_sidebar = Condition(lambda: self.sidebar_on and _terminal_size()[0] >= 100)
        header = ConditionalContainer(Window(self.header_control, height=lambda: Dimension.exact(max(1, self.header_control.rows())),
                                             dont_extend_height=True), filter=Condition(lambda: self.header_control.rows() > 0))
        chat = Window(self.chat_control, wrap_lines=False, always_hide_cursor=True)
        rule = Window(FormattedTextControl(self.rule_fragments), height=1)
        status_style = f"fg:{th['status_fg']} bg:{th['status_bg']}" if th["status_bg"] else "fg:#8a8a8a"
        status = Window(FormattedTextControl(self.status_fragments), height=1, style=status_style)
        main_col = HSplit([header, chat, rule,
                           ConditionalContainer(self.input_window, no_ask),
                           ConditionalContainer(self.ask_menu_window, is_menu),
                           ConditionalContainer(self.ask_text_window, is_text),
                           status])
        sidebar_sep = Window(width=1, char="│", style=th["sidebar_border"])
        sidebar_win = Window(FormattedTextControl(self.sidebar_fragments), width=30)
        body = VSplit([main_col, ConditionalContainer(VSplit([sidebar_sep, sidebar_win]), show_sidebar)])
        root = FloatContainer(body, floats=[Float(xcursor=True, ycursor=True, content=CompletionsMenu(
            max_height=8, scroll_offset=1, extra_filter=has_focus(buf)))])

        style = Style.from_dict({
            "completion-menu": f"bg:{th['menu_bg']} fg:{th['menu_fg']}",
            "completion-menu.completion.current": f"bg:{cfg_accent} fg:#ffffff",
            "completion-menu.meta.completion": f"bg:{th['meta_bg']} fg:{th['meta_fg']}",
            "completion-menu.meta.completion.current": f"bg:{cfg_accent} fg:#ffffff",
            "scrollbar.background": f"bg:{th['menu_bg']}",
            "scrollbar.button": "bg:#6b6b6b",
        })
        depth = ColorDepth.TRUE_COLOR if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit") and not os.environ.get("NO_COLOR") else None
        self.application = app = Application(
            layout=Layout(root, focused_element=self.input_window), key_bindings=self._keys(buf), full_screen=True,
            mouse_support=Condition(lambda: self.mouse_on), enable_page_navigation_bindings=False, style=style, color_depth=depth,
            terminal_size_polling_interval=None, min_redraw_interval=0.04, input=input, output=output)
        app.ttimeoutlen = 0.05
        app.timeoutlen = 0.3
        return app

    def _keys(self, buf: Buffer) -> KeyBindings:
        kb = KeyBindings()
        no_ask = Condition(lambda: self.ask is None)
        busy_no_ask = Condition(lambda: self.busy and self.ask is None)
        empty = Condition(lambda: not buf.text)

        @kb.add("enter", filter=no_ask)
        def _enter(event) -> None:
            if buf.complete_state:
                buf.cancel_completion()
                return
            text = buf.text.strip()
            if self.busy:
                return self.flash("busy · Esc interrupts")
            if text:
                buf.reset(append_to_history=True)
                self.submit(text)

        @kb.add("escape", "enter", filter=no_ask)
        def _newline(event) -> None:
            buf.insert_text("\n")

        kb.add("escape", filter=busy_no_ask & empty, eager=True)(lambda event: self.interrupt())
        kb.add("escape", filter=busy_no_ask & ~empty)(lambda event: self.interrupt())

        @kb.add("c-c")
        @kb.add("<sigint>")
        def _ctrl_c(event) -> None:
            if self.ask is not None:
                self._close("" if self.ask.kind == "text" else None)
            if self.busy:
                self.interrupt()
            elif buf.text:
                buf.reset()
            elif not self.ask:
                self.flash("Ctrl+D quits")

        @kb.add("c-d", filter=no_ask & ~Condition(lambda: self.busy) & empty)
        def _quit(event) -> None:
            self.cmd.should_exit = True
            event.app.exit()

        @kb.add("s-tab")
        def _auto(event) -> None:
            if self.agent:
                self.agent.auto_edits = not self.agent.auto_edits
            event.app.invalidate()

        def page(sign: int):
            def handler(event) -> None:
                self.chat_control.scroll(sign * max(1, self.chat_control._height - 2))
            return handler

        kb.add("pageup")(page(-1))
        kb.add("pagedown")(page(1))
        kb.add("c-home")(lambda event: self.chat_control.to_top())
        kb.add("c-end")(lambda event: self.chat_control.to_bottom())

        @kb.add("f2")
        def _mouse(event) -> None:
            self.mouse_on = not self.mouse_on
            self.flash("mouse capture on · F2 lets you select text" if self.mouse_on
                       else "mouse capture off · drag to select · F2 turns it back on")

        kb.add("f3")(lambda event: self.pet())

        @kb.add("c-b")
        def _sidebar(event) -> None:
            self.sidebar_on = not self.sidebar_on
            self.flash("sidebar on" if self.sidebar_on else "sidebar off")

        return kb


# ---------------------------------------------------------------- session replay and the leave summary

def replay(fe: FullUI, entries) -> None:
    """Show stored messages in the chat the way render_transcript does, without redaction and without a share header."""
    calls: dict[str, dict] = {}
    first = True
    for msg, info in entries:
        role, content = msg.get("role"), (msg.get("content") or "").strip()
        if role == "user":
            if not first:
                fe.console.print(Text(""))
            first = False
            fe.print_user(content.split("\n\n[attached ", 1)[0])
        elif role == "assistant":
            if content:
                fe.print_assistant(content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls[tc.get("id", "")] = args
                fe.print_tool_header(fn.get("name", "tool"), args)
        elif role == "tool":
            info = info or {}
            ok = not content.startswith(("ERROR", "BLOCKED", "User declined"))
            summary = info.get("summary") or (content.splitlines()[0] if content else "done")
            fe.print_tool_result(ok, summary, info.get("diff") or None, calls.get(msg.get("tool_call_id", ""), {}).get("path", ""))


def leave_summary(inline: RichUI, cmd: commands.App) -> None:
    stats = cmd.agent.session_stats
    n = lambda k, word: f"{k} {word}{'' if k == 1 else 's'}"
    inline.console.print(Text(f"✦ full-window: {n(stats.turns, 'turn')} · {n(stats.tool_calls, 'tool call')} · "
                              f"{n(stats.edits, 'edit')} · session {cmd.session.meta.get('id', '')}", style="dim"))
    reply = clipboard.last_reply(cmd.agent.history)
    if reply:
        lines = reply.splitlines()
        shown = "\n".join(lines[:30])
        inline.print_assistant(shown)
        if len(lines) > 30:
            inline.console.print(Text("… /copy copies it all", style="dim"))


class _Guard:
    """Restores the terminal at interpreter exit when the application did not end normally."""

    def __init__(self) -> None:
        self.saved = None
        self.fd = None
        try:
            import termios
            if sys.stdin.isatty():
                self.fd = sys.stdin.fileno()
                self.saved = termios.tcgetattr(self.fd)
        except (ImportError, OSError, ValueError):
            self.fd = self.saved = None

    def __call__(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            os.write(sys.__stdout__.fileno(), RESET_SEQUENCES.encode())
        if self.saved is not None:
            import termios
            with contextlib.suppress(OSError, termios.error):
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def run_fullwindow(cfg: dict, cmd: commands.App, inline: RichUI, history, *, input=None, output=None, on_ready=None) -> int:
    """Run the full-window view until the user leaves it. Returns 0, or 1 when the view stopped after an internal error.

    The caller checks that a real terminal is available; `input` and `output` let tests drive it with pipes."""
    fe = FullUI(cfg, cmd, inline, history)
    fe.on_ready = on_ready
    entries = cmd.session.entries[-MAX_REPLAY:]
    replay(fe, entries)
    agent = cmd.agent
    saved = (agent.frontend, cmd.out, cmd.editor, cmd.ui_hook)
    agent.frontend, cmd.out, cmd.ui_hook = fe, fe.out, fe.hook
    if saved[2] is not None:
        original_editor = saved[2]

        def edit(path: Path) -> None:
            with fe.suspend_live():
                original_editor(path)

        cmd.editor = edit
    app = fe.build(input, output)
    real_terminal = output is None
    guard = _Guard() if real_terminal else None
    if guard:
        atexit.register(guard)
    old_stderr = sys.stderr
    err = None
    if real_terminal:
        with contextlib.suppress(OSError):
            log_path().parent.mkdir(parents=True, exist_ok=True)
            sys.stderr = err = open(log_path(), "a", encoding="utf-8")
    failure: Exception | None = None
    try:
        app.run(pre_run=fe.on_start, set_exception_handler=False)
    except EOFError:
        cmd.should_exit = True
    except Exception as e:
        failure = e
        log_error(traceback.format_exc())
    finally:
        for ask in list(fe._pending):
            if not ask.done.done():
                ask.done.set_result(None)
        if fe.busy:
            agent.interrupt()
        if fe._worker is not None:
            fe._worker.join(3)
        agent.frontend, cmd.out, cmd.editor, cmd.ui_hook = saved
        inline.stats_last = fe.stats_last
        sys.stderr = old_stderr
        if err is not None:
            err.close()
        if guard:
            atexit.unregister(guard)
        leave_summary(inline, cmd)
        if failure is not None:
            inline.notice("error", f"XarenCode stopped after an internal error ({failure}); details are in {log_path()}")
    return 1 if failure is not None else 0
