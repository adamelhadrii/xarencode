"""Slash commands, '!' shell input and '@file' mentions. Stdlib only; output goes through App.out."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import clipboard, persona, sessions
from .agent import Agent
from .config import config_path, model_short_name, resolve_model
from .context import compact_history
from .safety import ALLOW, BLOCK, YOLO_NOTICE, classify_command, resolve_path
from .server import LlamaServer, ServerStartError, doctor
from .tools import ToolContext, devserver, vm
from .tools.shell import run_command


@dataclass
class App:
    cfg: dict
    agent: Agent
    server: LlamaServer
    session: sessions.Session
    out: Callable[[str], None]
    workdir: Path
    config_file: Path = field(default_factory=config_path)
    editor: Callable[[Path], None] | None = None  # runs $EDITOR with the terminal released
    should_exit: bool = False
    ui_hook: Callable[[str], None] | None = None  # called with "pet", "header" or "mascot"

    def notice(self, level: str, text: str) -> None:
        self.agent.frontend.notice(level, text)

    def bind_session(self, s: sessions.Session) -> None:
        self.session = s
        self.agent.sink = s.append

    def tctx(self) -> ToolContext:
        return self.agent.tool_context()


COMMANDS: dict[str, tuple[Callable[[App, str], None], str, str]] = {}


def command(name: str, usage: str, desc: str):
    def deco(fn):
        COMMANDS[name] = (fn, usage, desc)
        return fn
    return deco


def command_list() -> list[tuple[str, str]]:
    return [(f"/{n}", d) for n, (_, _, d) in COMMANDS.items()]


def dispatch(app: App, line: str) -> bool:
    """Handle '/cmd args'. Returns True if the line was a command (even an unknown one)."""
    if not line.startswith("/"):
        return False
    name, _, arg = line[1:].partition(" ")
    entry = COMMANDS.get(name.lower())
    if entry is None:
        app.notice("warn", f"unknown command /{name}. Try /help")
        return True
    try:
        entry[0](app, arg.strip())
    except (ServerStartError, sessions.ExportError, RuntimeError, ValueError, OSError) as e:
        app.notice("error", str(e))
    return True


# ---------------------------------------------------------------- conversation

@command("help", "/help", "show commands and tips")
def _help(app, arg):
    rows = [f"  {usage:<26} {desc}" for _, (_, usage, desc) in COMMANDS.items()]
    app.out("\n".join(["Commands:", *rows, "",
                       "Tips: ! runs a shell command, @path attaches a file, Alt+Enter adds a new line,",
                       "Shift+Tab toggles auto-accept for edits, Esc interrupts a reply,",
                       "PgUp/PgDn scroll, F2 mouse select, F3 pets the cat."]))


@command("clear", "/clear", "start a fresh conversation")
def _clear(app, arg):
    app.agent.history.clear()
    app.bind_session(sessions.Session.new(app.cfg, app.workdir))
    if app.agent.prefix_cache is not None:
        app.agent.prefix_cache.mark_dirty()
    app.notice("info", "conversation cleared")


@command("compact", "/compact [focus]", "summarize the conversation to free context")
def _compact(app, arg):
    if not app.agent.history:
        return app.notice("info", "nothing to compact yet")
    summary = compact_history(app.agent.client, app.agent.history, app.agent.sampling, arg)
    if not summary:
        return app.notice("warn", "the model returned an empty summary; history kept")
    app.agent.history[:] = [{"role": "user", "content": f"Summary of the earlier conversation:\n{summary}"}]
    app.session.rewrite(app.agent.history)
    if app.agent.prefix_cache is not None:
        app.agent.prefix_cache.mark_dirty()
    app.notice("info", f"compacted to {len(summary.split())} words")


@command("context", "/context", "show the token breakdown")
def _context(app, arg):
    a = app.agent
    cm = a.ctx
    tools = a.registry.schemas()
    sysn = -(-len(a.system) // 3)
    toolsn = -(-len(json.dumps(tools)) // 3)
    by_role: dict[str, int] = {}
    for m in a.history:
        by_role[m["role"]] = by_role.get(m["role"], 0) + cm.raw_tokens([m])
    total = cm.estimate(a.system, tools, a.history)
    rows = [f"  system prompt  ~{sysn}", f"  tool schemas   ~{toolsn}"] + [f"  {r:<14} ~{n}" for r, n in by_role.items()]
    app.out("\n".join(["Context (estimates):", *rows, f"  total          ~{total} of {cm.ctx_size} (budget {cm.budget}, calibration x{cm.ratio:.2f})"]))


@command("sessions", "/sessions", "list saved sessions")
def _sessions(app, arg):
    items = sessions.list_sessions()[:15]
    if not items:
        return app.out("no saved sessions")
    app.out("\n".join(f"  {i}. {s.created[:16]}  {s.title or '(empty)'}  [{Path(s.workdir).name}]" for i, s in enumerate(items, 1)))


@command("resume", "/resume [N]", "continue a saved session")
def _resume(app, arg):
    info = sessions.find_session(arg or None)
    if info is None:
        return app.notice("warn", "no such session")
    s = sessions.Session.load(info.path)
    app.agent.history[:] = s.messages()
    app.bind_session(s)
    app.notice("info", f"resumed: {info.title or info.id} ({len(s.entries)} messages)")


@command("exit", "/exit", "quit")
def _exit(app, arg):
    app.should_exit = True


# ---------------------------------------------------------------- model and settings

@command("model", "/model", "show the model (change it in the config)")
def _model(app, arg):
    m = resolve_model(app.cfg)
    size = f"{m.stat().st_size / 1024 ** 3:.2f} GiB" if m.is_file() else "not downloaded"
    app.out(f"  model   {model_short_name(app.cfg)}  ({m.name}, {size})\n  ctx     {app.cfg['server']['ctx_size']}\n"
            f"  swap    edit model.gguf in {app.config_file}, then restart")


@command("server", "/server [status|start|stop|restart|log]", "manage llama-server")
def _server(app, arg):
    srv = app.server
    act = arg or "status"
    if act == "status":
        app.out(f"llama-server: {srv.status()} (port {app.cfg['server']['port']})")
    elif act == "start":
        app.out("started" if srv.ensure(lambda s: None) else "already running")
    elif act == "stop":
        app.out(srv.stop())
    elif act == "restart":
        srv.stop()
        srv.ensure(lambda s: None)
        if app.agent.prefix_cache is not None:
            app.agent.prefix_cache.mark_dirty()
        app.out("restarted")
    elif act == "log":
        app.out(srv.log_tail(30))
    else:
        app.notice("warn", "usage: /server [status|start|stop|restart|log]")


@command("config", "/config [edit]", "show the config path or edit it")
def _config(app, arg):
    if arg == "edit":
        if app.editor is None:
            return app.notice("warn", "no editor available")
        return app.editor(app.config_file)
    app.out(f"config: {app.config_file}\n(use /config edit; changes apply after a restart)")


@command("doctor", "/doctor", "check the setup")
def _doctor(app, arg):
    mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}
    app.out("\n".join(f"  [{mark[s]}] {t}" for s, t in doctor(app.cfg)))


@command("cd", "/cd DIR", "change the working directory")
def _cd(app, arg):
    if not arg:
        return app.out(f"working directory: {app.workdir}")
    target = Path(os.path.realpath(os.path.expanduser(arg) if os.path.isabs(os.path.expanduser(arg)) else app.workdir / arg))
    if not target.is_dir():
        return app.notice("error", f"{target} is not a folder")
    if target == Path("/"):
        return app.notice("error", "refusing to work in /")
    app.workdir = target
    app.agent.workdir = target
    app.agent.system = persona.system_prompt(app.cfg, target)
    if target == Path.home():
        app.notice("warn", "working in your home folder; consider a project folder instead")
    app.notice("info", f"now working in {target}")


@command("permissions", "/permissions [reset]", "show or reset remembered approvals")
def _permissions(app, arg):
    a = app.agent
    if arg == "reset":
        a.approvals.clear()
        a.auto_edits = False
        return app.notice("info", "approvals reset")
    keys = ", ".join(sorted(a.approvals)) or "none"
    app.out(f"  always allowed this session: {keys}\n  auto-accept edits: {'on' if a.auto_edits else 'off'}"
            f"\n  yolo: {'on' if a.yolo else 'off'}")


@command("yolo", "/yolo [on|off]", "auto-approve every ask for this session (hard blocks and sudo still apply)")
def _yolo(app, arg):
    a = app.agent
    word = arg.strip().lower()
    if word not in ("", "on", "off"):
        return app.notice("warn", "usage: /yolo on|off")
    a.yolo = (word == "on") if word else a.yolo
    if word:
        app.notice("warn" if a.yolo else "info", YOLO_NOTICE if a.yolo else "yolo is off: commands that need approval will ask again")
    else:
        app.out(f"  yolo: {'on' if a.yolo else 'off'} (this session only; /yolo on|off)")


# ---------------------------------------------------------------- dev servers and VMs

@command("dev", "/dev [start|stop NAME]", "list, start or stop dev servers")
def _dev(app, arg):
    verb, _, name = arg.partition(" ")
    name = name.strip()
    if verb == "start" and name:
        tool = app.agent.registry.get("dev_server")
        args = {"action": "start", "name": name}
        plan = tool.plan(args, app.tctx())
        if plan.verdict == BLOCK:
            return app.notice("error", plan.reason)
        res = tool.run(args, app.tctx(), plan)
        return app.notice("celebrate" if res.ok else "error", res.content)
    if verb == "stop" and name:
        return app.out(devserver.stop_server(name))
    run = devserver.running()
    lines = [f"  running: {n}  {e.get('url', '')}" for n, e in run.items()] or ["  nothing running"]
    lines += [f"  preset:  {n}  {p.get('url', '')}" for n, p in app.cfg["dev"].items() if n not in run]
    app.out("\n".join(lines))


@command("vm", "/vm [start|stop NAME]", "list, start or stop virtual machines")
def _vm(app, arg):
    verb, _, name = arg.partition(" ")
    name = name.strip()
    machines = app.cfg["vm"]["machines"]
    if not machines:
        return app.out(vm.NONE_MSG)
    backend = vm.get_backend(app.cfg)
    if verb == "start" and name in machines:
        return app.out(backend.start(name, vm.spec_for(app.cfg, name)))
    if verb == "stop" and name in machines:
        return app.out(backend.stop(name, False))
    running = {s["name"] for s in backend.status()}
    app.out("\n".join(f"  {n}  {'running' if n in running else 'stopped'}" for n in machines))


# ---------------------------------------------------------------- social

@command("share", "/share [md|svg|html]", "save this session as a shareable file")
def _share(app, arg):
    fmt = arg or "md"
    path = sessions.export(app.session, fmt, Path(app.cfg["ui"]["share_dir"]), app.cfg)
    app.notice("celebrate", f"saved {path} (home folder and secrets redacted)")


@command("vibe", "/vibe [name]", "change Xaren's personality")
def _vibe(app, arg):
    if not arg:
        return app.out(f"  vibe: {app.cfg['persona']['vibe']}   options: {', '.join(persona.vibe_names())}")
    if arg not in persona.VIBES:
        return app.notice("warn", f"unknown vibe; options: {', '.join(persona.vibe_names())}")
    app.cfg["persona"]["vibe"] = arg
    app.agent.system = persona.system_prompt(app.cfg, app.workdir)
    app.notice("info", f"vibe set to {arg} for this session (edit the config to keep it)")


# ---------------------------------------------------------------- view

def _ui_cfg(app: App) -> dict:
    return app.cfg.setdefault("ui", {})


def _hook(app: App, action: str) -> None:
    if app.ui_hook:
        app.ui_hook(action)


@command("copy", "/copy [code|all]", "copy the last reply, its last code block, or the whole session")
def _copy(app, arg):
    word = arg.strip().lower()
    if word not in ("", "code", "all"):
        return app.notice("warn", "usage: /copy [code|all]")
    if word == "all":
        text = sessions.export_md(app.session.meta, app.session.entries, app.cfg["persona"]["name"]) if app.session.entries else ""
    elif word == "code":
        text = clipboard.last_code_block(app.agent.history)
    else:
        text = clipboard.last_reply(app.agent.history)
    if not text:
        return app.notice("info", "nothing to copy yet")
    where = clipboard.copy_text(text)
    app.notice("info", f"copied {len(text)} chars to {where}")


@command("pet", "/pet", "pet the cat")
def _pet(app, arg):
    if not _ui_cfg(app).get("mascot", True):
        return app.notice("info", "the mascot is off (/mascot on)")
    _hook(app, "pet")


@command("mascot", "/mascot [on|off]", "show or hide the cat for this session")
def _mascot(app, arg):
    word = arg.strip().lower()
    if word not in ("", "on", "off"):
        return app.notice("warn", "usage: /mascot on|off")
    if not word:
        return app.out(f"  mascot: {'on' if _ui_cfg(app).get('mascot', True) else 'off'} (this session only; /mascot on|off)")
    _ui_cfg(app)["mascot"] = word == "on"
    _hook(app, "mascot")


@command("header", "/header [full|small|off]", "change the size of the header for this session")
def _header(app, arg):
    word = arg.strip().lower()
    if word not in ("", "full", "small", "off"):
        return app.notice("warn", "usage: /header full|small|off")
    if not word:
        return app.out(f"  header: {_ui_cfg(app).get('header', 'full')} (this session only; /header full|small|off)")
    _ui_cfg(app)["header"] = word
    _hook(app, "header")


@command("theme", "/theme [dark|light|contrast|mono]", "change the UI colour theme for this session")
def _theme(app, arg):
    word = arg.strip().lower()
    if word not in ("", "dark", "light", "contrast", "mono"):
        return app.notice("warn", "usage: /theme dark|light|contrast|mono")
    if not word:
        return app.out(f"  theme: {_ui_cfg(app).get('theme', 'dark')} (this session only; /theme dark|light|contrast|mono)")
    _ui_cfg(app)["theme"] = word
    _hook(app, "theme")


@command("timestamps", "/timestamps [on|off]", "show or hide timestamps on chat messages")
def _timestamps(app, arg):
    word = arg.strip().lower()
    if word not in ("", "on", "off"):
        return app.notice("warn", "usage: /timestamps on|off")
    if not word:
        return app.out(f"  timestamps: {'on' if _ui_cfg(app).get('timestamps', False) else 'off'} (this session only; /timestamps on|off)")
    _ui_cfg(app)["timestamps"] = word == "on"
    _hook(app, "timestamps")


@command("sidebar", "/sidebar [on|off]", "show or hide the info sidebar (Ctrl+B)")
def _sidebar(app, arg):
    word = arg.strip().lower()
    if word not in ("", "on", "off"):
        return app.notice("warn", "usage: /sidebar on|off")
    if not word:
        return app.out(f"  sidebar: {'on' if _ui_cfg(app).get('sidebar', False) else 'off'} (Ctrl+B toggles)")
    _ui_cfg(app)["sidebar"] = word == "on"
    _hook(app, "sidebar")


# ---------------------------------------------------------------- '!' and '@'

def run_bang(app: App, cmd: str) -> None:
    """User-typed shell command: BLOCK rules still apply; other verdicts run without a prompt."""
    d = classify_command(cmd, app.workdir, app.cfg)
    if d.verdict == BLOCK:
        return app.notice("error", f"blocked: {d.reason}")
    fe = app.agent.frontend
    elevated = d.verdict.name == "ELEVATED"
    if elevated:
        with app.agent.tool_context().terminal():
            res = run_command(cmd, app.workdir, timeout=600, elevated=True)
        app.out(res.output)
    else:
        app.agent.cancelled.clear()  # a leftover Esc/Ctrl+C from an earlier turn must not kill this command
        res = run_command(cmd, app.workdir, timeout=600, on_output=fe.tool_output, cancelled=app.agent.cancelled)
    note = f"I ran `{cmd}` (exit {res.exit_code}):\n```\n{res.output[-1500:]}\n```"
    app.agent.add_user_note(note)
    app.notice("info" if res.exit_code == 0 else "warn", f"exit {res.exit_code} · {res.seconds:.1f}s")


MENTION = re.compile(r"(?<!\S)@([\w./~-]+)")
MAX_ATTACH = 3000


def expand_mentions(text: str, workdir: Path, cfg: dict) -> str:
    """Append a capped excerpt for each @path that names a readable text file."""
    extras = []
    for m in MENTION.finditer(text):
        path, verdict, _ = resolve_path(m.group(1), workdir, write=False, cfg=cfg)
        if verdict != ALLOW or not path.is_file():
            continue
        try:
            data = path.read_bytes()[:MAX_ATTACH * 2]
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        body = data.decode("utf-8", "replace")[:MAX_ATTACH]
        extras.append(f"[attached {m.group(1)}]\n```\n{body}\n```")
    return text + ("\n\n" + "\n\n".join(extras) if extras else "")


def at_completions(prefix: str, workdir: Path, limit: int = 30) -> list[str]:
    """File names for '@' completion, relative to the working directory."""
    skip = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    base, _, stem = prefix.rpartition("/")
    folder = (workdir / base) if base else workdir
    out = []
    try:
        for e in sorted(os.scandir(folder), key=lambda e: e.name):
            if e.name in skip or not e.name.lower().startswith(stem.lower()) or (e.name.startswith(".") and not stem.startswith(".")):
                continue
            out.append((base + "/" if base else "") + e.name + ("/" if e.is_dir() else ""))
            if len(out) >= limit:
                break
    except OSError:
        pass
    return out
