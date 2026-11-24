"""dev_server tool: start/stop/inspect long-running dev servers with a small JSON registry."""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

from ..config import state_dir
from ..safety import ALLOW, ASK, BLOCK, ELEVATED, classify_command
from ..server import proc_start_time
from . import Plan, Tool, ToolContext, ToolResult, error, schema

_procs: dict[int, subprocess.Popen] = {}  # keep Popen objects so exited children get reaped


def dev_dir() -> Path:
    d = state_dir() / "dev"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _registry_path() -> Path:
    return dev_dir() / "registry.json"


def load_registry() -> dict:
    try:
        return json.loads(_registry_path().read_text())
    except (OSError, ValueError):
        return {}


def _save(reg: dict) -> None:
    _registry_path().write_text(json.dumps(reg, indent=1))


def pid_alive(pid: int) -> bool:
    proc = _procs.get(pid)
    if proc is not None:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def is_ours(entry: dict) -> bool:
    """False when the pid now belongs to some unrelated process (a stale registry entry after a reboot).

    Entries record the process start time, which survives a process renaming itself (npm does). Older entries
    without one fall back to matching the command line.
    """
    pid, cmd = entry["pid"], entry.get("cmd", "")
    if pid in _procs:
        return True
    if entry.get("start") is not None:
        try:
            same_group = os.getpgid(pid) == entry.get("pgid", pid)
        except OSError:
            return False
        return same_group and proc_start_time(pid) == entry["start"]
    cl = _cmdline(pid)
    words = {os.path.basename(w) for w in cmd.split()}
    return bool(cl) and (cmd in " ".join(cl) or any(os.path.basename(a) in words for a in cl))


def running() -> dict:
    return {n: e for n, e in load_registry().items() if pid_alive(e["pid"])}


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _guess_port(cmd: str, url: str = "") -> int | None:
    for rx in (r"--port[= ](\d+)", r"\s-p[= ]?(\d+)", r"http\.server\s+(\d+)", r"localhost:(\d+)"):
        m = re.search(rx, cmd) or re.search(rx, url)
        if m:
            return int(m.group(1))
    m = re.search(r":(\d{2,5})\b", url)
    return int(m.group(1)) if m else None


def _slug(cmd: str, port: int | None) -> str:
    return f"dev{port}" if port else re.sub(r"\W+", "-", cmd)[:20].strip("-") or "dev"


def _plan(args: dict, tctx: ToolContext) -> Plan:
    action = args.get("action")
    if action != "start":
        return Plan(ALLOW, "read-only dev server action")
    presets = tctx.cfg.get("dev", {})
    name = args.get("name")
    if name in presets and not args.get("command"):
        cmd = presets[name]["cmd"]
        d = classify_command(cmd, tctx.workdir, tctx.cfg)
        if d.verdict == BLOCK:
            return Plan(BLOCK, d.reason)
        if d.verdict == ELEVATED:  # a preset is no way around typed sudo: it cannot be typed into a background server
            return Plan(BLOCK, "dev_server cannot run elevated commands; use run_shell")
        payload = {"cmd": cmd, "preset": presets[name], "name": name}
        if d.tag:  # a preset that downloads or deletes is still asked every time
            return Plan(ASK, d.tag, cmd, "bash", None, payload, tag=d.tag)
        return Plan(ALLOW, "preset from your config", cmd, "bash", payload=payload)
    cmd = args.get("command")
    if not cmd:
        return Plan(ALLOW, "invalid", payload={"error": "start needs a preset name or a command"
                    + (f" (presets: {', '.join(presets)})" if presets else "")})
    d = classify_command(cmd, tctx.workdir, tctx.cfg)
    if d.verdict == BLOCK:
        return Plan(BLOCK, d.reason)
    if d.verdict == ELEVATED:
        return Plan(BLOCK, "dev_server cannot run elevated commands; use run_shell")
    key = f"dev_server {d.approval_key}" if d.approval_key else None
    return Plan(ASK, d.tag or "starts a background server", cmd, "bash", key,
                {"cmd": cmd, "preset": None, "name": name}, tag=d.tag)


def _start(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    p = plan.payload
    if "error" in p:
        return error(p["error"])
    preset = p["preset"] or {}
    cmd = p["cmd"]
    cwd = Path(os.path.expanduser(preset["dir"])) if preset.get("dir") else tctx.workdir
    if not cwd.is_dir():
        return error(f"folder {cwd} does not exist")
    url = preset.get("url", "")
    port = args.get("port") or _guess_port(cmd, url)
    name = p["name"] or _slug(cmd, port)
    reg = load_registry()
    if name in reg and pid_alive(reg[name]["pid"]):
        return error(f"'{name}' is already running at {reg[name].get('url') or 'its port'}")
    if port and port_in_use(port):
        return error(f"port {port} is already in use")
    log = dev_dir() / f"{name}.log"
    with open(log, "wb") as lf:
        proc = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ, NO_COLOR="1", BROWSER="none"))
    _procs[proc.pid] = proc
    deadline = time.monotonic() + 8
    found = url
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log.read_text("utf-8", "replace").splitlines()[-15:])
            return error(f"'{name}' exited with code {proc.returncode}\n{tail}")
        if port and port_in_use(port):
            break
        m = re.search(r"https?://(?:localhost|127\.0\.0\.1):\d+", log.read_text("utf-8", "replace"))
        if m:
            found = found or m.group(0)
            break
        time.sleep(0.2)
    reg[name] = {"pid": proc.pid, "pgid": proc.pid, "start": proc_start_time(proc.pid), "cmd": cmd, "cwd": str(cwd), "url": found or (f"http://localhost:{port}" if port else "")}
    _save(reg)
    where = reg[name]["url"] or "(no URL detected yet)"
    return ToolResult(True, f"started '{name}' (pid {proc.pid}) at {where}", f"{name} up at {where}",
                      {"event": "server_up", "name": name, "url": where})


def stop_server(name: str) -> str:
    reg = load_registry()
    e = reg.get(name)
    if not e:
        return f"no server named '{name}' was started by XarenCode"
    pgid = e["pgid"]
    if pid_alive(e["pid"]) and not is_ours(e):
        reg.pop(name, None)  # the pid was reused by something else: signal nothing, drop the stale entry
        _save(reg)
        return f"stale entry for '{name}' removed (pid {e['pid']} is no longer that server)"
    if pid_alive(e["pid"]) and pgid > 1 and pgid != os.getpgrp():
        for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                break
            end = time.monotonic() + wait
            while time.monotonic() < end and pid_alive(e["pid"]):
                time.sleep(0.1)
            if not pid_alive(e["pid"]):
                break
    reg.pop(name, None)
    _save(reg)
    return f"stopped '{name}'"


def stop_all() -> list[str]:
    return [stop_server(n) for n in list(running())]


def _run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    action = args["action"]
    if action == "start":
        return _start(args, tctx, plan)
    reg = load_registry()
    name = args.get("name")
    if action == "status":
        if not reg:
            return ToolResult(True, "no dev servers started by XarenCode", "none running")
        lines = [f"{n}: {'running' if pid_alive(e['pid']) else 'stopped'} {e.get('url', '')} ({e['cmd']})" for n, e in reg.items()]
        return ToolResult(True, "\n".join(lines), f"{len(running())} running")
    if not name:
        return error(f"{action} needs a name")
    if action == "stop":
        msg = stop_server(name)
        return ToolResult(msg.startswith("stopped"), msg if msg.startswith("stopped") else f"ERROR: {msg}", msg)
    if action == "logs":
        log = dev_dir() / f"{name}.log"
        if name not in reg or not log.exists():
            return error(f"no logs for '{name}'")
        return ToolResult(True, "\n".join(log.read_text("utf-8", "replace").splitlines()[-60:]), "logs")
    return error(f"unknown action {action}")


def tool(cfg: dict) -> Tool:
    presets = list(cfg.get("dev", {}))
    desc = "Start, stop, check or read logs of a dev server."
    if presets:
        desc += f" Presets: {', '.join(presets)}."
    return Tool("dev_server", desc, schema({
        "action": {"type": "string", "enum": ["start", "stop", "status", "logs"]},
        "name": {"type": "string"}, "command": {"type": "string"}, "port": {"type": "integer"}}, ["action"]),
        _plan, _run, False, lambda a: f"{a.get('action', '')} {a.get('name') or a.get('command') or ''}".strip())
