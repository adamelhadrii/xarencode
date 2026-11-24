"""run_shell tool and the shared run_command helper (also used for '!' input)."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..safety import ALLOW, ASK, BLOCK, ELEVATED, classify_command, git_targets, nested_repos
from . import Plan, Tool, ToolContext, ToolResult, error, schema

SERVER_STYLE = re.compile(
    r"(?:^|[;&|]\s*)(?:npm\s+(?:run\s+)?(?:dev|start)|yarn\s+(?:run\s+)?(?:dev|start)|pnpm\s+(?:run\s+)?dev|npx\s+vite|vite"
    r"|python3?\s+-m\s+http\.server|serve|next\s+dev|uvicorn|flask\s+run)\b")
TEST_CMD = re.compile(r"\b(?:pytest|unittest|npm\s+(?:run\s+)?test|yarn\s+test|cargo\s+test|go\s+test|vitest|jest)\b")
MAX_CAPTURE = 200_000


@dataclass
class CommandResult:
    exit_code: int | None
    output: str
    seconds: float
    timed_out: bool = False
    interrupted: bool = False


def _kill_group(proc: subprocess.Popen, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except OSError:
            pass


# Config keys (found in the repositories a command uses) that name a program git would run. Each is overridden below.
GIT_HOOK_KEYS = re.compile(
    r"^(?:core\.(?:fsmonitor|pager|sshcommand|editor|askpass|hookspath|gitproxy)|diff\.external|diff\..+\.(?:textconv|command)"
    r"|filter\..+\.(?:clean|smudge|process)|merge\..+\.driver|gpg\..*program|credential\..*helper|pager\..+"
    r"|remote\..+\.(?:uploadpack|receivepack|vcs|proxy)|log\.showsignature)$", re.I)
# Overridden whether or not a repository sets them. (diff.external is not here: an empty value makes `git diff` fail.)
GIT_STATIC_OVERRIDES = {
    "core.fsmonitor": "false", "core.hookspath": "/dev/null", "core.pager": "cat", "core.sshcommand": "/bin/false",
    "core.editor": "true", "credential.helper": "", "log.showsignature": "false", "gpg.program": "/bin/false",
    "gpg.openpgp.program": "/bin/false", "gpg.x509.program": "/bin/false", "gpg.ssh.program": "/bin/false",
    # a committed nested repository (gitlink) is read with ITS config when the outer status or diff looks into it
    "diff.ignoresubmodules": "all", "status.submodulesummary": "false", "submodule.recurse": "false",
    "fetch.recursesubmodules": "false", "diff.submodule": "short",
}


def _override_for(key: str) -> str:
    k = key.lower()
    if k == "core.fsmonitor" or k == "log.showsignature":
        return "false"
    if k == "core.hookspath":
        return "/dev/null"
    if re.fullmatch(r"gpg\..*program", k) or k == "core.sshcommand":
        return "/bin/false"
    return ""


def git_hardening_env(workdir: Path, env: dict, command: str = "") -> dict:
    """Environment settings (they outrank .git/config) that stop a repository's own config from running programs.

    A read-only git command (status, diff, log, show, blame) can still execute core.fsmonitor, diff.external, a textconv
    or clean/smudge filter, or gpg.program named in the repo's config. The keys are found in every repository the command
    uses (the working directory and each `git -C DIR` / `--git-dir` target) and overridden with an empty or inert value.
    """
    base = {**env, "GIT_PAGER": "cat"}
    names: set[str] = set()

    def collect(cwd: Path, git_dir: Path | None, work_tree: Path | None) -> None:
        if not cwd.is_dir():
            return
        e = dict(base, **({"GIT_DIR": str(git_dir)} if git_dir else {}), **({"GIT_WORK_TREE": str(work_tree)} if work_tree else {}))
        try:
            listed = subprocess.run(["git", "config", "--list", "--name-only"], cwd=cwd, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=5, env=e).stdout.split()
        except (OSError, subprocess.SubprocessError):
            listed = []
        names.update(k for k in listed if GIT_HOOK_KEYS.match(k))

    for eff, git_dir, work_tree in git_targets(command, workdir) or [(workdir, None, None)]:
        collect(eff, git_dir, work_tree)
        for folder, gitdir in nested_repos(eff, git_dir, work_tree) or []:
            collect(folder, gitdir, folder)  # the nested repositories' own keys too
    overrides = dict(GIT_STATIC_OVERRIDES)
    overrides.update({k: _override_for(k) for k in names})
    try:
        n = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
    except ValueError:
        n = 0
    out = dict(env)
    for key, value in overrides.items():
        out[f"GIT_CONFIG_KEY_{n}"], out[f"GIT_CONFIG_VALUE_{n}"] = key, value
        n += 1
    out["GIT_CONFIG_COUNT"] = str(n)
    return out


def run_command(command: str, workdir: Path, *, timeout: int = 60, on_output: Callable[[str], None] | None = None,
                cancelled: threading.Event | None = None, elevated: bool = False, harden_git: bool = False) -> CommandResult:
    env = dict(os.environ, PAGER="cat", GIT_PAGER="cat", NO_COLOR="1", TERM="dumb")
    if harden_git and "git" in command:
        env = git_hardening_env(workdir, env, command)
    start = time.monotonic()
    kwargs = {} if elevated else {"stdin": subprocess.DEVNULL, "start_new_session": True}
    proc = subprocess.Popen(["bash", "-c", command], cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    chunks: list[str] = []
    size = 0

    def pump():
        nonlocal size
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if size < MAX_CAPTURE:
                chunks.append(line)
                size += len(line) + 1
            if on_output:
                on_output(line)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    timed_out = interrupted = False
    try:
        while proc.poll() is None:
            if time.monotonic() - start > timeout:
                timed_out = True
            elif cancelled is not None and cancelled.is_set():
                interrupted = True
            if timed_out or interrupted:
                _kill_group(proc)
                try:
                    proc.wait(3)
                except subprocess.TimeoutExpired:
                    _kill_group(proc, signal.SIGKILL)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        _kill_group(proc, signal.SIGKILL)
    proc.wait()
    t.join(2)
    proc.stdout.close()
    return CommandResult(proc.returncode, "\n".join(chunks), time.monotonic() - start, timed_out, interrupted)


def _plan(args: dict, tctx: ToolContext) -> Plan:
    cmd = args.get("command", "")
    d = classify_command(cmd, tctx.workdir, tctx.cfg)
    if d.verdict == BLOCK:
        return Plan(BLOCK, d.reason)
    if SERVER_STYLE.search(cmd):
        return Plan(ALLOW, "server command", payload={"error": "this looks like a long-running server; use the dev_server tool instead"})
    return Plan(d.verdict, d.reason, cmd, "bash", d.approval_key, {"elevated": d.verdict == ELEVATED}, tag=d.tag)


def _run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    if plan.payload.get("error"):
        return error(plan.payload["error"])
    timeout = max(1, min(600, int(args.get("timeout", tctx.cfg["agent"]["shell_timeout"]))))
    elevated = plan.payload["elevated"]
    cmd = args["command"]
    if elevated:
        with tctx.terminal():
            res = run_command(cmd, tctx.workdir, timeout=timeout, cancelled=tctx.cancelled, elevated=True)
        for ln in res.output.splitlines()[-8:]:
            tctx.output(ln)
    else:
        res = run_command(cmd, tctx.workdir, timeout=timeout, on_output=tctx.output, cancelled=tctx.cancelled,
                          harden_git=plan.verdict == ALLOW)  # approved commands are the user's call; read-only ones are neutralised
    n = len(res.output.splitlines())
    head = f"exit={res.exit_code} ({res.seconds:.1f}s)"
    if res.timed_out:
        return ToolResult(False, f"ERROR: timed out after {timeout}s and was killed\n{res.output}", f"timed out after {timeout}s", {"exit": None})
    if res.interrupted:
        return ToolResult(False, f"ERROR: interrupted by the user\n{res.output}", "interrupted", {"exit": None})
    detail = {"exit": res.exit_code, "seconds": res.seconds, "lines": n}
    if res.exit_code == 0 and TEST_CMD.search(cmd):
        detail["event"] = "tests_passed"
    return ToolResult(res.exit_code == 0, f"{head}\n{res.output}".rstrip(), f"exit {res.exit_code} · {res.seconds:.1f}s · {n} lines", detail)


def tool() -> Tool:
    return Tool(
        "run_shell", "Run a shell command in the working directory. Not for servers (use dev_server).",
        schema({"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
        _plan, _run, False, lambda a: a.get("command", ""))
