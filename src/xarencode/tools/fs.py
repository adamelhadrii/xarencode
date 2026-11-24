"""File tools: read_file, list_dir, search_code, write_file, edit_file."""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
from pathlib import Path

from ..safety import ALLOW, ASK, BLOCK, CREDENTIAL_REASON, resolve_path
from . import Plan, Tool, ToolContext, ToolResult, error, schema

MAX_READ_BYTES = 2 * 1024 * 1024
MAX_WRITE_BYTES = 512 * 1024
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}


def _rel(path: Path, wd: Path) -> str:
    wd = Path(os.path.realpath(wd))
    try:
        return str(path.relative_to(wd))
    except ValueError:
        return str(path)


def _blocked(reason: str) -> Plan:
    return Plan(BLOCK, reason)


def _read_plan(args: dict, tctx: ToolContext) -> Plan:
    path, verdict, why = resolve_path(args.get("path", "."), tctx.workdir, write=False, cfg=tctx.cfg)
    if verdict == BLOCK:
        return _blocked(why)
    return Plan(verdict, why or "read", approval_key=f"read:{path}" if verdict == ASK and not why.startswith(CREDENTIAL_REASON) else None, payload=path)


# -------------------------------------------------------------- read_file

def _read_run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    path: Path = plan.payload
    if not path.is_file():
        return error(f"{args['path']} is not a file" if path.exists() else f"{args['path']} does not exist")
    if path.stat().st_size > MAX_READ_BYTES:
        return error(f"{args['path']} is larger than 2 MB; use search_code or read it in a shell with head/tail")
    data = path.read_bytes()
    if b"\0" in data[:4096]:
        return error(f"{args['path']} looks like a binary file")
    lines = data.decode("utf-8", "replace").splitlines()
    total = len(lines)
    offset = max(1, int(args.get("offset", 1)))
    limit = max(1, min(200, int(args.get("limit", 80))))
    chunk = lines[offset - 1:offset - 1 + limit]
    if not chunk:
        return error(f"offset {offset} is past the end of the file ({total} lines)")
    last = offset - 1 + len(chunk)
    head = f"{args['path']} lines {offset}-{last} of {total}"
    return ToolResult(True, head + "\n" + "\n".join(chunk), f"Read {len(chunk)} lines", {"path": str(path)})


# -------------------------------------------------------------- list_dir

def _list_run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    root: Path = plan.payload
    if not root.is_dir():
        return error(f"{args.get('path', '.')} is not a directory")
    depth = max(1, min(3, int(args.get("depth", 1))))
    out: list[str] = []
    cap = 200

    def walk(d: Path, level: int):
        try:
            entries = sorted(os.scandir(d), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError as e:
            out.append("  " * level + f"[{e.strerror}]")
            return
        for e in entries:
            if len(out) >= cap:
                return
            is_dir = e.is_dir(follow_symlinks=False)
            if is_dir and e.name in SKIP_DIRS:
                continue
            out.append("  " * level + e.name + ("/" if is_dir else ""))
            if is_dir and level + 1 < depth:
                walk(Path(e.path), level + 1)

    walk(root, 0)
    note = f"\n[... capped at {cap} entries]" if len(out) >= cap else ""
    body = "\n".join(out) if out else "(empty)"
    return ToolResult(True, body + note, f"Listed {len(out)} entries", {"path": str(root)})


# -------------------------------------------------------------- search_code

def _search_run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    pattern = args["pattern"]
    target = str(plan.payload)
    ic = bool(args.get("ignore_case"))
    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", "--color", "never", "--with-filename"]
        if ic:
            cmd.append("-i")
        if args.get("glob"):
            cmd += ["-g", args["glob"]]
        cmd += ["-e", pattern, "--", target]
    else:
        cmd = ["grep", "-rnIE", "--color=never"] + (["-i"] if ic else [])
        cmd += [f"--exclude-dir={d}" for d in SKIP_DIRS]
        if args.get("glob"):
            cmd.append(f"--include={args['glob']}")
        cmd += ["-e", pattern, "--", target]
    try:
        r = subprocess.run(cmd, cwd=tctx.workdir, capture_output=True, text=True, errors="replace", timeout=20, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return error("search timed out")
    except OSError as e:
        return error(f"search failed: {e}")
    if r.returncode == 1 and not r.stdout:
        return ToolResult(True, "no matches", "No matches")
    if r.returncode > 1:
        return error(r.stderr.strip()[:300] or "search failed")
    wd = Path(os.path.realpath(tctx.workdir))
    lines = []
    for raw in r.stdout.splitlines()[:50]:
        m = re.match(r"^(.*?):(\d+):(.*)$", raw)
        if m:
            f = m.group(1)
            try:
                f = str(Path(os.path.realpath(f)).relative_to(wd))
            except ValueError:
                pass
            lines.append(f"{f}:{m.group(2)}: {m.group(3).strip()[:200]}")
    more = len(r.stdout.splitlines()) - len(lines)
    if more > 0:
        lines.append(f"[... {more} more matches]")
    return ToolResult(True, "\n".join(lines), f"{len(lines)} matches")


# -------------------------------------------------------------- write_file / edit_file

def _diff(old: str, new: str, name: str) -> tuple[str, int, int]:
    d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=2))
    add = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    rem = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    return "\n".join(d), add, rem


def _write_plan(args: dict, tctx: ToolContext) -> Plan:
    path, verdict, why = resolve_path(args.get("path", ""), tctx.workdir, write=True, cfg=tctx.cfg)
    if verdict == BLOCK:
        return _blocked(why)
    content = args.get("content", "")
    if len(content.encode()) > MAX_WRITE_BYTES:
        return _blocked("file is larger than 512 KB")
    if path.is_dir():
        return Plan(ALLOW, "directory", payload={"error": f"{args['path']} is a directory"})
    old = path.read_text("utf-8", "replace") if path.is_file() else ""
    diff, add, rem = _diff(old, content, _rel(path, tctx.workdir))
    lines = diff.splitlines()
    if not path.exists():
        lines = lines[:42] + ([f"... ({len(diff.splitlines()) - 42} more lines)"] if len(lines) > 42 else [])
    return Plan(ASK if verdict == ALLOW else verdict, why or ("new file" if not path.exists() else "overwrite"),
                "\n".join(lines), "diff", f"write:{_rel(path, tctx.workdir)}",
                {"path": path, "old": old, "diff": diff, "add": add, "rem": rem, "new": not path.exists()},
                hard_ask=verdict != ALLOW)


def _write_run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    p = plan.payload
    if "error" in p:
        return error(p["error"])
    path: Path = p["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], "utf-8")
    n = len(args["content"].splitlines())
    verb = "Created" if p["new"] else "Wrote"
    return ToolResult(True, f"{verb} {_rel(path, tctx.workdir)} ({n} lines)", f"{verb} {n} lines (+{p['add']} -{p['rem']})",
                      {"diff": p["diff"], "path": _rel(path, tctx.workdir), "add": p["add"], "rem": p["rem"], "event": "edit_applied"})


_NUM_PREFIX = re.compile(r"^\s*\d+[:|]\s?")


def _strip_prefixes(s: str) -> str:
    return "\n".join(_NUM_PREFIX.sub("", ln).rstrip() for ln in s.split("\n"))


def _closest(text: str, needle: str) -> str:
    first = next((ln.strip() for ln in needle.splitlines() if ln.strip()), "")
    lines = text.splitlines()
    best = difflib.get_close_matches(first, [ln.strip() for ln in lines], n=1, cutoff=0.5)
    if not best:
        return ""
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == best[0])
    return "closest match near line %d:\n%s" % (idx + 1, "\n".join(lines[max(0, idx - 1):idx + 3]))


def _edit_plan(args: dict, tctx: ToolContext) -> Plan:
    path, verdict, why = resolve_path(args.get("path", ""), tctx.workdir, write=True, cfg=tctx.cfg)
    if verdict == BLOCK:
        return _blocked(why)
    fail = lambda msg: Plan(ALLOW, "invalid edit", payload={"error": msg})  # noqa: E731  (no approval needed for a failing edit)
    if not path.is_file():
        return fail(f"{args['path']} does not exist")
    raw = path.read_bytes().decode("utf-8", "replace")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    old, new = args["old_string"].replace("\r\n", "\n"), args["new_string"].replace("\r\n", "\n")
    if not old:
        return fail("old_string is empty; use write_file to create or replace a whole file")
    count = text.count(old)
    if count == 0:
        o2 = _strip_prefixes(old)
        if o2 and text.count(o2):
            old, new, count = o2, _strip_prefixes(new), text.count(o2)
    if count == 0:
        hint = _closest(text, old)
        return fail("old_string was not found in the file" + (f"; {hint}" if hint else "; re-read the file with read_file"))
    if count > 1 and not args.get("replace_all"):
        return fail(f"old_string matches {count} places; add more surrounding lines or set replace_all")
    updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
    diff, add, rem = _diff(text, updated, _rel(path, tctx.workdir))
    final = updated.replace("\n", "\r\n") if crlf else updated
    return Plan(ASK, "edit", diff, "diff", f"edit:{_rel(path, tctx.workdir)}",
                {"path": path, "final": final, "diff": diff, "add": add, "rem": rem, "count": count},
                hard_ask=verdict != ALLOW)


def _edit_run(args: dict, tctx: ToolContext, plan: Plan) -> ToolResult:
    p = plan.payload
    if "error" in p:
        return error(p["error"])
    p["path"].write_bytes(p["final"].encode("utf-8"))
    return ToolResult(True, f"Edited {_rel(p['path'], tctx.workdir)} ({p['count']} replacement{'s' if p['count'] != 1 else ''})",
                      f"Edited (+{p['add']} -{p['rem']})",
                      {"diff": p["diff"], "path": _rel(p["path"], tctx.workdir), "add": p["add"], "rem": p["rem"], "event": "edit_applied"})


def tools() -> list[Tool]:
    S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
    return [
        Tool("read_file", "Read a text file (80 lines by default). Use offset/limit to page.",
             schema({"path": S, "offset": I, "limit": I}, ["path"]), _read_plan, _read_run, True, lambda a: a.get("path", "")),
        Tool("list_dir", "List a folder. Folders end in /.", schema({"path": S, "depth": I}), _read_plan, _list_run, True,
             lambda a: a.get("path", ".")),
        Tool("search_code", "Regex search in files. Returns file:line: text.",
             schema({"pattern": S, "path": S, "glob": S, "ignore_case": B}, ["pattern"]), _read_plan, _search_run, True,
             lambda a: a.get("pattern", "")),
        Tool("write_file", "Create or overwrite a whole file in the working directory.",
             schema({"path": S, "content": S}, ["path", "content"]), _write_plan, _write_run, False, lambda a: a.get("path", "")),
        Tool("edit_file", "Replace old_string with new_string in a file. old_string must match exactly once.",
             schema({"path": S, "old_string": S, "new_string": S, "replace_all": B}, ["path", "old_string", "new_string"]),
             _edit_plan, _edit_run, False, lambda a: a.get("path", "")),
    ]
