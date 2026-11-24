"""Turn streamed tool-call deltas (or text-embedded calls) into validated ToolCall objects."""
from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_ALIASES = {
    "bash": "run_shell", "shell": "run_shell", "exec": "run_shell",
    "cat": "read_file", "open_file": "read_file",
    "ls": "list_dir", "list_files": "list_dir",
    "grep": "search_code", "search": "search_code",
    "create_file": "write_file",
    "str_replace": "edit_file", "replace": "edit_file",
}
ARG_ALIASES = {
    "file_path": "path", "filepath": "path", "file": "path",
    "cmd": "command", "old": "old_string", "old_str": "old_string",
    "new": "new_string", "new_str": "new_string", "contents": "content", "text": "content",
}
LENGTH_ERROR = "cut off at the token limit; write smaller pieces or use edit_file"


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    raw: str = ""
    error: str | None = None
    source: str = "native"

    @property
    def history_args(self) -> str:
        """Always valid JSON, as the chat template parses it."""
        if self.error and not self.args:
            return "{}"
        try:
            return json.dumps(self.args)
        except (TypeError, ValueError):
            return "{}"


# ---------------------------------------------------------------- JSON repair

def _load(s: str) -> Any:
    v = json.loads(s)
    if isinstance(v, str):
        v = json.loads(v)
    return v


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", s, re.S)
    return m.group(1).strip() if m else s


def _close(s: str) -> str:
    """Close open strings/brackets of a truncated JSON object."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if esc:
        s = s[:-1]
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s)
    s = re.sub(r':\s*$', ': null', s)
    return s + "".join(reversed(stack))


def repair_json(s: str) -> tuple[dict | None, bool]:
    """Returns (object, lossy). Lossy means characters may have been invented or dropped."""
    if s is None:
        return None, False
    s = s.strip()
    if not s:
        return {}, False
    attempts = [s]
    fenced = _strip_fences(s)
    nocomma = re.sub(r",\s*([}\]])", r"\1", fenced)
    attempts += [fenced, nocomma]
    for a in attempts:
        try:
            v = _load(a)
            if isinstance(v, dict):
                return v, False
        except (ValueError, TypeError):
            pass
    for a in (fenced, nocomma):
        try:
            v = ast.literal_eval(a)
            if isinstance(v, dict):
                return v, False
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
    start = fenced.find("{")
    if start < 0:
        return None, True
    end = fenced.rfind("}")
    body = fenced[start:end + 1] if end > start else fenced[start:]
    for cand in (body, _close(body)):
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v, True
        except ValueError:
            continue
    return None, True


# ---------------------------------------------------------------- accumulator

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


class CallAccumulator:
    def __init__(self):
        self._parts: dict[int, dict] = {}
        self.reasoning = ""

    def feed(self, d) -> None:
        p = self._parts.setdefault(d.index, {"id": None, "name": "", "args": []})
        if d.id:
            p["id"] = d.id
        if d.name:
            p["name"] = d.name if not p["name"] else p["name"]
        if d.arguments:
            p["args"].append(d.arguments)

    def pending_bytes(self) -> int:
        return sum(len("".join(p["args"])) for p in self._parts.values())

    def finish(self, content: str, finish_reason: str | None, registry) -> tuple[str, list[ToolCall]]:
        content = content or ""
        thoughts = THINK_RE.findall(content)
        if thoughts:
            self.reasoning += "\n".join(t.strip() for t in thoughts)
            content = THINK_RE.sub("", content).strip()
        raw_calls: list[tuple[str, str, Any, str]] = []  # id, name, args(str|dict), source
        for i, idx in enumerate(sorted(self._parts)):
            p = self._parts[idx]
            raw_calls.append((p["id"] or f"call_{i + 1}", p["name"], "".join(p["args"]), "native"))
        if not raw_calls:
            content, text_calls = _extract_text_calls(content)
            for i, (name, args) in enumerate(text_calls):
                raw_calls.append((f"call_{i + 1}", name, args, "text"))
        calls = [self._build(cid, name, args, src, registry) for cid, name, args, src in raw_calls]
        if finish_reason == "length":
            for c in calls:
                c.error = f"{c.name or 'tool call'}: {LENGTH_ERROR}"
        return content, calls

    def _build(self, cid: str, name: str, args: Any, source: str, registry) -> ToolCall:
        raw = args if isinstance(args, str) else json.dumps(args)
        call = ToolCall(id=cid, name=name, args={}, raw=raw, source=source)
        lossy = False
        if isinstance(args, dict):
            obj = args
        else:
            obj, lossy = repair_json(args)
        name = TOOL_ALIASES.get(name, name)
        call.name = name
        tool = registry.get(name)
        if tool is None:
            hint = difflib.get_close_matches(name, registry.names(), n=1)
            call.error = f"unknown tool '{name}'" + (f"; did you mean '{hint[0]}'?" if hint else "") + f" Available: {', '.join(registry.names())}"
            return call
        if obj is None:
            call.error = f"{name}: could not parse arguments as JSON; resend them as one valid JSON object"
            return call
        if lossy and not getattr(tool, "read_only", False):
            call.error = f"{name}: arguments were malformed JSON; resend them as one valid JSON object"
            return call
        call.args, call.error = _normalize(obj, tool)
        return call


def _extract_text_calls(content: str) -> tuple[str, list[tuple[str, Any]]]:
    found: list[tuple[str, Any]] = []

    def take(body: str) -> bool:
        obj, _ = repair_json(body)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and ("arguments" in obj or "parameters" in obj):
            found.append((obj["name"], obj.get("arguments", obj.get("parameters"))))
            return True
        return False

    if "<tool_call>" in content:
        def sub(m):
            return "" if take(m.group(1)) else m.group(0)
        content = TOOL_BLOCK_RE.sub(sub, content).strip()
    elif content.strip():
        stripped = _strip_fences(content)
        if stripped.startswith("{") and stripped.endswith("}") and take(stripped):
            content = ""
    return content, found


def _coerce(value: Any, spec: dict) -> Any:
    t = spec.get("type")
    if t == "integer":
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            return int(value.strip())
    elif t == "number" and isinstance(value, str):
        return float(value.strip())
    elif t == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        raise ValueError
    elif t == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


def _type_ok(value: Any, spec: dict) -> bool:
    t = spec.get("type")
    return {
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
    }.get(t, lambda v: True)(value)


def _normalize(obj: dict, tool) -> tuple[dict, str | None]:
    props = tool.parameters.get("properties", {})
    out: dict = {}
    for k, v in obj.items():
        k = ARG_ALIASES.get(k, k) if k not in props else k
        if k in props and k not in out:
            out[k] = v
    for k, v in list(out.items()):
        spec = props[k]
        try:
            out[k] = _coerce(v, spec)
        except (ValueError, TypeError):
            return out, f"{tool.name}: argument '{k}' must be {spec.get('type')}"
        if not _type_ok(out[k], spec):
            return out, f"{tool.name}: argument '{k}' must be {spec.get('type')}"
        if "enum" in spec and out[k] not in spec["enum"]:
            return out, f"{tool.name}: '{k}' must be one of {', '.join(map(str, spec['enum']))}"
    missing = [r for r in tool.parameters.get("required", []) if r not in out]
    if missing:
        return out, f"{tool.name}: missing required argument(s): {', '.join(missing)}"
    return out, None
