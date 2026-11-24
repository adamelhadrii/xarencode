"""Tool framework: Tool/Plan/ToolResult/ToolContext and the registry."""
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..safety import Verdict


@dataclass
class Plan:
    verdict: Verdict
    reason: str = ""
    preview: str = ""
    preview_kind: str = "text"  # "diff" | "bash" | "text"
    approval_key: str | None = None
    payload: Any = None
    tag: str = ""            # "downloads from the internet" / "deletes files": asked every time, never remembered
    hard_ask: bool = False   # an ASK that yolo mode must still put to the user (a write outside the working directory)


@dataclass
class ToolResult:
    ok: bool
    content: str
    summary: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class ToolContext:
    workdir: Path
    cfg: dict
    output: Callable[[str], None] = lambda line: None
    cancelled: threading.Event = field(default_factory=threading.Event)
    terminal: Callable[[], Any] = contextlib.nullcontext  # context manager that frees the terminal (for sudo)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    plan: Callable[[dict, ToolContext], Plan]
    run: Callable[[dict, ToolContext, Plan], ToolResult]
    read_only: bool = False
    label: Callable[[dict], str] = lambda args: ""  # short text for the UI card header


def schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def error(msg: str) -> ToolResult:
    return ToolResult(False, f"ERROR: {msg}", msg)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in self._tools.values()]


def build_registry(cfg: dict) -> ToolRegistry:
    from . import devserver, fs, shell, vm

    tools = [*fs.tools(), shell.tool(), devserver.tool(cfg)]
    if cfg.get("vm", {}).get("machines"):  # no machines configured: the vm tool would only cost ~90 prompt tokens per request
        tools.append(vm.tool(cfg))
    return ToolRegistry(tools)
