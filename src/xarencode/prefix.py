"""Persistent prefix KV cache: the system prompt plus tool schemas are evaluated once, saved to disk and restored at launch."""
from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path
from typing import Callable

from .config import resolve_model, state_dir
from .llm import LlamaClient, LlmError, PromptProgress

KEEP_FILES = 4


def _arg_value(args: list[str], names: tuple[str, ...], default: str) -> str:
    for i, a in enumerate(args[:-1]):
        if a in names:
            return args[i + 1]
    return default


def kv_args(cfg: dict) -> dict[str, str]:
    """The options that change how the KV cache is laid out; a user's server.args overrides the config values."""
    s = cfg["server"]
    args = list(s["args"])
    kv = str(s.get("kv_cache", "q8_0"))
    return {"-c": _arg_value(args, ("-c", "--ctx-size"), str(s["ctx_size"])),
            "-ctk": _arg_value(args, ("-ctk", "--cache-type-k"), kv),
            "-ctv": _arg_value(args, ("-ctv", "--cache-type-v"), kv),
            "-fa": _arg_value(args, ("-fa", "--flash-attn"), "on")}


class PrefixCache:
    """Restore-or-prime of slot 0 with the static system prompt and the tool list.

    `warm()` always runs (launch); `ensure()` only when `dirty`, which a server start, /compact or a sleeping server sets.
    Restoring replaces the slot's whole KV state, so it is only ever called before a turn, never mid-conversation.
    """

    def __init__(self, cfg: dict, client: LlamaClient, tools: list[dict], static_system: str, version: str | None = None):
        self.cfg = cfg
        self.client = client
        self.tools = tools
        self.static_system = static_system
        self._version = version
        self.dirty = True
        self.dir = state_dir() / "kv"

    @property
    def enabled(self) -> bool:
        s = self.cfg["server"]
        return bool(s.get("prefix_cache", True)) and bool(s.get("manage", True))

    @cached_property
    def key(self) -> str:
        from .server import version_line  # lazy: server imports a lot at module level

        s = self.cfg["server"]
        version = self._version if self._version is not None else version_line(s["binary"])
        try:
            st = resolve_model(self.cfg).stat()
            model = [str(resolve_model(self.cfg)), st.st_size, st.st_mtime_ns]
        except OSError:
            model = [str(resolve_model(self.cfg)), 0, 0]
        blob = json.dumps({
            "model": model, "version": version, "kv": kv_args(self.cfg), "system": self.static_system,
            "tools": json.dumps(self.tools, sort_keys=True),
            "template": {"enable_thinking": bool(self.cfg["model"]["thinking"])},
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def filename(self) -> str:
        return f"prefix-{self.key[:24]}.bin"

    @property
    def path(self) -> Path:
        return self.dir / self.filename

    def mark_dirty(self) -> None:
        self.dirty = True

    def ensure(self, progress: Callable[[str], None] | None = None) -> None:
        """Restore (or prime) the prefix if something invalidated the slot since the last time; otherwise do nothing."""
        if self.dirty:
            self.warm(progress)

    def warm(self, progress: Callable[[str], None] | None = None) -> str:
        """Returns "restored", "primed" or "unavailable"."""
        self.dirty = False
        if not self.enabled:
            return "unavailable"
        try:
            if self.path.is_file():
                if progress:
                    progress("restoring prompt cache")
                try:
                    self.client.slot("restore", self.filename)
                    self.path.touch()
                    return "restored"
                except LlmError:
                    self.path.unlink(missing_ok=True)  # stale or corrupt: fall through and rebuild it
            return self._prime(progress)
        except (LlmError, OSError):
            return "unavailable"

    def _prime(self, progress) -> str:
        sampling = {"temperature": 0.0, "thinking": bool(self.cfg["model"]["thinking"])}
        msgs = [{"role": "system", "content": self.static_system}]
        for ev in self.client.stream_chat(msgs, self.tools, max_tokens=1, sampling=sampling, progress=progress is not None):
            if isinstance(ev, PromptProgress) and progress:
                progress(f"priming prompt {ev.processed}/{ev.total}")
        if progress:
            progress("saving prompt cache")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.client.slot("save", self.filename)
            self._prune()
        except (LlmError, OSError):
            pass  # the slot is primed in memory either way; only the disk copy failed
        return "primed"

    def _prune(self) -> None:
        files = sorted(self.dir.glob("prefix-*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[KEEP_FILES:]:
            old.unlink(missing_ok=True)
