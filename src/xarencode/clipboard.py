"""Copy text to the system clipboard. Standard library only."""
from __future__ import annotations

import os
import re
import shutil
import subprocess

from .config import state_dir

_FENCE = re.compile(r"^[ \t]*(?:```|~~~)[^\n]*\n(.*?)^[ \t]*(?:```|~~~)[ \t]*$", re.S | re.M)


def last_reply(history: list[dict]) -> str:
    """The last non-empty assistant message."""
    for msg in reversed(history):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            return msg["content"].strip()
    return ""


def last_code_block(history: list[dict]) -> str:
    """The body of the last fenced code block in the newest assistant message that has one."""
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        blocks = _FENCE.findall(msg.get("content") or "")
        if blocks:
            return blocks[-1].rstrip("\n")
    return ""


def _pipe_to(cmd: list[str], text: str) -> bool:
    # stdout and stderr go to /dev/null, never to a pipe: wl-copy leaves a background process holding them open
    try:
        done = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def copy_text(text: str) -> str:
    """Copy `text` and return where it went. Falls back to a private file in the state folder."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy") and _pipe_to(["wl-copy"], text):
        return "the clipboard (wl-copy)"
    if os.environ.get("DISPLAY"):
        if shutil.which("xclip") and _pipe_to(["xclip", "-selection", "clipboard"], text):
            return "the clipboard (xclip)"
        if shutil.which("xsel") and _pipe_to(["xsel", "-b", "-i"], text):
            return "the clipboard (xsel)"
    path = state_dir() / "clipboard.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)
    return str(path)
