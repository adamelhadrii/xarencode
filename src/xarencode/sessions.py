"""JSONL sessions on disk, listing/pruning, redaction and export (md/svg/html)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir, model_short_name


def sessions_dir() -> Path:
    d = data_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class SessionInfo:
    id: str
    path: Path
    created: str
    title: str
    workdir: str
    model: str


class Session:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.entries: list[tuple[dict, dict | None]] = []

    @classmethod
    def new(cls, cfg: dict, workdir: Path) -> "Session":
        sid = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        meta = {"t": "meta", "id": sid, "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "workdir": str(workdir), "model": model_short_name(cfg), "title": ""}
        return cls(sessions_dir() / f"{sid}.jsonl", meta)

    @classmethod
    def load(cls, path: Path) -> "Session":
        lines = path.read_text("utf-8").splitlines()
        s = cls(path, json.loads(lines[0]))
        for ln in lines[1:]:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("t") == "msg":
                s.entries.append((rec["msg"], rec.get("ui")))
        return s

    def messages(self) -> list[dict]:
        return [m for m, _ in self.entries]

    def append(self, msg: dict, ui: dict | None = None) -> None:
        self.entries.append((msg, ui))
        first = not self.path.exists()
        if first:
            self.meta["title"] = next((m["content"] for m, _ in self.entries if m["role"] == "user"), "")[:60].replace("\n", " ")
        with open(self.path, "a", encoding="utf-8") as f:
            if first:
                f.write(json.dumps(self.meta) + "\n")
            f.write(json.dumps({"t": "msg", "msg": msg, "ui": ui}) + "\n")

    def rewrite(self, messages: list[dict]) -> None:
        """Replace the stored conversation (after /compact or /clear)."""
        self.entries = [(m, None) for m in messages]
        if not messages and not self.path.exists():
            return
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.meta) + "\n")
            for m, ui in self.entries:
                f.write(json.dumps({"t": "msg", "msg": m, "ui": ui}) + "\n")


def list_sessions() -> list[SessionInfo]:
    out = []
    for p in sessions_dir().glob("*.jsonl"):
        try:
            with open(p, encoding="utf-8") as f:
                m = json.loads(f.readline())
        except (OSError, ValueError):
            continue
        out.append(SessionInfo(m.get("id", p.stem), p, m.get("created", ""), m.get("title", ""), m.get("workdir", ""), m.get("model", "")))
    return sorted(out, key=lambda s: s.created, reverse=True)


def prune(keep: int = 50) -> int:
    old = list_sessions()[keep:]
    for s in old:
        s.path.unlink(missing_ok=True)
    return len(old)


def find_session(ref: str | int | None) -> SessionInfo | None:
    items = list_sessions()
    if ref in (None, "", "last"):
        return items[0] if items else None
    ref = str(ref)
    if ref.isdigit() and 1 <= int(ref) <= len(items):
        return items[int(ref) - 1]
    return next((s for s in items if s.id.startswith(ref)), None)


# ---------------------------------------------------------------- redaction

# A name whose own word (split on _ - . or camelCase) is a secret word: DB_PASSWORD, api-key, accessToken, "secret".
# Plain prose and code such as author:, monkey:, max_tokens=, key=lambda and keyboard= do not match.
_SECRET_WORD = (r"(?:(?:api|access|private|secret|auth|ssh|signing|encryption|master|license|session|client)[_-]?key"
                r"|apikey|token|secret|passw(?:or)?d|credentials?)")
_NAME = (r"(?i)((?:\b|(?<=[_.-])|(?-i:(?<=[a-z])(?=[A-Z])))(?:[A-Za-z0-9]+[_.-])*" + _SECRET_WORD
         + r"(?![A-Za-z0-9])(?:[_.-][A-Za-z0-9]+)*")
_SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[redacted private key]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[redacted]"),
    (re.compile(r"\b(?:sk|pk|hf|ghp|gho|ghu|ghs|ghr|xox[abprs])[-_][A-Za-z0-9_-]{16,}"), "[redacted]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[redacted]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"), "Bearer [redacted]"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@"), r"\1[redacted]@"),
    (re.compile(r"(?i)\b(authorization[\"']?\s*[:=]\s*[\"']?)(?:(?:basic|bearer|token|digest|negotiate)\s+)?[^\s\"',;]+"),
     r"\1[redacted]"),
    (re.compile(_NAME + r"[\"']?\s*[:=]\s*)(?!\[redacted|\$)(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;&\"'}\])]+)"), r"\1[redacted]"),
]


def redact(text: str) -> str:
    if not text:
        return text
    home = str(Path.home())
    for h in {home, os.path.realpath(home)}:
        if len(h) > 1:
            text = text.replace(h, "~")
    for rx, repl in _SECRET_PATTERNS:
        text = rx.sub(repl, text)
    return text


# ---------------------------------------------------------------- export

class ExportError(RuntimeError):
    pass


def _tool_label(tc: dict) -> str:
    fn = tc.get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
        first = next((str(v) for v in args.values() if isinstance(v, str)), "")
    except ValueError:
        first = ""
    return f"{fn.get('name', 'tool')}({first[:80]})"


def export_md(meta: dict, entries: list[tuple[dict, dict | None]], name: str = "Xaren") -> str:
    out = [f"# XarenCode session", "", f"- model: {meta.get('model', '')}", f"- folder: {redact(meta.get('workdir', ''))}",
           f"- date: {meta.get('created', '')}", ""]
    for msg, ui in entries:
        role, content = msg.get("role"), (msg.get("content") or "").strip()
        if role == "user":
            out += [f"**You:** {content}", ""]
        elif role == "assistant":
            if content:
                out += [f"**{name}:** {content}", ""]
            for tc in msg.get("tool_calls") or []:
                out += [f"> `{_tool_label(tc)}`", ""]
        elif role == "tool":
            body = "\n".join(content.splitlines()[:30])
            summary = (ui or {}).get("summary") or ""
            out += [f"<sub>{summary}</sub>" if summary else "", "```", body, "```", ""]
    return redact("\n".join(out)).rstrip() + "\n"


def export(session: Session, fmt: str, out_dir: Path, cfg: dict) -> Path:
    if fmt not in ("md", "svg", "html"):
        raise ExportError("format must be md, svg or html")
    if not session.entries:
        raise ExportError("nothing to share yet")
    out_dir = Path(os.path.expanduser(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"xarencode-{session.meta['id']}.{fmt}"
    if fmt == "md":
        path.write_text(export_md(session.meta, session.entries, cfg["persona"]["name"]), "utf-8")
        return path
    try:
        from . import ui
    except ImportError as e:
        raise ExportError("svg/html export needs rich (sudo pacman -S python-rich)") from e
    console = ui.render_transcript(session.meta, session.entries, cfg)
    if fmt == "svg":
        console.save_svg(str(path), title="XarenCode")
    else:
        console.save_html(str(path))
    return path
