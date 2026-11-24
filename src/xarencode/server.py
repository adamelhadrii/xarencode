"""llama-server lifecycle, model download and the doctor checks (stdlib only)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .config import data_dir, parse_gguf, resolve_model, state_dir
from .llm import LlamaClient

USER_AGENT = "xarencode/0.1"
READY_TIMEOUT = 180
POLL_INTERVAL = 0.05
GIB = 1024 ** 3


class ServerStartError(RuntimeError):
    pass


# Tuning flags: each entry lists every spelling of one llama-server option, so a user's own server.args wins.
ALIASES = {
    "tb": ("-tb", "--threads-batch"),
    "repack": ("--no-repack", "--repack", "-nr"),
    "fa": ("-fa", "--flash-attn"),
    "ctk": ("-ctk", "--cache-type-k"),
    "ctv": ("-ctv", "--cache-type-v"),
    "cram": ("--cache-ram", "-cram"),
    "slots": ("--slot-save-path",),
    "sleep": ("--sleep-idle-seconds",),
    "reuse": ("--cache-reuse",),
}
_HELP_CACHE: dict[str, set[str] | None] = {}


def supported_flags(binary: str) -> set[str] | None:
    """Every option spelling in `<binary> --help`, parsed once per process. None if the probe fails (assume supported)."""
    if binary not in _HELP_CACHE:
        flags: set[str] | None = None
        try:
            r = subprocess.run([binary, "--help"], capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
            text = (r.stdout + r.stderr).decode("utf-8", "replace")
            found = set(re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", text))
            flags = found or None
        except (OSError, subprocess.SubprocessError):
            flags = None
        _HELP_CACHE[binary] = flags
    return _HELP_CACHE[binary]


def version_line(binary: str) -> str:
    """The `llama-server --version` line (part of the prefix-cache key); "" if it cannot be read."""
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
        lines = (r.stdout + r.stderr).decode("utf-8", "replace").splitlines()
    except (OSError, subprocess.SubprocessError):
        return ""
    return next((ln.strip() for ln in lines if re.search(r"version", ln, re.I)), lines[0].strip() if lines else "")


def proc_start_time(pid: int) -> int | None:
    """Start time of a process in clock ticks since boot (field 22 of /proc/<pid>/stat); None if it is gone.

    A pid plus its start time identifies one process for good, even if the pid is reused or the process renames itself.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text("utf-8", "replace")
        return int(stat[stat.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class LlamaServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        s = cfg["server"]
        self.client = LlamaClient(f"http://{s['host']}:{s['port']}")
        self.started_by_us = False
        self._aborted = threading.Event()  # set at shutdown so a boot thread cannot start a server nobody will stop
        self.log_path = state_dir() / "llama-server.log"
        self.pid_path = state_dir() / "server.pid"

    def command(self) -> list[str]:
        s = self.cfg["server"]
        base = [s["binary"], "-m", str(resolve_model(self.cfg)), "-c", str(s["ctx_size"]), "--host", s["host"],
                "--port", str(s["port"]), "-t", str(s["threads"]), *s["args"]]
        return base + self.tuning_flags()

    def kv_dir(self) -> Path:
        return state_dir() / "kv"

    def tuning_flags(self) -> list[str]:
        """Speed flags appended to the command line, minus any the user set in server.args or the build lacks."""
        s = self.cfg["server"]
        have = set(s["args"])
        known = supported_flags(s["binary"])
        kv = str(s.get("kv_cache", "q8_0"))
        want: list[tuple[str, list[str]]] = []
        if int(s.get("threads_batch", 0)) > 0:
            want.append(("tb", ["-tb", str(int(s["threads_batch"]))]))
        if not s.get("repack", False):
            want.append(("repack", ["--no-repack"]))
        want.append(("fa", ["-fa", "on"]))
        want.append(("ctk", ["-ctk", kv]))
        want.append(("ctv", ["-ctv", kv]))
        want.append(("cram", ["--cache-ram", str(int(s.get("cache_ram_mib", 0)))]))
        if s.get("prefix_cache", True):
            want.append(("slots", ["--slot-save-path", str(self.kv_dir())]))
        if s.get("keep_running"):
            want.append(("sleep", ["--sleep-idle-seconds", str(int(s.get("sleep_after", 600)))]))
        if int(s.get("cache_reuse", 0)) > 0:
            want.append(("reuse", ["--cache-reuse", str(int(s["cache_reuse"]))]))
        out: list[str] = []
        for key, flag in want:
            names = ALIASES[key]
            if any(a in have for a in names) or (known is not None and flag[0] not in known):
                continue
            out += flag
        if kv != "f16" and "-fa" not in out and not any(a in have for a in ALIASES["fa"]):
            # a quantized V cache needs flash attention; without it fall back to the default cache type
            if "-ctv" in out:
                i = out.index("-ctv")
                del out[i:i + 2]
        return out

    def status(self) -> str:
        return self.client.health()

    def log_tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text("utf-8", "replace").splitlines()[-n:])
        except OSError:
            return "(no log yet)"

    def ensure(self, progress: Callable[[str], None] | None = None, timeout: float = READY_TIMEOUT) -> bool:
        """Make sure a server answers on the configured port. Returns True if we started it."""
        if self.status() == "ok":
            return False
        if self.status() == "loading":
            self._wait(progress, timeout)
            return False
        if not self.cfg["server"]["manage"]:
            raise ServerStartError(f"no llama-server on port {self.cfg['server']['port']} and server.manage is false")
        self.start()
        try:
            self._wait(progress, timeout)
        except BaseException:
            self.stop()  # do not leave a half-started llama-server holding the model in RAM
            raise
        return True

    def abort(self) -> None:
        self._aborted.set()

    def start(self) -> None:
        if self._aborted.is_set():
            raise ServerStartError("shutting down")
        binary = self.cfg["server"]["binary"]
        if shutil.which(binary) is None and not os.access(binary, os.X_OK):
            raise ServerStartError(f"{binary} not found. Install it with: sudo pacman -S llama-cpp")
        model = resolve_model(self.cfg)
        if not model.is_file():
            raise ServerStartError(f"model file missing: {model}. Run: xarencode pull")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg["server"].get("prefix_cache", True):
            self.kv_dir().mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "wb")  # truncated on every start
        self.proc = subprocess.Popen(self.command(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)  # own session: Ctrl+C in the terminal does not reach it
        log.close()
        self.pid_path.write_text(f"{self.proc.pid} {proc_start_time(self.proc.pid) or ''}".strip())
        self.started_by_us = True

    def _wait(self, progress, timeout: float) -> None:
        end = time.monotonic() + timeout
        proc = getattr(self, "proc", None)
        while time.monotonic() < end:
            if self._aborted.is_set():
                raise ServerStartError("shutting down")
            st = self.status()
            if st == "ok":
                return
            if proc is not None and proc.poll() is not None:
                raise ServerStartError(f"llama-server exited with code {proc.returncode}:\n{self.log_tail()}")
            if progress:
                progress("loading model" if st == "loading" else "starting llama-server")
            time.sleep(POLL_INTERVAL)
        raise ServerStartError(f"llama-server was not ready after {int(timeout)}s:\n{self.log_tail()}")

    def _is_llama_server(self, pid: int, start: int | None = None) -> bool:
        if start is not None:
            return proc_start_time(pid) == start
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        want = os.path.basename(self.cfg["server"]["binary"])
        return any(os.path.basename(a.decode("utf-8", "replace")) == want for a in raw.split(b"\0") if a)

    def stop(self) -> str:
        proc = getattr(self, "proc", None)
        try:
            fields = self.pid_path.read_text().split()
            pid = int(fields[0])
            start = int(fields[1]) if len(fields) > 1 else None
        except (OSError, ValueError, IndexError):
            if proc is None or proc.poll() is not None:
                return "no server started by XarenCode"
            pid, start = proc.pid, None  # pid file lost but our own child is still running
        if proc is not None and proc.pid != pid:
            proc = None
        alive = lambda: proc.poll() is None if proc else _pid_alive(pid)  # noqa: E731
        if proc is None and alive() and not self._is_llama_server(pid, start):
            self.pid_path.unlink(missing_ok=True)  # pid reused by an unrelated process: leave it alone
            return f"stale pid file removed (pid {pid} is not llama-server)"
        if alive():
            for sig, wait in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
                try:
                    os.killpg(pid, sig)
                except OSError:
                    break
                end = time.monotonic() + wait
                while time.monotonic() < end and alive():
                    time.sleep(0.1)
                if not alive():
                    break
        try:
            self.pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        self.started_by_us = False
        return "llama-server stopped"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- pull

class PullError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _head(url: str) -> tuple[int, str | None]:
    """HEAD without following redirects; returns (size, sha256 or None)."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=30) as r:
            headers = r.headers
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise PullError(f"{url} returned HTTP {e.code}") from e
        headers = e.headers
    except urllib.error.URLError as e:
        raise PullError(f"cannot reach {url}: {e.reason}") from e
    size = headers.get("X-Linked-Size") or headers.get("Content-Length")
    if not size or not size.isdigit():
        raise PullError("server did not report the file size")
    etag = (headers.get("X-Linked-ETag") or "").strip('"')
    return int(size), etag if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag.lower()) else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pull_model(cfg: dict, out: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
               confirm: Callable[[str], bool] = lambda q: False) -> Path:
    parts = parse_gguf(cfg["model"]["gguf"])
    if parts is None:
        raise PullError("model.gguf is an absolute path; nothing to download")
    owner, repo, fname = parts
    url = f"{cfg['model']['hf_base'].rstrip('/')}/{owner}/{repo}/resolve/main/{fname}"
    dest_dir = resolve_model(cfg).parent
    dest = dest_dir / fname
    part = dest.with_name(fname + ".part")
    size, sha = _head(url)
    out(f"{fname}: {size / GIB:.2f} GiB")
    if dest.is_file() and dest.stat().st_size == size:
        out("already downloaded")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    others = [p for p in dest_dir.glob("*.gguf") if p.name != fname]
    budget = cfg["model"]["storage_budget_gb"] * GIB
    if sum(p.stat().st_size for p in others) + size + 0.1 * GIB > budget:
        names = ", ".join(p.name for p in others) or "none"
        if not others or not confirm(f"Over the {cfg['model']['storage_budget_gb']} GB storage budget. Delete other models ({names})?"):
            raise PullError(f"storage budget of {cfg['model']['storage_budget_gb']} GB would be exceeded; delete old models or raise model.storage_budget_gb")
        for p in others:
            p.unlink()
            out(f"deleted {p.name}")
    have = part.stat().st_size if part.exists() else 0
    free = shutil.disk_usage(dest_dir).free
    if free < size - have + GIB // 4:
        raise PullError(f"not enough free disk space ({free / GIB:.1f} GiB free, need {(size - have) / GIB:.1f} GiB)")
    _download(url, part, size, have, out)
    if sha:
        out("verifying sha256 ...")
        if sha256_file(part) != sha:
            part.unlink()
            raise PullError("sha256 mismatch; the partial file was deleted, run pull again")
    part.rename(dest)
    out(f"saved {dest}")
    return dest


def _download(url: str, part: Path, size: int, have: int, out) -> None:
    headers = {"User-Agent": USER_AGENT}
    if have:
        if have > size:
            part.unlink()
            have = 0
        elif have == size:
            return
        else:
            headers["Range"] = f"bytes={have}-"
            out(f"resuming at {have / GIB:.2f} GiB")
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.URLError as e:
        raise PullError(f"download failed: {e}") from e
    mode = "ab"
    if have and resp.status != 206:  # server ignored Range: start over
        have, mode = 0, "wb"
    t0, got = time.monotonic(), 0
    with resp, open(part, mode) as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
            got += len(block)
            done = have + got
            rate = got / max(time.monotonic() - t0, 1e-3)
            sys.stderr.write(f"\r{done * 100 // size:3d}%  {done / 1e6:8.1f}/{size / 1e6:.1f} MB  {rate / 1e6:5.1f} MB/s ")
            sys.stderr.flush()
    sys.stderr.write("\n")
    if part.stat().st_size != size:
        raise PullError(f"download incomplete ({part.stat().st_size}/{size} bytes); run pull again to resume")


# ---------------------------------------------------------------- doctor

def doctor(cfg: dict) -> list[tuple[str, str]]:
    """Returns [(status, text)] where status is ok|warn|fail."""
    res: list[tuple[str, str]] = []
    binary = cfg["server"]["binary"]
    if shutil.which(binary) or os.access(binary, os.X_OK):
        res.append(("ok", f"{binary} found"))
    else:
        res.append(("fail", f"llama-server not found. Install with: sudo pacman -S llama-cpp"))
    model = resolve_model(cfg)
    if model.is_file():
        res.append(("ok", f"model {model.name} ({model.stat().st_size / GIB:.2f} GiB)"))
    else:
        res.append(("fail", f"model missing: {model}. Run: xarencode pull"))
    st = LlamaClient(f"http://{cfg['server']['host']}:{cfg['server']['port']}").health()
    res.append(("ok" if st == "ok" else "warn", f"server on port {cfg['server']['port']}: {st}"))
    for mod, pkg in (("rich", "python-rich"), ("prompt_toolkit", "python-prompt_toolkit")):
        if importlib.util.find_spec(mod):
            res.append(("ok", f"{mod} installed"))
        else:
            res.append(("fail", f"{mod} missing. Install with: sudo pacman -S {pkg}"))
    res.append(("ok", "ripgrep found") if shutil.which("rg") else ("warn", "ripgrep not found (grep is used instead): sudo pacman -S ripgrep"))
    try:
        free = shutil.disk_usage(data_dir() if data_dir().exists() else Path.home()).free
        res.append(("ok" if free > 6 * GIB else "warn", f"{free / GIB:.1f} GiB free disk"))
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    gb = int(ln.split()[1]) * 1024 / GIB
                    res.append(("ok" if gb > 3.6 else "warn", f"{gb:.1f} GiB RAM available (model needs about 3.4 GiB)"))
    except OSError:
        pass
    if shutil.which("qemu-system-x86_64") is None:
        res.append(("warn", "qemu not installed (only needed for the vm tool)"))
    return res
