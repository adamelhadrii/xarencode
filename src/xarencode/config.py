"""Configuration: XDG paths, defaults, first-run template and loading."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

DEFAULT_GGUF = "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

DEFAULTS: dict = {
    "model": {
        "gguf": DEFAULT_GGUF,
        "hf_base": "https://huggingface.co",
        "storage_budget_gb": 5.0,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 1024,
        "thinking": False,
    },
    "server": {
        "manage": True,
        "binary": "llama-server",
        "host": "127.0.0.1",
        "port": 8091,
        "ctx_size": 4096,
        "threads": 4,
        "threads_batch": 6,
        "args": ["--jinja", "-np", "1"],
        "keep_running": False,
        "repack": False,          # true restores llama.cpp's default weight repacking (slower on this CPU)
        "kv_cache": "q8_0",       # KV cache type for -ctk/-ctv: "q8_0", "f16", ...
        "cache_ram_mib": 256,     # llama-server --cache-ram: saved prompt states kept in RAM (0 = none)
        "prefix_cache": True,     # save the system prompt + tools KV state to disk and restore it on launch
        "sleep_after": 600,       # keep_running only: seconds idle before llama-server unloads the model
        "cache_reuse": 256,       # llama-server --cache-reuse chunk size (0 = off)
    },
    "agent": {
        "workdir": "",
        "max_steps": 12,
        "tool_output_chars": 2000,
        "shell_timeout": 60,
        "keep_sessions": 50,
    },
    "safety": {"allow": [], "sudo": "ask", "write_outside_workdir": False},
    "persona": {"name": "Xaren", "user_name": "", "vibe": "friendly", "emoji": True},
    "ui": {"accent": "#0a84ff", "share_dir": "~/Documents/XarenCode", "header": "full", "mascot": True,
           "theme": "dark", "timestamps": False, "sidebar": False},
    "dev": {},
    "vm": {"backend": "qemu", "max_memory": "3G", "machines": {}},
}

TEMPLATE = '''\
# XarenCode configuration. Delete this file to get a fresh copy.

[model]
# The ONE line to change when swapping models: "owner/repo/file" or an absolute path.
gguf = "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
# gguf = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
hf_base = "https://huggingface.co"
storage_budget_gb = 5.0     # `xarencode pull` refuses to exceed this across all models
temperature = 0.7
top_p = 0.8
top_k = 20
max_tokens = 1024
thinking = false

[server]
manage = true               # start and stop llama-server for you
binary = "llama-server"
host = "127.0.0.1"
port = 8091
ctx_size = 4096
threads = 4
threads_batch = 6
args = ["--jinja", "-np", "1"]
keep_running = false        # true keeps the server (and its RAM) alive after exit
# Speed tuning (llama.cpp flags are skipped automatically if your build lacks them):
repack = false              # true = llama.cpp's default weight repacking (slower here, uses more RAM)
kv_cache = "q8_0"           # KV cache type for -ctk/-ctv ("f16" for the old behaviour)
cache_ram_mib = 256         # --cache-ram: MiB of saved prompt states kept in RAM
prefix_cache = true         # restore the system prompt + tools from a saved KV file at launch
sleep_after = 600           # with keep_running: seconds idle before the model is unloaded
cache_reuse = 256           # --cache-reuse chunk size, 0 = off

[agent]
workdir = ""                # empty means the folder you launch from
max_steps = 12
tool_output_chars = 2000
shell_timeout = 60
keep_sessions = 50

[safety]
allow = []                  # extra read-only command prefixes, e.g. ["npm test"]
sudo = "ask"                # "ask" or "never"
write_outside_workdir = false
# yolo is not a config setting: it can only be switched on per run (--yolo / -yolo) or per session (/yolo on)

[persona]
name = "Xaren"
user_name = ""
vibe = "friendly"           # friendly, hype, chill or focused
emoji = true

[ui]
accent = "#0a84ff"
share_dir = "~/Documents/XarenCode"
# header = "full"    # "full", "small" or "off": the cat and wordmark at the top
# mascot = true      # the small blue cat; false removes it
# theme = "dark"     # "dark", "light", "contrast" or "mono" (NO_COLOR forces mono)
# timestamps = false # dim right-aligned times on chat messages
# sidebar = false    # open the sidebar (Ctrl+B) at startup

# Dev server presets (uncomment the ones you want):
#
# [dev.xaren-workspace]
# dir = "~/Projects/xaren-workspace"
# cmd = "npm run dev -- --port 5173"
# url = "http://localhost:5173"
#
# [dev.hand-galaxy]
# dir = "~/Projects/hand-galaxy"
# cmd = "npm run dev -- --port 5174"
# url = "http://localhost:5174"
#
# [dev.naja7]
# dir = "~/Projects/naja7-tcsf.vercel.app"
# cmd = "python -m http.server 8001 --bind 127.0.0.1"
# url = "http://localhost:8001"
#
# [dev.retrorcade]
# dir = "~/Projects/retrorcade.netlify.app"
# cmd = "python -m http.server 8002 --bind 127.0.0.1"
# url = "http://localhost:8002"
#
# [dev.xarenchat]
# dir = "~/Projects/xarenchat.pages.dev"
# cmd = "python -m http.server 8003 --bind 127.0.0.1"
# url = "http://localhost:8003"
#
# [dev.xarenspace]
# dir = "~/Projects/xarenspace.pages.dev"
# cmd = "python -m http.server 8004 --bind 127.0.0.1"
# url = "http://localhost:8004"

[vm]
backend = "qemu"
max_memory = "3G"
# No machines are configured. Example:
# [vm.machines.arch-test]
# disk = "~/vms/arch-test.qcow2"
# iso = ""
# memory = "2G"
# cpus = 2
# uefi = true
# display = "gtk"           # or "none"
# ssh_port = 0
'''


def _xdg(var: str, fallback: str) -> Path:
    base = os.environ.get(var)
    return Path(base) if base else Path.home() / fallback


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "xarencode"


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "xarencode"


def state_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state") / "xarencode"


def config_path() -> Path:
    return config_dir() / "config.toml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def default_config() -> dict:
    return copy.deepcopy(DEFAULTS)


def load_config(path: str | Path | None = None) -> dict:
    """Load config, writing the commented template on first run of the default path."""
    explicit = path is not None
    p = Path(path) if explicit else config_path()
    if not p.exists():
        if explicit:
            raise FileNotFoundError(f"config not found: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(TEMPLATE, encoding="utf-8")
    with open(p, "rb") as f:
        user = tomllib.load(f)
    return _merge(DEFAULTS, user)


def parse_gguf(spec: str) -> tuple[str, str, str] | None:
    """'owner/repo/file' -> parts, or None for an absolute path."""
    if os.path.isabs(os.path.expanduser(spec)):
        return None
    parts = spec.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"model.gguf must be 'owner/repo/file' or an absolute path, got {spec!r}")
    return parts[0], parts[1], parts[2]


def resolve_model(cfg: dict) -> Path:
    spec = cfg["model"]["gguf"]
    parts = parse_gguf(spec)
    if parts is None:
        return Path(os.path.expanduser(spec))
    return data_dir() / "models" / parts[2]


def model_short_name(cfg: dict) -> str:
    stem = Path(cfg["model"]["gguf"]).name
    stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
    keep = []
    for tok in stem.split("-"):
        if tok.lower().startswith(("instruct", "chat", "q4", "q5", "q6", "q8", "iq")) or tok.isdigit() and len(tok) == 4:
            break
        keep.append(tok)
    return "-".join(keep) or stem
