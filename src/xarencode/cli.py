"""Command line entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .config import config_path, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xarencode", description="A lean terminal coding agent for a local llama.cpp model.",
                                allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"xarencode {__version__}")
    p.add_argument("-p", "--print", dest="prompt", metavar="TEXT", help="run one prompt non-interactively and print the answer")
    p.add_argument("--resume", nargs="?", const="last", metavar="ID", help="continue the last (or the given) session")
    p.add_argument("--workdir", metavar="DIR", help="folder to work in (default: the current folder)")
    p.add_argument("--yolo", "-yolo", dest="yolo", action="store_true",
                   help="auto-approve every ask for this run (hard blocks and sudo still apply)")
    p.add_argument("--fullscreen", action="store_true", help=argparse.SUPPRESS)  # the full-window view is the only view; accepted as a no-op
    p.add_argument("--config", metavar="FILE", help="config file (default: ~/.config/xarencode/config.toml)")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("pull", help="download the model configured in model.gguf")
    sub.add_parser("doctor", help="check the setup")
    srv = sub.add_parser("server", help="manage llama-server")
    srv.add_argument("action", choices=["start", "stop", "status", "log"])
    return p


def resolve_workdir(args, cfg) -> Path | None:
    raw = args.workdir or cfg["agent"]["workdir"] or os.getcwd()
    wd = Path(os.path.realpath(os.path.expanduser(raw)))
    if not wd.is_dir():
        print(f"xarencode: {wd} is not a folder", file=sys.stderr)
        return None
    if wd == Path("/"):
        print("xarencode: refusing to work in /", file=sys.stderr)
        return None
    return wd


def cmd_doctor(cfg) -> int:
    from .server import doctor
    marks = {"ok": "✔", "warn": "!", "fail": "✖"}
    results = doctor(cfg)
    for status, text in results:
        print(f" {marks[status]} {text}")
    return 1 if any(s == "fail" for s, _ in results) else 0


def cmd_pull(cfg) -> int:
    from .server import PullError, pull_model

    def confirm(q: str) -> bool:
        return sys.stdin.isatty() and input(f"{q} [y/N] ").strip().lower() in ("y", "yes")

    try:
        pull_model(cfg, lambda s: print(s, file=sys.stderr), confirm)
    except PullError as e:
        print(f"xarencode: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_server(cfg, action: str) -> int:
    from .server import LlamaServer, ServerStartError
    srv = LlamaServer(cfg)
    try:
        if action == "status":
            print(srv.status())
        elif action == "start":
            print("started" if srv.ensure(lambda s: None) else "already running")
        elif action == "stop":
            print(srv.stop())
        else:
            print(srv.log_tail(40))
    except ServerStartError as e:
        print(f"xarencode: {e}", file=sys.stderr)
        return 1
    return 0


def run_print(cfg, wd: Path, prompt: str, config_file: Path | None, yolo: bool = False) -> int:
    from . import persona, sessions
    from .agent import Agent
    from .plain import PlainFrontend
    from .safety import YOLO_NOTICE
    from .server import LlamaServer, ServerStartError
    from .tools import build_registry

    srv = LlamaServer(cfg)
    try:
        srv.ensure()
    except ServerStartError as e:
        print(f"xarencode: {e}", file=sys.stderr)
        return 1
    try:
        fe = PlainFrontend()
        if yolo:
            fe.notice("warn", YOLO_NOTICE)
        session = sessions.Session.new(cfg, wd)
        agent = Agent(cfg, srv.client, build_registry(cfg), fe, persona.system_prompt(cfg, wd), wd, sink=session.append, yolo=yolo)
        agent.run_turn(prompt)
        sys.stdout.write("\n")
        sessions.prune(cfg["agent"]["keep_sessions"])
        return 1 if fe.had_error else 0
    finally:
        if srv.started_by_us and not cfg["server"]["keep_running"]:
            srv.stop()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"xarencode: {e}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as e:  # includes tomllib.TOMLDecodeError
        print(f"xarencode: bad config: {e}", file=sys.stderr)
        return 2
    if "yolo" in cfg["safety"]:
        cfg["safety"].pop("yolo")  # yolo is never read from the config: only --yolo or /yolo turn it on
        print("xarencode: [safety] yolo in the config is ignored; use --yolo for one run or /yolo on for one session", file=sys.stderr)
    if args.command == "doctor":
        return cmd_doctor(cfg)
    if args.command == "pull":
        return cmd_pull(cfg)
    if args.command == "server":
        return cmd_server(cfg, args.action)
    wd = resolve_workdir(args, cfg)
    if wd is None:
        return 2
    cfg_file = Path(args.config) if args.config else config_path()
    if args.prompt is not None:
        return run_print(cfg, wd, args.prompt, cfg_file, args.yolo)
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM", "") in ("", "dumb"):
        print('XarenCode needs a real terminal; use `xarencode -p "prompt"` for one-shot output', file=sys.stderr)
        return 2
    try:
        from . import ui
    except ImportError as e:
        print(f"xarencode: the chat UI needs rich and prompt_toolkit ({e.name} is missing).\n"
              "  sudo pacman -S python-rich python-prompt_toolkit   (or: pip install rich prompt_toolkit)", file=sys.stderr)
        return 1
    return ui.run_chat(cfg, wd, args.resume, cfg_file, yolo=args.yolo)
