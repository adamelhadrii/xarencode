"""Plain frontend for -p mode: stdlib only, writes the answer to stdout, tool activity to stderr."""
from __future__ import annotations

import sys

from .agent import Approval, BaseFrontend


class PlainFrontend(BaseFrontend):
    def __init__(self, out=None, err=None, interactive: bool | None = None):
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.had_error = False

    def text(self, d):
        self.out.write(d)
        self.out.flush()

    def end_stream(self):
        pass

    def approve(self, call, plan):
        if not self.interactive:
            return Approval("feedback", "approval is unavailable in non-interactive mode; use a read-only approach or tell the user what to run")
        if plan.verdict.name == "ELEVATED":
            self.err.write(f"\n{call.name}: {plan.reason}\n{plan.preview}\nThis needs administrator rights. Type 'sudo' to allow: ")
            self.err.flush()
            return Approval("yes" if input().strip() == "sudo" else "no")
        tag = f" [{plan.tag}]" if plan.tag else ""
        self.err.write(f"\n{call.name}{tag}: {plan.reason}\n{plan.preview}\nAllow? [y/N] ")
        self.err.flush()
        return Approval("yes" if input().strip().lower() in ("y", "yes") else "no")

    def tool_start(self, call, plan):
        arg = next((str(v) for v in call.args.values()), "")
        self.err.write(f"\n[{call.name}] {arg[:80]}\n")

    def tool_done(self, call, result):
        self.err.write(f"  -> {result.summary or result.content[:80]}\n")

    def notice(self, level, text):
        self.had_error = self.had_error or level == "error"
        self.err.write(f"[{level}] {text}\n")
