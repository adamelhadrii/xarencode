"""The agent loop: model stream -> tool calls -> safety -> approval -> result -> repeat."""
from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from . import persona
from .context import ContextManager
from .llm import (ContextOverflow, Done, Interrupted, LlamaClient, LlmError, ReasoningDelta, ServerError, ServerUnavailable,
                  PromptProgress, TextDelta, ToolCallDelta)
from .safety import ALLOW, ASK, BLOCK, ELEVATED
from .toolcall import CallAccumulator, ToolCall
from .tools import Plan, ToolContext, ToolRegistry, ToolResult


@dataclass
class Approval:
    kind: str  # "yes" | "always" | "no" | "feedback"
    text: str = ""


@dataclass
class TurnStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tok_per_s: float = 0.0
    ctx_used: int = 0
    ctx_size: int = 0


class Frontend(Protocol):
    def begin_stream(self) -> None: ...
    def text(self, d: str) -> None: ...
    def reasoning(self, d: str) -> None: ...
    def tool_preparing(self, name: str, nbytes: int) -> None: ...
    def end_stream(self) -> None: ...
    def approve(self, call: ToolCall, plan: Plan) -> Approval: ...
    def tool_start(self, call: ToolCall, plan: Plan) -> None: ...
    def tool_output(self, line: str) -> None: ...
    def tool_done(self, call: ToolCall, result: ToolResult) -> None: ...
    def notice(self, level: str, text: str) -> None: ...
    def stats(self, s: TurnStats) -> None: ...


class BaseFrontend:
    """No-op defaults so frontends only override what they show."""

    def begin_stream(self): pass
    def text(self, d): pass
    def reasoning(self, d): pass
    def tool_preparing(self, name, nbytes): pass
    def end_stream(self): pass
    def approve(self, call, plan): return Approval("no")
    def tool_start(self, call, plan): pass
    def tool_output(self, line): pass
    def tool_done(self, call, result): pass
    def notice(self, level, text): pass
    def stats(self, s): pass


def cap_output(text: str, limit: int) -> str:
    lines = text.split("\n")
    if len(lines) > 50:
        text = "\n".join(lines[:15] + [f"[... {len(lines) - 50} lines omitted]"] + lines[-35:])
    if len(text) > limit:
        keep_head, keep_tail = int(limit * 0.4), int(limit * 0.6)
        text = text[:keep_head] + f"\n[... {len(text) - limit} chars omitted]\n" + text[-keep_tail:]
    return text


class Agent:
    def __init__(self, cfg: dict, client: LlamaClient, registry: ToolRegistry, frontend: Frontend, system,
                 workdir: Path, history: list[dict] | None = None, sink: Callable[[dict, dict | None], None] | None = None,
                 stats: persona.SessionStats | None = None, yolo: bool = False, prefix_cache=None):
        self.cfg = cfg
        self.client = client
        self.registry = registry
        self.frontend = frontend
        self.system = system
        self.workdir = Path(workdir)
        self.history: list[dict] = history if history is not None else []
        self.sink = sink
        self.session_stats = stats or persona.SessionStats()
        self.ctx = ContextManager(cfg["server"]["ctx_size"])
        self.prefix_cache = prefix_cache
        self.approvals: set[str] = set()
        self._call_ids: set[str] = set()  # every tool call id used in this session, so ids never repeat in history
        self.auto_edits = False
        # yolo: answer every ASK with "yes". Set only by the command-line flag or /yolo; never read from the config, the
        # environment or a tool argument.
        self.yolo = bool(yolo)
        self.cancelled = threading.Event()
        self.last_stats = TurnStats(ctx_size=cfg["server"]["ctx_size"])
        m = cfg["model"]
        self.sampling = {k: m[k] for k in ("temperature", "top_p", "top_k")} | {"thinking": m["thinking"]}

    # -- control
    def interrupt(self) -> None:
        self.cancelled.set()
        self.client.cancel()

    def tool_context(self) -> ToolContext:
        fe = self.frontend
        return ToolContext(self.workdir, self.cfg, fe.tool_output, self.cancelled, getattr(fe, "suspend_live", nullcontext))

    def _add(self, msg: dict, ui: dict | None = None) -> None:
        self.history.append(msg)
        if self.sink:
            self.sink(msg, ui)

    def add_user_note(self, text: str) -> None:
        self._add({"role": "user", "content": text})

    # -- the loop
    def run_turn(self, user_text: str) -> None:
        fe = self.frontend
        self.cancelled.clear()
        self._add({"role": "user", "content": user_text})
        self.session_stats.turns += 1
        totals = TurnStats(ctx_size=self.ctx.ctx_size)
        steps = errors = 0
        last_sig, repeats = None, 0
        tools = self.registry.schemas()
        t0 = time.monotonic()
        try:
            if self.prefix_cache is not None:
                self.prefix_cache.ensure()
                if self.cancelled.is_set():
                    return
            while True:
                msgs, est = self.ctx.fit(self.system, tools, self.history)
                result = self._stream(msgs, est, tools, totals)
                if result is None:
                    return
                content, calls, finish = result
                self._add({"role": "assistant", "content": content or "",
                           **({"tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.history_args}}
                                              for c in calls]} if calls else {})})
                if not calls:
                    return
                first_reply = len(self.history)  # tool replies for these calls land after this index
                steps += 1
                if steps > self.cfg["agent"]["max_steps"]:
                    for c in calls:
                        self._add_tool(c, "ERROR: stopped: step limit reached")
                    fe.notice("warn", f"Stopped after {steps - 1} tool rounds (agent.max_steps). Say 'continue' to go on.")
                    return
                try:
                    for i, call in enumerate(calls):
                        msg, ended, errors, last_sig, repeats = self._handle_call(call, errors, last_sig, repeats)
                        if ended or self.cancelled.is_set():
                            for rest in calls[i + 1:]:
                                self._add_tool(rest, "skipped")
                            return
                except KeyboardInterrupt:
                    # every tool_calls entry needs a tool message, or the next request is rejected
                    self.cancelled.set()
                    answered = {m.get("tool_call_id") for m in self.history[first_reply:] if m.get("role") == "tool"}
                    for c in calls:
                        if c.id not in answered:
                            self._add_tool(c, "ERROR: interrupted")
                    fe.notice("warn", "interrupted")
                    return
        finally:
            if totals.completion_tokens or totals.prompt_tokens:
                dt = time.monotonic() - t0
                if not totals.tok_per_s and dt > 0:
                    totals.tok_per_s = totals.completion_tokens / dt
                totals.ctx_used = self.ctx.last_est
                self.last_stats = totals
                self.session_stats.tokens += totals.completion_tokens
                fe.stats(totals)

    def _stream(self, msgs, est, tools, totals: TurnStats):
        fe = self.frontend
        for attempt in (0, 1):
            acc = CallAccumulator()
            parts: list[str] = []
            finish = None
            usage = timings = None
            deltas = 0
            started = time.monotonic()
            first = None
            fe.begin_stream()
            on_progress = getattr(fe, "prompt_progress", None)  # optional hook: a frontend that shows prompt-reading progress
            try:
                max_tokens = self.ctx.reply_tokens(self.cfg["model"]["max_tokens"], est)
                for ev in self.client.stream_chat(msgs, tools, max_tokens=max_tokens, sampling=self.sampling, progress=on_progress is not None):
                    if isinstance(ev, PromptProgress):
                        on_progress(ev)
                    elif isinstance(ev, TextDelta):
                        parts.append(ev.text)
                        fe.text(ev.text)
                        deltas += 1
                        first = first or time.monotonic()
                    elif isinstance(ev, ReasoningDelta):
                        fe.reasoning(ev.text)
                    elif isinstance(ev, ToolCallDelta):
                        acc.feed(ev)
                        fe.tool_preparing(acc._parts[ev.index]["name"], acc.pending_bytes())
                        first = first or time.monotonic()
                        deltas += 1 if ev.arguments else 0
                    elif isinstance(ev, Done):
                        finish = ev.finish_reason or finish
                        usage = ev.usage or usage
                        timings = ev.timings or timings
            except Interrupted:
                fe.end_stream()
                partial = "".join(parts)
                self._add({"role": "assistant", "content": (partial + " [interrupted]").strip()})
                fe.notice("warn", "interrupted")
                return None
            except ContextOverflow:
                fe.end_stream()
                if attempt == 1:
                    fe.notice("error", "The conversation no longer fits in the context window. Try /compact or /clear.")
                    return None
                fe.notice("warn", "Context was full; trimming older messages and retrying.")
                est = self.ctx.force_trim(self.system, tools, self.history)
                msgs = self.ctx.system_messages(self.system) + self.history
                continue
            except (ServerUnavailable, ServerError) as e:
                fe.end_stream()
                if (isinstance(e, ServerError) and "system" in str(e).lower() and not self.ctx.single_system
                        and len(msgs) > 1 and msgs[1].get("role") == "system" and attempt == 0):
                    self.ctx.single_system = True  # the chat template wants one system message: send them joined
                    msgs = self.ctx.system_messages(self.system) + self.history
                    continue
                fe.notice("error", str(e))
                return None
            except KeyboardInterrupt:
                fe.end_stream()
                self.cancelled.set()
                self.client.cancel()
                self._add({"role": "assistant", "content": ("".join(parts) + " [interrupted]").strip()})
                fe.notice("warn", "interrupted")
                return None
            fe.end_stream()
            if usage:
                self.ctx.observe(usage.get("prompt_tokens"))
                totals.prompt_tokens = usage.get("prompt_tokens", 0)
                totals.completion_tokens += usage.get("completion_tokens", 0)
            else:
                totals.completion_tokens += deltas
            if timings and timings.get("predicted_per_second"):
                totals.tok_per_s = timings["predicted_per_second"]
            elif first and deltas:
                totals.tok_per_s = deltas / max(time.monotonic() - first, 1e-3)
            content, calls = acc.finish("".join(parts), finish, self.registry)
            self._unique_ids(calls)
            return content, calls, finish
        return None

    def _unique_ids(self, calls: list[ToolCall]) -> None:
        """Text-fallback calls are numbered call_1... every time, and some servers reuse ids; make each id unique."""
        used = self._call_ids | {tc.get("id") for m in self.history for tc in m.get("tool_calls") or []}
        n = len(used)
        for c in calls:
            if c.id in used:
                n += 1
                while f"call_{n}" in used:
                    n += 1
                c.id = f"call_{n}"
            used.add(c.id)
        self._call_ids = used

    def _add_tool(self, call: ToolCall, text: str, ui: dict | None = None) -> None:
        self._add({"role": "tool", "tool_call_id": call.id, "content": text}, ui)

    def _handle_call(self, call: ToolCall, errors: int, last_sig, repeats):
        """Returns (message, end_turn, errors, last_sig, repeats)."""
        fe = self.frontend
        self.session_stats.tool_calls += 1

        def finish_error(text: str):
            nonlocal errors
            errors += 1
            self._add_tool(call, text)
            fe.tool_done(call, ToolResult(False, text, text))
            if errors >= 3:
                fe.notice("warn", "Three tool errors in a row; stopping this turn.")
                return text, True, errors, last_sig, repeats
            return text, False, errors, last_sig, repeats

        if call.error:
            return finish_error(f"ERROR: {call.error}")
        sig = (call.name, json.dumps(call.args, sort_keys=True))
        repeats = repeats + 1 if sig == last_sig else 1
        last_sig = sig
        if repeats >= 3:
            return finish_error("ERROR: repeated call; try another approach or answer")
        tool = self.registry.get(call.name)
        tctx = self.tool_context()
        try:
            plan = tool.plan(call.args, tctx)
        except Exception as e:  # a tool bug must not kill the session
            return finish_error(f"ERROR: {call.name} failed while planning: {e}")
        if plan.verdict == BLOCK:
            text = f"BLOCKED: {plan.reason}"
            self._add_tool(call, text)
            fe.tool_done(call, ToolResult(False, text, text))
            return text, False, errors, last_sig, repeats
        feedback = ""
        needs_ask = plan.verdict in (ASK, ELEVATED)
        # downloads, deletes and writes outside the working directory are asked every time: no remembered key, no auto-edits
        remembered = plan.verdict == ASK and not plan.tag and not plan.hard_ask
        if remembered and plan.approval_key and plan.approval_key in self.approvals:
            needs_ask = False
        if remembered and self.auto_edits and call.name in ("write_file", "edit_file"):
            needs_ask = False
        if plan.verdict == ASK and self.yolo and not plan.hard_ask:
            needs_ask = False  # BLOCK returned above, ELEVATED never gets here: both stay as they are in yolo
        if needs_ask:
            ap = fe.approve(call, plan)
            if ap.kind == "no":
                text = "User declined"
                self._add_tool(call, text)
                fe.tool_done(call, ToolResult(False, text, text))
                return text, True, errors, last_sig, repeats
            if ap.kind == "always" and plan.approval_key and plan.verdict == ASK and not plan.tag and not plan.hard_ask:
                self.approvals.add(plan.approval_key)
            if ap.kind == "feedback":
                text = f"User declined and said: {ap.text}"
                self._add_tool(call, text)
                fe.tool_done(call, ToolResult(False, text, text))
                return text, False, errors, last_sig, repeats
        fe.tool_start(call, plan)
        try:
            res = tool.run(call.args, tctx, plan)
        except Exception as e:
            res = ToolResult(False, f"ERROR: {call.name} crashed: {e}", str(e))
        fe.tool_done(call, res)
        text = cap_output(res.content, self.cfg["agent"]["tool_output_chars"])
        ui = {"summary": res.summary, "diff": res.detail.get("diff"), "tool": call.name}
        self._add_tool(call, text, ui)
        errors = errors + 1 if text.startswith("ERROR:") else 0
        event = res.detail.get("event")
        if event == "edit_applied":
            self.session_stats.edits += 1
        if event and res.ok:
            line = persona.celebrate(event, self.cfg)
            if line:
                fe.notice("celebrate", line)
        if errors >= 3:
            fe.notice("warn", "Three tool errors in a row; stopping this turn.")
            return text, True, errors, last_sig, repeats
        return text, False, errors, last_sig, repeats
