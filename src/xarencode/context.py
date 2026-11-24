"""Token budgeting, history trimming (in place, in big steps) and /compact."""
from __future__ import annotations

import json
import math

COMPACT_PROMPT = (
    "Summarize the conversation so far in at most 120 words. Cover the user's goals, decisions made, "
    "files touched and open TODOs. Plain text, no preamble."
)
TRIM_NOTE = "[earlier messages trimmed — /compact to summarize]"
RESERVE_REPLY = 768
MARGIN = 64


def msg_chars(m: dict) -> int:
    n = len(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        n += len(fn.get("name", "")) + len(fn.get("arguments", ""))
    return n


def system_parts(system) -> tuple[str, str]:
    """(static, dynamic) from a (static, dynamic) pair or a persona.SystemPrompt; a plain string has no dynamic part."""
    if isinstance(system, tuple):
        return system[0], system[1]
    static = getattr(system, "static", None)
    return (static, system.dynamic) if static is not None else (str(system), "")


def system_text(system) -> str:
    return "\n".join(p for p in system_parts(system) if p)


class ContextManager:
    def __init__(self, ctx_size: int):
        self.ctx_size = ctx_size
        self.single_system = False  # set when the server's template rejects two system messages
        self.ratio = 1.0
        self._last_raw = 0
        self.last_est = 0
        self.last_trim_note = False

    @property
    def budget(self) -> int:
        return self.ctx_size - RESERVE_REPLY - MARGIN

    @staticmethod
    def raw_tokens(msgs: list[dict]) -> int:
        return sum(math.ceil(msg_chars(m) / 3) + 6 for m in msgs)

    def system_messages(self, system) -> list[dict]:
        static, dynamic = system_parts(system)
        if self.single_system or not dynamic:
            return [{"role": "system", "content": system_text(system)}]
        return [{"role": "system", "content": static}, {"role": "system", "content": dynamic}]

    def estimate(self, system, tools: list[dict], history: list[dict]) -> int:
        raw = self.raw_tokens(history) + math.ceil(len(system_text(system)) / 3) + 6 + math.ceil(len(json.dumps(tools)) / 3)
        self._last_raw = raw
        return math.ceil(raw * self.ratio)

    def observe(self, prompt_tokens: int | None) -> None:
        """Calibrate the estimate against the server's real prompt token count."""
        if prompt_tokens and self._last_raw:
            sample = prompt_tokens / self._last_raw
            self.ratio = min(1.6, max(0.6, 0.7 * self.ratio + 0.3 * sample))

    def fit(self, system, tools: list[dict], history: list[dict]) -> tuple[list[dict], int]:
        """Trim `history` in place when over budget; returns (messages incl. system, estimate)."""
        est = self.estimate(system, tools, history)
        if est > self.budget:
            self.trim(system, tools, history, 0.6)
            est = self.estimate(system, tools, history)
        self.last_est = est
        return self.system_messages(system) + history, est

    def force_trim(self, system, tools: list[dict], history: list[dict]) -> int:
        """After a server-side overflow: assume our estimate was too low and trim harder."""
        self.ratio = min(1.6, self.ratio * 1.25)
        self.trim(system, tools, history, 0.35)
        self.last_est = self.estimate(system, tools, history)
        return self.last_est

    def reply_tokens(self, max_tokens: int, est: int) -> int:
        return max(64, min(max_tokens, self.ctx_size - est - 32))

    # -- trimming
    def trim(self, system, tools: list[dict], history: list[dict], fraction: float) -> None:
        target = int(self.budget * fraction)
        over = lambda: self.estimate(system, tools, history) > target  # noqa: E731
        self._elide_old_tool_results(history, over)
        if over():
            self._drop_old_exchanges(history, over)
        if over():
            self._shorten_large(history, over)

    @staticmethod
    def _tool_labels(history: list[dict]) -> dict[str, str]:
        labels = {}
        for m in history:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    first = next((str(v) for v in args.values() if isinstance(v, str)), "")
                except (ValueError, AttributeError):
                    first = ""
                labels[tc.get("id", "")] = f"{fn.get('name', 'tool')} {first[:60]}".strip()
        return labels

    def _elide_old_tool_results(self, history, over) -> None:
        tool_idx = [i for i, m in enumerate(history) if m.get("role") == "tool"]
        labels = self._tool_labels(history)
        for i in tool_idx[:-2]:
            m = history[i]
            if (m.get("content") or "").startswith("[elided"):
                continue
            m["content"] = f"[elided: {labels.get(m.get('tool_call_id', ''), 'tool result')}]"
            if not over():
                return

    def _drop_old_exchanges(self, history, over) -> None:
        def starts():
            return [i for i, m in enumerate(history) if m.get("role") == "user" and m.get("content") != TRIM_NOTE]

        while over():
            s = starts()
            if len(s) < 2:
                return
            # drop everything before the second real user message, keeping the trim note out
            cut = s[1]
            del history[:cut]
            history.insert(0, {"role": "user", "content": TRIM_NOTE})
            if len(history) > 1 and history[1].get("content") == TRIM_NOTE:
                del history[1]

    def _shorten_large(self, history, over) -> None:
        for m in sorted(history, key=lambda m: history.index(m)):
            c = m.get("content") or ""
            if len(c) > 1500:
                m["content"] = c[:700] + "\n[... trimmed ...]\n" + c[-500:]
                if not over():
                    return


def render_transcript_text(history: list[dict], limit: int = 6000) -> str:
    """Plain transcript for summarizing; tool results are cut short."""
    parts = []
    for m in history:
        role = m.get("role")
        c = (m.get("content") or "").strip()
        if role == "tool":
            c = c[:300]
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            c += f" [called {fn.get('name')} {fn.get('arguments', '')[:120]}]"
        parts.append(f"{role}: {c}")
    text = "\n".join(parts)
    return text if len(text) <= limit else text[:limit // 3] + "\n[...]\n" + text[-(2 * limit // 3):]


def compact_history(client, history: list[dict], sampling: dict, focus: str = "") -> str:
    prompt = COMPACT_PROMPT + (f" Focus on: {focus}." if focus else "")
    msgs = [{"role": "system", "content": "You write short, factual summaries."},
            {"role": "user", "content": f"{prompt}\n\n---\n{render_transcript_text(history)}"}]
    return client.complete(msgs, max_tokens=300, sampling=sampling)
