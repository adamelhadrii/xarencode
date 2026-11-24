"""The Xaren persona: vibes, system prompt, greetings, spinner verbs, celebrations and session stats."""
from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, field
from pathlib import Path

VIBES = {
    "friendly": "Be warm and encouraging; keep replies short.",
    "hype": "Be upbeat and playful; celebrate wins; keep replies short.",
    "chill": "Be relaxed and casual; keep replies very short.",
    "focused": "Be terse and precise; no small talk.",
}
VERBS = ["Cooking", "Brewing", "Noodling", "Vibing", "Scheming", "Conjuring", "Tinkering", "Pondering",
         "Simmering", "Hatching", "Doodling", "Whittling", "Percolating", "Spelunking"]
CELEBRATIONS = {
    "tests_passed": ["Tests are green. Nice work!", "All tests pass. Ship it!", "Green across the board."],
    "server_up": ["Server is up and running.", "Dev server is live. Go take a look!", "It's alive!"],
    "edit_applied": ["Edit applied.", "Change landed.", "Done and dusted."],
}
EMOJI = {"tests_passed": "🎉", "server_up": "🚀", "edit_applied": "✨"}


def vibe_names() -> list[str]:
    return list(VIBES)


class SystemPrompt(str):
    """The joined system prompt that remembers its stable part (`static`) and per-session part (`dynamic`)."""
    static: str
    dynamic: str

    def __new__(cls, static: str, dynamic: str):
        obj = super().__new__(cls, f"{static}\n{dynamic}")
        obj.static, obj.dynamic = static, dynamic
        return obj


def static_system(cfg: dict) -> str:
    """Name and rules only: identical on every launch, so its KV state can be cached on disk."""
    return (
        f"You are {cfg['persona']['name']}, a coding assistant in the user's terminal.\n"
        "Use tools to look before you act: read files before editing, prefer edit_file for changes, "
        "run tests after changing code. Never run destructive commands. Ask if a request is unclear."
    )


def dynamic_system(cfg: dict, workdir: Path | str) -> str:
    p = cfg["persona"]
    vibe = VIBES.get(p["vibe"], VIBES["friendly"])
    who = f"The user is {p['user_name']}. " if p.get("user_name") else ""
    return f"{who}Working directory: {workdir}.\n{vibe}"


def system_prompt(cfg: dict, workdir: Path | str) -> SystemPrompt:
    return SystemPrompt(static_system(cfg), dynamic_system(cfg, workdir))


def time_of_day(now: _dt.datetime | None = None) -> str:
    h = (now or _dt.datetime.now()).hour
    return "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 18 else "evening" if 18 <= h < 23 else "night"


def greeting(cfg: dict, now: _dt.datetime | None = None) -> str:
    p = cfg["persona"]
    part = time_of_day(now)
    who = f", {p['user_name']}" if p.get("user_name") else ""
    text = {
        "morning": f"Good morning{who}.",
        "afternoon": f"Good afternoon{who}.",
        "evening": f"Good evening{who}.",
        "night": f"Burning the midnight oil{who}?",
    }[part]
    tail = {"hype": " Let's build something great!", "chill": " Take it easy.", "focused": " What are we shipping?"}.get(p["vibe"], " What are we working on?")
    return text + tail + (" 👋" if p.get("emoji") else "")


def spinner_verb(rng: random.Random | None = None) -> str:
    return (rng or random).choice(VERBS)


def celebrate(event: str, cfg: dict, rng: random.Random | None = None) -> str:
    lines = CELEBRATIONS.get(event)
    if not lines:
        return ""
    text = (rng or random).choice(lines)
    if cfg["persona"].get("emoji"):
        text = f"{EMOJI[event]} {text}"
    return text


@dataclass
class SessionStats:
    started: float = field(default_factory=lambda: _dt.datetime.now().timestamp())
    turns: int = 0
    tool_calls: int = 0
    edits: int = 0
    tokens: int = 0

    def minutes(self) -> float:
        return (_dt.datetime.now().timestamp() - self.started) / 60


def farewell(stats: SessionStats, cfg: dict) -> str:
    name = cfg["persona"]["name"]
    bits = [f"{stats.turns} turn{'s' if stats.turns != 1 else ''}", f"{stats.tool_calls} tool call{'s' if stats.tool_calls != 1 else ''}"]
    if stats.edits:
        bits.append(f"{stats.edits} edit{'s' if stats.edits != 1 else ''}")
    if stats.tokens:
        bits.append(f"{stats.tokens} tokens")
    bits.append(f"{stats.minutes():.0f} min")
    tail = "See you next time!" if cfg["persona"]["vibe"] != "focused" else "Bye."
    return f"{name}: session wrapped, {' · '.join(bits)}. {tail}"
