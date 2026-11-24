"""The XarenCode mascot: detailed blue pixel cat, block-letter wordmark, animation state machine and header layout.

Standard library only. Everything that depends on time takes the time as an argument, so tests can drive a fake clock.
A "Line" is one screen row of prompt_toolkit style fragments: a list of (style, text) pairs.
"""
from __future__ import annotations

import colorsys
import os
import random
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math

from . import mascot_art, persona
from .config import model_short_name

Line = list[tuple[str, str]]

CAT_W, CAT_H = mascot_art.W, mascot_art.H // 2        # 22 x 10 cells (22 x 20 pixels)
SMALL_CAT_W, SMALL_CAT_H = mascot_art.SW, mascot_art.SH // 2  # 12 x 6 cells (12 x 12 pixels)
DEFAULT_ACCENT = "#0a84ff"
YOLO_STYLE = "bold fg:#ffffff bg:#d70040"
YOLO_TEXT = " auto-approve is on. Hard blocks and sudo still apply."
INLINE_TIPS = "/help for commands · ! runs a shell command · @file attaches a file · esc interrupts"
FULL_TIPS = "/help · ! shell · @file · Esc stop · PgUp/Dn · Ctrl+B sidebar · F2 select · F3 pet"

CAT = mascot_art.FRAMES
GLYPHS = mascot_art.GLYPHS
TINY = mascot_art.TINY

# ---------------------------------------------------------------- the wordmark

WORD: dict[str, tuple[str, str, str, str, str]] = {
    "X": ("█   █", " █ █ ", "  █  ", " █ █ ", "█   █"),
    "A": (" ▄█▄ ", "█   █", "█▀▀▀█", "█   █", "█   █"),
    "R": ("█▀▀▀▄", "█   █", "█▀▀█▀", "█  ▀▄", "█   █"),
    "E": ("█▀▀▀▀", "█    ", "█▀▀▀ ", "█    ", "█▄▄▄▄"),
    "N": ("█▄  █", "█▀▄ █", "█ ▀▄█", "█  ▀█", "█   █"),
    "C": ("▄▀▀▀▀", "█    ", "█    ", "█    ", "▀▄▄▄▄"),
    "O": ("▄▀▀▀▄", "█   █", "█   █", "█   █", "▀▄▄▄▀"),
    "D": ("█▀▀▀▄", "█   █", "█   █", "█   █", "█▄▄▄▀"),
}
WORDMARK = "XARENCODE"
WORD_W = len(WORDMARK) * 5 + len(WORDMARK) - 1   # 53

# ---------------------------------------------------------------- animation sequences

SEQ: dict[str, tuple[tuple[str, float], ...]] = {
    "click_jump": (("crouch", .12), ("jump", .25), ("crouch", .1), ("happy", .3), ("hearts", .45)),
    "click_stretch": (("stretch", .4), ("happy", .35), ("hearts", .45)),
    "click_purr": (("purr", .2), ("hearts", .2)) * 3,
    "celebrate": (("sparkle", .2), ("happy", .2)) * 4,
    "shocked": (("shocked", 1.2),),
    "wave": (("wave1", .15), ("wave2", .15)) * 3,
    "yawn": (("yawn", .7), ("blink", .2)),
    "lick": (("lick1", .22), ("lick2", .22)) * 2,
}
CLICKS = ("click_jump", "click_stretch", "click_purr")

SLEEP_AFTER = 180.0
SLEEP_STEP = 1.5
THINK_IDLE, THINK_TWITCH = 0.8, 0.2
TYPE_STEP = 0.4
BLINK_LEN, TAIL_LEN = 0.15, 0.3
SHIMMER_LEN = 0.8


# ---------------------------------------------------------------- colours and cell widths

def _rgb(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except (ValueError, IndexError):
        return _rgb(DEFAULT_ACCENT)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(v))) for v in rgb))


def mix(a: str, b: str, t: float) -> str:
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))


def light(accent: str, amount: float = .45) -> str:
    """A lighter tint of the accent: the lightness is raised by `amount` of the way to white."""
    r, g, b = (v / 255 for v in _rgb(accent))
    h, lum, s = colorsys.rgb_to_hls(r, g, b)
    return _hex(tuple(v * 255 for v in colorsys.hls_to_rgb(h, lum + (1 - lum) * amount, s)))


def cell_width(s: str) -> int:
    """Terminal cells a string takes: wide and fullwidth characters count 2, combining characters 0."""
    n = 0
    for ch in s:
        if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def line_width(line: Line) -> int:
    return sum(cell_width(t) for _, t in line)


def fit(line: Line, width: int) -> Line:
    """Cut a line to `width` cells; a cut line ends in an ellipsis."""
    if width <= 0:
        return []
    if line_width(line) <= width:
        return list(line)
    out: Line = []
    used = 0
    style = ""
    for style, text in line:
        keep = []
        for ch in text:
            w = cell_width(ch)
            if used + w > width - 1:
                out.append((style, "".join(keep) + "…"))
                return out
            keep.append(ch)
            used += w
        out.append((style, text))
    return out


def _fg(color: str) -> str:
    return f"fg:{color}"


@lru_cache(maxsize=256)
def _cat(frame: str, accent: str, small: bool = False) -> tuple[tuple[tuple[str, str], ...], ...]:
    if small:
        sframe = mascot_art.SMALL_MAP.get(frame, "idle")
        px = mascot_art.SMALL_FRAMES.get(sframe, mascot_art.SMALL_IDLE)
        glyph_list = mascot_art.SMALL_GLYPHS.get(sframe, ())
        cw, ch = SMALL_CAT_W, SMALL_CAT_H
    else:
        px = mascot_art.FRAMES.get(frame, mascot_art.IDLE)
        glyph_list = mascot_art.GLYPHS.get(frame, ())
        cw, ch = CAT_W, CAT_H

    glyph_map = {(r, c): txt for r, c, txt in glyph_list}
    tint = light(accent, 0.42)
    paint = {
        "O": "#181c26",
        "B": accent,
        "S": mix(accent, "#0a101d", 0.38),
        "H": tint,
        "C": "#eef2f7",
        "P": "#ff7eb6",
        "W": "#ffffff",
        "K": "#14161c",
        "L": light(accent, 0.65),
        "R": "#e5484d",
        "G": "#ffd166",
    }
    glyph_colors = {"♥": "#ff7eb6", "✧": "#ffd166", "⋆": "#ffd166", "!": "#ff5c5c"}
    rows = []
    for r in range(ch):
        top, bot = px[2 * r], px[2 * r + 1]
        line: list[tuple[str, str]] = []
        for c in range(cw):
            a, b = paint.get(top[c]), paint.get(bot[c])
            if a is None and b is None:
                g = glyph_map.get((r, c), " ")
                line.append((_fg(glyph_colors.get(g, tint)), g) if g != " " else ("", " "))
            elif a == b:
                line.append((_fg(a), "█"))
            elif b is None:
                line.append((_fg(a), "▀"))
            elif a is None:
                line.append((_fg(b), "▄"))
            else:
                line.append((f"fg:{a} bg:{b}", "▀"))
        rows.append(tuple(line))
    return tuple(rows)


def cat_lines(frame: str, accent: str = DEFAULT_ACCENT, small: bool = False) -> list[Line]:
    return [list(r) for r in _cat(frame, accent, small)]


def wordmark_lines(accent: str = DEFAULT_ACCENT, shimmer: int | None = None) -> list[Line]:
    tint = light(accent)
    flash = mix(accent, "#ffffff", .75)
    rows: list[Line] = [[] for _ in range(5)]
    for i, letter in enumerate(WORDMARK):
        hot = shimmer is not None and i in (shimmer - 1, shimmer)
        style = f"bold fg:{flash if hot else mix(accent, tint, i / (len(WORDMARK) - 1))}"
        for r in range(5):
            if i:
                rows[r].append((style, " "))
            rows[r].append((style, WORD[letter][r]))
    return rows


# ---------------------------------------------------------------- settings

def _ui(cfg: dict) -> dict:
    return cfg.get("ui", {}) if isinstance(cfg, dict) else {}


def accent_of(cfg: dict) -> str:
    return str(_ui(cfg).get("accent") or DEFAULT_ACCENT)


def on(cfg: dict) -> bool:
    """The [ui] mascot switch alone (text parts of the mascot work without colour)."""
    return bool(_ui(cfg).get("mascot", True))


def enabled(cfg: dict) -> bool:
    """True when the cat may be drawn and animated: the switch is on and colour is not disabled."""
    return on(cfg) and not os.environ.get("NO_COLOR")


def farewell_prefix(cfg: dict) -> str:
    return "=^.^=/ " if on(cfg) else ""


def pick_form(pref: str, cols: int, rows: int | None, no_color: bool) -> str:
    if pref == "off":
        return "off"
    if cols < 40 or no_color:
        return "tiny"
    if pref == "small" or cols < 82 or (rows is not None and rows < 30):
        return "small"
    return "full"


# ---------------------------------------------------------------- header layout

@dataclass
class Header:
    lines: list[Line]
    cat: tuple[int, int, int, int] | None    # click rectangle: x, y, w, h
    word: tuple[int, int, int, int] | None


def _home(path: Path | str) -> str:
    return str(path).replace(str(Path.home()), "~")


def banner(cfg: dict, workdir: Path | str, *, yolo: bool, width: int, rows: int | None = None, frame: str = "idle",
           shimmer: int | None = None, fullscreen: bool = False) -> Header:
    from . import __version__
    accent = accent_of(cfg)
    pref = str(_ui(cfg).get("header", "full"))
    form = pick_form(pref, width, rows, bool(os.environ.get("NO_COLOR")))
    if form == "off" and not fullscreen:
        form = "tiny"
    has_cat = on(cfg)
    title = f"XarenCode {__version__}"
    where = _home(workdir)
    ctx = cfg.get("server", {}).get("ctx_size", 0)
    greeting = persona.greeting(cfg)
    info = f"model {model_short_name(cfg)} · ctx {ctx} · {where}" if form == "small" else \
        f"{title} · {model_short_name(cfg)} · ctx {ctx} · {where}"
    tips = FULL_TIPS if fullscreen else INLINE_TIPS
    yolo_line: Line = [(YOLO_STYLE, " YOLO "), ("bold fg:ansired", YOLO_TEXT)]
    dim, bold = "dim", f"bold fg:{accent}"
    rect_cat = rect_word = None
    lines: list[Line] = []

    if form == "off":
        return Header([], None, None)

    if form == "tiny":
        row: Line = [("", " ")]
        if has_cat:
            glyph = TINY.get(frame, TINY["idle"])
            row += [(_fg(accent), glyph + " " * max(0, 6 - cell_width(glyph))), ("", " ")]
            rect_cat = (1, 0, 6, 1)
        row += [(bold, title), (dim, f" · {model_short_name(cfg)} · {where}")]
        if yolo:
            row += [("", " "), (YOLO_STYLE, " YOLO ")]
        lines.append(row)
        if not fullscreen:
            lines += [[("", " "), ("", greeting)], [("", " "), (dim, tips)]]
            if yolo:
                lines.append([("", " ")] + yolo_line)
    elif form == "full":
        art = wordmark_lines(accent, shimmer)
        if has_cat:
            cat = cat_lines(frame, accent, small=False)
            left = 1 + CAT_W + 2
            rule_w = min(WORD_W, max(10, width - left - 1))
            right_rows: list[Line] = [
                *art,
                [(dim, "─" * rule_w)],
                [("", greeting)],
                [(dim, info)],
                [(dim, tips)],
                yolo_line if yolo else [],
            ]
            for r in range(CAT_H):
                lines.append([("", " ")] + cat[r] + [("", "  ")] + right_rows[r])
            rect_cat = (1, 0, CAT_W, CAT_H)
            rect_word = (left, 0, WORD_W, 5)
        else:
            for r in range(5):
                lines.append([("", " ")] + art[r])
            rect_word = (1, 0, WORD_W, 5)
            lines += [[("", " "), ("", greeting)], [("", " "), (dim, info)], [("", " "), (dim, tips)]]
            if yolo:
                lines.append([("", " ")] + yolo_line)
        if fullscreen:
            lines.append([])
    else:  # small: the 12x6 cat with six lines of text to its right
        text: list[Line] = [[(bold, title)], [("", greeting)], [(dim, info)], [(dim, tips)], yolo_line if yolo else [], []]
        cat = cat_lines(frame, accent, small=True) if has_cat else [[] for _ in range(SMALL_CAT_H)]
        for r in range(SMALL_CAT_H):
            row_tail = text[r]
            if not has_cat and not row_tail:
                continue
            lines.append([("", " ")] + (cat[r] + [("", "  ")] if has_cat else []) + row_tail)
        rect_cat = (1, 0, SMALL_CAT_W, SMALL_CAT_H) if has_cat else None
        if fullscreen:
            lines.append([])
    return Header([fit(ln, width) for ln in lines], rect_cat, rect_word)


# ---------------------------------------------------------------- the state machine

class Mascot:
    """Chooses the cat's frame. Every method that depends on time is given `now` (seconds, any monotonic clock)."""

    def __init__(self, rng: random.Random | None = None, now: float = 0.0):
        self._lock = threading.RLock()
        self._rng = rng or random.Random()
        self._busy = self._alert = self._yolo = False
        self._busy_since = now
        self._last_input = now
        self._one: tuple[str, float] | None = None       # (sequence name, start time)
        self._blink_at = now + self._rng.uniform(4, 6)
        self._tail_at = now + self._rng.uniform(12, 20)
        self._shimmer_at: float | None = None

    # -- inputs
    def update(self, now: float, *, busy: bool, alert: bool, yolo: bool) -> None:
        with self._lock:
            if busy and not self._busy:
                self._busy_since = now
            if self._busy and not busy:
                self._last_input = max(self._last_input, now)   # finishing a turn is activity: no instant nap
            self._busy, self._alert, self._yolo = busy, alert, yolo
            self._advance(now)

    def poke(self, now: float) -> None:
        with self._lock:
            self._last_input = now

    def trigger(self, kind: str, now: float) -> bool:
        with self._lock:
            self._advance(now)
            self._last_input = now
            if kind == "click":
                if self._one is not None and self._one[0] in CLICKS:
                    return False
                name = self._rng.choice(CLICKS)
            elif kind in SEQ and kind not in CLICKS:
                name = kind
            else:
                return False
            self._one = (name, now)
            return True

    def shimmer(self, now: float) -> None:
        with self._lock:
            self._shimmer_at = now
            self._last_input = now

    # -- outputs
    def frame(self, now: float) -> str:
        with self._lock:
            self._advance(now)
            return self._frame(now)

    def shimmer_col(self, now: float) -> int | None:
        with self._lock:
            if self._shimmer_at is None:
                return None
            t = now - self._shimmer_at
            if t < 0 or t >= SHIMMER_LEN:
                return None
            return int(t / SHIMMER_LEN * (len(WORDMARK) + 1))

    def next_change(self, now: float) -> float:
        """Absolute time of the next moment the frame or the shimmer may change; math.inf when nothing is scheduled."""
        with self._lock:
            self._advance(now)
            times: list[float] = []
            if self._one is not None:
                name, start = self._one
                t = start
                for _, dur in SEQ[name]:
                    t += dur
                    if t > now:
                        times.append(t)
                        break
            if self._shimmer_at is not None and now < self._shimmer_at + SHIMMER_LEN:
                step = SHIMMER_LEN / (len(WORDMARK) + 1)
                times.append(self._shimmer_at + (int((now - self._shimmer_at) / step) + 1) * step)
            if self._one is None and not self._alert:
                if self._busy:
                    period = THINK_IDLE + THINK_TWITCH
                    k = int((now - self._busy_since) // period)
                    base = self._busy_since + k * period
                    for edge in (base + TYPE_STEP, base + THINK_IDLE, base + period):
                        if edge > now:
                            times.append(edge)
                            break
                elif self._asleep(now):
                    since = now - (self._last_input + SLEEP_AFTER)
                    times.append(now + SLEEP_STEP - (since % SLEEP_STEP))
                else:
                    times.append(self._last_input + SLEEP_AFTER)
                    if not self._yolo:
                        times += [t for t in (self._blink_at, self._blink_at + BLINK_LEN) if t > now]
                    times += [t for t in (self._tail_at, self._tail_at + TAIL_LEN) if t > now]
            return min(times) if times else math.inf

    # -- internals
    def _asleep(self, now: float) -> bool:
        return not self._busy and not self._alert and now - self._last_input >= SLEEP_AFTER

    def _advance(self, now: float) -> None:
        """Retire a finished one-shot and schedule the next blink and tail flick."""
        if self._one is not None:
            name, start = self._one
            if now - start >= sum(d for _, d in SEQ[name]):
                self._one = None
        if now >= self._blink_at + BLINK_LEN:
            self._blink_at = max(self._blink_at + BLINK_LEN, now) + self._rng.uniform(4, 6)
        if now >= self._tail_at + TAIL_LEN:
            self._tail_at = max(self._tail_at + TAIL_LEN, now) + self._rng.uniform(12, 20)

    def _frame(self, now: float) -> str:
        if self._one is not None:
            name, start = self._one
            t = now - start
            for frame, dur in SEQ[name]:
                if t < dur:
                    return frame
                t -= dur
        if self._alert:
            return "alert"
        calm = "cool" if self._yolo else "idle"
        if self._busy:
            phase = (now - self._busy_since) % (THINK_IDLE + THINK_TWITCH)
            if phase >= THINK_IDLE:
                return "twitch"
            return "type1" if phase < TYPE_STEP else "type2"
        if self._asleep(now):
            return "sleep1" if int((now - self._last_input - SLEEP_AFTER) / SLEEP_STEP) % 2 == 0 else "sleep2"
        if not self._yolo and self._blink_at <= now < self._blink_at + BLINK_LEN:
            return "blink"
        if self._tail_at <= now < self._tail_at + TAIL_LEN:
            return "tail"
        return calm
