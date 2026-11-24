"""Bridge from Rich renderables to prompt_toolkit style fragments.

Imports rich only (never prompt_toolkit). A "Line" is one screen row: a list of (prompt_toolkit style, text) pairs.
"""
from __future__ import annotations

import io
import re
from functools import lru_cache

from rich.color import Color, ColorType
from rich.console import Console, RenderableType
from rich.style import Style
from rich.text import Text

Line = list[tuple[str, str]]

_ANSI = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
_PT_NAMES = ["ansiblack", "ansired", "ansigreen", "ansiyellow", "ansiblue", "ansimagenta", "ansicyan", "ansigray",
             "ansibrightblack", "ansibrightred", "ansibrightgreen", "ansibrightyellow", "ansibrightblue", "ansibrightmagenta",
             "ansibrightcyan", "ansiwhite"]
_RICH_NAMES = {pt: (_ANSI[i] if i < 8 else "bright_" + _ANSI[i - 8]) for i, pt in enumerate(_PT_NAMES)}


def make_console(width: int = 100) -> Console:
    return Console(file=io.StringIO(), width=width, force_terminal=True, color_system="truecolor", highlight=False)


def _pt_color(color: Color | None) -> str:
    if color is None or color.type == ColorType.DEFAULT:
        return ""
    if color.type == ColorType.STANDARD and color.number is not None and color.number < 16:
        return _PT_NAMES[color.number]
    r, g, b = color.get_truecolor()
    return f"#{r:02x}{g:02x}{b:02x}"


@lru_cache(maxsize=2048)
def pt_style(style: Style | None) -> str:
    if not style:
        return ""
    parts = []
    fg, bg = _pt_color(style.color), _pt_color(style.bgcolor)
    if fg:
        parts.append(f"fg:{fg}")
    if bg:
        parts.append(f"bg:{bg}")
    for attr in ("bold", "italic", "underline", "strike", "reverse", "dim"):
        if getattr(style, attr):
            parts.append(attr)
    return " ".join(parts)


def to_lines(renderable: RenderableType, width: int, console: Console | None = None) -> list[Line]:
    """Render to screen rows of at most `width` cells; control segments are dropped and tabs become 4 spaces."""
    console = console or make_console(width)
    options = console.options.update_width(max(1, width))
    out: list[Line] = []
    for row in console.render_lines(renderable, options, pad=False):
        line: Line = []
        for seg in row:
            if seg.control or not seg.text:
                continue
            line.append((pt_style(seg.style), seg.text.replace("\t", "    ")))
        out.append(line)
    return out


_STRIP = re.compile("[‍️\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Drop what Rich and prompt_toolkit measure differently (variation selector, joiner) and terminal control characters."""
    if not text:
        return text
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _STRIP.sub("", text)


def split_stable(md: str) -> tuple[str, str]:
    """Split Markdown at the last blank line that is outside a code fence: (stable prefix, still-changing rest)."""
    fenced = False
    end = pos = 0
    for line in md.splitlines(keepends=True):
        pos += len(line)
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
        elif not stripped and not fenced:
            end = pos
    return md[:end], md[end:]


@lru_cache(maxsize=512)
def _rich_style(pt: str) -> Style:
    fg = bg = None
    flags: dict[str, bool] = {}
    for tok in pt.split():
        if tok.startswith("fg:"):
            fg = _RICH_NAMES.get(tok[3:], tok[3:])
        elif tok.startswith("bg:"):
            bg = _RICH_NAMES.get(tok[3:], tok[3:])
        elif tok in ("bold", "italic", "underline", "strike", "reverse", "dim"):
            flags[tok] = True
    return Style(color=fg, bgcolor=bg, **flags)


def to_rich_text(lines: list[Line]) -> Text:
    out = Text(no_wrap=True, overflow="crop")
    for i, line in enumerate(lines):
        if i:
            out.append("\n")
        for style, text in line:
            out.append(text, _rich_style(style) if style else None)
    return out
