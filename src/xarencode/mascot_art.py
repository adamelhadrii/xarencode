"""Pixel data for the mascot: palette symbols, the full 22x20 cat, the 12x12 small cat, every frame and the text cats.

Standard library only. A pixel grid is a tuple of equal-length strings; two pixel rows make one terminal cell row (half blocks).

Symbols:  O outline   B base   S shadow and stripes   H highlight   C chest and muzzle   P pink (ears, nose, blush)
          W white     K pupils and mouth   L whiskers   R collar   G bell   . transparent
"""
from __future__ import annotations

SYMBOLS = "OBSHCPWKLRG"
W, H = 22, 20            # the full cat in pixels: 22 columns x 10 cell rows
SW, SH = 12, 12          # the small cat in pixels: 12 columns x 6 cell rows

IDLE: tuple[str, ...] = (
    "......................",
    "...O.........O........",
    "..OHO.......OHO.......",
    "..OPBO.....OBPO.......",
    ".OPPBBOOOOOBBPPO......",
    ".OHHBBSBSBSBBBSO......",
    ".OHBBWKBBBWKBBSO......",
    ".OBBBKKBBBKKBBSO..OO..",
    "LLBBBCCCPCCCBBSLL.OHO.",
    ".OBBBCCKCKCCBBSO..OBBO",
    "LLOBBBCCCCCBBSOLL.OSSO",
    "...ORRRRGRRRRO....OBHO",
    "...OBBCCGCCBBO....OSSO",
    "..OHBBCCCCCBBSO...OBHO",
    "..OHBBCCCCCBBSO...OSSO",
    ".OHBBBCCCCCBBBSO..OBHO",
    ".OBBBBBCCCBBBBSO.OSSO.",
    ".OBBBBBBBBBBBBSOSBBO..",
    ".OBCCCBBBBCCCBOSSSO...",
    "..OOOOOOOOOOOOOOOO....",
)

SMALL_IDLE: tuple[str, ...] = (
    "............",
    "..O.....O...",
    ".OPO...OPO..",
    "OBBBOOOBBBO.",
    "OBWKBBBWKBO.",
    "LBBBCPCBBBL.",
    ".OBBCCCBBO..",
    "..ORRGRRO.O.",
    "..OBCCCBO.OO",
    ".OBBCCCBBOSO",
    ".OBBBBBBBOSO",
    ".OCCOOOCCOO.",
)


def edit(rows: tuple[str, ...], *cells: tuple[int, int, str]) -> tuple[str, ...]:
    """Copy of a grid with strings written at (row, column); characters running off the right edge are dropped."""
    out = [list(r) for r in rows]
    for r, c, s in cells:
        for i, ch in enumerate(s):
            if 0 <= c + i < len(out[r]):
                out[r][c + i] = ch
    return tuple("".join(r) for r in out)


def _crouch(rows: tuple[str, ...]) -> tuple[str, ...]:
    blank = "." * len(rows[0])
    return (blank,) + rows[:16] + rows[17:]


def _jump(rows: tuple[str, ...]) -> tuple[str, ...]:
    return rows[1:] + ("." * len(rows[0]),)


def _shift(rows: tuple[str, ...], r0: int, r1: int, c0: int, c1: int, d: int) -> tuple[str, ...]:
    """Slide the columns c0..c1 (inclusive) of rows r0..r1 sideways by d; the vacated edge repeats its neighbour."""
    out = [list(r) for r in rows]
    for r in range(r0, r1 + 1):
        old = out[r][c0:c1 + 1]
        if d < 0:
            new = old[-d:] + [old[-1]] * -d
        else:
            new = [old[0]] * d + old[:len(old) - d]
        out[r][c0:c1 + 1] = new
    return tuple("".join(r) for r in out)


def _tail_up(rows: tuple[str, ...], n: int) -> tuple[str, ...]:
    """Lengthen the vertical part of the tail upwards by n rows (rows 7-15, columns 18-21)."""
    out = [list(r) for r in rows]
    for r in range(7, 16):
        out[r - n][18:22] = rows[r][18:22]
    for r in range(16 - n, 16):
        out[r][18:22] = rows[15 - (r - (16 - n)) % 2][18:22]
    return tuple("".join(r) for r in out)


_EYES_BLINK = ((6, 5, "BB"), (6, 10, "BB"), (7, 5, "KK"), (7, 10, "KK"))
_EYES_WIDE = ((6, 5, "WW"), (6, 10, "WW"), (7, 5, "WK"), (7, 10, "WK"))
_BLINK = edit(IDLE, *_EYES_BLINK)
_HAPPY = edit(IDLE, (6, 5, "KK"), (6, 10, "KK"), (7, 5, "BB"), (7, 10, "BB"), (8, 3, "P"), (8, 13, "P"), (9, 8, "P"))
_SLEEP_BASE = edit(_BLINK, (8, 0, ".O"), (8, 15, "O."), (9, 0, "L"), (9, 16, "L"), (7, 18, "...."), (8, 18, "...."), (9, 18, "OOOO"))
_SLEEP = _crouch(_SLEEP_BASE)
_TYPE_EYES = ((6, 5, "SS"), (6, 10, "SS"), (7, 5, "WK"), (7, 10, "WK"), (18, 2, "OLSLSLSLSLSO"))

FRAMES: dict[str, tuple[str, ...]] = {
    "idle": IDLE,
    "blink": _BLINK,
    "tail": _shift(IDLE, 7, 9, 17, 21, -1),
    "twitch": edit(IDLE, (1, 13, ".O"), (2, 12, ".OHO")),
    "alert": edit(IDLE, (0, 3, "O"), (0, 13, "O"), (1, 2, "OHO"), (1, 12, "OHO"), *_EYES_WIDE),
    "crouch": _crouch(IDLE),
    "jump": _jump(edit(IDLE, (18, 3, "BBB"), (18, 10, "BBB"))),
    "stretch": _crouch(edit(_tail_up(IDLE, 2), (18, 0, "OCC"))),
    "happy": _HAPPY,
    "hearts": _HAPPY,
    "purr": edit(_BLINK, (8, 3, "P"), (8, 13, "P"), (8, 0, ".LL"), (8, 14, "LL."), (10, 0, ".L"), (10, 15, "L.")),
    "sparkle": edit(IDLE, (6, 5, "WW"), (6, 10, "WW"), (7, 5, "WW"), (7, 10, "WW")),
    "shocked": edit(IDLE, (1, 3, "."), (1, 13, "."), *_EYES_WIDE, (9, 7, "KPK")),
    "cool": edit(IDLE, (6, 3, "KKKKKKKKKKK"), (7, 4, "KKK"), (7, 10, "KKK"), (6, 4, "W"), (6, 10, "W")),
    "sleep1": _SLEEP,
    "sleep2": edit(_SLEEP, (9, 0, "L")),
    "wave1": edit(IDLE, (18, 10, "BBB"), (12, 15, "OCCO"), (13, 15, "OCCO")),
    "wave2": edit(IDLE, (18, 10, "BBB"), (11, 16, "OCCO"), (12, 16, "OCCO")),
    "yawn": edit(_BLINK, (9, 7, "KKK"), (10, 7, "KPK")),
    "lick1": edit(IDLE, (9, 4, "OCCO"), (10, 4, "OCCO"), (9, 8, "P")),
    "lick2": edit(_BLINK, (9, 4, "OCCO"), (10, 4, "OCCO")),
    "look_left": _shift(IDLE, 5, 9, 2, 14, -1),
    "look_right": _shift(IDLE, 5, 9, 2, 14, 1),
    "type1": edit(IDLE, *_TYPE_EYES, (17, 4, "CC"), (16, 10, "CC")),
    "type2": edit(IDLE, *_TYPE_EYES, (16, 4, "CC"), (17, 10, "CC")),
}

# (cell row, column, text) drawn on cells whose two pixels are both clear
GLYPHS: dict[str, tuple[tuple[int, int, str], ...]] = {
    "hearts": ((0, 17, "♥"), (0, 20, "♥"), (1, 18, "♥"), (2, 16, "♥")),
    "purr": ((0, 17, "♥"), (1, 19, "♥")),
    "sparkle": ((0, 0, "✧"), (0, 19, "✧"), (1, 20, "⋆"), (3, 0, "⋆")),
    "jump": ((0, 1, "⋆"), (0, 20, "⋆"), (1, 18, "⋆"), (9, 0, "⋆")),
    "shocked": ((0, 17, "!"), (1, 19, "!")),
    "sleep1": ((0, 17, "z"),),
    "sleep2": ((0, 17, "z"), (0, 19, "Z")),
}

_S_BLINK = edit(SMALL_IDLE, (4, 2, "KK"), (4, 7, "KK"))
_S_SLEEP_BASE = _S_BLINK

SMALL_FRAMES: dict[str, tuple[str, ...]] = {
    "idle": SMALL_IDLE,
    "blink": _S_BLINK,
    "happy": edit(_S_BLINK, (5, 1, "P"), (5, 9, "P")),
    "alert": edit(SMALL_IDLE, (0, 2, "O"), (0, 8, "O"), (4, 2, "WW"), (4, 7, "WW")),
    "cool": edit(SMALL_IDLE, (4, 1, "KKKKKKKKK"), (4, 2, "W"), (4, 8, "W")),
    "sleep": (("." * SW),) + _S_SLEEP_BASE[:10] + (_S_SLEEP_BASE[11],),
    "type1": edit(SMALL_IDLE, (3, 2, "SS"), (3, 7, "SS"), (10, 2, "CC")),
    "type2": edit(SMALL_IDLE, (3, 2, "SS"), (3, 7, "SS"), (10, 7, "CC")),
}
SMALL_GLYPHS: dict[str, tuple[tuple[int, int, str], ...]] = {
    "happy": ((0, 10, "♥"),),
    "sleep": ((0, 10, "z"),),
}
# every frame name -> the small frame that stands in for it; anything not listed uses the small idle cat
SMALL_MAP: dict[str, str] = {
    "blink": "blink", "purr": "blink", "yawn": "blink", "lick2": "blink", "happy": "happy", "hearts": "happy",
    "sparkle": "happy", "lick1": "happy", "alert": "alert", "shocked": "alert", "cool": "cool", "sleep1": "sleep",
    "sleep2": "sleep", "type1": "type1", "type2": "type2",
}

TINY: dict[str, str] = {
    "idle": "=^.^=", "blink": "=-.-=", "tail": "=^.^=~", "twitch": "=^.<=", "alert": "=O.O=!", "crouch": "=^_^=",
    "jump": "=^o^=", "stretch": "=^_^=~", "happy": "=^ω^=", "hearts": "=^ω^=♥", "purr": "=-ω-=♥", "sparkle": "=✧.✧=",
    "shocked": "=O.O=!", "cool": "=■.■=", "sleep1": "=-.-=z", "sleep2": "=-_-=z", "wave1": "=^.^=/", "wave2": "=^.^=\\",
    "yawn": "=-O-=", "lick1": "=^.^;", "lick2": "=-.-;", "look_left": "=<.<=", "look_right": "=>.>=",
    "type1": "=^.^=,", "type2": "=^.^=.",
}
