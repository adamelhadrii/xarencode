"""Mascot tests: frames, wordmark, state machine on a fake clock, and the header layouts. Standard library only."""
import math
import os
import random
import unittest
from unittest import mock

from xarencode import mascot
from xarencode.config import default_config


def text(line):
    return "".join(t for _, t in line)


def rows(header):
    return [text(ln) for ln in header.lines]


class FrameTests(unittest.TestCase):
    def test_cat_frames_are_20_by_22_and_use_known_pixels(self):
        self.assertEqual(len(mascot.CAT["idle"]), 20)
        for name, grid in mascot.CAT.items():
            self.assertEqual(len(grid), 20, name)
            for row in grid:
                self.assertEqual(len(row), 22, name)
                self.assertLessEqual(set(row), set("." + mascot.mascot_art.SYMBOLS), name)
        for name in ("idle", "blink", "tail", "twitch", "alert", "crouch", "jump", "stretch", "happy", "hearts", "purr", "sparkle",
                     "shocked", "cool", "sleep1", "sleep2", "wave1", "wave2", "yawn", "lick1", "lick2", "look_left", "look_right",
                     "type1", "type2"):
            self.assertIn(name, mascot.CAT)
        self.assertEqual(mascot.CAT["idle"][0], "." * 22)  # headroom
        self.assertIn("K", "".join(mascot.CAT["cool"]))    # sunglasses

    def test_overlays_and_tiny_frames(self):
        for name, glyphs in mascot.GLYPHS.items():
            self.assertIn(name, mascot.CAT)
            for r, c, ch in glyphs:
                self.assertTrue(0 <= r < mascot.CAT_H)
                self.assertTrue(0 <= c < mascot.CAT_W)
                self.assertIn(ch, "♥✧⋆zZ!")
        self.assertEqual(set(mascot.TINY), set(mascot.CAT))
        for name, tiny in mascot.TINY.items():
            self.assertLessEqual(mascot.cell_width(tiny), 6, name)
        self.assertEqual(mascot.TINY["idle"], "=^.^=")

    def test_sequences_use_existing_frames(self):
        for name, steps in mascot.SEQ.items():
            for frame, secs in steps:
                self.assertIn(frame, mascot.CAT, name)
                self.assertGreater(secs, 0)
        for name in mascot.CLICKS:
            self.assertLessEqual(sum(s for _, s in mascot.SEQ[name]), 1.6)

    def test_wordmark_letters_and_width(self):
        for letter, glyph in mascot.WORD.items():
            self.assertEqual(len(glyph), 5, letter)
            for row in glyph:
                self.assertEqual(len(row), 5, letter)
                self.assertLessEqual(set(row), set(" ▀▄█"), letter)
        lines = mascot.wordmark_lines("#0a84ff")
        self.assertEqual(len(lines), 5)
        for ln in lines:
            self.assertEqual(mascot.line_width(ln), 53)
            self.assertTrue(all("bold" in style for style, _ in ln))
        self.assertEqual(text(lines[0])[:5], "█   █")

    def test_wordmark_shimmer_marks_two_letters(self):
        plain = mascot.wordmark_lines("#0a84ff")[2]
        shim = mascot.wordmark_lines("#0a84ff", shimmer=3)[2]
        changed = [i for i, (a, b) in enumerate(zip(plain, shim)) if a != b]
        self.assertGreaterEqual(len(changed), 2)
        self.assertEqual(text(plain), text(shim))

    def test_cat_lines_shape_and_colours(self):
        lines = mascot.cat_lines("idle", "#0a84ff")
        self.assertEqual(len(lines), mascot.CAT_H)
        for ln in lines:
            self.assertEqual(mascot.line_width(ln), mascot.CAT_W)
            for style, ch in ln:
                if "bg:" in style:
                    self.assertEqual(ch, "▀")
                self.assertIn(ch, " ▀▄█")
        self.assertTrue(any("bg:" in s for ln in lines for s, _ in ln))  # the white eyes sit on the blue head
        self.assertTrue(any("#0a84ff" in s for ln in lines for s, _ in ln))
        lines[0].append(("x", "y"))
        self.assertEqual(len(mascot.cat_lines("idle", "#0a84ff")[0]), 22)  # the cache is not mutated

    def test_small_cat_lines_shape_and_colours(self):
        lines = mascot.cat_lines("idle", "#0a84ff", small=True)
        self.assertEqual(len(lines), mascot.SMALL_CAT_H)
        for ln in lines:
            self.assertEqual(mascot.line_width(ln), mascot.SMALL_CAT_W)
        self.assertTrue(any("#ff7eb6" in s for ln in lines for s, _ in ln))  # pink inner ears/nose
        self.assertTrue(any("#e5484d" in s for ln in lines for s, _ in ln))  # red collar

    def test_overlay_only_on_clear_cells(self):
        hearts = mascot.cat_lines("hearts", "#0a84ff")
        self.assertIn("♥", text(hearts[0]) + text(hearts[1]))
        for ln in hearts:
            self.assertEqual(mascot.line_width(ln), 22)

    def test_light_and_fit(self):
        self.assertEqual(mascot.light("#0a84ff")[0], "#")
        self.assertNotEqual(mascot.light("#0a84ff"), "#0a84ff")
        self.assertEqual(mascot.cell_width("ab"), 2)
        self.assertEqual(mascot.cell_width("日本"), 4)
        self.assertEqual(mascot.cell_width("é"), 1)
        line = [("a", "hello "), ("b", "world")]
        self.assertEqual(mascot.fit(line, 20), line)
        cut = mascot.fit(line, 8)
        self.assertEqual(text(cut), "hello w…")
        self.assertEqual(mascot.line_width(mascot.fit([("", "日本語日本語")], 7)), 7)


class MascotTests(unittest.TestCase):
    def make(self, seed=1):
        return mascot.Mascot(random.Random(seed), now=0.0)

    def run_frames(self, m, start, end, step=0.05, **flags):
        seen = {}
        t = start
        flags = {"busy": False, "alert": False, "yolo": False} | flags
        while t <= end:
            m.update(t, **flags)
            seen.setdefault(m.frame(t), t)
            t += step
        return seen

    def test_idle_blinks_within_six_seconds(self):
        seen = self.run_frames(self.make(), 0, 6.2)
        self.assertIn("blink", seen)
        self.assertGreaterEqual(seen["blink"], 4.0)
        self.assertIn("idle", seen)

    def test_tail_flick_within_twenty_seconds(self):
        self.assertIn("tail", self.run_frames(self.make(), 0, 20.5))

    def test_click_plays_and_second_click_is_refused(self):
        m = self.make()
        self.assertTrue(m.trigger("click", 0.5))
        self.assertFalse(m.trigger("click", 0.7))
        seen = self.run_frames(m, 0.5, 1.9)
        self.assertTrue({"hearts", "happy"} & set(seen) or {"purr", "hearts"} <= set(seen))
        self.assertEqual(m.frame(2.2), "idle")                 # back to idle within 1.6 s of the click
        self.assertTrue(m.trigger("click", 2.3))               # and it can be clicked again

    def test_each_click_sequence_ends_in_the_base_frame(self):
        for seed in range(12):
            m = self.make(seed)
            m.trigger("click", 1.0)
            self.assertEqual(m.frame(2.7), "idle", seed)

    def test_other_kinds_replace_the_current_one_shot(self):
        m = self.make()
        m.trigger("click", 0.0)
        self.assertTrue(m.trigger("shocked", 0.1))
        self.assertEqual(m.frame(0.2), "shocked")
        self.assertTrue(m.trigger("celebrate", 0.3))
        self.assertEqual(m.frame(0.3), "sparkle")
        self.assertTrue(m.trigger("wave", 2.0))
        self.assertEqual(m.frame(2.05), "wave1")
        self.assertFalse(m.trigger("nonsense", 3.0))

    def test_thinking_alert_and_yolo(self):
        m = self.make()
        seen = self.run_frames(m, 0, 3, busy=True)
        self.assertIn("twitch", seen)
        self.assertNotIn("blink", seen)
        m.update(3.0, busy=False, alert=True, yolo=False)
        self.assertEqual(m.frame(3.0), "alert")
        self.assertEqual(m.next_change(3.0), math.inf)
        m2 = self.make()
        seen = self.run_frames(m2, 0, 25, yolo=True)
        self.assertIn("cool", seen)
        self.assertNotIn("idle", seen)
        self.assertNotIn("blink", seen)

    def test_sleepy_after_three_minutes_and_poke_wakes(self):
        m = self.make()
        m.update(100.0, busy=False, alert=False, yolo=False)
        self.assertNotIn(m.frame(100.0), ("sleep1", "sleep2"))
        m.update(181.0, busy=False, alert=False, yolo=False)
        a = m.frame(181.0)
        b = m.frame(182.6)
        self.assertEqual({a, b}, {"sleep1", "sleep2"})
        m.poke(183.0)
        self.assertEqual(m.frame(183.1), "idle")
        m.update(183.1, busy=True, alert=False, yolo=False)
        self.assertNotIn(m.frame(400.0), ("sleep1", "sleep2"))

    def test_shimmer_advances_then_ends(self):
        m = self.make()
        self.assertIsNone(m.shimmer_col(1.0))
        m.shimmer(1.0)
        cols = [m.shimmer_col(1.0 + i * 0.08) for i in range(10)]
        self.assertEqual(cols[0], 0)
        self.assertTrue(all(b >= a for a, b in zip(cols, cols[1:])))
        self.assertGreater(cols[-1], 5)
        self.assertIsNone(m.shimmer_col(1.9))
        self.assertGreater(m.next_change(1.0), 1.0)

    def test_next_change_is_in_the_future_and_finite_when_idle(self):
        m = self.make()
        nxt = m.next_change(0.0)
        self.assertGreater(nxt, 0.0)
        self.assertLess(nxt, 7.0)
        m.trigger("click", 0.0)
        self.assertAlmostEqual(m.next_change(0.0), 0.12, places=2)

    def test_frame_only_changes_at_next_change(self):
        m = self.make(3)
        m.update(0, busy=False, alert=False, yolo=False)
        t, frame = 0.0, m.frame(0.0)
        for _ in range(60):
            nxt = m.next_change(t)
            if nxt == math.inf:
                break
            self.assertEqual(m.frame((t + nxt) / 2), frame)   # nothing changes before the announced time
            t = nxt + 0.001
            frame = m.frame(t)


class SettingsTests(unittest.TestCase):
    def test_pick_form(self):
        self.assertEqual(mascot.pick_form("full", 120, 40, False), "full")
        self.assertEqual(mascot.pick_form("full", 80, 24, False), "small")
        self.assertEqual(mascot.pick_form("full", 81, 40, False), "small")
        self.assertEqual(mascot.pick_form("full", 82, 40, False), "full")
        self.assertEqual(mascot.pick_form("full", 39, 40, False), "tiny")
        self.assertEqual(mascot.pick_form("full", 120, 40, True), "tiny")
        self.assertEqual(mascot.pick_form("small", 120, 40, False), "small")
        self.assertEqual(mascot.pick_form("off", 120, 40, False), "off")
        self.assertEqual(mascot.pick_form("full", 120, None, False), "full")

    def test_enabled_and_prefix(self):
        cfg = default_config()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NO_COLOR", None)
            self.assertTrue(mascot.enabled(cfg))
            self.assertEqual(mascot.farewell_prefix(cfg), "=^.^=/ ")
            cfg["ui"]["mascot"] = False
            self.assertFalse(mascot.enabled(cfg))
            self.assertEqual(mascot.farewell_prefix(cfg), "")
            cfg["ui"]["mascot"] = True
            os.environ["NO_COLOR"] = "1"
            self.assertFalse(mascot.enabled(cfg))
            self.assertEqual(mascot.farewell_prefix(cfg), "=^.^=/ ")
        self.assertTrue(mascot.enabled({}))  # a config without a [ui] block still works


class BannerTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {"HOME": "/home/tester"})
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop("NO_COLOR", None)
        self.cfg = default_config()

    def banner(self, width, rows=None, **kw):
        kw.setdefault("yolo", False)
        return mascot.banner(self.cfg, "/home/tester/proj", width=width, rows=rows, **kw)

    def test_full_header_in_the_full_window(self):
        h = self.banner(120, 40, fullscreen=True)
        r = rows(h)
        self.assertEqual(len(r), 11)
        self.assertEqual(h.cat, (1, 0, 22, 10))
        self.assertEqual(h.word, (25, 0, 53, 5))
        self.assertEqual(r[0][25:30], "█   █")
        self.assertIn("Good ", r[6])
        self.assertIn("XarenCode 0.1.0 · Qwen3-4B · ctx 4096 · ~/proj", r[7])
        self.assertIn("F3 pet", r[8])
        self.assertEqual(r[10], "")
        self.assertIn("YOLO", rows(self.banner(120, 40, fullscreen=True, yolo=True))[9])

    def test_full_header_inline_has_no_spacer(self):
        h = self.banner(100)
        r = rows(h)
        self.assertEqual(len(r), 10)
        self.assertIn("@file attaches a file", r[8])
        y = rows(self.banner(100, yolo=True))
        self.assertEqual(len(y), 10)
        self.assertIn("auto-approve is on. Hard blocks and sudo still apply.", y[9])

    def test_small_header(self):
        h = self.banner(80, 24, fullscreen=True)
        r = rows(h)
        self.assertEqual(len(r), 7)
        self.assertEqual(h.cat, (1, 0, 12, 6))
        self.assertIsNone(h.word)
        self.assertIn("XarenCode 0.1.0", r[0])
        self.assertIn("Good ", r[1])
        self.assertIn("model Qwen3-4B · ctx 4096 · ~/proj", r[2])
        self.assertEqual(r[6], "")

    def test_tiny_header(self):
        h = self.banner(39, 40, fullscreen=True)
        self.assertEqual(len(h.lines), 1)
        self.assertEqual(h.cat, (1, 0, 6, 1))
        self.assertTrue(rows(h)[0].startswith(" =^.^=  XarenCode"))
        inline = rows(self.banner(39))
        self.assertEqual(len(inline), 3)

    def test_off_in_the_full_window_and_tiny_inline(self):
        self.cfg["ui"]["header"] = "off"
        self.assertEqual(self.banner(120, 40, fullscreen=True).lines, [])
        self.assertIn("=^.^=", rows(self.banner(120))[0])

    def test_mascot_off_drops_the_cat_columns(self):
        self.cfg["ui"]["mascot"] = False
        h = self.banner(120, 40, fullscreen=True)
        self.assertIsNone(h.cat)
        self.assertEqual(h.word, (1, 0, 53, 5))
        self.assertTrue(rows(h)[0].startswith(" █   █"))
        self.assertNotIn("=^", "".join(rows(self.banner(39, 40, fullscreen=True))))

    def test_no_color_gives_the_text_cat_only(self):
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            r = rows(self.banner(120, 40, fullscreen=True))
        self.assertEqual(len(r), 1)
        self.assertIn("=^.^=", r[0])
        self.assertNotIn("█", r[0])

    def test_no_line_is_wider_than_the_screen(self):
        for width in (20, 40, 60, 69, 70, 80, 120):
            for yolo in (False, True):
                for full in (False, True):
                    for rows_ in (None, 24, 40):
                        h = self.banner(width, rows_, yolo=yolo, fullscreen=full)
                        for ln in h.lines:
                            self.assertLessEqual(mascot.line_width(ln), width, (width, yolo, full, rows_, text(ln)))

    def test_frame_and_shimmer_change_only_the_art(self):
        a = self.banner(120, 40, fullscreen=True)
        b = self.banner(120, 40, fullscreen=True, frame="jump", shimmer=4)
        self.assertNotEqual(a.lines[0], b.lines[0])
        self.assertEqual(a.lines[7][1 + mascot.CAT_W:], b.lines[7][1 + mascot.CAT_W:])


if __name__ == "__main__":
    unittest.main()
