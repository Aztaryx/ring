"""
editor_core.py - the chart editor engine. No pygame, no audio.

Owns everything that changes a chart: snapping, placing patterns, erasing,
undo/redo, per-bar copy/paste/clear. The front-end (editor.py) only turns key
presses and the audio clock into calls on this class.
"""
from __future__ import annotations

import bisect
import math

from circle_core import BEATS_PER_BAR, Chart, clamp, snap_time

SNAPS = [1, 2, 3, 4, 6, 8, 12, 16, 0]  # subdivisions of a beat, 0 = free
DUP_EPS = 0.004                        # two dots closer than 4 ms count as the same dot


class EditorEngine:
    def __init__(self, chart: Chart, snap_i: int = 3):
        self.chart = chart
        self.snap_i = snap_i % len(SNAPS)
        self.undo_stack = []   # (kind, [times]) kind = "add" | "rem"
        self.redo_stack = []
        self.clip = []         # copied bar: dot offsets from the bar start
        self._merge_erase = False

    # ------------------------------------------------------------- snap
    @property
    def snap_div(self) -> int:
        return SNAPS[self.snap_i]

    @property
    def snap_step(self) -> float:
        d = self.snap_div
        return self.chart.beat / d if d else 0.0

    @property
    def snap_label(self) -> str:
        d = self.snap_div
        return f"1/{d * 4}" if d else "off"

    def cycle_snap(self, d: int = 1):
        self.snap_i = (self.snap_i + d) % len(SNAPS)

    def snapped(self, t: float) -> float:
        return snap_time(t, self.chart.offset, self.snap_step)

    # --------------------------------------------------------- positions
    def bar_of(self, t: float) -> int:
        return int(math.floor((t - self.chart.offset) / self.chart.bar))

    def beat_of(self, t: float) -> int:
        return int(((t - self.chart.offset) / self.chart.beat) % BEATS_PER_BAR) + 1

    def bar_range(self, bar: int):
        t0 = self.chart.offset + bar * self.chart.bar
        return t0, t0 + self.chart.bar

    def visible(self, now: float):
        """Dots drawn on the circle: exactly one lap, a little behind the hand."""
        ch = self.chart
        lo = bisect.bisect_left(ch.dots, now - 0.15 * ch.bar)
        hi = bisect.bisect_right(ch.dots, now + 0.85 * ch.bar)
        return ch.dots[lo:hi]

    # ---------------------------------------------------------- internals
    def _push(self, kind, xs):
        if xs:
            self.undo_stack.append((kind, list(xs)))
            self.redo_stack.clear()

    def _insert(self, xs):
        ch, added = self.chart, []
        for x in xs:
            x = round(x, 4)
            i = bisect.bisect_left(ch.dots, x - DUP_EPS)
            if i < len(ch.dots) and abs(ch.dots[i] - x) < DUP_EPS:
                continue
            ch.dots.insert(i, x)
            added.append(x)
        return added

    def _remove(self, xs):
        ch = self.chart
        for x in xs:
            i = bisect.bisect_left(ch.dots, x - 1e-6)
            if i < len(ch.dots) and abs(ch.dots[i] - x) < 1e-5:
                ch.dots.pop(i)

    def _take(self, lo, hi):
        ch = self.chart
        i = bisect.bisect_left(ch.dots, lo)
        j = bisect.bisect_left(ch.dots, hi)
        out = ch.dots[i:j]
        del ch.dots[i:j]
        return out

    # ----------------------------------------------------------- editing
    def place(self, n: int, t: float):
        """Place an n-dot pattern anchored at the snapped press time, spaced like
        the n-button: n dots evenly across one pulse. Returns the dots added."""
        self._merge_erase = False
        t0 = self.snapped(t)
        added = self._insert([t0 + k * self.chart.beat / n for k in range(n)])
        self._push("add", added)
        return added

    def erase(self, lo: float, hi: float):
        """Remove dots in [lo, hi). Calls in a row merge into one undo step."""
        out = self._take(lo, hi)
        if out:
            if self._merge_erase and self.undo_stack and self.undo_stack[-1][0] == "rem":
                self.undo_stack[-1][1].extend(out)
            else:
                self._push("rem", out)
            self._merge_erase = True
        return out

    def break_erase(self):
        self._merge_erase = False

    def undo(self) -> bool:
        self._merge_erase = False
        if not self.undo_stack:
            return False
        kind, xs = self.undo_stack.pop()
        if kind == "add":
            self._remove(xs)
        else:
            self._insert(xs)
        self.redo_stack.append((kind, xs))
        return True

    def redo(self) -> bool:
        self._merge_erase = False
        if not self.redo_stack:
            return False
        kind, xs = self.redo_stack.pop()
        if kind == "add":
            self._insert(xs)
        else:
            self._remove(xs)
        self.undo_stack.append((kind, xs))
        return True

    # ------------------------------------------------------------ bars
    def copy_bar(self, bar: int) -> int:
        lo, hi = self.bar_range(bar)
        i = bisect.bisect_left(self.chart.dots, lo - 1e-6)
        j = bisect.bisect_left(self.chart.dots, hi - 1e-6)
        self.clip = [d - lo for d in self.chart.dots[i:j]]
        return len(self.clip)

    def paste_bar(self, bar: int):
        self._merge_erase = False
        lo, _ = self.bar_range(bar)
        added = self._insert([lo + x for x in self.clip])
        self._push("add", added)
        return added

    def clear_bar(self, bar: int):
        self._merge_erase = False
        lo, hi = self.bar_range(bar)
        out = self._take(lo - 1e-6, hi - 1e-6)
        self._push("rem", out)
        return out

    # ----------------------------------------------------------- timing
    def nudge_bpm(self, d: float):
        self.chart.bpm = round(clamp(self.chart.bpm + d, 30, 400), 3)

    def nudge_offset(self, d: float):
        self.chart.offset = round(self.chart.offset + d, 4)
