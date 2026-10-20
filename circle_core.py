"""
circle_core.py - game logic for the circle rhythm prototype.

No pygame in here. Everything is plain Python + numpy so it can be tested
without a window or an audio device.
"""
from __future__ import annotations

import bisect
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

BEATS_PER_BAR = 4  # 4/4 only for the prototype

# ----------------------------------------------------------------- AR / OD
# Editable anchors: (AR, look-ahead in degrees). Starting point for playtests.
AR_POINTS = [(0, 359), (5, 215), (7, 150), (8, 115), (9, 85), (10, 60), (11, 30)]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lookahead_deg(ar: float) -> float:
    ar = clamp(float(ar), 0.0, 11.0)
    for (a0, d0), (a1, d1) in zip(AR_POINTS, AR_POINTS[1:]):
        if ar <= a1:
            return d0 + (d1 - d0) * (ar - a0) / (a1 - a0)
    return AR_POINTS[-1][1]


def windows(od: float):
    """(perfect, great, good) half-windows in seconds. osu!standard formulas.
    Not rounded, so float OD works."""
    od = clamp(float(od), 0.0, 11.0)
    return (80 - 6 * od) / 1000.0, (140 - 8 * od) / 1000.0, (200 - 10 * od) / 1000.0


# ------------------------------------------------------------------- chart
@dataclass
class Chart:
    audio: str = ""
    bpm: float = 120.0       # quarter notes per minute
    offset: float = 0.0      # seconds: where bar 0 starts in the song
    ar: float = 8.0
    od: float = 7.0
    dots: list = field(default_factory=list)

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar(self) -> float:
        return self.beat * BEATS_PER_BAR

    @property
    def path(self) -> str:
        return self.audio + ".circle.json"

    def phase(self, t: float) -> float:
        """0..1 around the circle, 0 = 12 o'clock."""
        return ((t - self.offset) / self.bar) % 1.0

    def angle(self, t: float) -> float:
        return self.phase(t) * 360.0

    def segment(self, t: float) -> int:
        return int(self.phase(t) * BEATS_PER_BAR) % BEATS_PER_BAR

    def save(self):
        if not self.audio:
            return
        data = dict(bpm=round(self.bpm, 4), offset=round(self.offset, 4), ar=self.ar,
                    od=self.od, dots=[round(d, 4) for d in self.dots])
        with open(self.path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, audio: str) -> "Chart":
        c = cls(audio=audio)
        if os.path.exists(c.path):
            try:
                with open(c.path) as f:
                    d = json.load(f)
                c.bpm = float(d.get("bpm", c.bpm))
                c.offset = float(d.get("offset", c.offset))
                c.ar = float(d.get("ar", c.ar))
                c.od = float(d.get("od", c.od))
                c.dots = sorted(float(x) for x in d.get("dots", []))
            except Exception:
                pass
        return c


def snap_time(t: float, offset: float, step: float) -> float:
    if step <= 0:
        return t
    return offset + round((t - offset) / step) * step


# ------------------------------------------------------------------ presses
BUTTON_KEYS = "DFJK"


@dataclass
class Press:
    t0: float
    n: int
    times: list
    slot: int = 0

    @property
    def last(self) -> float:
        return self.times[-1]


def make_press(t0: float, n: int, beat: float, slot: int = 0) -> Press:
    """n hits spread evenly across one pulse (= one quarter in 4/4),
    first hit exactly at the press."""
    return Press(t0, n, [t0 + k * beat / n for k in range(n)], slot)


# -------------------------------------------------------------------- judge
TIER_BASE = {"perfect": 300, "great": 200, "good": 100}
TIER_ACC = {"perfect": 1.0, "great": 0.7, "good": 0.35}
TIER_HEAL = {"perfect": 0.015, "great": 0.010, "good": 0.005}
MISS_DRAIN = 0.08


class Judge:
    """Per-dot judging. A hit clears the nearest uncleared dot inside the Good
    window. Misses drain health. Stray hits are free."""

    def __init__(self, times, od):
        self.times = list(times)
        self.state = [0] * len(self.times)  # 0 pending, 1 cleared, 2 missed
        self.perfect, self.great, self.good = windows(od)
        self.miss_ptr = 0
        self.counts = {"perfect": 0, "great": 0, "good": 0, "miss": 0}
        self.signed = []  # signed error (s) of every clear, + = late
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.health = 1.0
        self.acc_sum = 0.0
        self.presses = 0
        self.hits_fired = 0
        self.strays = 0

    # --- queries
    def nearest_free(self, t):
        lo = bisect.bisect_left(self.times, t - self.good)
        hi = bisect.bisect_right(self.times, t + self.good)
        best, bd = None, 1e9
        for i in range(lo, hi):
            if self.state[i] == 0:
                d = abs(self.times[i] - t)
                if d < bd:
                    best, bd = i, d
        return best

    def tier(self, dt):
        a = abs(dt)
        if a <= self.perfect:
            return "perfect"
        if a <= self.great:
            return "great"
        return "good"

    # --- events
    def hit(self, t):
        """Resolve one scheduled hit. Returns (dot_index, tier, signed_dt) or None for a stray."""
        self.hits_fired += 1
        i = self.nearest_free(t)
        if i is None:
            self.strays += 1
            return None
        self.state[i] = 1
        dt = t - self.times[i]
        tier = self.tier(dt)
        self.counts[tier] += 1
        self.signed.append(dt)
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        self.score += int(TIER_BASE[tier] * (1 + min(self.combo, 100) / 50))
        self.acc_sum += TIER_ACC[tier]
        self.health = min(1.0, self.health + TIER_HEAL[tier])
        return i, tier, dt

    def sweep(self, now):
        """Mark dots whose Good window has fully passed as missed."""
        out = []
        while self.miss_ptr < len(self.times) and self.times[self.miss_ptr] + self.good < now:
            i = self.miss_ptr
            if self.state[i] == 0:
                self.state[i] = 2
                self.counts["miss"] += 1
                self.combo = 0
                self.health = max(0.0, self.health - MISS_DRAIN)
                out.append(i)
            self.miss_ptr += 1
        return out

    # --- results
    @property
    def done(self):
        return self.miss_ptr >= len(self.times)

    @property
    def accuracy(self):
        n = len(self.times)
        return 100.0 * self.acc_sum / n if n else 100.0

    @property
    def mean_error_ms(self):
        return 1000.0 * sum(self.signed) / len(self.signed) if self.signed else 0.0

    def grade(self):
        a, m = self.accuracy, self.counts["miss"]
        if a >= 98 and m == 0:
            return "SS"
        if a >= 95:
            return "S"
        if a >= 90:
            return "A"
        if a >= 80:
            return "B"
        if a >= 70:
            return "C"
        return "D"


# --------------------------------------------------------------------- audio
def tone(sr, freq, dur, vol, decay=60.0, noise=0.0):
    t = np.arange(int(sr * dur)) / sr
    w = np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)
    if noise:
        w = w + noise * np.random.uniform(-1, 1, len(t)) * np.exp(-t * decay * 2)
    w = np.clip(w * vol, -1, 1)
    mono = (w * 32767).astype(np.int16)
    return np.ascontiguousarray(np.stack([mono, mono], 1))


def stretch(base: np.ndarray, rate: float) -> np.ndarray:
    """Play-speed change. Uses audiotsm (pitch-preserving) when installed,
    otherwise falls back to tape-style resampling (pitch follows speed)."""
    if abs(rate - 1.0) < 1e-3:
        return base
    try:
        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter

        x = np.ascontiguousarray((base.astype(np.float32) / 32768.0).T)
        writer = ArrayWriter(2)
        wsola(2, speed=rate).run(ArrayReader(x), writer)
        y = np.clip(writer.data.T, -1, 1)
        return np.ascontiguousarray((y * 32767).astype(np.int16))
    except Exception:
        n = len(base)
        m = int(n / rate)
        idx = np.linspace(0, n - 1, m)
        src = np.arange(n)
        out = np.stack([np.interp(idx, src, base[:, c]) for c in range(2)], 1)
        return np.ascontiguousarray(out.astype(np.int16))


def bake_buffer(arr, sr, t_start, rate, lead, clicks=None):
    """Build the buffer that gets played: `lead` seconds of silence, then the
    (already stretched) audio from chart time t_start. Optional metronome clicks
    are mixed in at exact sample positions so they can't drift against the music.

    clicks = (offset, beat, click_normal, click_accent) or None.
    Chart time of buffer sample j = t_start + (j - lead*sr) / sr * rate."""
    lead_i = int(lead * sr)
    i0 = int(round(t_start / rate * sr))
    pad = 0
    if i0 < 0:
        pad = -i0
        i0 = 0
    body = arr[i0:]
    buf = np.zeros((lead_i + pad + len(body), 2), np.int16)
    buf[lead_i + pad:] = body
    if clicks:
        offset, beat, c_norm, c_acc = clicks
        t_end = t_start + (len(buf) / sr - lead) * rate
        k0 = math.ceil((t_start - offset) / beat)
        k1 = math.floor((t_end - offset) / beat)
        for k in range(k0, k1 + 1):
            tb = offset + k * beat
            j = lead_i + int(round((tb - t_start) / rate * sr))
            c = c_acc if k % BEATS_PER_BAR == 0 else c_norm
            if j < 0 or j >= len(buf):
                continue
            seg = buf[j:j + len(c)]
            mixed = seg.astype(np.int32) + c[:len(seg)].astype(np.int32)
            buf[j:j + len(seg)] = np.clip(mixed, -32768, 32767)
    return buf


def tap_bpm(taps):
    """Tap tempo. taps = list of perf_counter times. Returns BPM or None."""
    if len(taps) < 4:
        return None
    gaps = [b - a for a, b in zip(taps, taps[1:])][-7:]
    return round(60.0 / (sum(gaps) / len(gaps)), 2)
