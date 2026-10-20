"""
circle_common.py - pygame pieces shared by the game (circle.py) and the
editor (editor.py): display setup, sprites, audio playback clock, drawing.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pygame

from circle_core import BEATS_PER_BAR, bake_buffer, clamp, stretch, tone, windows

# ------------------------------------------------------------------ config
W, H = 1280, 760
CX, CY, R = 640, 350, 225
FPS_CAP = 240
LEAD_IN = 1.2  # seconds of silence before the music starts

BG = (8, 9, 18)
# Hue per bar segment (one per beat in 4/4). Buttons stay neutral.
SEG_COLORS = [(255, 92, 162), (255, 190, 84), (72, 222, 255), (160, 128, 255)]
KEYMAP = {pygame.K_d: 1, pygame.K_f: 2, pygame.K_j: 3, pygame.K_k: 4}
BTN_X = [W // 2 + (i - 1.5) * 112 for i in range(4)]
BTN_Y, BTN_S = 662, 88

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "circle_settings.json")
DEFAULTS = dict(latency_ms=0.0, nofail=False, hitsound=True, bloom=True, play_rate=1.0,
                edit_rate=0.7, start_bar=0, last_song="", snap_i=3)


def load_settings():
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH) as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s


def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(s, f, indent=1)
    except Exception:
        pass


# --------------------------------------------------------------- helpers
def mul(c, k):
    return (int(clamp(c[0] * k, 0, 255)), int(clamp(c[1] * k, 0, 255)), int(clamp(c[2] * k, 0, 255)))


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def pos_at(ang, r):
    a = math.radians(ang)
    return (CX + r * math.sin(a), CY - r * math.cos(a))


def arc_pts(r, a0, a1, s=1.0):
    n = max(2, int(abs(a1 - a0) / 2.0) + 2)
    pts = []
    for i in range(n):
        x, y = pos_at(a0 + (a1 - a0) * i / (n - 1), r)
        pts.append((x * s, y * s))
    return pts


def band(r0, r1, a0, a1, s=1.0):
    return arc_pts(r1, a0, a1, s) + arc_pts(r0, a0, a1, s)[::-1]


def to_surface(rgb_float):
    """(h, w, 3) float 0..1 -> display-format Surface."""
    arr = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1))).convert()


def radii(size):
    c = size / 2
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.hypot(x - c + 0.5, y - c + 0.5)


def make_dot_sprite(col, size=72):
    r = radii(size)
    halo = 0.42 * np.exp(-(r / 19.0) ** 2)
    ring = np.exp(-((r - 15.0) / 1.7) ** 2)
    core = 1 / (1 + np.exp((r - 6.2) / 1.0))
    c = np.array(col, np.float32) / 255
    rgb = c * (halo + ring)[..., None] + (c * 0.35 + 0.65) * core[..., None]
    return to_surface(rgb)


def make_ring_sprite(col, size=44, radius=11.0):
    r = radii(size)
    ring = np.exp(-((r - radius) / 1.6) ** 2)
    halo = 0.35 * np.exp(-(r / 14.0) ** 2)
    c = np.array(col, np.float32) / 255
    return to_surface(c * (ring + halo)[..., None])


def make_glow(col, size=120, sigma=26.0, gain=0.9):
    r = radii(size)
    g = gain * np.exp(-(r / sigma) ** 2)
    return to_surface((np.array(col, np.float32) / 255) * g[..., None])


class Sprites:
    def __init__(self):
        self.dots = []
        for col in SEG_COLORS:
            base = make_dot_sprite(col)
            steps = []
            for i in range(8):
                p = (i + 1) / 8
                s = 1 + 2.70158 * (p - 1) ** 3 + 1.70158 * (p - 1) ** 2  # ease-out-back
                sz = max(6, int(72 * s))
                steps.append(base if i == 7 else pygame.transform.smoothscale(base, (sz, sz)))
            self.dots.append(steps)
        self.tick = [make_ring_sprite(c) for c in SEG_COLORS]
        self.glow = [make_glow(c) for c in SEG_COLORS]
        self.ring = self.build_ring()
        self.vignette = self.build_vignette()
        self.snap_cache = {}

    @staticmethod
    def build_ring():
        S = 2  # supersample for clean edges
        surf = pygame.Surface((W * S, H * S), pygame.SRCALPHA)
        for i, col in enumerate(SEG_COLORS):
            a0, a1 = i * 90 + 1.4, (i + 1) * 90 - 1.4
            pygame.draw.polygon(surf, (*col, 36), band(R - 10, R + 10, a0, a1, S))
            pygame.draw.polygon(surf, (*col, 170), band(R - 3.5, R + 3.5, a0, a1, S))
        for k in range(16):  # sixteenth-note ticks, faint
            a = k * 22.5
            if k % 4:
                p0, p1 = pos_at(a, R - 9), pos_at(a, R + 9)
                pygame.draw.line(surf, (255, 255, 255, 45), (p0[0] * S, p0[1] * S), (p1[0] * S, p1[1] * S), 2 * S)
        for k in range(4):  # beat lines
            a = k * 90
            p0, p1 = pos_at(a, R - 18), pos_at(a, R + 18)
            pygame.draw.line(surf, (255, 255, 255, 170 if k == 0 else 85),
                             (p0[0] * S, p0[1] * S), (p1[0] * S, p1[1] * S), (3 if k == 0 else 2) * S)
        pygame.draw.circle(surf, (255, 255, 255, 28), (CX * S, CY * S), int(R * 0.3 * S), 2 * S)
        return pygame.transform.smoothscale(surf, (W, H)).convert_alpha()

    @staticmethod
    def build_vignette():
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot((x - CX) / (W * 0.62), (y - CY) / (H * 0.68))
        v = np.clip(1.0 - 0.55 * d ** 2, 0.35, 1.0)
        return to_surface(np.repeat(v[..., None], 3, axis=2))

    def snap_ticks(self, div):
        if div in self.snap_cache:
            return self.snap_cache[div]
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        if div:
            for k in range(div * BEATS_PER_BAR):
                a = k * 360.0 / (div * BEATS_PER_BAR)
                big = (k % div == 0)
                p0, p1 = pos_at(a, R + 14), pos_at(a, R + (28 if big else 21))
                pygame.draw.line(surf, (255, 255, 255, 130 if big else 70), p0, p1, 2)
        self.snap_cache[div] = surf.convert_alpha()
        return self.snap_cache[div]


# ------------------------------------------------------------- audio side
class Song:
    def __init__(self, path):
        snd = pygame.mixer.Sound(path)
        arr = pygame.sndarray.array(snd)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], 1)
        self.path = path
        self.base = np.ascontiguousarray(arr.astype(np.int16))
        self.sr = pygame.mixer.get_init()[0]
        self.length = len(self.base) / self.sr
        self.cache = {1.0: self.base}

    def prepare(self, rate):
        if rate not in self.cache:
            self.cache[rate] = stretch(self.base, rate)


class Playback:
    """Own clock on top of perf_counter. The buffer is built with LEAD_IN of
    silence in front, so audio start and clock start are the same instant."""

    def __init__(self):
        self.chan = pygame.mixer.Channel(0)
        self.song = None
        self.snd = None
        self.rate = 1.0
        self.t_start = 0.0
        self.start_perf = 0.0
        self.hold = 0.0
        self.paused = True
        self.latency = 0.0  # seconds
        sr = pygame.mixer.get_init()[0]
        self.c_norm = tone(sr, 1500, 0.03, 0.55, decay=90)
        self.c_acc = tone(sr, 2300, 0.03, 0.7, decay=90)

    def start(self, t, rate, metro=None):
        sr = self.song.sr
        clicks = (metro[0], metro[1], self.c_norm, self.c_acc) if metro else None
        buf = bake_buffer(self.song.cache[rate], sr, t, rate, LEAD_IN, clicks)
        self.snd = pygame.sndarray.make_sound(buf)
        self.chan.stop()
        self.chan.play(self.snd)
        self.start_perf = time.perf_counter()
        self.t_start, self.rate, self.paused = t, rate, False

    def time(self):
        if self.paused:
            return self.hold
        return self.t_start + (time.perf_counter() - self.start_perf - LEAD_IN - self.latency) * self.rate

    def pause(self):
        if not self.paused:
            self.hold = self.time()
            self.chan.stop()
            self.paused = True

    def stop(self):
        self.pause()


def pick_file():
    """Native file dialog in a subprocess (tkinter + pygame in one process is flaky on macOS)."""
    code = ("import tkinter as tk\nfrom tkinter import filedialog\n"
            "r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
            "print(filedialog.askopenfilename(filetypes=[('Audio','*.mp3 *.ogg *.wav *.flac')]))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    except Exception:
        return ""


def init_display(caption):
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512, allowedchanges=0)
    except Exception:
        pygame.mixer.init()
    pygame.init()
    pygame.display.set_caption(caption)
    try:
        screen = pygame.display.set_mode((W, H), pygame.SCALED, vsync=0)
    except Exception:
        screen = pygame.display.set_mode((W, H))
    pygame.mixer.set_num_channels(40)
    pygame.mixer.set_reserved(1)
    return screen


def make_hitsounds():
    sr = pygame.mixer.get_init()[0]
    clear = pygame.sndarray.make_sound(tone(sr, 1900, 0.05, 0.55, decay=70, noise=0.1))
    stray = pygame.sndarray.make_sound(tone(sr, 900, 0.04, 0.22, decay=80))
    return clear, stray


# ---------------------------------------------------------------- drawing
class Gfx:
    """Everything that draws the circle, hand, buttons and effects. Game and
    Editor both draw through one of these."""

    def __init__(self, screen):
        self.screen = screen
        self.f_huge = pygame.font.Font(None, 150)
        self.f_big = pygame.font.Font(None, 74)
        self.f_mid = pygame.font.Font(None, 38)
        self.f_small = pygame.font.Font(None, 28)
        self.f_tiny = pygame.font.Font(None, 22)
        self.tcache = {}
        self.spr = Sprites()
        self.dim = pygame.Surface((W, H), pygame.SRCALPHA)
        self.dim.fill((4, 5, 12, 175))
        self.trail = pygame.Surface((2 * (R + 12), 2 * (R + 12)), pygame.SRCALPHA)
        self.rings, self.sparks = [], []
        self.btn_flash = [0.0] * 4

    # ---- text
    def txt(self, s, font, col, pos, anchor="center", alpha=255):
        key = (s, id(font), col)
        surf = self.tcache.get(key)
        if surf is None:
            if len(self.tcache) > 600:
                self.tcache.clear()
            surf = font.render(s, True, col)
            self.tcache[key] = surf
        rect = surf.get_rect(**{anchor: pos})
        if alpha < 255:
            surf = surf.copy()
            surf.set_alpha(alpha)
        self.screen.blit(surf, rect)

    # ---- post
    def bloom(self):
        scr = self.screen
        a = pygame.transform.smoothscale(scr, (W // 4, H // 4))
        b = pygame.transform.smoothscale(a, (W // 12, H // 12))
        a.fill((120, 120, 120), special_flags=pygame.BLEND_RGB_MULT)
        b.fill((170, 170, 170), special_flags=pygame.BLEND_RGB_MULT)
        scr.blit(pygame.transform.smoothscale(a, (W, H)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        scr.blit(pygame.transform.smoothscale(b, (W, H)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # ---- world
    def draw_base(self, now, ch, snap_div=None):
        """Background, ring, optional snap ticks, sweep trail. Returns (hand angle, segment)."""
        scr = self.screen
        ang = ch.angle(now)
        seg = ch.segment(now)
        beat_ph = ((now - ch.offset) / ch.beat) % 1.0
        pulse = math.exp(-beat_ph * 5.0)
        tint = SEG_COLORS[seg]
        scr.fill(tuple(int(BG[i] + tint[i] * (0.012 + 0.075 * pulse)) for i in range(3)))
        scr.blit(self.spr.vignette, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        scr.blit(self.spr.ring, (0, 0))
        if snap_div is not None:
            scr.blit(self.spr.snap_ticks(snap_div), (0, 0))
        self.draw_trail(ang)
        return ang, seg

    def draw_trail(self, ang):
        ts = self.trail
        ts.fill((0, 0, 0, 0))
        half = ts.get_width() / 2
        n, span = 30, 85.0
        for i in range(n):
            a1 = ang - i * span / n
            a0 = ang - (i + 1) * span / n
            col = SEG_COLORS[int((a1 % 360) // 90) % 4]
            alpha = int(95 * (1 - i / n) ** 2)
            pts = []
            for r_, a_ in ((R * 0.34, a0), (R + 4, a0), (R + 4, a1), (R * 0.34, a1)):
                x, y = pos_at(a_, r_)
                pts.append((x - CX + half, y - CY + half))
            pygame.draw.polygon(ts, (*col, alpha), pts)
        self.screen.blit(ts, (CX - half, CY - half))

    def draw_window(self, ang, seg, ch):
        p, g, b = windows(ch.od)
        col = SEG_COLORS[seg]
        for w, k in ((b, 0.22), (g, 0.42), (p, 0.9)):
            da = w / ch.bar * 360.0
            pygame.draw.polygon(self.screen, mul(mix(col, (255, 255, 255), 0.35), k),
                                band(R - 7, R + 7, ang - da, ang + da))

    def draw_hand(self, ang, seg):
        scr = self.screen
        col = mix(SEG_COLORS[seg], (255, 255, 255), 0.7)
        p0, p1 = pos_at(ang, R * 0.34), pos_at(ang, R + 8)
        pygame.draw.line(scr, mul(SEG_COLORS[seg], 0.35), p0, p1, 9)
        pygame.draw.line(scr, col, p0, p1, 3)
        g = self.spr.glow[seg]
        tip = pos_at(ang, R)
        scr.blit(g, g.get_rect(center=(int(tip[0]), int(tip[1]))), special_flags=pygame.BLEND_RGB_ADD)

    def draw_dot(self, pos, seg, step=7):
        spr = self.spr.dots[seg][step]
        self.screen.blit(spr, spr.get_rect(center=(int(pos[0]), int(pos[1]))), special_flags=pygame.BLEND_RGB_ADD)

    def draw_idle(self, ch):
        """Slowly sweeping circle behind menus and overlays."""
        ang, seg = self.draw_base(time.perf_counter() * 0.8, ch)
        self.draw_fx()
        self.draw_hand(ang, seg)

    # ---- fx
    def ripple(self, ang, seg):
        x, y = pos_at(ang, R)
        self.rings.append([x, y, time.perf_counter(), 0.28, SEG_COLORS[seg], 20, "ripple"])

    def draw_fx(self):
        scr, t = self.screen, time.perf_counter()
        keep = []
        for f in self.rings:
            p = (t - f[2]) / f[3]
            if p >= 1:
                continue
            keep.append(f)
            x, y, _, _, col, size, kind = f
            if kind == "miss":
                rad = size * (1 - 0.65 * p)
            else:
                rad = size * (0.3 + 0.7 * (1 - (1 - p) ** 3))
            pygame.draw.circle(scr, mul(col, (1 - p) ** 1.3), (int(x), int(y)), max(2, int(rad)),
                               max(1, int((5 if kind == "burst" else 3) * (1 - p))))
        self.rings = keep
        keep = []
        for s in self.sparks:
            p = (t - s[4]) / s[5]
            if p >= 1:
                continue
            keep.append(s)
            dt = t - s[4]
            x, y = s[0] + s[2] * dt, s[1] + s[3] * dt
            pygame.draw.circle(scr, mul(s[6], 1 - p), (int(x), int(y)), 2)
        self.sparks = keep

    # ---- buttons
    def draw_buttons(self):
        scr, t = self.screen, time.perf_counter()
        for i in range(4):
            f = max(0.0, 1 - (t - self.btn_flash[i]) / 0.18)
            rect = pygame.Rect(0, 0, BTN_S, BTN_S)
            rect.center = (int(BTN_X[i]), BTN_Y)
            pygame.draw.rect(scr, mix((26, 28, 44), (225, 230, 255), f * 0.9), rect, border_radius=20)
            pygame.draw.rect(scr, mix((84, 90, 118), (255, 255, 255), f), rect, 3, border_radius=20)
            dot_col = mix((205, 210, 235), (20, 22, 40), f)
            offs = {1: [(0, 0)], 2: [(-13, 0), (13, 0)], 3: [(0, -13), (-14, 10), (14, 10)],
                    4: [(-13, -13), (13, -13), (-13, 13), (13, 13)]}[i + 1]
            for ox, oy in offs:
                pygame.draw.circle(scr, dot_col, (rect.centerx + ox, rect.centery + oy), 7)
            self.txt("DFJK"[i], self.f_tiny, (110, 116, 145), (rect.centerx, rect.bottom + 14))
