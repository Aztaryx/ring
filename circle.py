#!/usr/bin/env python3
"""
CIRCLE (working title) - prototype.

    pip install -r requirements.txt
    python circle.py [song.mp3]

Drop an audio file on the window (or press Enter on the Song row) to start.
Menu: Up/Down pick a row, Left/Right change it (Shift = fine, Ctrl = coarse),
      T = tap tempo, P = play, E = record/edit, Esc = quit.
Play: D F J K = buttons (1-4 hits), R = retry, Esc = back.
Edit: Space or D/F/J/K place 1-4 dots at the (snapped) moment you press.
      [ ] snap, Z undo, Backspace (hold) erase, Enter pause, Left/Right seek 1 bar,
      Up/Down seek 8 bars, M metronome, , . offset, - = BPM, Esc save and exit.
"""
from __future__ import annotations

import bisect
import heapq
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np
import pygame

from circle_core import (BEATS_PER_BAR, Chart, Judge, clamp, lookahead_deg, make_press,
                         bake_buffer, snap_time, stretch, tap_bpm, tone, windows)

# ------------------------------------------------------------------ config
W, H = 1280, 760
CX, CY, R = 640, 350, 225
FPS_CAP = 240
LEAD_IN = 1.2  # seconds of silence before the music starts

BG = (8, 9, 18)
# Hue per bar segment (one per beat in 4/4). Buttons stay neutral.
SEG_COLORS = [(255, 92, 162), (255, 190, 84), (72, 222, 255), (160, 128, 255)]
TIER_COLOR = {"perfect": (255, 255, 255), "great": None, "good": None}
TIER_SIZE = {"perfect": 52, "great": 38, "good": 26}
TIER_TEXT = {"perfect": "PERFECT", "great": "GREAT", "good": "GOOD"}
SNAPS = [1, 2, 3, 4, 6, 8, 12, 16, 0]  # subdivisions of a beat, 0 = free
KEYMAP = {pygame.K_d: 1, pygame.K_f: 2, pygame.K_j: 3, pygame.K_k: 4}
BTN_X = [W // 2 + (i - 1.5) * 112 for i in range(4)]
BTN_Y, BTN_S = 662, 88

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "circle_settings.json")
DEFAULTS = dict(latency_ms=0.0, nofail=False, hitsound=True, bloom=True, play_rate=1.0,
                edit_rate=0.7, start_bar=0, last_song="")

ROWS = ["song", "bpm", "offset", "ar", "od", "start", "prate", "erate", "latency",
        "hitsound", "nofail", "bloom", "play", "edit"]


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
        self.glow_white = make_glow((255, 255, 255), sigma=22.0)
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


# -------------------------------------------------------------------- game
class Run:
    """State of one play session."""

    def __init__(self, judge, t_start, end_time):
        self.judge = judge
        self.t_start = t_start
        self.end_time = end_time
        self.heap = []       # (hit_time, press_id, k)
        self.presses = []
        self.next_id = 0
        self.failed = False


class Game:
    def __init__(self, song_path=None):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512, allowedchanges=0)
        except Exception:
            pygame.mixer.init()
        pygame.init()
        pygame.display.set_caption("Circle (working title)")
        try:
            self.screen = pygame.display.set_mode((W, H), pygame.SCALED, vsync=0)
        except Exception:
            self.screen = pygame.display.set_mode((W, H))
        pygame.mixer.set_num_channels(40)
        pygame.mixer.set_reserved(1)
        self.clock = pygame.time.Clock()
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
        self.pb = Playback()
        sr = pygame.mixer.get_init()[0]
        self.hs_clear = pygame.sndarray.make_sound(tone(sr, 1900, 0.05, 0.55, decay=70, noise=0.1))
        self.hs_stray = pygame.sndarray.make_sound(tone(sr, 900, 0.04, 0.22, decay=80))

        self.s = dict(DEFAULTS)
        try:
            with open(SETTINGS_PATH) as f:
                self.s.update(json.load(f))
        except Exception:
            pass
        self.song = None
        self.chart = Chart()
        self.state = "menu"
        self.sel = 0
        self.notice = ("", 0.0)
        self.taps = []
        self.now = 0.0
        self.run = None
        self.loading = None
        self.result = None
        self.btn_flash = [0.0] * 4
        self.rings, self.sparks = [], []
        self.jtext = None
        self.miss_flash = 0.0
        # editor
        self.snap_i = 3
        self.metro = True
        self.undo = []
        self.rebuild_at = None
        pygame.key.set_repeat(300, 45)

        path = song_path or self.s.get("last_song")
        if path and os.path.exists(path):
            self.load_song(path)

    # ------------------------------------------------------------- misc
    def save_settings(self):
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(self.s, f, indent=1)
        except Exception:
            pass

    def flash(self, msg):
        self.notice = (msg, time.perf_counter())

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

    def load_song(self, path):
        try:
            self.flash("Loading audio...")
            self.draw()
            pygame.display.flip()
            self.song = Song(path)
        except Exception as e:  # noqa: BLE001
            self.song = None
            self.flash(f"Couldn't load that file ({e.__class__.__name__})")
            return
        self.pb.song = self.song
        self.chart = Chart.load(path)
        self.s["last_song"] = path
        self.save_settings()
        self.flash(f"Loaded {os.path.basename(path)}")

    # ------------------------------------------------------------- loop
    def run_forever(self):
        while True:
            self.frame()

    def frame(self):
        for e in pygame.event.get():
            self.handle_event(e)
        if self.state == "play":
            self.update_play()
        elif self.state == "edit":
            self.update_edit()
        elif self.state == "loading":
            self.update_loading()
        self.draw()
        pygame.display.flip()
        self.clock.tick(FPS_CAP)

    def quit(self):
        self.chart.save()
        self.save_settings()
        pygame.quit()
        sys.exit(0)

    def event_time(self, e):
        """Chart time of a key event. Uses the SDL timestamp when the build exposes
        one (removes up to a frame of input jitter), otherwise the poll time."""
        now = self.pb.time()
        ts = getattr(e, "timestamp", None)
        if ts:
            age = pygame.time.get_ticks() - ts
            if 0 <= age <= 100:
                return now - age / 1000.0 * self.pb.rate
        return now

    # ------------------------------------------------------------ events
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            self.quit()
        elif e.type == pygame.DROPFILE and self.state == "menu":
            self.load_song(e.file)
        elif e.type == pygame.KEYDOWN:
            if self.state == "menu":
                self.key_menu(e)
            elif self.state == "play":
                self.key_play(e)
            elif self.state == "edit":
                self.key_edit(e)
            elif self.state == "results":
                if e.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    self.to_menu()
                elif e.key == pygame.K_r:
                    self.begin("play")
            elif self.state == "loading" and e.key == pygame.K_ESCAPE:
                self.to_menu()

    def to_menu(self):
        self.pb.stop()
        self.state = "menu"
        self.loading = None
        pygame.key.set_repeat(300, 45)

    # -------------------------------------------------------------- menu
    def key_menu(self, e):
        k, mod = e.key, e.mod
        rid = ROWS[self.sel]
        if k == pygame.K_ESCAPE:
            self.quit()
        elif k == pygame.K_UP:
            self.sel = (self.sel - 1) % len(ROWS)
        elif k == pygame.K_DOWN:
            self.sel = (self.sel + 1) % len(ROWS)
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            self.adjust(rid, 1 if k == pygame.K_RIGHT else -1, mod)
        elif k == pygame.K_RETURN:
            if rid == "song":
                self.open_dialog()
            elif rid in ("play", "edit"):
                self.begin(rid)
            else:
                self.adjust(rid, 1, mod)
        elif k == pygame.K_o:
            self.open_dialog()
        elif k == pygame.K_p:
            self.begin("play")
        elif k == pygame.K_e:
            self.begin("edit")
        elif k == pygame.K_t:
            now = time.perf_counter()
            if self.taps and now - self.taps[-1] > 2.0:
                self.taps = []
            self.taps.append(now)
            bpm = tap_bpm(self.taps)
            if bpm:
                self.chart.bpm = clamp(bpm, 30, 400)

    def open_dialog(self):
        p = pick_file()
        pygame.event.clear()
        if p:
            self.load_song(p)

    def adjust(self, rid, d, mod):
        shift, ctrl = bool(mod & pygame.KMOD_SHIFT), bool(mod & pygame.KMOD_CTRL)

        def step(base, fine, coarse):
            return coarse if ctrl else fine if shift else base

        ch, s = self.chart, self.s
        if rid == "bpm":
            ch.bpm = round(clamp(ch.bpm + d * step(0.1, 0.01, 1.0), 30, 400), 3)
        elif rid == "offset":
            ch.offset = round(ch.offset + d * step(0.01, 0.001, 0.1), 4)
        elif rid == "ar":
            ch.ar = round(clamp(ch.ar + d * step(0.1, 0.01, 1.0), 0, 11), 3)
        elif rid == "od":
            ch.od = round(clamp(ch.od + d * step(0.1, 0.01, 1.0), 0, 11), 3)
        elif rid == "start":
            s["start_bar"] = int(max(0, s["start_bar"] + d * step(1, 1, 8)))
        elif rid in ("prate", "erate"):
            key = "play_rate" if rid == "prate" else "edit_rate"
            s[key] = round(clamp(s[key] + d * step(0.05, 0.01, 0.25), 0.3, 1.5), 2)
        elif rid == "latency":
            s["latency_ms"] = round(s["latency_ms"] + d * step(5, 1, 25), 1)
        elif rid == "hitsound":
            s["hitsound"] = not s["hitsound"]
        elif rid == "nofail":
            s["nofail"] = not s["nofail"]
        elif rid == "bloom":
            s["bloom"] = not s["bloom"]

    def row_info(self, rid):
        ch, s = self.chart, self.s
        p, g, b = windows(ch.od)
        return {
            "song": ("Song", os.path.basename(ch.audio) if self.song else "drop a file here, or press Enter"),
            "bpm": ("BPM (quarter notes)", f"{ch.bpm:.2f}      T = tap tempo"),
            "offset": ("Bar-0 offset", f"{ch.offset * 1000:.0f} ms"),
            "ar": ("AR", f"{ch.ar:.2f}     {lookahead_deg(ch.ar):.0f} deg ahead"),
            "od": ("OD", f"{ch.od:.2f}     {p * 1000:.0f} / {g * 1000:.0f} / {b * 1000:.0f} ms"),
            "start": ("Start at bar", str(s["start_bar"])),
            "prate": ("Play speed", f"{s['play_rate']:.2f}x"),
            "erate": ("Record speed", f"{s['edit_rate']:.2f}x"),
            "latency": ("Audio latency", f"{s['latency_ms']:.0f} ms   (late hits? raise it)"),
            "hitsound": ("Hit sounds", "on" if s["hitsound"] else "off"),
            "nofail": ("No fail", "on" if s["nofail"] else "off"),
            "bloom": ("Bloom", "on" if s["bloom"] else "off"),
            "play": ("PLAY", f"{len(ch.dots)} dots"),
            "edit": ("RECORD / EDIT", "tap-record at slow speed"),
        }[rid]

    # ----------------------------------------------------------- sessions
    def begin(self, mode):
        if not self.song:
            self.flash("Load a song first (drop a file on the window).")
            return
        if mode == "play" and not self.chart.dots:
            self.flash("This chart is empty. Record some dots first (E).")
            return
        rate = round(self.s["play_rate"] if mode == "play" else self.s["edit_rate"], 2)
        self.chart.save()
        self.save_settings()
        if rate not in self.song.cache:
            th = threading.Thread(target=self.song.prepare, args=(rate,), daemon=True)
            th.start()
            self.loading = (mode, rate, th)
            self.state = "loading"
            return
        self.launch(mode, rate)

    def update_loading(self):
        mode, rate, th = self.loading
        if not th.is_alive():
            self.loading = None
            self.launch(mode, rate)

    def start_time(self):
        b = self.s["start_bar"]
        return 0.0 if b == 0 else self.chart.offset + b * self.chart.bar

    def launch(self, mode, rate):
        ch = self.chart
        self.pb.latency = self.s["latency_ms"] / 1000.0
        self.rings, self.sparks, self.jtext = [], [], None
        pygame.key.set_repeat(0)
        t0 = self.start_time()
        if mode == "play":
            times = [d for d in ch.dots if d >= t0 - LEAD_IN * rate]
            end = (times[-1] if times else 0) + 1.5
            self.run = Run(Judge(times, ch.od), t0, end)
            self.pb.start(t0, rate)
            self.state = "play"
        else:
            self.undo = []
            self.rebuild_at = None
            self.pb.start(t0, rate, (ch.offset, ch.beat) if self.metro else None)
            self.state = "edit"
        self.now = self.pb.time()

    # --------------------------------------------------------------- play
    def key_play(self, e):
        if e.key == pygame.K_ESCAPE:
            self.to_menu()
        elif e.key == pygame.K_r:
            self.begin("play")
        elif e.key in KEYMAP:
            self.press(KEYMAP[e.key], self.event_time(e))

    def press(self, n, t):
        r, ch = self.run, self.chart
        used = {p.slot for p in r.presses}
        slot = 0
        while slot in used:
            slot += 1
        p = make_press(t, n, ch.beat, slot)
        p.lit = [None] * n
        pid = r.next_id
        r.next_id += 1
        r.presses.append(p)
        r.judge.presses += 1
        for k, ht in enumerate(p.times):
            heapq.heappush(r.heap, (ht, pid, k))
        self.btn_flash[n - 1] = time.perf_counter()

    def update_play(self):
        r, ch = self.run, self.chart
        j = r.judge
        now = self.pb.time()
        self.now = now
        while r.heap and r.heap[0][0] <= now:
            self.fire_hit(heapq.heappop(r.heap)[0])
        for i in j.sweep(now):
            self.fx_miss(i)
        r.presses = [p for p in r.presses if now - p.last < 0.25]
        if j.health <= 0 and not self.s["nofail"]:
            self.finish(True)
        elif j.done and now > r.end_time:
            self.finish(False)

    def fire_hit(self, ht):
        r, ch = self.run, self.chart
        j = r.judge
        res = j.hit(ht)
        self.fx_ripple(ch.angle(ht), ch.segment(ht))
        if res is None:
            if self.s["hitsound"]:
                self.hs_stray.play()
            return
        i, tier, dt = res
        td = j.times[i]
        self.fx_burst(tier, ch.angle(td), ch.segment(td), dt)
        if self.s["hitsound"]:
            self.hs_clear.play()

    def finish(self, failed):
        j = self.run.judge
        self.result = dict(failed=failed, score=j.score, acc=j.accuracy, grade="F" if failed else j.grade(),
                           counts=dict(j.counts), combo=j.max_combo, presses=j.presses,
                           hits=j.hits_fired, strays=j.strays, err=j.mean_error_ms, total=len(j.times))
        self.pb.stop()
        self.state = "results"
        pygame.key.set_repeat(0)

    # ---------------------------------------------------------------- fx
    def fx_ripple(self, ang, seg):
        x, y = pos_at(ang, R)
        self.rings.append([x, y, time.perf_counter(), 0.28, SEG_COLORS[seg], 20, "ripple"])

    def fx_burst(self, tier, ang, seg, dt):
        x, y = pos_at(ang, R)
        base = SEG_COLORS[seg]
        col = mix(base, (255, 255, 255), 0.65) if tier == "perfect" else base if tier == "great" else mul(base, 0.55)
        t = time.perf_counter()
        self.rings.append([x, y, t, 0.45, col, TIER_SIZE[tier], "burst"])
        for _ in range({"perfect": 9, "great": 5, "good": 2}[tier]):
            a = random.uniform(0, math.tau)
            v = random.uniform(60, 190)
            self.sparks.append([x, y, math.cos(a) * v, math.sin(a) * v, t, random.uniform(0.3, 0.55), col])
        px, py = pos_at(ang, R * 0.6)
        self.jtext = (TIER_TEXT[tier], col, px, py, t)

    def fx_miss(self, i):
        ch = self.chart
        td = self.run.judge.times[i]
        x, y = pos_at(ch.angle(td), R)
        t = time.perf_counter()
        grey = (120, 124, 140)
        self.rings.append([x, y, t, 0.4, grey, 22, "miss"])
        for _ in range(5):
            a = random.uniform(0, math.tau)
            v = random.uniform(20, 70)
            self.sparks.append([x, y, math.cos(a) * v, math.sin(a) * v + 30, t, 0.5, grey])
        px, py = pos_at(ch.angle(td), R * 0.6)
        self.jtext = ("MISS", grey, px, py, t)
        self.miss_flash = t

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

    # ------------------------------------------------------------- editor
    def key_edit(self, e):
        k, ch = e.key, self.chart
        shift = bool(e.mod & pygame.KMOD_SHIFT)
        if k == pygame.K_ESCAPE:
            ch.save()
            self.to_menu()
        elif k == pygame.K_RETURN:
            if self.pb.paused:
                self.restart_audio(self.pb.hold)
            else:
                self.pb.pause()
        elif k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
            bars = {pygame.K_LEFT: -1, pygame.K_RIGHT: 1, pygame.K_UP: 8, pygame.K_DOWN: -8}[k]
            self.seek(self.pb.time() + bars * ch.bar)
        elif k == pygame.K_LEFTBRACKET:
            self.snap_i = (self.snap_i - 1) % len(SNAPS)
        elif k == pygame.K_RIGHTBRACKET:
            self.snap_i = (self.snap_i + 1) % len(SNAPS)
        elif k == pygame.K_m:
            self.metro = not self.metro
            self.restart_audio(self.pb.time())
        elif k == pygame.K_z:
            self.undo_last()
        elif k in (pygame.K_COMMA, pygame.K_PERIOD):
            ch.offset = round(ch.offset + (0.001 if shift else 0.005) * (1 if k == pygame.K_PERIOD else -1), 4)
            self.rebuild_at = time.perf_counter() + 0.35
        elif k in (pygame.K_MINUS, pygame.K_EQUALS):
            ch.bpm = round(clamp(ch.bpm + (0.01 if shift else 0.05) * (1 if k == pygame.K_EQUALS else -1), 30, 400), 3)
            self.rebuild_at = time.perf_counter() + 0.35
        elif (k == pygame.K_SPACE or k in KEYMAP) and not self.pb.paused:
            self.place(1 if k == pygame.K_SPACE else KEYMAP[k], self.event_time(e))

    def seek(self, t):
        t = clamp(t, 0.0, self.song.length)
        self.restart_audio(t)

    def restart_audio(self, t):
        ch = self.chart
        self.rebuild_at = None
        self.pb.start(t, self.pb.rate, (ch.offset, ch.beat) if self.metro else None)

    def place(self, n, t):
        ch = self.chart
        div = SNAPS[self.snap_i]
        step = ch.beat / div if div else 0
        t0 = snap_time(t, ch.offset, step)
        added = []
        for k in range(n):
            x = round(t0 + k * ch.beat / n, 4)
            i = bisect.bisect_left(ch.dots, x - 0.004)
            if i < len(ch.dots) and abs(ch.dots[i] - x) < 0.004:
                continue
            ch.dots.insert(i, x)
            added.append(x)
        if added:
            self.undo.append(("add", added))
        self.btn_flash[n - 1] = time.perf_counter()
        if self.s["hitsound"]:
            self.hs_clear.play()

    def undo_last(self):
        if not self.undo:
            return
        kind, xs = self.undo.pop()
        ch = self.chart
        if kind == "add":
            for x in xs:
                i = bisect.bisect_left(ch.dots, x - 1e-6)
                if i < len(ch.dots) and abs(ch.dots[i] - x) < 1e-5:
                    ch.dots.pop(i)
        else:
            for x in xs:
                bisect.insort(ch.dots, x)

    def update_edit(self):
        ch = self.chart
        now = self.pb.time()
        self.now = now
        if self.rebuild_at and time.perf_counter() >= self.rebuild_at and not self.pb.paused:
            self.restart_audio(now)
            self.now = self.pb.time()
        if not self.pb.paused and pygame.key.get_pressed()[pygame.K_BACKSPACE]:
            lo = bisect.bisect_left(ch.dots, now - 0.07)
            hi = bisect.bisect_right(ch.dots, now + 0.07)
            if hi > lo:
                self.undo.append(("rem", ch.dots[lo:hi]))
                del ch.dots[lo:hi]
        if not self.pb.paused and now > self.song.length + 1.0:
            self.pb.pause()

    # ------------------------------------------------------------ drawing
    def draw(self):
        st = self.state
        if st == "play" or st == "edit":
            self.draw_world(self.now, st)
            if self.s["bloom"]:
                self.bloom()
            self.draw_hud(st)
        else:
            self.draw_world(time.perf_counter() * 0.8, "idle")
            if self.s["bloom"]:
                self.bloom()
            self.screen.blit(self.dim, (0, 0))
            if st == "menu":
                self.draw_menu()
            elif st == "loading":
                self.draw_loading()
            elif st == "results":
                self.draw_results()

    def bloom(self):
        scr = self.screen
        a = pygame.transform.smoothscale(scr, (W // 4, H // 4))
        b = pygame.transform.smoothscale(a, (W // 12, H // 12))
        a.fill((120, 120, 120), special_flags=pygame.BLEND_RGB_MULT)
        b.fill((170, 170, 170), special_flags=pygame.BLEND_RGB_MULT)
        scr.blit(pygame.transform.smoothscale(a, (W, H)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        scr.blit(pygame.transform.smoothscale(b, (W, H)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def draw_world(self, now, mode):
        scr, ch = self.screen, self.chart
        ang = ch.angle(now)
        seg = ch.segment(now)
        beat_ph = ((now - ch.offset) / ch.beat) % 1.0
        pulse = math.exp(-beat_ph * 5.0)
        tint = SEG_COLORS[seg]
        scr.fill(tuple(int(BG[i] + tint[i] * (0.012 + 0.075 * pulse)) for i in range(3)))
        scr.blit(self.spr.vignette, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        scr.blit(self.spr.ring, (0, 0))
        if mode == "edit":
            scr.blit(self.spr.snap_ticks(SNAPS[self.snap_i]), (0, 0))
        self.draw_trail(ang)
        if mode in ("play", "edit"):
            self.draw_window(ang, seg)
        if mode == "play":
            self.draw_play_dots(now)
            self.draw_previews(now)
        elif mode == "edit":
            self.draw_edit_dots(now)
        self.draw_fx()
        self.draw_hand(ang, seg)

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

    def draw_window(self, ang, seg):
        ch = self.chart
        p, g, b = windows(ch.od)
        col = SEG_COLORS[seg]
        for w, k in ((b, 0.22), (g, 0.42), (p, 0.9)):
            da = w / ch.bar * 360.0
            pygame.draw.polygon(self.screen, mul(mix(col, (255, 255, 255), 0.35), k), band(R - 7, R + 7, ang - da, ang + da))

    def draw_hand(self, ang, seg):
        scr = self.screen
        col = mix(SEG_COLORS[seg], (255, 255, 255), 0.7)
        p0, p1 = pos_at(ang, R * 0.34), pos_at(ang, R + 8)
        pygame.draw.line(scr, mul(SEG_COLORS[seg], 0.35), p0, p1, 9)
        pygame.draw.line(scr, col, p0, p1, 3)
        g = self.spr.glow[seg]
        tip = pos_at(ang, R)
        scr.blit(g, g.get_rect(center=(int(tip[0]), int(tip[1]))), special_flags=pygame.BLEND_RGB_ADD)

    def draw_play_dots(self, now):
        j, ch = self.run.judge, self.chart
        look = lookahead_deg(ch.ar) / 360.0 * ch.bar
        i = j.miss_ptr
        n = len(j.times)
        scr = self.screen
        while i < n and j.times[i] - look <= now:
            if j.state[i] == 0:
                t = j.times[i]
                anim = (now - (t - look)) / 0.1
                step = 7 if anim >= 1 else max(0, int(anim * 8))
                spr = self.spr.dots[ch.segment(t)][step]
                x, y = pos_at(ch.angle(t), R)
                scr.blit(spr, spr.get_rect(center=(int(x), int(y))), special_flags=pygame.BLEND_RGB_ADD)
            i += 1

    def draw_edit_dots(self, now):
        ch, scr = self.chart, self.screen
        lo = bisect.bisect_left(ch.dots, now - 0.15 * ch.bar)
        hi = bisect.bisect_right(ch.dots, now + 0.85 * ch.bar)
        for t in ch.dots[lo:hi]:
            spr = self.spr.dots[ch.segment(t)][7]
            x, y = pos_at(ch.angle(t), R)
            scr.blit(spr, spr.get_rect(center=(int(x), int(y))), special_flags=pygame.BLEND_RGB_ADD)

    def dashed(self, pos, col):
        x, y = int(pos[0]), int(pos[1])
        rect = pygame.Rect(x - 8, y - 8, 16, 16)
        for k in range(6):
            a0 = math.radians(k * 60)
            pygame.draw.arc(self.screen, col, rect, a0, a0 + math.radians(34), 2)

    def draw_previews(self, now):
        r, ch, scr = self.run, self.chart, self.screen
        j = r.judge
        for p in r.presses:
            rad = R + 30 + 12 * p.slot
            frm = max(now, p.t0)
            if p.last > frm:
                a0 = ch.angle(frm)
                pts = arc_pts(rad, a0, a0 + (p.last - frm) / ch.bar * 360.0)
                pygame.draw.lines(scr, (105, 112, 140), False, pts, 2)
            for k, tk in enumerate(p.times):
                age = now - tk
                if age > 0.2:
                    continue
                x, y = pos_at(ch.angle(tk), rad)
                if age <= 0:
                    idx = j.nearest_free(tk)
                    p.lit[k] = ch.segment(j.times[idx]) if idx is not None else -1
                    if idx is not None:
                        s = self.spr.tick[p.lit[k]]
                        scr.blit(s, s.get_rect(center=(int(x), int(y))), special_flags=pygame.BLEND_RGB_ADD)
                    else:
                        self.dashed((x, y), (110, 116, 145))
                else:
                    f = 1 - age / 0.2
                    lit = p.lit[k]
                    col = (95, 100, 125) if lit in (None, -1) else SEG_COLORS[lit]
                    pygame.draw.circle(scr, mul(col, f), (int(x), int(y)), int(10 + age * 70), 2)

    # ------------------------------------------------------------- HUD
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

    def draw_hud(self, mode):
        scr, ch, t = self.screen, self.chart, time.perf_counter()
        self.draw_buttons()
        self.txt(f"{ch.bpm:.1f}", self.f_mid, (200, 205, 230), (CX, CY + 40))
        self.txt("BPM  |  pulse 1/4", self.f_tiny, (110, 116, 145), (CX, CY + 66))
        if self.song:
            prog = clamp(self.now / max(1.0, self.song.length), 0, 1)
            pygame.draw.rect(scr, (60, 64, 90), (0, 0, int(W * prog), 4))
        if mode == "play":
            j = self.run.judge
            hp = j.health
            pygame.draw.rect(scr, (24, 26, 42), (40, 26, 420, 14), border_radius=7)
            hcol = mix((255, 70, 90), (90, 230, 255), clamp(hp * 1.3, 0, 1))
            if t - self.miss_flash < 0.15:
                hcol = (255, 255, 255)
            pygame.draw.rect(scr, hcol, (40, 26, int(420 * hp), 14), border_radius=7)
            self.txt(f"{j.score:08d}", self.f_big, (235, 238, 255), (W - 40, 18), "topright")
            self.txt(f"{j.accuracy:.2f}%", self.f_small, (150, 156, 185), (W - 40, 72), "topright")
            if j.combo > 1:
                self.txt(str(j.combo), self.f_big, (255, 255, 255), (CX, CY - 8))
                self.txt("combo", self.f_tiny, (120, 126, 155), (CX, CY + 22))
            if self.jtext:
                s, col, x, y, t0 = self.jtext
                age = t - t0
                if age < 0.5:
                    self.txt(s, self.f_small, col, (int(x), int(y - 14 * age)), alpha=int(255 * (1 - age / 0.5)))
            self.txt(f"AR {ch.ar:.1f}   OD {ch.od:.1f}   {self.pb.rate:.2f}x", self.f_tiny, (95, 100, 128), (40, H - 28), "topleft")
        else:
            div = SNAPS[self.snap_i]
            bar_n = int(math.floor((self.now - ch.offset) / ch.bar))
            beat_n = int(((self.now - ch.offset) / ch.beat) % BEATS_PER_BAR) + 1
            pygame.draw.circle(scr, (255, 70, 90), (52, 36), 8)
            self.txt("REC" if not self.pb.paused else "PAUSED", self.f_mid, (235, 238, 255), (72, 24), "topleft")
            lines = [f"bar {bar_n}  beat {beat_n}",
                     f"{len(ch.dots)} dots    speed {self.pb.rate:.2f}x",
                     f"snap 1/{div * 4}" if div else "snap off",
                     f"metronome {'on' if self.metro else 'off'}",
                     f"BPM {ch.bpm:.2f}   offset {ch.offset * 1000:.0f} ms"]
            for i, line in enumerate(lines):
                self.txt(line, self.f_small, (170, 176, 205), (40, 70 + i * 26), "topleft")
            hint = ["Space/DFJK place 1-4 dots   Z undo   hold Backspace erase",
                    "[ ] snap   M metronome   Enter pause   arrows seek",
                    ", . offset   - = BPM   Esc save + exit"]
            for i, line in enumerate(hint):
                self.txt(line, self.f_tiny, (100, 106, 135), (W - 40, 24 + i * 22), "topright")

    def draw_menu(self):
        self.txt("CIRCLE", self.f_huge, (255, 255, 255), (CX, 82))
        self.txt("working title", self.f_tiny, (120, 126, 155), (CX, 140))
        y0 = 190
        for i, rid in enumerate(ROWS):
            label, val = self.row_info(rid)
            y = y0 + i * 36 + (14 if rid in ("play", "edit") else 0)
            sel = i == self.sel
            col = (255, 255, 255) if sel else (150, 156, 185)
            if rid in ("play", "edit"):
                col = SEG_COLORS[0] if rid == "play" else SEG_COLORS[2]
                col = col if sel else mul(col, 0.7)
            if sel:
                pygame.draw.rect(self.screen, (30, 34, 60), (CX - 330, y - 15, 660, 32), border_radius=10)
            self.txt(label, self.f_small, col, (CX - 310, y), "midleft")
            self.txt(val, self.f_small, col, (CX + 20, y), "midleft")
        self.txt("Up/Down select   Left/Right change (Shift fine, Ctrl coarse)   P play   E record   Esc quit",
                 self.f_tiny, (100, 106, 135), (CX, H - 24))
        msg, t0 = self.notice
        if msg and time.perf_counter() - t0 < 4:
            self.txt(msg, self.f_mid, (255, 214, 120), (CX, H - 62))

    def draw_loading(self):
        mode, rate, _ = self.loading
        self.txt(f"Preparing {rate:.2f}x audio...", self.f_big, (255, 255, 255), (CX, CY - 20))
        self.txt("stretching once, then it's cached for this session   (Esc cancels)", self.f_small, (140, 146, 175), (CX, CY + 30))
        a = time.perf_counter() * 5
        pygame.draw.arc(self.screen, SEG_COLORS[2], pygame.Rect(CX - 24, CY + 70, 48, 48), a, a + 4.2, 4)

    def draw_results(self):
        r = self.result
        col = (255, 90, 110) if r["failed"] else SEG_COLORS[1] if r["grade"] in ("SS", "S") else SEG_COLORS[2]
        self.txt("FAILED" if r["failed"] else r["grade"], self.f_huge, col, (CX, 130))
        self.txt(f"{r['score']:,}", self.f_big, (255, 255, 255), (CX, 220))
        self.txt(f"{r['acc']:.2f}%    max combo {r['combo']}", self.f_mid, (200, 205, 230), (CX, 268))
        c = r["counts"]
        rows = [("Perfect", c["perfect"], (255, 255, 255)), ("Great", c["great"], SEG_COLORS[1]),
                ("Good", c["good"], SEG_COLORS[2]), ("Miss", c["miss"], (150, 154, 170))]
        for i, (name, n, col2) in enumerate(rows):
            self.txt(name, self.f_mid, col2, (CX - 150, 340 + i * 40), "midleft")
            self.txt(str(n), self.f_mid, col2, (CX + 150, 340 + i * 40), "midright")
        err = r["err"]
        lean = "early" if err < -0.5 else "late" if err > 0.5 else "dead on"
        self.txt(f"average timing {err:+.1f} ms ({lean})", self.f_small, (150, 156, 185), (CX, 524))
        self.txt(f"{r['presses']} presses   {r['hits']} hits   {r['strays']} strays   {r['total']} dots",
                 self.f_small, (150, 156, 185), (CX, 556))
        self.txt("R retry    Enter menu", self.f_small, (110, 116, 145), (CX, 640))


if __name__ == "__main__":
    Game(sys.argv[1] if len(sys.argv) > 1 else None).run_forever()
