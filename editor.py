#!/usr/bin/env python3
"""
CIRCLE editor - standalone:   python editor.py [song.mp3]
The game also embeds this same Editor (menu -> RECORD / EDIT).

Place:  Space or D/F/J/K = 1-4 dots at the snapped moment you press
Edit:   Z undo   Y redo   hold Backspace = erase under the hand
Bars:   C copy this bar   V paste into this bar   X clear this bar
Time:   Enter pause   arrows seek (1 bar / 8 bars)   S cycle record speed
Sync:   M metronome   , . offset (Shift = fine)   - = BPM (Shift = fine)
Grid:   [ ] snap
Esc saves and exits. Charts are stored next to the song as <song>.circle.json.
"""
from __future__ import annotations

import sys
import threading
import time

import pygame

from circle_common import (FPS_CAP, KEYMAP, R, SEG_COLORS, W, H, CX, CY, Gfx, Playback, Song, init_display,
                           load_settings, make_hitsounds, pick_file, pos_at, save_settings)
from circle_core import Chart, clamp
from editor_core import EditorEngine

RATES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class Editor:
    """Pygame front-end for EditorEngine: keys + audio clock in, drawing out."""

    def __init__(self, gfx, pb, settings, hs_clear):
        self.g, self.pb, self.s, self.hs = gfx, pb, settings, hs_clear
        self.song = None
        self.chart = None
        self.eng = None
        self.metro = True
        self.done = True
        self.loading = None
        self.rebuild_at = None
        self.now = 0.0
        self.msg = ("", 0.0)

    # --------------------------------------------------------- lifecycle
    @property
    def active(self):
        return self.song is not None and not self.done

    def open(self, song, chart, start_t=0.0):
        self.song, self.chart = song, chart
        self.pb.song = song
        self.eng = EditorEngine(chart, self.s.get("snap_i", 3))
        self.done = False
        self.rebuild_at = None
        pygame.key.set_repeat(0)
        self._begin(start_t, round(self.s["edit_rate"], 2))

    def close(self):
        self.pb.stop()
        self.loading = None
        if self.chart:
            self.chart.save()
        if self.eng:
            self.s["snap_i"] = self.eng.snap_i
        save_settings(self.s)
        self.done = True

    def _begin(self, t, rate):
        self.pb.latency = self.s["latency_ms"] / 1000.0
        self.s["edit_rate"] = rate
        if rate not in self.song.cache:
            self.pb.stop()
            th = threading.Thread(target=self.song.prepare, args=(rate,), daemon=True)
            th.start()
            self.loading = (t, rate, th)
            return
        self.loading = None
        self._start(t, rate)

    def _metro(self):
        return (self.chart.offset, self.chart.beat) if self.metro else None

    def _start(self, t, rate):
        self.pb.start(clamp(t, 0.0, self.song.length), rate, self._metro())
        self.now = self.pb.time()

    def restart_audio(self, t):
        self.rebuild_at = None
        self.pb.start(clamp(t, 0.0, self.song.length), self.pb.rate, self._metro())

    def say(self, msg):
        self.msg = (msg, time.perf_counter())

    # ------------------------------------------------------------ input
    def event_time(self, e):
        now = self.pb.time()
        ts = getattr(e, "timestamp", None)
        if ts:
            age = pygame.time.get_ticks() - ts
            if 0 <= age <= 100:
                return now - age / 1000.0 * self.pb.rate
        return now

    def handle_event(self, e):
        if e.type != pygame.KEYDOWN:
            return
        k, eng, ch = e.key, self.eng, self.chart
        if self.loading:
            if k == pygame.K_ESCAPE:
                self.close()
            return
        shift = bool(e.mod & pygame.KMOD_SHIFT)
        t = self.pb.time()
        bar = eng.bar_of(t)
        if k == pygame.K_ESCAPE:
            self.close()
        elif k == pygame.K_RETURN:
            if self.pb.paused:
                self.restart_audio(self.pb.hold)
            else:
                self.pb.pause()
        elif k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
            bars = {pygame.K_LEFT: -1, pygame.K_RIGHT: 1, pygame.K_UP: 8, pygame.K_DOWN: -8}[k]
            self.seek(t + bars * ch.bar)
        elif k == pygame.K_LEFTBRACKET:
            eng.cycle_snap(-1)
        elif k == pygame.K_RIGHTBRACKET:
            eng.cycle_snap(1)
        elif k == pygame.K_m:
            self.metro = not self.metro
            self.restart_audio(t)
        elif k == pygame.K_s:
            i = min(range(len(RATES)), key=lambda j: abs(RATES[j] - self.pb.rate))
            self._begin(t, RATES[(i + 1) % len(RATES)])
        elif k == pygame.K_z:
            if not eng.undo():
                self.say("nothing to undo")
        elif k == pygame.K_y:
            if not eng.redo():
                self.say("nothing to redo")
        elif k == pygame.K_c:
            self.say(f"copied bar {bar} ({eng.copy_bar(bar)} dots)")
        elif k == pygame.K_v:
            if eng.clip:
                self.say(f"pasted {len(eng.paste_bar(bar))} dots into bar {bar}")
            else:
                self.say("nothing copied yet (C)")
        elif k == pygame.K_x:
            self.say(f"cleared bar {bar} ({len(eng.clear_bar(bar))} dots)")
        elif k in (pygame.K_COMMA, pygame.K_PERIOD):
            eng.nudge_offset((0.001 if shift else 0.005) * (1 if k == pygame.K_PERIOD else -1))
            self.rebuild_at = time.perf_counter() + 0.35
        elif k in (pygame.K_MINUS, pygame.K_EQUALS):
            eng.nudge_bpm((0.01 if shift else 0.05) * (1 if k == pygame.K_EQUALS else -1))
            self.rebuild_at = time.perf_counter() + 0.35
        elif (k == pygame.K_SPACE or k in KEYMAP) and not self.pb.paused:
            n = 1 if k == pygame.K_SPACE else KEYMAP[k]
            for x in eng.place(n, self.event_time(e)):
                self.g.ripple(ch.angle(x), ch.segment(x))
            self.g.btn_flash[n - 1] = time.perf_counter()
            if self.s["hitsound"]:
                self.hs.play()

    def seek(self, t):
        self.restart_audio(clamp(t, 0.0, self.song.length))

    # ----------------------------------------------------------- update
    def update(self):
        if self.loading:
            t, rate, th = self.loading
            if not th.is_alive():
                self.loading = None
                self._start(t, rate)
            return
        now = self.pb.time()
        self.now = now
        if self.rebuild_at and time.perf_counter() >= self.rebuild_at and not self.pb.paused:
            self.restart_audio(now)
            self.now = self.pb.time()
        if not self.pb.paused and pygame.key.get_pressed()[pygame.K_BACKSPACE]:
            self.eng.erase(now - 0.07, now + 0.07)
        else:
            self.eng.break_erase()
        if not self.pb.paused and now > self.song.length + 1.0:
            self.pb.pause()

    # ------------------------------------------------------------- draw
    def draw(self):
        g, ch, eng = self.g, self.chart, self.eng
        if self.loading:
            g.draw_idle(ch)
            g.screen.blit(g.dim, (0, 0))
            _, rate, _ = self.loading
            g.txt(f"Preparing {rate:.2f}x audio...", g.f_big, (255, 255, 255), (CX, CY - 20))
            g.txt("stretching once, then it's cached for this session   (Esc cancels)", g.f_small,
                  (140, 146, 175), (CX, CY + 30))
            a = time.perf_counter() * 5
            pygame.draw.arc(g.screen, SEG_COLORS[2], pygame.Rect(CX - 24, CY + 70, 48, 48), a, a + 4.2, 4)
            return
        now = self.now
        ang, seg = g.draw_base(now, ch, eng.snap_div)
        g.draw_window(ang, seg, ch)
        for t in eng.visible(now):
            g.draw_dot(pos_at(ch.angle(t), R), ch.segment(t))
        g.draw_fx()
        g.draw_hand(ang, seg)
        if self.s["bloom"]:
            g.bloom()
        self.draw_hud()

    def draw_hud(self):
        g, ch, eng, scr = self.g, self.chart, self.eng, self.g.screen
        g.draw_buttons()
        g.txt(f"{ch.bpm:.1f}", g.f_mid, (200, 205, 230), (CX, CY + 40))
        g.txt("BPM  |  pulse 1/4", g.f_tiny, (110, 116, 145), (CX, CY + 66))
        prog = clamp(self.now / max(1.0, self.song.length), 0, 1)
        pygame.draw.rect(scr, (60, 64, 90), (0, 0, int(W * prog), 4))
        pygame.draw.circle(scr, (255, 70, 90), (52, 36), 8)
        g.txt("PAUSED" if self.pb.paused else "REC", g.f_mid, (235, 238, 255), (72, 24), "topleft")
        lines = [f"bar {eng.bar_of(self.now)}  beat {eng.beat_of(self.now)}",
                 f"{len(ch.dots)} dots    speed {self.pb.rate:.2f}x",
                 f"snap {eng.snap_label}",
                 f"metronome {'on' if self.metro else 'off'}",
                 f"BPM {ch.bpm:.2f}   offset {ch.offset * 1000:.0f} ms"]
        if eng.clip:
            lines.append(f"clipboard: {len(eng.clip)} dots")
        for i, line in enumerate(lines):
            g.txt(line, g.f_small, (170, 176, 205), (40, 70 + i * 26), "topleft")
        hint = ["Space/DFJK place 1-4 dots   Z undo   Y redo   hold Backspace erase",
                "C copy bar   V paste bar   X clear bar   [ ] snap   S speed",
                "M metronome   Enter pause   arrows seek   , . offset   - = BPM   Esc save + exit"]
        for i, line in enumerate(hint):
            g.txt(line, g.f_tiny, (100, 106, 135), (W - 40, 24 + i * 22), "topright")
        msg, t0 = self.msg
        if msg and time.perf_counter() - t0 < 2.5:
            g.txt(msg, g.f_mid, (255, 214, 120), (CX, H - 24))


# ------------------------------------------------------------ standalone
class EditorApp:
    def __init__(self, path=None):
        self.screen = init_display("Circle editor")
        self.g = Gfx(self.screen)
        self.pb = Playback()
        self.s = load_settings()
        hs, _ = make_hitsounds()
        self.ed = Editor(self.g, self.pb, self.s, hs)
        self.clock = pygame.time.Clock()
        self.chart = Chart()
        self.notice = ""
        path = path or self.s.get("last_song")
        if path:
            self.load(path)

    def load(self, path):
        try:
            song = Song(path)
        except Exception as e:  # noqa: BLE001
            self.notice = f"Couldn't load that file ({e.__class__.__name__})"
            return
        self.notice = ""
        self.chart = Chart.load(path)
        self.s["last_song"] = path
        save_settings(self.s)
        if self.ed.active:
            self.ed.close()
        self.ed.open(song, self.chart, 0.0)

    def quit(self):
        if self.ed.active:
            self.ed.close()
        save_settings(self.s)
        pygame.quit()
        sys.exit(0)

    def frame(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit()
            elif e.type == pygame.DROPFILE:
                self.load(e.file)
            elif self.ed.active:
                self.ed.handle_event(e)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_o:
                    p = pick_file()
                    pygame.event.clear()
                    if p:
                        self.load(p)
                elif e.key == pygame.K_ESCAPE:
                    self.quit()
        if self.ed.active:
            self.ed.update()
            self.ed.draw()
        elif self.ed.song is not None:
            self.quit()  # Esc inside the editor = save and exit the app
        else:
            self.g.draw_idle(self.chart)
            self.screen.blit(self.g.dim, (0, 0))
            self.g.txt("CIRCLE editor", self.g.f_big, (255, 255, 255), (CX, CY - 30))
            self.g.txt("drop an audio file on the window, or press O", self.g.f_small, (150, 156, 185), (CX, CY + 20))
            if self.notice:
                self.g.txt(self.notice, self.g.f_small, (255, 214, 120), (CX, CY + 60))
        pygame.display.flip()
        self.clock.tick(FPS_CAP)

    def run(self):
        while True:
            self.frame()


if __name__ == "__main__":
    EditorApp(sys.argv[1] if len(sys.argv) > 1 else None).run()
