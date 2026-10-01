"""Snake.io autopilot v2 — capture -> perception -> pilot -> touch control.

State machine: launch game, tap orange PLAY, then per frame: perceive the
Scene, ask the Pilot for (angle, boost), drive the held virtual joystick and
boost button. On death, tap PLAY / green REVIVE and continue. Relaunches the
app if gameplay stalls.

Usage:  python bot.py [--secs N] [--debug-dir DIR]
"""
import argparse
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np

import perception
from pilot import Pilot
from scrcpy_link import ScrcpyLink, ACTION_DOWN, ACTION_MOVE, ACTION_UP

HERE = os.path.dirname(os.path.abspath(__file__))


def find_button(frame, h_lo, h_hi, s_min=110, v_min=150, min_frac=0.010,
                y_lo=0.60, y_hi=0.95):
    """Centre of the largest solid button-coloured blob in the bottom band.
    Buttons live in the lower ~third; restricting to y_lo..y_hi keeps arena
    snakes (which share the orange/blue hues) from being mistaken for a button.
    Real buttons measure frac ~0.02-0.03; min_frac=0.010 clears snake bits."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    hs = small.shape[0]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
         (hsv[:, :, 1] > s_min) & (hsv[:, :, 2] > v_min)).astype(np.uint8)
    m[:int(hs * y_lo)] = 0
    m[int(hs * y_hi):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cents = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < min_frac * m.size:
        return None
    return int(cents[i][0] * 2), int(cents[i][1] * 2)


class Joystick:
    def __init__(self, link, cfg):
        self.link = link
        self.ax_f, self.ay_f = cfg["ui"]["joystick_anchor"]
        self.r_f = cfg["ui"]["joystick_radius_frac"]
        self.down = False

    def steer(self, frame, angle):
        h, w = frame.shape[:2]
        ax, ay, r = int(w * self.ax_f), int(h * self.ay_f), int(w * self.r_f)
        if not self.down:
            self.link.touch(ACTION_DOWN, ax, ay, 1)
            self.down = True
        self.link.touch(ACTION_MOVE, ax + int(r * math.cos(angle)),
                        ay + int(r * math.sin(angle)), 1)

    def release(self, frame):
        if self.down:
            h, w = frame.shape[:2]
            self.link.touch(ACTION_UP, int(w * self.ax_f),
                            int(h * self.ay_f), 1)
            self.down = False


class Booster:
    def __init__(self, link, cfg):
        self.link = link
        self.bx_f, self.by_f = cfg["ui"]["boost_button"]
        self.down = False

    def set(self, frame, on):
        h, w = frame.shape[:2]
        x, y = int(w * self.bx_f), int(h * self.by_f)
        if on and not self.down:
            self.link.touch(ACTION_DOWN, x, y, 2)
            self.down = True
        elif not on and self.down:
            self.link.touch(ACTION_UP, x, y, 2)
            self.down = False


class Bot:
    def __init__(self, cfg, secs, debug_dir=None, dense=False):
        self.cfg = cfg
        self.secs = secs
        self.debug_dir = debug_dir
        self.dense = dense
        self._fno = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        self.link = ScrcpyLink(cfg["adb"], cfg["server_jar"],
                               cfg["server_version"], cfg["port"],
                               cfg["max_size"], cfg["max_fps"],
                               cfg["video_bit_rate"])
        self.per = perception.Perception()
        self.pil = Pilot()
        self.revive_max = cfg.get("revive_max", 2)

    def log(self, m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    def _reconnect(self):
        """Rebuild the scrcpy link after a dropped control/video socket (phone
        or USB hiccup). Returns fresh Joystick/Booster — the old ones hold stale
        sockets and pointer-down state. Gives up after 10 tries."""
        self._reconnects = getattr(self, "_reconnects", 0) + 1
        if self._reconnects > 10:
            raise RuntimeError("too many reconnects -- device gone?")
        self.log(f"link dropped -> reconnect #{self._reconnects}")
        try:
            self.link.stop()
        except Exception:
            pass
        time.sleep(2.0)
        try:
            self.link.launch_app(self.cfg["package"])
        except Exception:
            pass
        time.sleep(6.0)
        self.link.start()
        self.link.clear_pointers()
        return Joystick(self.link, self.cfg), Booster(self.link, self.cfg)

    def run(self):
        subprocess.run([self.cfg["adb"], "shell", "monkey", "-p",
                        self.cfg["package"], "-c",
                        "android.intent.category.LAUNCHER", "1"],
                       capture_output=True)
        time.sleep(6)
        self.link.start()
        self.link.clear_pointers()
        joy, boost = Joystick(self.link, self.cfg), Booster(self.link, self.cfg)
        frame, _ = self.link.frame()
        self.log(f"video {frame.shape[1]}x{frame.shape[0]}")

        period = 1.0 / self.cfg["strategy"]["decision_hz"]
        t0 = time.time()
        life_t0 = None
        last_play = t0
        last_hb = 0.0
        games = 0
        last_tap = 0.0
        last_dead_save = 0.0
        # debounce state: the frame right after a PLAY tap / at death flickers
        # between playing and menu. Require a short streak before acting so we
        # don't start/end a "game" on a single garbage transition frame.
        play_streak = dead_streak = 0
        in_game = False
        spawned = False
        try:
            while time.time() - t0 < self.secs:
                t = time.time()
                if not getattr(self.link, "ctrl_ok", True):
                    joy, boost = self._reconnect()
                    play_streak = dead_streak = 0
                    in_game = False
                    last_play = time.time()
                    continue
                frame, seq = self.link.frame()
                if frame is None:              # video not up yet -> wait
                    time.sleep(0.1)
                    continue
                state = perception.classify_state(frame)
                if state == perception.STATE_PLAYING:
                    play_streak += 1
                    dead_streak = 0
                else:
                    dead_streak += 1
                    play_streak = 0

                if play_streak >= 2:
                    last_play = t
                    if not in_game:
                        in_game = True
                        life_t0 = t
                        spawned = False
                        self.per.reset()
                        self.pil = Pilot()
                    sc = self.per.process(frame)
                    if not spawned:
                        # A real spawn snake is TINY. During the death->restart
                        # flicker perception can grab a big non-self blob; steering
                        # that = instant bogus death (the 5s "games"). Wait for a
                        # plausibly small self, or a short timeout, before trusting
                        # it. Spawn invulnerability makes going straight safe.
                        # Count/log the game only once spawn is CONFIRMED, so the
                        # 0-1s classify flickers don't log phantom games.
                        if (sc.self_head is not None and sc.self_area < 1600) \
                                or t - life_t0 > 2.0:
                            spawned = True
                            games += 1
                            self.log(f"game #{games} start")
                        else:
                            joy.release(frame)
                    if spawned and sc.self_head is not None:
                        grace = (t - life_t0) < 2.5
                        ang, b = self.pil.decide(sc, grace=grace)
                        joy.steer(frame, ang)
                        boost.set(frame, b)
                        if t - last_hb > 2.0:
                            self.log(f"hb mode={self.pil.mode} "
                                     f"minClear={self.pil.last_min_clear:.0f} "
                                     f"selfA={sc.self_area} "
                                     f"enemy={len(sc.enemies)} "
                                     f"food={len(sc.food)} boost={int(b)}")
                            last_hb = t
                elif dead_streak >= 3:
                    joy.release(frame)
                    boost.set(frame, False)
                    if in_game:
                        if spawned:      # only a CONFIRMED game counts as "over"
                            self.log(f"game #{games} over (~{t - life_t0:.0f}s)")
                        in_game = False
                        life_t0 = None
                    if t - last_hb > 3.0:
                        self.log(f"waiting (state={state})")
                        last_hb = t
                    if t - last_tap > 1.3:
                        h, w = frame.shape[:2]
                        # Get back into a game by DETECTING the button, not by
                        # guessing coords -- its position differs per screen
                        # (stats PLAY ~0.72, menu PLAY ~0.90). The yellow PLAY
                        # starts a game directly and shows on both the stats
                        # screen and the menu; blue NEXT only advances the early
                        # stats screen. Priority PLAY > NEXT > blind fallback.
                        # Revive is skipped on purpose -- it triggers an ad and
                        # hangs the loop.
                        play_btn = find_button(frame, 15, 35, 110, 170)
                        next_btn = find_button(frame, 100, 125, 80, 150)
                        if play_btn:
                            self.link.tap(*play_btn)
                        elif next_btn:
                            self.link.tap(*next_btn)
                        else:
                            self.link.tap(
                                int(w * self.cfg["ui"]["menu_play"][0]),
                                int(h * self.cfg["ui"]["menu_play"][1]))
                        last_tap = t
                    if self.debug_dir and t - last_dead_save > 5.0:
                        cv2.imwrite(os.path.join(self.debug_dir,
                                    f"dead_{int(t)}.png"), frame)
                        last_dead_save = t
                    if t - last_play > 45:
                        self.log("no gameplay 45s -> relaunch app")
                        subprocess.run(
                            [self.cfg["adb"], "shell", "monkey", "-p",
                             self.cfg["package"], "-c",
                             "android.intent.category.LAUNCHER", "1"],
                            capture_output=True)
                        last_play = t
                        time.sleep(6)

                if self.debug_dir and state == perception.STATE_PLAYING:
                    if self.dense:   # every play frame -> consecutive replay
                        cv2.imwrite(os.path.join(self.debug_dir,
                                    f"img_{self._fno:06d}.jpg"), frame)
                        self._fno += 1
                    elif int(t * 2) % 10 == 0:
                        cv2.imwrite(os.path.join(self.debug_dir,
                                    f"play_{int(t)}.png"), frame)
                dt = time.time() - t
                if dt < period:
                    time.sleep(period - dt)
        finally:
            try:
                f0 = self.link.frame()[0]
                if f0 is not None:
                    joy.release(f0)
                    boost.set(f0, False)
            except Exception:
                pass
            try:
                self.link.stop()
            except Exception:
                pass
        self.log(f"done: {games} games")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=1e9)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--dense", action="store_true",
                    help="save every play frame (img_NNNNNN.jpg) for replay")
    args = ap.parse_args()
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    Bot(cfg, args.secs, args.debug_dir, args.dense).run()


if __name__ == "__main__":
    main()
