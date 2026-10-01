"""Self-driving data grab: launch Snake.io, wander the snake around, and
capture gameplay frames for the segmentation set. Steering is deliberately
dumb (smooth random exploration) — we want VARIED frames (enemies, loot,
borders, coils), not good play. Deaths are fine: it taps PLAY and continues.

Usage:  python live_capture.py [--secs 120] [--out <dir>]
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import perception
from scrcpy_link import ScrcpyLink, ACTION_DOWN, ACTION_MOVE, ACTION_UP

HERE = os.path.dirname(os.path.abspath(__file__))


def find_button(frame, h_lo, h_hi, s_min=120, v_min=130, min_frac=0.0015):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
         (hsv[:, :, 1] > s_min) & (hsv[:, :, 2] > v_min)).astype(np.uint8)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    out = args.out or os.path.join(HERE, "sessions",
                                   time.strftime("live_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    link = ScrcpyLink(cfg["adb"], cfg["server_jar"], cfg["server_version"],
                      cfg["port"], cfg["max_size"], cfg["max_fps"],
                      cfg["video_bit_rate"])
    # make sure the game is foregrounded
    import subprocess
    subprocess.run([cfg["adb"], "shell", "monkey", "-p", cfg["package"],
                    "-c", "android.intent.category.LAUNCHER", "1"],
                   capture_output=True)
    time.sleep(6)
    link.start()
    link.clear_pointers()
    joy = Joystick(link, cfg)
    fidx = open(os.path.join(out, "frames.jsonl"), "w", encoding="utf-8")

    heading = 0.0
    target = 0.0
    t_turn = 0.0
    n = 0
    plays = 0
    t0 = time.time()
    last_tap = 0.0
    print(f"capturing -> {out}", flush=True)
    try:
        while time.time() - t0 < args.secs:
            frame, seq = link.frame()
            state = perception.classify_state(frame)
            t = time.time()
            if state == perception.STATE_PLAYING:
                # smooth random wander: re-pick a target heading periodically
                if t - t_turn > 2.5:
                    target = np.random.uniform(-math.pi, math.pi)
                    t_turn = t
                d = math.atan2(math.sin(target - heading),
                               math.cos(target - heading))
                heading += 0.12 * d
                joy.steer(frame, heading)
                name = f"f{n:06d}.jpg"
                cv2.imwrite(os.path.join(out, name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                fidx.write(json.dumps({"t": round(t, 3), "file": name,
                                       "state": state}) + "\n")
                fidx.flush()
                n += 1
                if n % 100 == 0:
                    print(f"{n} frames ({int(t - t0)}s)", flush=True)
                time.sleep(0.11)
            else:
                joy.release(frame)
                if t - last_tap > 1.2:
                    btn = (find_button(frame, 10, 30) or       # orange PLAY
                           find_button(frame, 40, 75))         # green REVIVE
                    if btn:
                        link.tap(*btn)
                        plays += 1
                    else:
                        h, w = frame.shape[:2]
                        link.tap(int(w * cfg["ui"]["menu_play"][0]),
                                 int(h * cfg["ui"]["menu_play"][1]))
                    last_tap = t
                time.sleep(0.4)
    finally:
        joy.release(link.frame()[0])
        fidx.close()
        link.stop()
    print(json.dumps({"frames": n, "play_taps": plays, "out": out}))


if __name__ == "__main__":
    main()
