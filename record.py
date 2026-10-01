"""Passive imitation recorder: YOU play Snake.io on the phone, this only
watches. Captures frames (via the scrcpy video socket) and your raw touches
(via `adb shell getevent`), time-aligned, into a session directory that
train.py turns into a steering dataset.

Usage:  python record.py [--out sessions] [--fps 10]
Stop with Ctrl+C. Play normally; several long games are ideal.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time

import cv2

import perception as vision
from scrcpy_link import ScrcpyLink

HERE = os.path.dirname(os.path.abspath(__file__))


class TouchLogger(threading.Thread):
    """Parses `getevent -lt` into (t, slot, x, y, down) rows.
    Raw ABS coords are in the touchscreen's units — train.py normalizes."""

    def __init__(self, adb, out_path):
        super().__init__(daemon=True)
        self.proc = subprocess.Popen(
            [adb, "shell", "getevent", "-lt"],
            stdout=subprocess.PIPE, text=True, errors="ignore", bufsize=1)
        self.f = open(out_path, "w", encoding="utf-8")
        self.t0_dev = None
        self.t0_host = None
        self.count = 0

    def run(self):
        slot, x, y, down = 0, None, None, False
        # getevent -lt lines: "[ 1234.5678] /dev/input/event4: EV_ABS ABS_MT_POSITION_X 0000021f"
        # NOTE the type field (EV_ABS/EV_SYN) sits BETWEEN device and code —
        # an earlier version parsed the type as the code and logged nothing.
        pat = re.compile(
            r"\[\s*(\d+\.\d+)\]\s+\S+\s+(EV_\w+)\s+(\S+)\s+(\S+)")
        self.count = 0
        for line in self.proc.stdout:
            m = pat.search(line)
            if not m:
                continue
            ts_dev, _etype, code, val = m.groups()
            ts_dev = float(ts_dev)
            if self.t0_dev is None:
                self.t0_dev, self.t0_host = ts_dev, time.time()
            t = self.t0_host + (ts_dev - self.t0_dev)
            if code == "ABS_MT_SLOT":
                slot = int(val, 16)
            elif code == "ABS_MT_POSITION_X":
                x = int(val, 16)
            elif code == "ABS_MT_POSITION_Y":
                y = int(val, 16)
            elif code == "ABS_MT_TRACKING_ID":
                down = val != "ffffffff"
            elif code == "SYN_REPORT" and x is not None and y is not None:
                self.f.write(json.dumps({
                    "t": round(t, 4), "slot": slot,
                    "x": x, "y": y, "down": down}) + "\n")
                self.f.flush()
                self.count += 1

    def stop(self):
        self.proc.terminate()
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "sessions"))
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    sess = os.path.join(args.out, time.strftime("rec_%Y%m%d_%H%M%S"))
    os.makedirs(sess, exist_ok=True)

    link = ScrcpyLink(cfg["adb"], cfg["server_jar"], cfg["server_version"],
                      cfg["port"], cfg["max_size"], cfg["max_fps"],
                      cfg["video_bit_rate"])
    link.start()
    frame, _ = link.frame()
    print(f"recording {frame.shape[1]}x{frame.shape[0]} at {args.fps} fps "
          f"-> {sess}\nPLAY NOW. Ctrl+C to stop.")

    touches = TouchLogger(cfg["adb"], os.path.join(sess, "touches.jsonl"))
    touches.start()

    period = 1.0 / args.fps
    n = 0
    # frames index written INCREMENTALLY — a hard kill must not lose it
    fidx = open(os.path.join(sess, "frames.jsonl"), "w", encoding="utf-8")
    try:
        while True:
            t = time.time()
            frame, seq = link.frame()
            state = vision.classify_state(frame)
            name = f"f{n:06d}.jpg"
            cv2.imwrite(os.path.join(sess, name),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            fidx.write(json.dumps({"t": round(t, 4), "file": name,
                                   "state": state}) + "\n")
            fidx.flush()
            n += 1
            if n == 50 and touches.count == 0:
                print("WARNING: 50 frames captured but ZERO touch events — "
                      "touch that screen; if this persists the touch "
                      "logger is broken and training will have no labels!",
                      flush=True)
            if n % 100 == 0:
                print(f"{n} frames, {touches.count} touch events, "
                      f"state={state}", flush=True)
            dt = time.time() - t
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        touches.stop()
        fidx.close()
        link.stop()
        print(f"\nsaved {n} frames + {touches.count} touch events -> {sess}")
        print("next: python train.py --data", sess)


if __name__ == "__main__":
    main()
