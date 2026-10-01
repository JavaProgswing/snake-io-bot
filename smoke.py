"""Link smoke test: frame grab + state classify + optional tap."""
import json
import os
import sys
import time

import cv2

import perception as vision
from scrcpy_link import ScrcpyLink

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "smoke.png")

with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)

link = ScrcpyLink(cfg["adb"], cfg["server_jar"], cfg["server_version"],
                  cfg["port"], cfg["max_size"], cfg["max_fps"],
                  cfg["video_bit_rate"],
                  log_path=os.path.join(HERE, "server.log"))
link.start()
frame, seq = link.frame()
print("frame:", frame.shape, "seq:", seq)
print("state:", vision.classify_state(frame))
cv2.imwrite(OUT, frame)

if "--tap-play" in sys.argv:
    h, w = frame.shape[:2]
    link.tap(int(w * cfg["ui"]["menu_play"][0]), int(h * cfg["ui"]["menu_play"][1]))
    print("tapped play")
    time.sleep(3)
    frame, _ = link.frame()
    print("state after tap:", vision.classify_state(frame))
    cv2.imwrite(OUT.replace(".png", "_after.png"), frame)

link.stop()
