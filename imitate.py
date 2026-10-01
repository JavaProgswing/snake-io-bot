"""Passive imitation logger: YOU play Snake.io on the phone, this only
watches and records (features -> your heading) pairs into imit_data.jsonl.

The label is scene.self_heading (the direction the snake actually moved =
what you commanded), so no touch capture is needed. It does NOT steer or tap
anything — you have full manual control. Run several strong games.

Usage:  python imitate.py [--secs N]   (Ctrl+C to stop)
Then:   python train_policy.py
        python drive_seg.py --policy
"""
import argparse
import json
import math
import os
import time

import cv2

import perception as P
from perceive_model import SegModel
from features import scene_features
from scrcpy_link import ScrcpyLink

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=1e9)
    ap.add_argument("--out", default=os.path.join(HERE, "imit_data.jsonl"))
    args = ap.parse_args()
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    link = ScrcpyLink(cfg["adb"], cfg["server_jar"], cfg["server_version"],
                      cfg["port"], cfg["max_size"], cfg["max_fps"],
                      cfg["video_bit_rate"])
    link.start()
    md = SegModel()
    # append so multiple sessions accumulate
    fout = open(args.out, "a", encoding="utf-8")
    prev_heading = 0.0
    n, kept = 0, 0
    t0 = time.time()
    print("PASSIVE — play manually. logging (features -> your heading).",
          flush=True)
    try:
        while time.time() - t0 < args.secs:
            frame, seq = link.frame()
            if P.classify_state(frame) != P.STATE_PLAYING:
                prev_heading = None
                time.sleep(0.15)
                continue
            sc = md.process(frame)
            n += 1
            # need a confident self + a real movement direction to label
            if (sc.self_heading is not None and sc.self_conf > 0.5 and
                    sc.self_area > 200):
                if prev_heading is None:
                    prev_heading = sc.self_heading
                # features encode where we WERE going; label = where the
                # human turned to THIS frame
                feat = scene_features(sc, prev_heading)
                fout.write(json.dumps({
                    "x": [round(float(v), 4) for v in feat],
                    "y": round(float(sc.self_heading), 4)}) + "\n")
                fout.flush()
                kept += 1
                prev_heading = sc.self_heading
            if n % 100 == 0:
                print(f"{n} frames, {kept} labeled ({int(time.time()-t0)}s)",
                      flush=True)
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        fout.close()
        link.stop()
        print(json.dumps({"frames": n, "labeled": kept, "out": args.out}))


if __name__ == "__main__":
    main()
