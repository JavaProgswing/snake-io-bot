"""Offline pilot check: run recorded frames through perception -> pilot and
overlay the chosen steer direction (thick blue arrow) + mode, so steering
decisions can be eyeballed before any live control.

Usage: python drive_offline.py <frames_dir> [--start N] [--count M]
                               [--stride S] [--every E] [--out DIR]
"""
import argparse
import glob
import math
import os

import cv2
import numpy as np

import perception
from pilot import Pilot, _ray_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(args.src, "*.png")))
    files = files[args.start:args.start + args.count * args.stride:args.stride]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "drive_out")
    os.makedirs(out, exist_ok=True)

    per = perception.Perception()
    pil = Pilot()
    saved = 0
    for k, fp in enumerate(files):
        img = cv2.imread(fp)
        if img is None or perception.classify_state(img) != \
                perception.STATE_PLAYING:
            per.reset()
            continue
        sc = per.process(img)
        ang, boost = pil.decide(sc)
        if k % args.every:
            continue
        vis = perception.render(sc, img, hud=(
            f"mode={pil.mode} boost={int(boost)} "
            f"minClear={pil.last_min_clear:.0f} selfA={sc.self_area} "
            f"enemy={len(sc.enemies)} food={len(sc.food)} loot={len(sc.loot)}"))
        if sc.self_head:
            cx, cy = sc.self_head
            ex = int(cx + 74 * math.cos(ang))
            ey = int(cy + 74 * math.sin(ang))
            col = (255, 120, 0) if not boost else (255, 0, 0)
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), col, 3, tipLength=0.3)
        cv2.imwrite(os.path.join(out, f"d{k:05d}.png"), vis)
        saved += 1
    print(f"saved {saved} drive overlays -> {out}")


if __name__ == "__main__":
    main()
