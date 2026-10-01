"""Auto-label session frames with the classical perception detector (the
teacher) to build a segmentation training set. We keep only HIGH-CONFIDENCE
frames — the model learns clean examples and generalizes to the hard cases
(coils, occlusion) that morphology fragments.

Label map (single channel, class ids):
  0 background   1 self   2 enemy   3 border   4 food   5 loot

Output: <out>/img_XXXXXX.jpg (train-size BGR) + <out>/lab_XXXXXX.png (ids),
plus meta.json. Frames processed IN ORDER so the temporal tracker is valid.

Usage:
  python autolabel.py <session_dir> [--out seg_data] [--max 6000]
                      [--every 2] [--tw 320] [--th 144]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import perception

CLASS_SELF, CLASS_ENEMY, CLASS_BORDER, CLASS_FOOD, CLASS_LOOT = 1, 2, 3, 4, 5


def label_mask(sc, tw, th):
    """Rasterize a Scene into a class-id map at half scale, then resize to
    (tw, th) with nearest-neighbour."""
    m = np.zeros((sc.h, sc.w), np.uint8)
    m[sc.border_mask > 0] = CLASS_BORDER
    m[sc.enemy_mask > 0] = CLASS_ENEMY
    for lx, ly, lw in sc.loot:
        cv2.circle(m, (int(lx), int(ly)), 7, CLASS_LOOT, -1)
    for (x, y), wt in zip(sc.food, sc.food_w):
        cv2.circle(m, (int(x), int(y)), 3, CLASS_FOOD, -1)
    m[sc.self_mask > 0] = CLASS_SELF     # self painted last: never occluded
    return cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)


def keep(sc):
    """High-confidence gate — clean teacher labels only."""
    return (sc.self_head is not None and sc.self_conf >= 0.9
            and 300 <= sc.self_area <= 9000 and sc.scroll_conf >= 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max", type=int, default=6000)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--tw", type=int, default=320)
    ap.add_argument("--th", type=int, default=144)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "seg_data")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(args.src, "*.png")))

    per = perception.Perception()
    kept = seen = 0
    cls_px = np.zeros(6, np.int64)
    for k, fp in enumerate(files):
        if kept >= args.max:
            break
        img = cv2.imread(fp)
        if img is None:
            continue
        if perception.classify_state(img) != perception.STATE_PLAYING:
            per.reset()
            continue
        sc = per.process(img)               # every frame: keep tracker warm
        if k % args.every:
            continue
        seen += 1
        if not keep(sc):
            continue
        lab = label_mask(sc, args.tw, args.th)
        frame = cv2.resize(img, (args.tw, args.th), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, f"img_{kept:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(out, f"lab_{kept:06d}.png"), lab)
        for c in range(6):
            cls_px[c] += int((lab == c).sum())
        kept += 1
        if kept % 500 == 0:
            print(f"kept {kept} (scanned {k})", flush=True)

    meta = {"kept": kept, "gate_seen": seen, "tw": args.tw, "th": args.th,
            "class_pixel_frac": [round(float(c) / max(1, cls_px.sum()), 4)
                                 for c in cls_px], "src": args.src}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
