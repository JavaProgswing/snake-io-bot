"""Offline perception test: feed a recorded session (or any frame folder)
through Perception IN ORDER (the tracker needs the sequence) and dump
overlay images + a per-frame detection summary.

Usage:
  python run_perception.py <frames_dir_or_glob> [--start N] [--count M]
                           [--stride S] [--out DIR] [--every E]
"""
import argparse
import glob
import json
import os

import cv2

import perception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--every", type=int, default=5, help="save every Nth overlay")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if os.path.isdir(args.src):
        files = sorted(glob.glob(os.path.join(args.src, "*.jpg"))) or \
            sorted(glob.glob(os.path.join(args.src, "*.png")))
    else:
        files = sorted(glob.glob(args.src))
    files = files[args.start:args.start + args.count * args.stride:args.stride]
    if not files:
        raise SystemExit("no frames")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "perc_out")
    os.makedirs(out, exist_ok=True)

    per = perception.Perception()
    summ = {"frames": 0, "self_seen": 0, "enemy_total": 0, "food_total": 0,
            "loot_total": 0, "scroll_conf_sum": 0.0}
    rows = []
    saved = 0
    for k, fp in enumerate(files):
        img = cv2.imread(fp)
        if img is None:
            continue
        st = perception.classify_state(img)
        if st != perception.STATE_PLAYING:
            per.reset()
            continue
        sc = per.process(img, dt=1.0)
        summ["frames"] += 1
        summ["self_seen"] += 1 if sc.self_head else 0
        summ["enemy_total"] += len(sc.enemies)
        summ["food_total"] += len(sc.food)
        summ["loot_total"] += len(sc.loot)
        summ["scroll_conf_sum"] += sc.scroll_conf
        rows.append({
            "file": os.path.basename(fp),
            "self": bool(sc.self_head), "self_conf": sc.self_conf,
            "self_area": sc.self_area,
            "enemies": len(sc.enemies), "food": len(sc.food),
            "loot": len(sc.loot), "scroll_conf": sc.scroll_conf,
            "enemy_heads": [e.head for e in sc.enemies],
            "enemy_speed": [round(e.speed, 1) for e in sc.enemies],
        })
        if k % args.every == 0:
            cv2.imwrite(os.path.join(out, f"p{k:05d}.png"),
                        perception.render(sc, img))
            saved += 1

    with open(os.path.join(out, "detections.jsonl"), "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = max(1, summ["frames"])
    print(json.dumps({
        "playing_frames": summ["frames"],
        "self_seen_pct": round(100 * summ["self_seen"] / n, 1),
        "avg_enemies": round(summ["enemy_total"] / n, 2),
        "avg_food": round(summ["food_total"] / n, 2),
        "avg_loot": round(summ["loot_total"] / n, 2),
        "avg_scroll_conf": round(summ["scroll_conf_sum"] / n, 2),
        "overlays_saved": saved, "out": out,
    }, indent=2))


if __name__ == "__main__":
    main()
