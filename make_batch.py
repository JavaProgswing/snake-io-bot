"""Dump raw|overlay composites for a sampled subset of the SAME item list
label_perception builds, so my labels map to the tool's indices. Also emits
batch_meta.json with per-item detection counts for cross-checking."""
import base64
import json
import os
import sys

import cv2
import numpy as np

import label_perception as L

src, start, count, stride, step, out = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6])
os.makedirs(out, exist_ok=True)


def dec(b64):
    raw = base64.b64decode(b64.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


items = L.build(src, start, count, stride)
meta = {}
for i in range(0, len(items), step):
    it = items[i]
    raw, ovl = dec(it["raw"]), dec(it["ovl"])
    comp = cv2.hconcat([raw, ovl])
    cv2.putText(comp, f"idx{i}", (4, comp.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, f"idx{i:03d}.png"), comp)
    meta[i] = {"name": it["name"], "self_conf": it["self_conf"],
               "n_enemy": it["n_enemy"], "n_food": it["n_food"],
               "n_loot": it["n_loot"]}
with open(os.path.join(out, "batch_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(json.dumps({"items": len(items), "saved": len(meta), "out": out}))
