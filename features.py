"""Shared Scene -> feature-vector for the imitation policy.

Used identically by imitate.py (logging your manual play) and drive_seg.py
(inference), so there is zero train/serve skew. Features are screen-frame
(the camera is head-locked, so screen frame is consistent across frames):

  [ threat clearance per ray (48) ,
    food attraction per ray  (48) ,
    loot attraction per ray  (48) ,
    sin(cur_heading), cos(cur_heading), self_area_norm ]   = 147 dims

The LABEL when logging is scene.self_heading — the direction the snake
actually moved, i.e. exactly what you commanded.
"""
import math

import cv2
import numpy as np

NRAYS = 48
FEAT_DIM = NRAYS * 3 + 3


def _ray_dist(mask, cx, cy, angles, ray_len, step=4, skip=4):
    h, w = mask.shape
    ts = np.arange(skip, int(ray_len), step, dtype=np.float32)
    xs = np.clip((cx + np.cos(angles)[:, None] * ts).astype(np.int32), 0, w - 1)
    ys = np.clip((cy + np.sin(angles)[:, None] * ts).astype(np.int32), 0, h - 1)
    hit = mask[ys, xs] > 0
    first = np.argmax(hit, axis=1)
    return np.where(hit.any(1), ts[first], ray_len).astype(np.float32)


def scene_features(sc, heading):
    """sc: perception.Scene. heading: current commanded heading (rad).
    Returns a FEAT_DIM float32 vector."""
    h, w = sc.h, sc.w
    cx, cy = sc.self_head or (w // 2, h // 2)
    diag = math.hypot(w, h)
    ray_len = diag * 0.42
    ang = np.linspace(-math.pi, math.pi, NRAYS, endpoint=False)

    walls = sc.obstacle_mask
    heads = np.zeros((h, w), np.uint8)
    for e in sc.enemies:
        hx, hy = e.head
        sp = e.speed
        k = min(10.0, 70.0 / sp) if sp > 1.0 else 3.0
        cv2.line(heads, (hx, hy),
                 (int(hx + e.vel[0] * k), int(hy + e.vel[1] * k)), 255, 15)
        cv2.circle(heads, (hx, hy), 11, 255, -1)
    threat = cv2.bitwise_or(walls, heads)
    d = _ray_dist(threat, cx, cy, ang, ray_len)

    food_g = np.zeros(NRAYS, np.float32)
    if len(sc.food):
        fx = sc.food[:, 0] - cx
        fy = sc.food[:, 1] - cy
        dist = np.hypot(fx, fy) + 1e-3
        fa = np.arctan2(fy, fx)
        wt = sc.food_w * np.exp(-0.008 * dist)
        for i, a in enumerate(ang):
            dd = np.abs(np.angle(np.exp(1j * (fa - a))))
            food_g[i] = float((wt * np.exp(-2.2 * dd)).sum())

    loot_g = np.zeros(NRAYS, np.float32)
    for lx, ly, lw in sc.loot:
        la = math.atan2(ly - cy, lx - cx)
        dist = math.hypot(lx - cx, ly - cy) + 1e-3
        dd = np.abs(np.angle(np.exp(1j * (ang - la))))
        loot_g += (lw / 150.0) * math.exp(-0.005 * dist) * np.exp(-2.5 * dd)

    feat = np.concatenate([
        d / ray_len,
        food_g / (food_g.max() + 1e-3),
        loot_g / (loot_g.max() + 1e-3),
        [math.sin(heading), math.cos(heading),
         min(sc.self_area / 10000.0, 3.0)],
    ]).astype(np.float32)
    return feat
