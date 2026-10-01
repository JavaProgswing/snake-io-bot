"""Live bot: drives Snake.io from the learned U-Net perception (seg_model.pt).

Perception -> Scene (perceive_model.SegModel), then a compact ray-caster
steers: openness + food/loot attraction - enemy-head risk - border - turn
cost, argmax over safe rays, exponential-smoothed heading. Emergency hard
turn when the forward cone is blocked. Menu/revive handled by button color.

Usage:  python drive_seg.py [--secs N] [--boost] [--debug-dir DIR]
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import perception as P
from perceive_model import SegModel
from features import scene_features
from live_capture import Joystick, find_button
from scrcpy_link import ScrcpyLink, ACTION_DOWN, ACTION_UP

HERE = os.path.dirname(os.path.abspath(__file__))
NRAYS = 48


def ray_dist(mask, cx, cy, angles, ray_len, step=4, skip=4):
    h, w = mask.shape
    ts = np.arange(skip, int(ray_len), step, dtype=np.float32)
    xs = np.clip((cx + np.cos(angles)[:, None] * ts).astype(np.int32), 0, w - 1)
    ys = np.clip((cy + np.sin(angles)[:, None] * ts).astype(np.int32), 0, h - 1)
    hit = mask[ys, xs] > 0
    first = np.argmax(hit, axis=1)
    return np.where(hit.any(1), ts[first], ray_len).astype(np.float32)


LIVE = os.path.join(HERE, "live")


def read_nudge():
    """live/nudge.json written by the viewer when the user drags on the live
    canvas. Returns override angle (rad) while fresh, else None."""
    try:
        p = os.path.join(LIVE, "nudge.json")
        with open(p, encoding="utf-8") as f:
            n = json.load(f)
        if time.time() - n.get("t", 0) < n.get("ttl", 1.2):
            return math.radians(float(n["angle"])), bool(n.get("boost", 0))
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


class PolicyBrain:
    """Loads policy.pt (train_policy.py) -> heading from Scene features."""
    def __init__(self, path):
        import torch
        from train_policy import PolicyNet
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = PolicyNet().to(self.dev)
        self.net.load_state_dict(torch.load(path, map_location=self.dev))
        self.net.eval()

    def __call__(self, feat):
        x = self.torch.from_numpy(feat).float().unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            o = self.net(x)[0]
        return math.atan2(float(o[0]), float(o[1]))


class Pilot:
    def __init__(self, policy=None):
        self.heading = 0.0
        self._boost = False
        self.mode = "cruise"
        self.policy = policy

    def decide(self, sc):
        h, w = sc.h, sc.w
        cx, cy = sc.self_head or (w // 2, h // 2)
        diag = math.hypot(w, h)
        ray_len = diag * 0.42
        safe = diag * 0.11
        panic = diag * 0.055
        ang = np.linspace(-math.pi, math.pi, NRAYS, endpoint=False)
        cos, sin = np.cos(ang), np.sin(ang)

        # bodies/border are hard walls; enemy HEADS get a forward-swept cone
        # (where the head will be), kept SEPARATE so we still thread lanes
        # past static bodies to reach loot.
        walls = sc.obstacle_mask
        heads = np.zeros((h, w), np.uint8)
        for e in sc.enemies:
            hx, hy = e.head
            sp = e.speed
            # project further ahead of FAST heads (they close the gap in the
            # ~0.15s of capture->touch latency) — this is the decisive dodge
            k = min(10.0, 70.0 / sp) if sp > 1.0 else 3.0
            cv2.line(heads, (hx, hy),
                     (int(hx + e.vel[0] * k), int(hy + e.vel[1] * k)),
                     255, 15)
            cv2.circle(heads, (hx, hy), 11, 255, -1)
        threat = cv2.bitwise_or(walls, heads)

        d = ray_dist(threat, cx, cy, ang, ray_len)
        dwall = ray_dist(walls, cx, cy, ang, ray_len)
        openn = (d + np.roll(d, 1) + np.roll(d, -1)) / 3.0
        blocked = d < panic
        blockw = blocked | np.roll(blocked, 1) | np.roll(blocked, -1)
        mind = float(d.min())

        # --- food/loot attraction. loot (kill orbs) pulls hard.
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

        # --- LOOT RUSH: pick the best pile reachable by a CLEAR lane (ray
        # clearance past the pile). If one exists, commit to it and boost —
        # this is how the snake actually grows. No global timidity.
        best_loot, best_score = None, 0.0
        for lx, ly, lw in sc.loot:
            dist = math.hypot(lx - cx, ly - cy) + 1e-3
            la = math.atan2(ly - cy, lx - cx)
            ri = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - la))))))
            # lane must be open a bit past the pile, and not through a head cone
            if dwall[ri] < dist + safe * 0.3 or d[ri] < dist * 0.85:
                continue
            val = (lw / 150.0) * math.exp(-0.004 * dist)
            if val > best_score:
                best_score, best_loot = val, (la, ri, dist)

        loot_g = np.zeros(NRAYS, np.float32)
        for lx, ly, lw in sc.loot:
            la = math.atan2(ly - cy, lx - cx)
            dist = math.hypot(lx - cx, ly - cy) + 1e-3
            dd = np.abs(np.angle(np.exp(1j * (ang - la))))
            loot_g += 7.0 * (lw / 150.0) * math.exp(-0.005 * dist) * np.exp(-2.5 * dd)

        turn = np.abs(np.angle(np.exp(1j * (ang - self.heading))))
        danger = np.where(d < safe, (safe - d) / safe, 0.0) ** 2 * 8.0
        score = food_g + loot_g + openn / diag * 1.2 - danger - turn * 0.35
        score[blockw] = -1e9

        boost = False
        cur = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - self.heading))))))
        cone = [(cur + kk) % NRAYS for kk in (-2, -1, 0, 1, 2)]
        emerg = d[cone].min() < panic * 1.35

        if emerg or (blockw).all() or (score < -1e8).all():
            # DECISIVE ESCAPE (hard seatbelt, overrides the policy): widest
            # neighbour-averaged gap, full turn.
            esc = openn.copy()
            esc[blocked] = -1e9
            if (esc < -1e8).all():
                esc = d.copy()              # fully boxed: least-bad single ray
            target = float(ang[int(np.argmax(esc))])
            smooth = 1.0
            emerg = True
        elif self.policy is not None:
            # LEARNED policy drives normal play (cloned from your manual
            # games). Blocked rays near its choice still get vetoed below.
            pa = self.policy(scene_features(sc, self.heading))
            pray = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - pa))))))
            if blockw[pray]:                # policy points into a wall: nudge
                cand = np.where(~blockw)[0]  # to nearest safe ray
                if len(cand):
                    pray = int(cand[np.argmin(np.abs(
                        np.angle(np.exp(1j * (ang[cand] - pa)))))])
                    pa = float(ang[pray])
            target = pa
            smooth = 0.6
            boost = (best_loot is not None and mind > safe * 1.6 and
                     sc.self_area > 2600)
            self.mode = "policy"
        elif best_loot is not None and best_score > 0.4:
            target = best_loot[0]           # commit to the loot lane
            smooth = 0.7
            boost = (mind > safe * 1.6 and sc.self_area > 2600)
        else:
            target = float(ang[int(np.argmax(score))])
            smooth = 0.45

        dth = math.atan2(math.sin(target - self.heading),
                         math.cos(target - self.heading))
        self.heading = math.atan2(math.sin(self.heading + smooth * dth),
                                  math.cos(self.heading + smooth * dth))
        self._boost = boost
        return self.heading, mind, emerg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=1e9)
    ap.add_argument("--boost", action="store_true")
    ap.add_argument("--policy", action="store_true",
                    help="steer with the imitation policy (policy.pt)")
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        os.makedirs(os.path.join(HERE, "live"), exist_ok=True)

    link = ScrcpyLink(cfg["adb"], cfg["server_jar"], cfg["server_version"],
                      cfg["port"], cfg["max_size"], cfg["max_fps"],
                      cfg["video_bit_rate"])
    import subprocess
    subprocess.run([cfg["adb"], "shell", "monkey", "-p", cfg["package"],
                    "-c", "android.intent.category.LAUNCHER", "1"],
                   capture_output=True)
    time.sleep(6)
    link.start()
    link.clear_pointers()
    joy = Joystick(link, cfg)
    md = SegModel()
    policy = None
    if args.policy:
        pp = os.path.join(HERE, "policy.pt")
        if os.path.isfile(pp):
            policy = PolicyBrain(pp)
            print("policy loaded — steering from your cloned play", flush=True)
        else:
            print("policy.pt missing — run imitate.py + train_policy.py first",
                  flush=True)
    pilot = Pilot(policy)
    print("seg model loaded, driving live", flush=True)

    bx, by = cfg["ui"].get("boost_button", [0.9, 0.5])
    life_t0, games, deaths, last_tap, last_live = time.time(), 0, 0, 0.0, 0.0
    t0 = time.time()
    boost_down = False
    try:
        while time.time() - t0 < args.secs:
            frame, seq = link.frame()
            t = time.time()
            state = P.classify_state(frame)
            if state == P.STATE_PLAYING:
                sc = md.process(frame)
                heading, mind, emerg = pilot.decide(sc)
                nud = read_nudge()
                nudged = nud is not None
                if nudged:
                    heading, _nb = nud
                    pilot.heading = heading
                joy.steer(frame, heading)
                # boost when the pilot's loot-rush asks for it (safe lane +
                # body mass), never in emergencies
                want_b = args.boost and not emerg and getattr(pilot, "_boost",
                                                              False)
                fh, fw = frame.shape[:2]
                if want_b and not boost_down:
                    link.touch(ACTION_DOWN, int(fw * bx), int(fh * by), 2)
                    boost_down = True
                elif not want_b and boost_down:
                    link.touch(ACTION_UP, int(fw * bx), int(fh * by), 2)
                    boost_down = False
                if games == 0 or life_t0 is None:
                    life_t0 = t
                    games += 1
                    print(f"game #{games}", flush=True)
                if t - last_live > 0.2:
                    last_live = t
                    os.makedirs(LIVE, exist_ok=True)
                    tag = f"seg minE{mind:.0f} {'EMERG' if emerg else ''}" \
                          f"{' NUDGE' if nudged else ''}"
                    ov = P.render(sc, frame, hud=tag)
                    cv2.imwrite(os.path.join(LIVE, "overlay.jpg"), ov,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                    cv2.imwrite(os.path.join(LIVE, "raw.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 70])
                    meta = {"t": round(t, 2), "game": games,
                            "life_s": round(t - (life_t0 or t), 1),
                            "mode": "EMERG" if emerg else "cruise",
                            "min_enemy_px": round(mind, 1),
                            "self_area": int(sc.self_area),
                            "enemies": len(sc.enemies), "food": len(sc.food),
                            "loot": len(sc.loot), "boost": boost_down,
                            "nudged": nudged,
                            "heading_deg": round(math.degrees(heading), 1)}
                    tmp = os.path.join(LIVE, "meta.tmp")
                    with open(tmp, "w", encoding="utf-8") as f2:
                        json.dump(meta, f2)
                    os.replace(tmp, os.path.join(LIVE, "meta.json"))
                time.sleep(0.05)
            else:
                if boost_down:
                    fh, fw = frame.shape[:2]
                    link.touch(ACTION_UP, int(fw * bx), int(fh * by), 2)
                    boost_down = False
                joy.release(frame)
                if life_t0 is not None and games:
                    print(f"death (life {t - life_t0:.0f}s)", flush=True)
                    deaths += 1
                    life_t0 = None
                if t - last_tap > 1.2:
                    btn = (find_button(frame, 10, 30) or
                           find_button(frame, 40, 75))
                    if btn:
                        link.tap(*btn)
                    else:
                        h, w = frame.shape[:2]
                        link.tap(int(w * cfg["ui"]["menu_play"][0]),
                                 int(h * cfg["ui"]["menu_play"][1]))
                    last_tap = t
                time.sleep(0.35)
    finally:
        if boost_down:
            fr = link.frame()[0]
            fh, fw = fr.shape[:2]
            link.touch(ACTION_UP, int(fw * bx), int(fh * by), 2)
        joy.release(link.frame()[0])
        link.stop()
    print(json.dumps({"games": games, "deaths": deaths}))


if __name__ == "__main__":
    main()
