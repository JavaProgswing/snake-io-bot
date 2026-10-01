"""Steering brain — turns a perception.Scene into (steer_angle, boost).

Built on the v2 perception, which gives things the old bot never had: real
enemy HEADS and WORLD-RELATIVE velocities. So avoidance is PREDICTIVE — we
stamp each enemy's projected future position into the hazard field and steer
around where it's *going*, not just where it is.

Rays are cast from our head over a hazard mask (enemies + border + our own
far body + predicted enemy paths). Each direction is scored by openness +
food/loot pull − danger − turn cost. Modes: cruise / seek / flee / trapped.
Returns (angle_rad, boost_bool); .mode is set for debugging.
"""
import math

import cv2
import numpy as np


class Pilot:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.num_rays = cfg.get("num_rays", 48)
        self.ray_len_frac = cfg.get("ray_len_frac", 0.45)
        self.safe_frac = cfg.get("safe_frac", 0.12)
        self.panic_frac = cfg.get("panic_frac", 0.055)
        self.food_decay = cfg.get("food_decay", 0.012)
        self.smooth = cfg.get("turn_smooth", 0.4)
        self.boost_min_area = cfg.get("boost_min_area", 2600)
        self.heading = 0.0
        self.mode = "cruise"
        self.last_min_clear = 1e9
        self._boost = False

    # ------------------------------------------------------------------

    def decide(self, sc, grace=False):
        """sc: perception.Scene. grace=True during spawn invulnerability
        (still avoid — invulnerability is short and ghosts kill)."""
        if sc.self_head is None:
            return self.heading, False
        cx, cy = sc.self_head
        diag = math.hypot(sc.w, sc.h)
        ray_len = diag * self.ray_len_frac
        safe = diag * self.safe_frac
        panic = diag * self.panic_frac

        hazard = self._hazard(sc, cx, cy, diag)
        angles = np.linspace(-math.pi, math.pi, self.num_rays, endpoint=False)
        # cast from slightly AHEAD to compensate capture->act latency
        look = 10 if not self._boost else 18
        ox = int(np.clip(cx + look * math.cos(self.heading), 0, sc.w - 1))
        oy = int(np.clip(cy + look * math.sin(self.heading), 0, sc.h - 1))
        clear = _ray_dist(hazard, ox, oy, angles, ray_len)
        self.last_min_clear = float(clear.min()) if len(clear) else ray_len

        # neighbour-smoothed openness: a gap is only real if its sides are open
        openn = (clear + np.roll(clear, 1) + np.roll(clear, -1)) / 3.0
        blocked = clear < panic
        # a lone open ray between two blocked ones is a needle gap — no good
        blocked_w = blocked | np.roll(blocked, 1) | np.roll(blocked, -1)

        food = _attract(sc.food, sc.food_w, cx, cy, angles, self.food_decay)
        loot = _loot_attract(sc.loot, cx, cy, angles)
        # crowd pressure: a broad repulsion away from where enemy HEADS cluster.
        # `danger` only sees the nearest obstacle per ray; this makes us actively
        # DRIFT toward emptier map instead of wandering into a dense pocket and
        # dying there (the main death mode once local dodging works).
        crowd = _enemy_pressure(sc.enemies, cx, cy, angles)

        # turn cost: bias toward holding our current heading. Higher weight =>
        # commit to a lane instead of zig-zagging (the erratic self-juking).
        turn = np.abs(np.angle(np.exp(1j * (angles - self.heading)))) * 0.75
        closeness = np.clip((safe - clear) / safe, 0.0, 1.0)
        danger = closeness ** 2 * 9.0
        if grace:
            danger *= 0.5

        # --- reflex: hazard in the FORWARD cone (in our PATH) — not merely
        # something close behind/beside us. Hard-turn to the widest open gap.
        ih = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles -
                                                        self.heading))))))
        cone = [(ih + k) % self.num_rays for k in range(-5, 6)]
        ahead = float(clear[cone].min())
        if ahead < panic:
            self.mode = "reflex"
            esc = openn.copy()
            esc[blocked_w] = -1e9
            if (esc < -1e8).all():
                esc = openn.copy()
                esc[blocked] = -1e9           # forced: take any non-wall ray
            if (esc < -1e8).all():
                self.mode = "trapped"
                self.heading = self._trapped(angles, clear, sc, cx, cy)
                return self.heading, False
            self._commit(angles, esc)
            # NEVER boost while dodging -- accelerating mid-panic drives us into
            # the very thing we're escaping. Turn first, boost later.
            self._boost = False
            return self.heading, False

        # --- cruise / seek ---
        self.mode = "seek" if (food.max() > 6 or loot.max() > 6) else "cruise"
        greed = 1.6 if grace else 1.0
        score = (food * greed + loot * 3.0 + openn / diag * 1.3
                 - danger - turn - crowd * 2.5)
        score = 0.5 * score + 0.25 * np.roll(score, 1) + 0.25 * np.roll(score, -1)
        score[blocked_w] = -1e9
        if (score < -1e8).all():
            score2 = (openn / diag * 1.3 - danger - turn)
            score2[blocked] = -1e9
            if (score2 < -1e8).all():
                self.mode = "trapped"
                self.heading = self._trapped(angles, clear, sc, cx, cy)
                return self.heading, False
            score = score2
        self._commit(angles, score)

        # BOOST is dangerous -- it amplifies any perception gap into a death.
        # We boosted straight into an adjacent enemy head that self-merge had
        # erased from the hazard field. So only boost when the field is
        # genuinely EMPTY: cruise mode (nothing to chase nearby), a long clear
        # runway ahead, nothing close in any direction, and no enemy head near.
        i = int(np.argmax(score))
        near_enemy = any(math.hypot(e.head[0] - cx, e.head[1] - cy)
                         < 0.25 * diag for e in sc.enemies)
        want_boost = (not grace and
                      sc.self_area > self.boost_min_area and
                      clear[i] > safe * 2.0 and
                      self.last_min_clear > safe * 1.4 and
                      not near_enemy)
        self._boost = want_boost
        return self.heading, want_boost

    # ------------------------------------------------------------------

    def _hazard(self, sc, cx, cy, diag):
        hz = sc.obstacle_mask.copy()
        head_r = max(6, int(0.045 * diag))
        for e in sc.enemies:
            hx, hy = e.head
            # the HEAD is the lethal point in a head-to-head. Stamp it SOLID so
            # an enemy head right beside ours registers as hazard even when its
            # body is thin or self-merge has eaten part of it -- this is what we
            # kept ramming. Then stamp the PREDICTIVE path (head + vel*t).
            cv2.circle(hz, (int(hx), int(hy)), head_r, 1, -1)
            r = 8 + int(min(e.speed, 6) * 1.5)
            for t in (3, 6, 10, 14):
                px = int(hx + e.vel[0] * t)
                py = int(hy + e.vel[1] * t)
                if 0 <= px < sc.w and 0 <= py < sc.h:
                    cv2.circle(hz, (px, py), r, 1, -1)
        # our OWN body: only the FAR body is a real self-collision risk. The
        # neck trailing right behind the head can never be hit (can't turn
        # 180 in one tick), so clear a GENEROUS disk around the head -- a small
        # hole made the bot panic at its own neck every frame (perpetual
        # reflex). Keep only body beyond ~0.14*diag as hazard.
        if sc.self_mask.any():
            own_far = sc.self_mask.copy()
            cv2.circle(own_far, (int(cx), int(cy)), int(0.14 * diag), 0, -1)
            hz = cv2.bitwise_or(hz, own_far)
        return hz

    def _commit(self, angles, score):
        target = float(angles[int(np.argmax(score))])
        d = math.atan2(math.sin(target - self.heading),
                       math.cos(target - self.heading))
        self.heading = math.atan2(math.sin(self.heading + self.smooth * d),
                                  math.cos(self.heading + self.smooth * d))

    def _trapped(self, angles, clear, sc, cx, cy):
        """Every lane is blocked. Head for the MOST-OPEN direction — it buys the
        most time and often threads a gap that's about to open as enemies move.
        Only when we're truly walled in on every side (max clearance ~nothing)
        fall back to curling around our own body, so we don't straighten into a
        wall."""
        if float(clear.max()) < 0.04 * math.hypot(sc.w, sc.h):
            return self._coil(sc, cx, cy)
        self._commit(angles, clear)
        return self.heading

    def _coil(self, sc, cx, cy):
        """Boxed in: curl tightly around our own body centroid."""
        ys, xs = np.nonzero(sc.self_mask)
        if len(xs) > 30:
            ang = math.atan2(float(ys.mean()) - cy, float(xs.mean()) - cx)
        else:
            ang = self.heading + math.pi / 2
        turn = math.radians(90)
        d = math.atan2(math.sin(ang - self.heading),
                       math.cos(ang - self.heading))
        sign = 1.0 if d > 0 else -1.0
        self.heading = math.atan2(math.sin(self.heading + sign * turn),
                                  math.cos(self.heading + sign * turn))
        return self.heading


# ----------------------------------------------------------- ray / attract

def _ray_dist(mask, cx, cy, angles, ray_len, step=4, skip=4):
    h, w = mask.shape
    ts = np.arange(skip, int(ray_len), step, dtype=np.float32)
    cos = np.cos(angles)[:, None]
    sin = np.sin(angles)[:, None]
    xs = np.clip((cx + cos * ts).astype(np.int32), 0, w - 1)
    ys = np.clip((cy + sin * ts).astype(np.int32), 0, h - 1)
    hit = mask[ys, xs] > 0
    first = np.argmax(hit, axis=1)
    any_hit = hit.any(axis=1)
    return np.where(any_hit, ts[first], ray_len).astype(np.float32)


def _attract(pts, wts, cx, cy, angles, decay):
    gain = np.zeros(len(angles), np.float32)
    if len(pts) == 0:
        return gain
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dist = np.hypot(dx, dy) + 1e-3
    ang = np.arctan2(dy, dx)
    w = wts * np.exp(-decay * dist)
    for i, a in enumerate(angles):
        dd = np.abs(np.angle(np.exp(1j * (ang - a))))
        gain[i] = float((w * np.exp(-2.2 * dd)).sum())
    return gain


def _loot_attract(loot, cx, cy, angles):
    gain = np.zeros(len(angles), np.float32)
    for lx, ly, lw in loot:
        dist = math.hypot(lx - cx, ly - cy) + 1e-3
        ang = math.atan2(ly - cy, lx - cx)
        wgt = (lw / 300.0) ** 1.2 * math.exp(-0.004 * dist)
        dd = np.abs(np.angle(np.exp(1j * (angles - ang))))
        gain += wgt * np.exp(-3.0 * dd)
    return gain


def _enemy_pressure(enemies, cx, cy, angles):
    """Per-direction enemy-head density. Nearer heads push harder; a WIDE
    angular spread so it's a smooth drift away from clusters, not a sharp dodge
    (that's `danger`'s job). Subtracted from the steering score."""
    gain = np.zeros(len(angles), np.float32)
    for e in enemies:
        ex, ey = e.head
        dist = math.hypot(ex - cx, ey - cy) + 1e-3
        ang = math.atan2(ey - cy, ex - cx)
        w = math.exp(-0.006 * dist)
        dd = np.abs(np.angle(np.exp(1j * (angles - ang))))
        gain += w * np.exp(-1.5 * dd)
    return gain
