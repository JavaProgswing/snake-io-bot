"""Steering brain: ray-scored direction picking, coil defense, boost logic."""

import math
import time

import cv2
import numpy as np


class Pilot:
    def __init__(self, cfg, weights=None):
        """weights: optional dict overriding strategy weights — this is the
        genome the score-driven evolver tunes (see evolver.PARAM_SPACE)."""
        s = cfg["strategy"]
        w = weights or {}
        self.num_rays = s["num_rays"]
        self.ray_len_frac = s["ray_len_frac"]
        self.safe_frac = w.get("safe_dist_frac", s["safe_dist_frac"])
        self.panic_frac = w.get("panic_dist_frac", s["panic_dist_frac"])
        self.food_decay = s["food_decay"]
        self.smooth = s["turn_smooth"]
        self.boost_min_score = w.get("boost_min_score", s["boost_min_score"])
        self.coil_turn = math.radians(s["coil_turn_deg"])

        self.reflex_frac = w.get("reflex_dist_frac",
                                 s.get("reflex_dist_frac", 0.075))
        self.coil_min_area = w.get("coil_min_area",
                                   s.get("coil_min_area", 1500))
        self.corridor_width_frac = s.get("corridor_width_frac", 0.055)

        self.w_food = w.get("w_food", 1.0)
        self.w_loot = w.get("w_loot", 5.0)
        self.w_crowd = w.get("w_crowd", 1.0)
        self.w_open = w.get("w_open", 1.1)
        self.w_corridor = w.get("w_corridor", 1.9)
        self.w_danger = w.get("w_danger", 1.0)
        self.w_headrisk = w.get("w_headrisk", 1.0)
        self.w_turn = w.get("w_turn", 0.45)
        self.greed_early = w.get("greed_early", 2.6)

        self.rl = None            # OnlineRL instance when --learn rl
        self.last_rl = None       # (features, probs, chosen) of last tick

        self.heading = 0.0       # our current commanded angle (rad)
        self.coiling = 0         # frames left in coil mode
        self.boost = False
        # debug/heartbeat info
        self.last_min_d = 1e9
        self.mode = "cruise"
        # boost burst limiting: boost costs points, use in short decisive
        # bursts only (max ~1.2s on, then >=3s cooldown)
        self._boost_on_since = 0.0
        self._boost_off_since = 0.0
        # decaying memory of where obstacles have BEEN — a moving snake's
        # recent sweep marks space it may still occupy when we arrive
        self._danger = None
        self._own_ema = 0.0      # sanity reference for trap/coil gating

    # ------------------------------------------------------------------

    def decide(self, a, score_est, grace=False, advice=None):
        """a: vision.Analysis. grace=True while spawn invulnerability is
        active — only the border can kill us, so farm greedily.
        Returns (target_angle_rad, boost_bool)."""
        cx, cy = a.center
        diag = math.hypot(a.w, a.h)
        ray_len = diag * self.ray_len_frac
        safe = diag * self.safe_frac
        panic = diag * self.panic_frac
        # honest OCR tiers are 5/80/220/1000 — the 220 tier means a real
        # score of 200-999, which is no longer "early"
        early = score_est < 200 or a.own_area < self.coil_min_area * 0.8

        # obstacle memory: OR in a decayed copy of recent obstacle masks so
        # rays see where snakes are sweeping, not just where they are now.
        # The memory lives in screen space — translate it by this frame's
        # camera shift first, or our own motion smears it into fake danger.
        obs = a.obstacle.astype(np.float32)
        if self._danger is not None and self._danger.shape == obs.shape:
            sx, sy = getattr(a, "cam_shift", (0.0, 0.0))
            if abs(sx) >= 1 or abs(sy) >= 1:
                m = np.float32([[1, 0, sx], [0, 1, sy]])
                self._danger = cv2.warpAffine(
                    self._danger, m, (obs.shape[1], obs.shape[0]))
            # when registration failed the memory's positions are stale in
            # an unknown direction — a long-lived wrong memory parks fake
            # danger on our head (minE pinned at 6px). Decay it hard.
            decay = 0.90 if getattr(a, "cam_ok", True) else 0.55
            self._danger = np.maximum(obs, self._danger * decay)
        else:
            self._danger = obs
        # even under spawn grace the full hazard stays live: invulnerability
        # runs out sooner than we think, and ghost enemies can still kill —
        # grace only softens the cruise danger weight below, never blinds us
        hazard = (self._danger > 0.45).astype(np.uint8)
        if self.boost:
            # extra safety bubble at boost speed
            hazard = cv2.dilate(hazard, np.ones((7, 7), np.uint8))
        # ENEMY-ONLY hazard, before our own body is added: threat metrics
        # (min_enemy, crowded_close) must ignore our own tail — the backward
        # ray always grazes it at ~6px, which used to pin min_enemy near
        # zero and lock the bot in panic modes for entire lives.
        hazard_enemy = hazard.copy()
        # Our own body is lethal too (for steering). Leave only the
        # immediate head/neck disk soft.
        own_far = a.own_mask.copy()
        cv2.circle(own_far, (int(cx), int(cy)),
                   int(max(12.0, safe * 0.22)), 0, -1)
        if own_far.any():
            k = 5 if early else 9
            hazard = cv2.bitwise_or(
                hazard, cv2.dilate(own_far, np.ones((k, k), np.uint8)))
        fast_threat = np.zeros_like(hazard)
        for name, k in (("front_mask", 25), ("ghost_mask", 17),
                        ("chain_mask", 13), ("head_proj", 11)):
            m = getattr(a, name, None)
            if m is not None and m.shape == hazard.shape and m.any():
                fast_threat = cv2.bitwise_or(
                    fast_threat, cv2.dilate(m, np.ones((k, k), np.uint8)))
        if fast_threat.any():
            hazard = cv2.bitwise_or(hazard, fast_threat)
            hazard_enemy = cv2.bitwise_or(hazard_enemy, fast_threat)

        # cast rays from a point slightly AHEAD of us: compensates capture->
        # inject latency (~0.2s); at boost speed we're further along by then
        look = 18 if not self.boost else 30
        ox = cx + look * math.cos(self.heading)
        oy = cy + look * math.sin(self.heading)
        ox = int(max(0, min(a.w - 1, ox)))
        oy = int(max(0, min(a.h - 1, oy)))

        angles = np.linspace(-math.pi, math.pi, self.num_rays, endpoint=False)
        head_risk = self._head_ray_risk(a, cx, cy, angles, safe)
        obstacle_d = self._ray_distances(hazard, ox, oy, angles, ray_len)
        enemy_d = self._ray_distances(hazard_enemy, ox, oy, angles, ray_len)
        food_block_d = self._ray_distances(a.obstacle, cx, cy, angles, ray_len)
        # corridors measured between ENEMIES/walls only — measuring against
        # our own body made every post-turn corridor look needle-thin and
        # kept the bot in panic modes for whole lives
        corridor_w = self._corridor_widths(
            hazard_enemy, ox, oy, angles, enemy_d, ray_len,
            diag * self.corridor_width_frac)
        # own body is also lethal for head-on hits, but we must be able to
        # coil over/next to it — treat it as a soft obstacle at short range
        own_d = self._ray_distances(a.own_mask, cx, cy, angles, ray_len,
                                    skip_near=8)

        food_gain = self._food_gain(a, cx, cy, angles)
        has_food = len(a.food_pts) > 0
        loot_gain, loot_ray, loot_dist = self._loot_gain(a, cx, cy, angles)
        if early:
            self.coiling = 0

        min_enemy = enemy_d.min() if len(enemy_d) else ray_len
        self.last_min_d = float(min_enemy)
        # mode triggers see enemies/walls; own-body avoidance is handled by
        # own_pen and the _safe_heading guard on the chosen ray
        blocked = enemy_d < panic
        blocked = blocked | (corridor_w < panic * 0.45)
        blocked = blocked | (own_d < panic * 0.6)   # own neck-fold guard
        # a single open ray between two blocked ones is a needle-thin gap
        # in a snake's body — threading it is how we bonk. Treat a ray as
        # blocked unless its neighbours are open too.
        blocked_w = blocked | np.roll(blocked, 1) | np.roll(blocked, -1)
        # gap openness: a ray is only truly open if its neighbours are too
        open_n = (obstacle_d + np.roll(obstacle_d, 1) +
                  np.roll(obstacle_d, -1)) / 3.0
        corridor_n = (corridor_w + np.roll(corridor_w, 1) +
                      np.roll(corridor_w, -1)) / 3.0
        current_ray = int(np.argmin(np.abs(np.angle(
            np.exp(1j * (angles - self.heading))))))
        front = [(current_ray + k) % len(angles) for k in (-2, -1, 0, 1, 2)]

        surrounded = blocked.mean() > (0.68 if early else 0.55)
        threatened = (not grace and min_enemy < panic and a.enemy_area > 400)
        front_close = enemy_d[front].min() < panic * 1.18
        crowded_close = (
            min_enemy < panic * 0.58 or
            (enemy_d < safe * 0.50).mean() > 0.18 or
            (corridor_w < panic * 0.72).mean() > 0.42
        )
        close_quarters = (front_close or crowded_close) and not grace

        edge_escape = self._edge_escape_gain(hazard, angles)

        # Hard wall guard comes before greed. A border only needs to occupy
        # one front/side lane to kill us, so don't wait for a huge arc average.
        border_d = None
        if a.border_mask.any():
            border_d = self._ray_distances(a.border_mask, ox, oy, angles,
                                           ray_len)
            # trigger only when the wall is genuinely CLOSE. The old 1.95x
            # radius fired whenever a wall was merely visible, so the bot
            # spent whole lives in wall-escape instead of eating.
            hard_wall = (
                border_d.min() < safe * 0.95 or
                border_d[current_ray] < safe * 1.25 or
                border_d[front].min() < safe * 1.05 or
                (border_d < safe * 1.2).mean() > 0.28
            )
            if hard_wall:
                # Rank rays by border clearance PLUS how squarely they point
                # away from the wall. Clearance alone let the bot run
                # parallel to the wall forever (parallel rays never hit it)
                # until something pinned it there.
                push = self._border_push_angle(a)
                away = (np.cos(angles - push) if push is not None
                        else np.zeros(len(angles), np.float32))
                esc = (border_d / diag * 3.0 + away * 1.8 +
                       open_n / diag * 1.5 + corridor_n / diag * 2.2 -
                       head_risk * 2.2)
                esc[border_d < safe * 1.05] = -1e9
                esc[blocked] = -1e9
                if (esc < -1e8).all():
                    esc = border_d * 2.0 + away * panic + open_n + corridor_n * 1.2
                    esc[blocked] = -1e9
                if (esc < -1e8).all():
                    esc = border_d + obstacle_d + away * panic
                self._commit(float(angles[int(np.argmax(esc))]),
                             "wall-escape", angles, obstacle_d, corridor_w,
                             panic, smooth=1.0, strict=True)
                # enemies pressing us against the wall: burst out if open
                self.boost = (min_enemy < safe and
                              self._boost_decision(a, score_est, obstacle_d,
                                                   flee=True))
                return self.heading, self.boost

        if close_quarters:
            esc = (open_n / diag * 1.7 + corridor_n / diag * 2.4 +
                   loot_gain * 0.55 + food_gain * (0.22 if early else 0.08))
            esc -= head_risk * 2.0
            if edge_escape is not None:
                esc += edge_escape * 1.15
            esc[blocked] = -1e9
            if (esc < -1e8).all():
                esc = open_n + corridor_n * 1.4 - head_risk * panic
                esc[blocked] = -1e9
            if (esc > -1e8).any():
                self._commit(float(angles[int(np.argmax(esc))]), "hug-break",
                             angles, obstacle_d, corridor_w, panic,
                             smooth=1.0)
                # squeezed by enemies: a short burst out of the pocket beats
                # getting sealed in (gates: score, body size, open lane)
                self.boost = self._boost_decision(a, score_est, obstacle_d,
                                                  flee=True)
                return self.heading, self.boost

        # --- reflex BEFORE any target chasing: something dead ahead
        # (someone cut across / flicked into our path) -> hard turn to the
        # widest gap. This used to run after eat/loot, so a committed meal
        # run would sail straight into an arriving head.
        reflex_d = diag * self.reflex_frac * (1.7 if self.boost else 1.0)
        ih = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - self.heading))))))
        cone = [(ih + k) % len(angles) for k in (-2, -1, 0, 1, 2)]
        ahead = obstacle_d[cone].min()
        if ahead < reflex_d:
            self.mode = "reflex"
            food_pull = 0.55 if early else 0.15
            esc = (open_n * 1.20 + corridor_n * 1.15 + food_gain * food_pull
                   - np.where(own_d < safe * 0.5, 3.0, 0.0))
            esc -= head_risk * panic * 1.1
            if edge_escape is not None:
                esc += edge_escape * panic * 0.9
            esc_w = esc.copy()
            esc_w[blocked if early else blocked_w] = -1e9
            if (esc_w > -1e8).any():
                esc = esc_w              # prefer wide gaps...
            else:
                esc[blocked] = -1e9      # ...thread a needle only if forced
            if (esc < -1e8).all() and early:
                self.mode = "graze-run"
                esc = food_gain * 0.35 + loot_gain * 1.8 + open_n / diag + corridor_n / diag
                esc[blocked] = -1e9
                if (esc < -1e8).all():
                    esc = open_n + corridor_n
                self._commit(float(angles[int(np.argmax(esc))]), "graze-run",
                             angles, obstacle_d, corridor_w, panic, smooth=1.0)
                self.boost = False
                return self.heading, False
            if (esc < -1e8).all():
                self.coiling = 14
                self._commit(self._coil(a), "coil", angles, obstacle_d,
                             corridor_w, panic, smooth=1.0)
                return self.heading, False
            self._commit(float(angles[int(np.argmax(esc))]), "reflex",
                         angles, obstacle_d, corridor_w, panic, smooth=1.0)
            self.boost = self._boost_decision(a, score_est, obstacle_d,
                                              flee=True)
            return self.heading, self.boost

        loot_target = self._best_loot_target(a, hazard, angles, obstacle_d,
                                             corridor_w, safe, panic)
        self._rush_risk = 0.0
        if loot_target is not None and not grace:
            self._commit(loot_target, "loot", angles, obstacle_d, corridor_w,
                         panic, smooth=0.75)
            self.boost = self._boost_decision(a, score_est, obstacle_d,
                                              flee=False, rush=True)
            return self.heading, self.boost

        # Early game is food-limited, not territory-limited. If there is an
        # edible ray with even modest clearance, take it before reflex/panic
        # logic convinces us to orbit empty space.
        if early and has_food and food_gain.max() > 0.8:
            target = self._best_food_target(a, hazard, angles, food_block_d,
                                            corridor_w, own_d, safe, panic)
            if target is not None:
                self._commit(target, "eat", angles, obstacle_d, corridor_w,
                             panic, smooth=max(self.smooth, 0.70))
                self.boost = False
                return self.heading, False

        # Softer wall escape after early food logic catches cases where the
        # wall is beside us rather than directly in the current path.
        if border_d is not None:
            if (border_d < safe * 1.35).mean() > 0.30:
                self.mode = "wall-escape"
                push = self._border_push_angle(a)
                away = (np.cos(angles - push) if push is not None
                        else np.zeros(len(angles), np.float32))
                esc = (border_d / diag * 2.4 + away * 1.4 +
                       open_n / diag + corridor_n / diag * 1.8)
                esc[border_d < safe * 1.2] = -1e9
                esc[blocked] = -1e9
                if (esc < -1e8).all():
                    esc = border_d + away * panic + open_n + corridor_n
                    esc[blocked] = -1e9
                if (esc < -1e8).all():
                    esc = border_d + obstacle_d + away * panic
                self._commit(float(angles[int(np.argmax(esc))]), "wall-escape",
                             angles, obstacle_d, corridor_w, panic,
                             smooth=1.0, strict=True)
                self.boost = False
                return self.heading, False

        # --- trap: a smaller enemy is inside our loop and we're big — keep
        # circling to wall them in; they die on our body, we take the loot
        if self._own_ema == 0.0:
            self._own_ema = float(a.own_area)
        else:
            self._own_ema = 0.9 * self._own_ema + 0.1 * a.own_area
        if (not early and a.enemy_inside and a.own_area > self.coil_min_area and
                a.own_area < self._own_ema * 1.5 + 500 and
                a.enemy_area < a.own_area * 0.5 and not grace):
            self.mode = "trap"
            cxb, cyb = a.center
            ys, xs = np.nonzero(a.own_mask)
            if len(xs) > 50:
                ang_c = math.atan2(float(ys.mean()) - cyb,
                                   float(xs.mean()) - cxb)
                d = math.atan2(math.sin(ang_c - self.heading),
                               math.cos(ang_c - self.heading))
                sign = 1.0 if d > 0 else -1.0
                turn = self.coil_turn * 0.6      # wide, sustained circle
                cand = math.atan2(math.sin(self.heading + sign * turn),
                                  math.cos(self.heading + sign * turn))
                ray = int(np.argmin(np.abs(np.angle(
                    np.exp(1j * (angles - cand))))))
                if obstacle_d[ray] > panic * 1.2:   # don't circle into a wall
                    self._commit(cand, "trap", angles, obstacle_d,
                                 corridor_w, panic, smooth=0.85)
                    self.boost = False
                    return self.heading, False

        juke = self._juke_target(a, score_est, safe)
        if juke is not None and not grace:
            self._commit(juke, "juke", angles, obstacle_d, corridor_w,
                         panic, smooth=0.9)
            self.boost = False
            return self.heading, False

        # --- coil: only worthwhile once we have body mass to circle with;
        # a tiny snake coils into a death spiral, better to just run
        if surrounded and early:
            self.mode = "graze-run"
            esc = (open_n / diag * 1.4 + corridor_n / diag * 1.8
                   + food_gain * 0.45 + loot_gain * 2.2)
            esc -= head_risk * 1.4
            if edge_escape is not None:
                esc += edge_escape * 0.9
            esc -= np.where(own_d < safe * 0.45, 2.5, 0.0)
            # Small snakes need calories more than perfect lanes. Only hard
            # veto the ray itself; neighbour-wide vetoes make it starve in
            # normal food fields where every tasty dot sits near some body.
            esc[blocked] = -1e9
            if (esc < -1e8).all():
                esc = food_gain * 0.35 + loot_gain * 1.8 + open_n / diag + corridor_n / diag
                esc[blocked] = -1e9
            if (esc < -1e8).all():
                esc = open_n + corridor_n
            self._commit(float(angles[int(np.argmax(esc))]), "graze-run",
                         angles, obstacle_d, corridor_w, panic, smooth=1.0)
            self.boost = False
            return self.heading, False

        if (not early and
                (surrounded or (threatened and a.own_area >= self.coil_min_area
                                and open_n.max() < safe))):
            self.coiling = 12
        if self.coiling > 0:
            self.coiling -= 1
            self.mode = "coil"
            self.boost = self._boost_decision(a, score_est, obstacle_d,
                                              flee=True)
            self._commit(self._coil(a), "coil", angles, obstacle_d,
                         corridor_w, panic, smooth=1.0)
            return self.heading, self.boost

        # --- online RL ray choice: same features the heuristic uses, but
        # the weights learn from every tick. Hard vetoes stay — the policy
        # chooses among SAFE rays only.
        if self.rl is not None and not grace:
            closeness_rl = np.clip((safe - obstacle_d) / safe, 0.0, 1.0)
            turn_rl = np.abs(np.angle(np.exp(1j * (angles - self.heading))))
            crowd_rl = self._crowd_gain(a, cx, cy, angles, safe)
            x = np.stack([food_gain, loot_gain, open_n / diag,
                          corridor_n / diag, closeness_rl ** 2, head_risk,
                          turn_rl, crowd_rl], axis=1).astype(np.float32)
            veto = (blocked if early else blocked_w).copy()
            if not veto.all():
                c, p = self.rl.choose(x, veto)
                probe = min(obstacle_d[c] * 0.6, safe * 2.0)
                pxp = cx + probe * math.cos(angles[c])
                pyp = cy + probe * math.sin(angles[c])
                if self._enclosed(hazard, pxp, pyp):
                    veto[c] = True          # pocket: re-choose once
                    if not veto.all():
                        c, p = self.rl.choose(x, veto)
                self.last_rl = (x, p, c)
                self._commit(float(angles[c]), "rl", angles, obstacle_d,
                             corridor_w, panic, smooth=self.smooth)
                self.boost = self._boost_decision(
                    a, score_est, obstacle_d, flee=False,
                    food_gain=food_gain)
                return self.heading, self.boost

        # --- cruise: score each direction
        self.mode = "grace" if grace else "cruise"
        # quadratic hug penalty: skimming a snake at half the safe distance
        # must hurt a lot more than being at 90% of it
        closeness = np.clip((safe - obstacle_d) / safe, 0.0, 1.0)
        danger_pen = closeness ** 2 * (6.0 if early else 9.0) * self.w_danger
        danger_pen += head_risk * (4.5 if early else 7.0) * self.w_headrisk
        narrow = np.clip((safe * 0.72 - corridor_w) / (safe * 0.72),
                         0.0, 1.0)
        danger_pen += narrow ** 2 * (2.2 if early else 5.5) * self.w_danger
        if grace:
            danger_pen *= 0.5       # braver while invulnerable, never blind
        own_pen = np.where(own_d < safe * 0.6,
                           (safe * 0.6 - own_d) / (safe * 0.6) * 2.5, 0.0)
        turn_cost = np.abs(np.angle(np.exp(
            1j * (angles - self.heading)))) * self.w_turn

        # crowd drift: with no local food/loot worth chasing and no threat
        # nearby, migrate toward distant enemy mass — that's where the big
        # snakes (incl. the crowned king) fight and drop point bursts
        crowd = self._crowd_gain(a, cx, cy, angles, safe)
        if grace:
            # a revive drops us exactly where we died — usually a crowded
            # pit. Spend the grace window LEAVING the hot zone, not farming
            crowd = -1.5 * crowd
        elif early:
            # a small snake in traffic is prey: farm the quiet zones and
            # only wade into crowds once we have the mass to trade
            crowd = -1.2 * crowd
        elif min_enemy < safe or loot_gain.max() > 0.5:
            crowd *= 0.0

        greed = self.greed_early if (grace or early) else 1.6
        edge_bonus = edge_escape * 1.2 if edge_escape is not None else 0.0
        # cloned-net advice: a mild preference toward the direction the
        # imitation net would take — style without authority (danger terms
        # are 5-10x stronger and every safety veto still applies)
        advice_bonus = (0.9 * np.cos(angles - advice)
                        if advice is not None else 0.0)
        score = ((food_gain * self.w_food + loot_gain * self.w_loot) * greed
                 + crowd * self.w_crowd + edge_bonus + advice_bonus
                 + open_n / diag * self.w_open
                 + corridor_n / diag * self.w_corridor
                 - danger_pen - own_pen - turn_cost)
        # angular smoothing kills single-ray jitter (the "jiggle")
        score = 0.5 * score + 0.25 * np.roll(score, 1) + 0.25 * np.roll(score, -1)
        score[blocked if early else blocked_w] = -1e9
        if (score < -1e8).all():
            score = ((food_gain + loot_gain * 5.0) * greed
                     + open_n / diag * 1.2 - danger_pen - own_pen - turn_cost)
            score[blocked] = -1e9

        if (score < -1e8).all() and early:
            self.mode = "graze-run"
            score = food_gain * 0.35 + loot_gain * 1.8 + open_n / diag + corridor_n / diag
            score[blocked] = -1e9
            if (score < -1e8).all():
                score = open_n + corridor_n
        if (score < -1e8).all():          # fully boxed in -> coil
            self.coiling = 14
            self._commit(self._coil(a), "coil", angles, obstacle_d,
                         corridor_w, panic, smooth=1.0)
            return self.heading, False

        # pocket veto: a direction can be locally open yet lead INTO a
        # space enclosed by a snake's coil — entering is suicide once the
        # loop closes (grace made this worse: bot strolled into circled-up
        # snakes while "invulnerable" and died inside)
        best = int(np.argmax(score))
        for _ in range(3):
            probe = min(obstacle_d[best] * 0.6, safe * 2.0)
            pxp = cx + probe * math.cos(angles[best])
            pyp = cy + probe * math.sin(angles[best])
            if not self._enclosed(hazard, pxp, pyp):
                break
            score[best] = -1e9
            score[(best + 1) % len(angles)] = -1e9
            score[(best - 1) % len(angles)] = -1e9
            if (score < -1e8).all() and early:
                self.mode = "graze-run"
                score = food_gain * 0.35 + loot_gain * 1.8 + open_n / diag + corridor_n / diag
                score[blocked] = -1e9
                if (score < -1e8).all():
                    score = open_n + corridor_n
                break
            if (score < -1e8).all():
                self.coiling = 14
                self._commit(self._coil(a), "coil", angles, obstacle_d,
                             corridor_w, panic, smooth=1.0)
                return self.heading, False
            best = int(np.argmax(score))
        target = float(angles[best])
        # exponential smoothing on the circle
        self._commit(target, self.mode, angles, obstacle_d, corridor_w,
                     panic, smooth=self.smooth)

        # loot rush: chosen direction is the loot direction, the corridor to
        # it is clearly open, and the burst is close enough to claim
        ray_gap = 0 if loot_ray is None else min(
            abs(best - loot_ray), len(angles) - abs(best - loot_ray))
        # boost-to-loot when the corridor is open a bit past the loot —
        # loot beside a snake's BODY is claimable (the vision layer already
        # dropped anything a head or moving front could contest)
        rushing_loot = (loot_ray is not None and ray_gap <= 2 and
                        obstacle_d[loot_ray] > loot_dist + safe * 0.3 and
                        corridor_w[loot_ray] > safe * 0.55 and
                        loot_dist < ray_len)
        if rushing_loot and not grace:
            self.mode = "loot"

        if grace:
            boost = False           # score too low to spend on boost anyway
        else:
            boost = self._boost_decision(a, score_est, obstacle_d,
                                         flee=False, food_gain=food_gain,
                                         rush=rushing_loot)
        self.boost = boost
        return self.heading, boost

    def _crowd_gain(self, a, cx, cy, angles, safe):
        """Mild per-ray pull toward DISTANT enemy mass (beyond 1.8x safe).
        Crowded zones mean big snakes, kills, and loot chains — but the pull
        is capped so it can never outweigh danger or real food nearby."""
        gain = np.zeros(len(angles), np.float32)
        em = a.enemy_mask[::4, ::4]
        ys, xs = np.nonzero(em)
        if len(xs) < 20:
            return gain
        xs = xs.astype(np.float32) * 4 - cx
        ys = ys.astype(np.float32) * 4 - cy
        d = np.hypot(xs, ys)
        far = d > safe * 1.8
        if far.sum() < 20:
            return gain
        ang = np.arctan2(ys[far], xs[far])
        for i, a_i in enumerate(angles):
            dd = np.abs(np.angle(np.exp(1j * (ang - a_i))))
            gain[i] = float((np.exp(-2.0 * dd)).sum())
        m = gain.max()
        if m > 0:
            gain *= 1.2 / m        # cap: at most a gentle preference
        return gain

    def _loot_gain(self, a, cx, cy, angles):
        """Per-ray attraction toward loot clusters. Also returns the ray
        index and distance of the strongest cluster (or None)."""
        gain = np.zeros(len(angles), np.float32)
        best_ray, best_dist = None, 0.0
        for lx, ly, lw in a.loot:
            dx, dy = lx - cx, ly - cy
            dist = math.hypot(dx, dy) + 1e-3
            ang = math.atan2(dy, dx)
            # superlinear in cluster weight: a big snake's death chain
            # dominates scattered spawn food
            wgt = 7.0 * (lw / 300.0) ** 1.45 * math.exp(-0.0025 * dist)
            dd = np.abs(np.angle(np.exp(1j * (angles - ang))))
            gain += wgt * np.exp(-3.0 * dd)
            if best_ray is None:      # a.loot is heaviest-first
                best_ray = int(np.argmin(dd))
                best_dist = dist
        return gain, best_ray, best_dist

    def _best_loot_target(self, a, hazard, angles, obstacle_d, corridor_w, safe, panic):
        if not a.loot:
            return None
        best_score = -1e9
        best_ang = None
        cx, cy = a.center
        for lx, ly, lw in a.loot:
            dx, dy = lx - cx, ly - cy
            dist = math.hypot(dx, dy) + 1e-3
            ang = math.atan2(dy, dx)
            ray = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - ang))))))
            # Do not chase a bounty through a body. The old check only asked
            # for a short clear segment, so "loot" could aim straight through
            # another snake if the payout was attractive enough.
            needed_clear = min(dist + safe * 0.22, obstacle_d.max() * 0.96)
            if obstacle_d[ray] < needed_clear:
                continue
            if corridor_w[ray] < max(panic * 0.78, safe * 0.36):
                continue
            turn = abs(math.atan2(math.sin(ang - self.heading),
                                  math.cos(ang - self.heading)))
            risk = self._target_risk(a, lx, ly, dist)
            space = self._target_space(hazard, lx, ly)
            score = ((lw / 180.0) ** 1.65 * math.exp(-0.0020 * dist)
                     + 0.95 * space - 2.25 * risk - 0.42 * turn)
            if score > best_score:
                best_score = score
                best_ang = ang
                self._rush_risk = risk
        # gate was 0.7 when head velocities were noisy (camera motion mixed
        # in); with camera-compensated tracking the risk term is trustworthy,
        # so rush more of the bounties it clears
        return best_ang if best_score > 0.45 else None

    def _best_food_target(self, a, hazard, angles, obstacle_d, corridor_w,
                          own_d, safe, panic):
        cx, cy = a.center
        best_score = -1e9
        best_ang = None
        for (x, y), wt in zip(a.food_pts, a.food_w):
            dx, dy = float(x - cx), float(y - cy)
            dist = math.hypot(dx, dy) + 1e-3
            ang = math.atan2(dy, dx)
            ray = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - ang))))))
            # Food is cheap; a body between us and it is never worth the line.
            needed_clear = max(panic * 1.15, dist - 8.0)
            if obstacle_d[ray] < needed_clear:
                continue
            if corridor_w[ray] < panic * 0.72:
                continue
            if own_d[ray] < max(18.0, panic * 0.55):
                continue
            turn = abs(math.atan2(math.sin(ang - self.heading),
                                  math.cos(ang - self.heading)))
            edge = 0.0
            if x < a.w * 0.20 or x > a.w * 0.84:
                edge += 1.2
            if y < a.h * 0.12 or y > a.h * 0.82:
                edge += 1.8
            risk = self._target_risk(a, x, y, dist)
            space = self._target_space(hazard, x, y)
            score = ((wt ** 0.7) * math.exp(-0.006 * dist)
                     + 0.65 * space - 1.15 * risk - 0.45 * turn - edge)
            if score > best_score:
                best_score = score
                best_ang = ang
        return best_ang

    @staticmethod
    def _target_space(hazard, x, y, radius=34):
        h, w = hazard.shape
        xi, yi = int(x), int(y)
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        crop = hazard[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        return float(1.0 - crop.mean())

    @staticmethod
    def _target_risk(a, x, y, our_dist):
        heads = getattr(a, "enemy_heads", None) or []
        if not heads:
            return 0.0
        risk = 0.0
        for head in heads:
            hx, hy, area, dist_c, *vel = head
            hvx, hvy = (vel + [0.0, 0.0])[:2]
            hdist = math.hypot(x - hx, y - hy) + 1e-3
            # If the enemy head is closer to the target than we are, or its
            # rough velocity points at the target, the pellet/bounty is hot.
            contest = max(0.0, (our_dist * 1.15 - hdist) / max(30.0, our_dist))
            vmag = math.hypot(hvx, hvy)
            approach = 0.0
            if vmag > 0.5:
                tx, ty = (x - hx) / hdist, (y - hy) / hdist
                approach = max(0.0, (hvx * tx + hvy * ty) / vmag)
            near = max(0.0, (130.0 - hdist) / 130.0)
            risk = max(risk, contest * 1.2 + approach * near)
        return min(2.5, risk)

    def _juke_target(self, a, score_est, safe):
        """Present our body sideways to a nearby enemy head when we are big
        enough to survive the setup. Conservative: no tiny-snake hero plays."""
        if score_est < 500 or a.own_area < self.coil_min_area:
            return None
        if not getattr(a, "enemy_heads", None):
            return None
        cx, cy = a.center
        for head in a.enemy_heads:
            hx, hy, area, dist = head[:4]
            if not (safe * 0.9 < dist < safe * 3.4):
                continue
            incoming = math.atan2(cy - hy, cx - hx)
            rel = abs(math.atan2(math.sin(incoming - self.heading),
                                 math.cos(incoming - self.heading)))
            if rel > math.radians(125):
                continue
            side = incoming + (math.pi / 2 if math.sin(incoming - self.heading) >= 0
                               else -math.pi / 2)
            return math.atan2(math.sin(side), math.cos(side))
        return None

    def _commit(self, target, mode, angles, obstacle_d, corridor_w, panic,
                smooth=1.0, strict=False):
        safe_target = (target if strict else
                       self._safe_heading(target, angles, obstacle_d,
                                          corridor_w, panic))
        d = math.atan2(math.sin(safe_target - self.heading),
                       math.cos(safe_target - self.heading))
        current_ray = int(np.argmin(np.abs(np.angle(
            np.exp(1j * (angles - self.heading))))))
        emergency = obstacle_d[current_ray] < panic * 1.35
        if strict:
            max_turn = math.radians(132)
        else:
            max_turn = math.radians(82 if obstacle_d[current_ray] < 14 else
                                    58 if emergency else 24)
        step = max(-max_turn, min(max_turn, smooth * d))
        self.heading = self.heading + step
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        self.mode = mode
        return self.heading

    @staticmethod
    def _safe_heading(target, angles, obstacle_d, corridor_w, panic):
        ray = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - target))))))
        min_clear = panic * 1.18
        min_width = panic * 0.55
        if obstacle_d[ray] >= min_clear and corridor_w[ray] >= min_width:
            return float(angles[ray])
        clearance = obstacle_d / max(1.0, obstacle_d.max())
        width = corridor_w / max(1.0, corridor_w.max())
        turn = np.abs(np.angle(np.exp(1j * (angles - target))))
        score = clearance * 2.2 + width * 1.5 - turn * 0.55
        score[obstacle_d < panic] = -1e9
        if (score < -1e8).all():
            score = obstacle_d + corridor_w * 0.8 - turn * panic
        return float(angles[int(np.argmax(score))])

    # ------------------------------------------------------------------

    @staticmethod
    def _edge_escape_gain(hazard, angles):
        """Steer back inside when a large hazard mass touches the screen edge.

        Arena borders and huge snake bodies both produce this pattern; either
        way, food chasing should pause until the head is back in open space.
        """
        h, w = hazard.shape
        band_x = max(6, int(w * 0.055))
        band_y = max(6, int(h * 0.075))
        left = hazard[:, :band_x].mean()
        right = hazard[:, w - band_x:].mean()
        top = hazard[:band_y, :].mean()
        bottom = hazard[h - band_y:, :].mean()
        vx = left - right
        vy = top - bottom
        strength = abs(vx) + abs(vy)
        if strength < 0.22:
            return None
        target = math.atan2(vy, vx)
        gain = np.zeros(len(angles), np.float32)
        for i, ang in enumerate(angles):
            dd = abs(math.atan2(math.sin(ang - target),
                                math.cos(ang - target)))
            gain[i] = math.exp(-2.5 * dd) * strength
        return gain

    @staticmethod
    def _border_push_angle(a):
        ys, xs = np.nonzero(a.border_mask)
        if len(xs) < 20:
            return None
        cx, cy = a.center
        # Use the nearest border pixels, not the whole wall/dead-zone mass.
        # Otherwise a huge left/bottom border can average into a bad diagonal.
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        d2 = dx * dx + dy * dy
        keep = d2 <= np.percentile(d2, 45)
        if keep.sum() < 10:
            keep = d2 <= np.percentile(d2, 70)
        bx = float(xs[keep].mean())
        by = float(ys[keep].mean())
        vx, vy = cx - bx, cy - by
        if abs(vx) + abs(vy) < 1.0:
            return None
        return math.atan2(vy, vx)

    @staticmethod
    def _head_ray_risk(a, cx, cy, angles, safe):
        risk = np.zeros(len(angles), np.float32)
        heads = getattr(a, "enemy_heads", None) or []
        for head in heads:
            hx, hy, area, dist = head[:4]
            if dist > safe * 4.0:
                continue
            ang = math.atan2(hy - cy, hx - cx)
            width = 0.55 if dist < safe * 1.6 else 0.34
            strength = max(0.0, (safe * 4.0 - dist) / (safe * 4.0))
            dd = np.abs(np.angle(np.exp(1j * (angles - ang))))
            risk = np.maximum(risk, np.exp(-(dd / width) ** 2) * strength)
        return risk

    # ------------------------------------------------------------------

    def _coil(self, a):
        """Tight circle around our own body mass. Turn direction curls the
        head toward our body centroid (tuck inward) when threats are outside;
        away from it (push outward) when an enemy is inside our loop."""
        cx, cy = a.center
        ys, xs = np.nonzero(a.own_mask)
        if len(xs) > 50:
            ang_c = math.atan2(float(ys.mean()) - cy, float(xs.mean()) - cx)
        else:
            ang_c = self.heading + math.pi / 2
        d = math.atan2(math.sin(ang_c - self.heading),
                       math.cos(ang_c - self.heading))
        sign = 1.0 if d > 0 else -1.0
        if a.enemy_inside:
            sign = -sign
        self.heading = math.atan2(math.sin(self.heading + sign * self.coil_turn),
                                  math.cos(self.heading + sign * self.coil_turn))
        return self.heading

    def _boost_decision(self, a, score_est, obstacle_d, flee,
                        food_gain=None, rush=False):
        want = self._boost_want(a, score_est, obstacle_d, flee, food_gain, rush)
        now = time.time()
        if want:
            if not self.boost:
                # respect cooldown before starting a new burst
                if now - self._boost_off_since < 3.0:
                    return False
                self._boost_on_since = now
            elif not flee and now - self._boost_on_since > 1.2:
                self._boost_off_since = now
                return False          # burst over (fleeing may run longer)
            return True
        if self.boost:
            self._boost_off_since = now
        return False

    def _boost_want(self, a, score_est, obstacle_d, flee, food_gain,
                    rush=False):
        if score_est < self.boost_min_score:
            return False        # never burn points in the early game
        if a.own_area < 2200:
            # body size is the honest score signal (the on-screen OCR
            # sometimes reads 220 while we're actually tiny) — a small
            # snake must never pay points for speed
            return False
        if rush:
            if getattr(self, "_rush_risk", 1.0) > 0.55:
                return False
            return obstacle_d.max() > obstacle_d.mean() * 1.35
        if flee:
            # only rush out if the chosen escape is actually open —
            # boosting into a tight pocket just gets us killed faster
            return (self.last_min_d > 0 and
                    obstacle_d.max() > obstacle_d.mean() * 1.5)
        if food_gain is None:
            return False
        i = int(np.argmax(food_gain))
        return (food_gain[i] > 9.0 and
                obstacle_d[i] > obstacle_d.mean() and
                obstacle_d.min() > obstacle_d.max() * 0.25)

    # ------------------------------------------------------------------

    @staticmethod
    def _enclosed(hazard, x, y, cap=1500):
        """True if the free space reachable from (x, y) is a small pocket
        (e.g. the inside of another snake's coil). Quarter-scale flood."""
        m = (hazard[::2, ::2] > 0).astype(np.uint8)
        hq, wq = m.shape
        xi, yi = int(x / 2), int(y / 2)
        if not (0 <= xi < wq and 0 <= yi < hq) or m[yi, xi]:
            return False        # probe inside an obstacle: rays handle that
        ff_mask = np.zeros((hq + 2, wq + 2), np.uint8)
        cv2.floodFill(m, ff_mask, (xi, yi), 1,
                      flags=8 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        return int(ff_mask.sum()) < cap

    @staticmethod
    def _ray_distances(mask, cx, cy, angles, ray_len, skip_near=6, step=4):
        h, w = mask.shape
        ts = np.arange(skip_near, int(ray_len), step, dtype=np.float32)
        cos = np.cos(angles)[:, None]
        sin = np.sin(angles)[:, None]
        xs = np.clip((cx + cos * ts).astype(np.int32), 0, w - 1)
        ys = np.clip((cy + sin * ts).astype(np.int32), 0, h - 1)
        hits = mask[ys, xs] > 0                     # (rays, steps)
        first = np.argmax(hits, axis=1)
        any_hit = hits.any(axis=1)
        d = np.where(any_hit, ts[first], ray_len)
        return d.astype(np.float32)

    @classmethod
    def _corridor_widths(cls, mask, cx, cy, angles, obstacle_d, ray_len,
                         side_len):
        """Approximate corridor width at a lookahead point on each ray.

        Long straight clearance alone is misleading in Snake.io: a one-body
        slit can look open down the center but close before our head clears it.
        This samples left/right clearance at the point we are about to enter.
        """
        widths = np.zeros(len(angles), np.float32)
        h, w = mask.shape
        for i, ang in enumerate(angles):
            probe = min(max(18.0, obstacle_d[i] * 0.45), ray_len * 0.55)
            px = int(max(0, min(w - 1, cx + probe * math.cos(ang))))
            py = int(max(0, min(h - 1, cy + probe * math.sin(ang))))
            side = np.array([ang + math.pi / 2, ang - math.pi / 2],
                            dtype=np.float32)
            d = cls._ray_distances(mask, px, py, side, side_len,
                                   skip_near=2, step=3)
            widths[i] = float(min(d[0], d[1]))
        return widths

    def _food_gain(self, a, cx, cy, angles):
        gain = np.zeros(len(angles), np.float32)
        if len(a.food_pts) == 0:
            return gain
        dx = a.food_pts[:, 0] - cx
        dy = a.food_pts[:, 1] - cy
        dist = np.hypot(dx, dy) + 1e-3
        fang = np.arctan2(dy, dx)
        wgt = a.food_w * np.exp(-self.food_decay * dist)
        for i, ang in enumerate(angles):
            dd = np.abs(np.angle(np.exp(1j * (fang - ang))))
            gain[i] = float((wgt * np.exp(-2.2 * dd)).sum())
        return gain
