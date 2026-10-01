"""Score-driven self-learning: a (mu, lambda) evolution strategy over the
strategy weights. Each life is one (noisy) fitness sample, so candidates are
evaluated over several lives before the generation updates. State persists in
es.json so learning continues across bot restarts.

Fitness = survival seconds + a bonus for the honest score tier reached.
"""

import json
import os
import random

# name -> (default, relative mutation scale, lo, hi)
PARAM_SPACE = {
    "w_food":        (1.0,  0.25, 0.2, 4.0),
    "w_loot":        (5.0,  0.25, 0.5, 15.0),
    "w_crowd":       (1.0,  0.30, 0.0, 3.0),
    "w_open":        (1.1,  0.25, 0.2, 4.0),
    "w_corridor":    (1.9,  0.25, 0.2, 5.0),
    "w_danger":      (1.0,  0.25, 0.3, 3.0),
    "w_headrisk":    (1.0,  0.25, 0.3, 3.0),
    "w_turn":        (0.45, 0.25, 0.1, 1.2),
    "greed_early":   (2.6,  0.20, 1.0, 5.0),
    "safe_dist_frac":   (0.105, 0.12, 0.07, 0.17),
    "panic_dist_frac":  (0.060, 0.12, 0.040, 0.10),
    "reflex_dist_frac": (0.085, 0.12, 0.055, 0.14),
    "boost_min_score":  (220,  0.15, 220, 450),
    "coil_min_area":    (1500, 0.20, 800, 4000),
}

POP = 6           # candidates per generation
LIVES_PER_CAND = 2
ELITE = 2         # top candidates averaged into the next base
MIN_LIFE_S = 8.0  # spawn-luck deaths don't count as evaluations

TIER_BONUS = {0: 0, 5: 0, 10: 0, 30: 5, 80: 25, 100: 35, 220: 80, 1000: 300}


def _clamp(name, v):
    _, _, lo, hi = PARAM_SPACE[name]
    return max(lo, min(hi, v))


class Evolver:
    def __init__(self, path, log=print):
        self.path = path
        self.log = log
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.s = json.load(f)
            self.log(f"evolver: resumed gen {self.s['gen']} "
                     f"(best fitness so far {self.s.get('best_fit', 0):.0f})")
        else:
            base = {k: v[0] for k, v in PARAM_SPACE.items()}
            self.s = {"gen": 0, "base": base, "sigma": 1.0,
                      "best_fit": 0.0, "best": dict(base),
                      "pop": self._spawn(base, 1.0), "idx": 0}
            self._save()

    # ------------------------------------------------------------------

    def _spawn(self, base, sigma):
        pop = [{"params": dict(base), "fits": []}]      # elite carry-over
        while len(pop) < POP:
            p = {}
            for k, (d, rel, lo, hi) in PARAM_SPACE.items():
                span = (hi - lo) * rel * sigma
                p[k] = _clamp(k, base[k] + random.gauss(0, span))
            pop.append({"params": p, "fits": []})
        return pop

    def current_params(self):
        return dict(self.s["pop"][self.s["idx"]]["params"])

    def life_result(self, duration, max_score_tier, stats=None):
        """Called once per (debounced, real) life. Returns True if the
        active candidate changed (bot should rebuild its Pilot).

        Shaped fitness: survival + score tier (scaled up when reached FAST)
        + orbs eaten (body growth) + loot-rush growth (double weight — the
        whole point of kills) + successful dodges + crowd time survived.
        """
        if duration < MIN_LIFE_S:
            return False
        stats = stats or {}
        tier = int(max_score_tier)
        tier_t = stats.get("tier_t", {}).get(str(tier),
                                             stats.get("tier_t", {}).get(tier))
        speed = 1.0 + (30.0 / max(30.0, tier_t)) if tier_t else 1.0
        fit = (duration
               + TIER_BONUS.get(tier, 0) * speed
               + stats.get("growth", 0.0) / 50.0
               + stats.get("loot_growth", 0.0) / 25.0
               + min(stats.get("dodges", 0), 40) * 2.0
               + min(stats.get("crowd_s", 0.0), 60.0) * 0.8)
        # avoidable deaths cost fitness — the genome answers for HOW it died
        fit -= {"boost-suicide": 30, "wall": 25, "boxed-in": 20,
                "head-on": 12}.get(stats.get("cause", ""), 0)
        fit = max(0.0, fit)
        cand = self.s["pop"][self.s["idx"]]
        cand["fits"].append(round(fit, 1))
        if fit > self.s["best_fit"]:
            self.s["best_fit"] = fit
            self.s["best"] = dict(cand["params"])
        moved = False
        if len(cand["fits"]) >= LIVES_PER_CAND:
            self.s["idx"] += 1
            moved = True
        if self.s["idx"] >= len(self.s["pop"]):
            self._next_generation()
            moved = True
        self._save()
        return moved

    def _next_generation(self):
        scored = [(sum(c["fits"]) / max(1, len(c["fits"])), c)
                  for c in self.s["pop"] if c["fits"]]
        scored.sort(key=lambda x: -x[0])
        if scored:
            elite = scored[:ELITE]
            base = {}
            for k in PARAM_SPACE:
                base[k] = sum(c["params"][k] for _, c in elite) / len(elite)
            self.log(f"evolver: gen {self.s['gen']} done, "
                     f"top fitness {scored[0][0]:.0f}, "
                     f"new base {{'safe': {base['safe_dist_frac']:.3f}, "
                     f"'danger': {base['w_danger']:.2f}, "
                     f"'loot': {base['w_loot']:.2f}}}")
        else:
            base = self.s["base"]
        self.s["gen"] += 1
        self.s["base"] = base
        self.s["sigma"] = max(0.35, self.s["sigma"] * 0.96)
        self.s["pop"] = self._spawn(base, self.s["sigma"])
        self.s["idx"] = 0

    def best_params(self):
        return dict(self.s.get("best") or self.s["base"])

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.s, f, indent=1)
