"""Online self-tuning: after every death, nudge the strategy parameters in
the direction the evidence points, within safe bounds. Persists to
learn.json so the bot keeps improving across restarts."""

import json
import os
import time

BOUNDS = {
    "safe_dist_frac":   (0.075, 0.105),
    "panic_dist_frac":  (0.04, 0.06),
    "reflex_dist_frac": (0.055, 0.085),
    "boost_min_score":  (300, 500),
    "turn_smooth":      (0.30, 0.60),
}

TUNED_KEYS = list(BOUNDS)


def _clamp(key, v):
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, v))


class Tuner:
    def __init__(self, cfg, path, log=print):
        self.cfg = cfg
        self.path = path
        self.log = log
        self.history = []          # recent life records
        self.good_streak = 0
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    saved = json.load(f)
                changed = False
                for k, v in saved.get("params", {}).items():
                    if k in BOUNDS:
                        cv = _clamp(k, v)
                        cfg["strategy"][k] = cv
                        changed = changed or cv != v
                self.history = saved.get("history", [])[-200:]
                self.log(f"tuner: restored params {self._params()} "
                         f"({len(self.history)} lives on record)")
                if changed:
                    self._save()
            except (json.JSONDecodeError, OSError):
                pass

    def _params(self):
        return {k: round(self.cfg["strategy"][k], 4) for k in TUNED_KEYS}

    def death(self, duration, mode, boosting, max_score):
        """Called once per life that ended in a death."""
        s = self.cfg["strategy"]
        rec = {"t": int(time.time()), "dur": round(duration, 1),
               "mode": mode, "boost": bool(boosting), "score": max_score}
        self.history.append(rec)
        changed = []

        # Learn from every death. Very short lives are noisy, so the nudge is
        # small, but they still expose panic/reflex crashes that used to be
        # thrown away.
        if duration < 10:
            if mode in ("reflex", "graze-run", "edge-escape", "coil"):
                s["panic_dist_frac"] = _clamp("panic_dist_frac",
                                              s["panic_dist_frac"] * 1.03)
                changed.append("panic+")
            if boosting:
                s["boost_min_score"] = _clamp("boost_min_score",
                                              s["boost_min_score"] + 20)
                changed.append("boost_min_score+")
            self._save()
            if changed:
                self.log(f"tuner: {'/'.join(changed)} -> {self._params()}")
            return
        self._short_count = getattr(self, "_short_count", 0)
        if duration < 45:
            self._short_count += 1
            if self._short_count % 2:
                self._save()
                return
        else:
            self._short_count = 0

        if duration < 45:
            self.good_streak = 0
            if boosting:
                # died while boosting -> boost later + demand more room
                s["boost_min_score"] = _clamp("boost_min_score",
                                              s["boost_min_score"] + 25)
                changed.append("boost_min_score+")
            if mode == "reflex":
                # reflex fired but too late -> look further ahead
                s["reflex_dist_frac"] = _clamp("reflex_dist_frac",
                                               s["reflex_dist_frac"] * 1.08)
                changed.append("reflex+")
            if mode == "cruise":
                # blindsided in open play -> bigger safety margins
                s["safe_dist_frac"] = _clamp("safe_dist_frac",
                                             s["safe_dist_frac"] * 1.05)
                s["panic_dist_frac"] = _clamp("panic_dist_frac",
                                              s["panic_dist_frac"] * 1.05)
                changed.append("margins+")
        else:
            self.good_streak += 1
            if self.good_streak >= 3:
                # three long lives in a row -> we can afford more greed
                self.good_streak = 0
                s["safe_dist_frac"] = _clamp("safe_dist_frac",
                                             s["safe_dist_frac"] * 0.97)
                s["boost_min_score"] = _clamp("boost_min_score",
                                              s["boost_min_score"] - 10)
                changed.append("greedier")

        self._save()
        if changed:
            self.log(f"tuner: {'/'.join(changed)} -> {self._params()}")

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"params": {k: self.cfg["strategy"][k]
                                  if k not in BOUNDS else _clamp(k, self.cfg["strategy"][k])
                                  for k in TUNED_KEYS},
                       "target_score": 20000,
                       "history": self.history[-200:]}, f, indent=1)
