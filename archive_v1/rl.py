"""Per-decision online RL for ray selection — learns during play, updates
every tick instead of every life (thousands of gradient steps per hour).

Policy: softmax over rays, logits = w . x_ray where x_ray are the SAME
features the heuristic scores with. w is warm-started to mimic the
heuristic, then REINFORCE with baseline pushes it toward what actually
grows the snake and away from what precedes deaths. Hard safety vetoes
(blocked rays) are applied before sampling — the learner can prefer, not
suicide. State persists in rl.json.
"""

import json
import math
import os

import numpy as np

FEATS = ["food", "loot", "open", "corridor", "close2", "headrisk",
         "align", "crowd"]
W0 = np.array([1.0, 5.0, 1.1, 1.9, -7.0, -6.0, -0.45, 0.5], np.float32)
W_MIN = W0 - np.abs(W0) * 1.5 - 0.5
W_MAX = W0 + np.abs(W0) * 1.5 + 0.5

LR = 0.02
GAMMA = 0.97
TEMP = 0.35            # softmax temperature: mostly greedy, some explore
DEATH_PENALTY = -6.0
TRACE_TICKS = 30       # death blame reaches ~1.5s back at 20 Hz


class OnlineRL:
    def __init__(self, path, log=print):
        self.path = path
        self.log = log
        self.w = W0.copy()
        self.baseline = 0.0
        self.episodes = 0
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    s = json.load(f)
                self.w = np.array(s["w"], np.float32)
                self.baseline = float(s.get("baseline", 0.0))
                self.episodes = int(s.get("episodes", 0))
                self.log(f"rl: resumed after {self.episodes} lives, "
                         f"w={[round(float(v), 2) for v in self.w]}")
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        self.buf = []          # per-tick (x_all, probs, chosen, reward)

    # ------------------------------------------------------------------

    def choose(self, x, blocked):
        """x: (rays, feats) feature matrix. Returns chosen ray index."""
        logits = x @ self.w
        logits[blocked] = -1e9
        z = logits / TEMP
        z -= z.max()
        p = np.exp(z)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return int(np.argmax(logits)), None
        p /= s
        i = int(np.random.choice(len(p), p=p))
        return i, p

    def record(self, x, p, chosen, reward):
        if p is not None:
            self.buf.append((x, p, chosen, float(reward)))
            if len(self.buf) > 4000:
                self.buf = self.buf[-4000:]

    def end_life(self, died=True):
        """REINFORCE over the buffered episode."""
        if not self.buf:
            return
        n = len(self.buf)
        rewards = np.array([b[3] for b in self.buf], np.float32)
        if died:
            # blame the tail: the choices right before death caused it
            k = min(TRACE_TICKS, n)
            rewards[-k:] += DEATH_PENALTY * np.linspace(0.2, 1.0, k)
        # discounted returns, then advantage vs running baseline
        ret = np.zeros(n, np.float32)
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc = rewards[i] + GAMMA * acc
            ret[i] = acc
        self.baseline = 0.98 * self.baseline + 0.02 * float(ret.mean())
        adv = ret - self.baseline
        grad = np.zeros_like(self.w)
        for (x, p, c, _), a in zip(self.buf, adv):
            grad += a * (x[c] - p @ x)
        self.w += LR * grad / max(1, n)
        self.w = np.clip(self.w, W_MIN, W_MAX)
        self.episodes += 1
        self.buf = []
        if self.episodes % 5 == 0:
            self.log(f"rl: {self.episodes} lives, baseline "
                     f"{self.baseline:.1f}, w="
                     f"{[round(float(v), 2) for v in self.w]}")
        self._save()

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"w": [float(v) for v in self.w],
                           "baseline": self.baseline,
                           "episodes": self.episodes}, f, indent=1)
        except OSError:
            pass
