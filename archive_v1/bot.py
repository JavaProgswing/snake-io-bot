"""Snake.io autopilot. Usage:  python bot.py [--debug-dir DIR]

State machine: menu -> playing -> revive/dead -> menu. Steers via a virtual
joystick (held pointer 1), boosts with pointer 2, taps UI with pointer 4.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import cv2

import vision
from evolver import Evolver
from scrcpy_link import ScrcpyLink, ACTION_DOWN, ACTION_MOVE, ACTION_UP, KEYCODE_BACK
from strategy import Pilot
from tuner import Tuner

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(HERE, "live")


class Joystick:
    """Pointer held down at an anchor; MOVEs around it set direction."""

    def __init__(self, link, cfg):
        self.link = link
        ax, ay = cfg["ui"]["joystick_anchor"]
        self.ax_f, self.ay_f = ax, ay
        self.r_frac = cfg["ui"]["joystick_radius_frac"]
        self.down = False

    def _geom(self):
        frame, _ = self.link.frame()
        h, w = frame.shape[:2]
        return int(w * self.ax_f), int(h * self.ay_f), int(w * self.r_frac)

    def steer(self, angle):
        ax, ay, r = self._geom()
        x = ax + r * math.cos(angle)
        y = ay + r * math.sin(angle)
        if not self.down:
            self.link.touch(ACTION_DOWN, ax, ay, pointer_id=1)
            self.down = True
        self.link.touch(ACTION_MOVE, x, y, pointer_id=1)

    def release(self):
        if self.down:
            ax, ay, _ = self._geom()
            self.link.touch(ACTION_UP, ax, ay, pointer_id=1)
            self.down = False


class Booster:
    def __init__(self, link, cfg):
        self.link = link
        self.bx_f, self.by_f = cfg["ui"]["boost_button"]
        self.down = False

    def set(self, on):
        frame, _ = self.link.frame()
        h, w = frame.shape[:2]
        x, y = int(w * self.bx_f), int(h * self.by_f)
        if on and not self.down:
            self.link.touch(ACTION_DOWN, x, y, pointer_id=2)
            self.down = True
        elif not on and self.down:
            self.link.touch(ACTION_UP, x, y, pointer_id=2)
            self.down = False


class Bot:
    def __init__(self, cfg, debug_dir=None, learn="off", brain="none"):
        self.cfg = cfg
        self.learn = learn
        self.debug_dir = debug_dir
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        self.link = ScrcpyLink(
            cfg["adb"], cfg["server_jar"], cfg["server_version"],
            cfg["port"], cfg["max_size"], cfg["max_fps"],
            cfg["video_bit_rate"],
            log_path=os.path.join(debug_dir, "server.log") if debug_dir else None)
        self.joy = None
        self.boost = None
        self.revives_used = 0
        self.games = 0
        self.score_est = cfg["score"]["start_score"]
        self.play_t0 = None
        self.last_state = None
        self.state_since = time.time()
        self.last_debug = 0.0
        self.last_hb = 0.0
        self.hb_frames = 0
        self.frames_done = 0
        # learning backends: 'active' = score-driven evolution over the
        # strategy weights (each life is a fitness sample); 'off' = play the
        # best genome found so far. The old heuristic tuner only runs when
        # not evolving (they would fight over the same knobs).
        self.evolver = None
        self.rl = None
        weights = None
        if learn == "rl":
            from rl import OnlineRL
            self.rl = OnlineRL(os.path.join(HERE, "rl.json"), self.log)
            self.tuner = None
        elif learn == "active":
            self.evolver = Evolver(os.path.join(HERE, "es.json"), self.log)
            weights = self.evolver.current_params()
            self.tuner = None
        else:
            es_path = os.path.join(HERE, "es.json")
            if os.path.exists(es_path):
                self.evolver = Evolver(es_path, self.log)
                weights = self.evolver.best_params()
                self.evolver = None      # play best genome, no learning
                self.log("using best evolved weights")
            self.tuner = Tuner(cfg, os.path.join(HERE, "learn.json"), self.log)
        self.weights = weights
        self.pilot = Pilot(cfg, self.weights)
        self.pilot.rl = self.rl
        self.brain = None
        if brain == "bc":
            import brain as brain_mod
            self.brain = brain_mod.load_if_available(HERE)
            self.log("brain: bc_policy loaded" if self.brain
                     else "brain: bc_policy.pt missing -> heuristic only")
        self.life_t0 = None
        self.game_max_score = 0
        self.score_reads = []
        self.score_cand = None
        self.score_cand_t = 0.0
        # rolling last ~15s of raw frames plus numeric analysis layers; dumped
        # on every death so each death can be reviewed/recolored frame by frame
        self.replay = deque(maxlen=30)
        self.last_replay = 0.0
        self.deaths = 0
        self.pending_death = None
        self.life_stats = self._fresh_stats()
        # rolling tail of play ticks for the death autopsy
        self.mode_hist = deque(maxlen=80)
        os.makedirs(LIVE_DIR, exist_ok=True)
        self.last_live = 0.0
        self._nudge_mtime = 0.0
        self._nudge = None
        self.death_db_path = os.path.join(HERE, "deaths.json")
        try:
            with open(self.death_db_path, encoding="utf-8") as f:
                self.death_db = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.death_db = {}

    @staticmethod
    def _fresh_stats():
        return {"growth": 0.0, "loot_growth": 0.0, "dodges": 0,
                "crowd_s": 0.0, "peak_area": 0, "tier_t": {},
                "last_mode": "", "last_tick": None}

    def read_nudge(self):
        """The web viewer writes live/nudge.json when the user drags on the
        live canvas. Obey it while fresh; each packet steers ~1.2s."""
        p = os.path.join(LIVE_DIR, "nudge.json")
        try:
            mt = os.path.getmtime(p)
            if mt != self._nudge_mtime:
                self._nudge_mtime = mt
                with open(p, encoding="utf-8") as f:
                    self._nudge = json.load(f)
            n = self._nudge
            if n and time.time() - n.get("t", 0) < n.get("ttl", 1.2):
                return n
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def publish_live(self, frame, a, angle, t):
        """Drop raw + overlay JPEGs and a meta json for the web viewer."""
        if t - self.last_live < 0.15:
            return
        self.last_live = t
        try:
            cv2.imwrite(os.path.join(LIVE_DIR, "raw.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 72])
            hud = (f"{self.pilot.mode} sc{self.score_est:.0f} "
                   f"minE{self.pilot.last_min_d:.0f} B{int(self.pilot.boost)}")
            ov = vision.overlay_debug(frame, a, angle,
                                      danger=self.pilot._danger, hud=hud)
            cv2.imwrite(os.path.join(LIVE_DIR, "overlay.jpg"), ov,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            meta = {
                "t": round(t, 2), "game": self.games,
                "life_s": round(t - (self.life_t0 or t), 1),
                "mode": self.pilot.mode, "score": int(self.score_est),
                "min_enemy_px": round(float(self.pilot.last_min_d), 1),
                "own_area": int(a.own_area), "food": int(len(a.food_pts)),
                "loot": int(len(a.loot)),
                "heads": len(getattr(a, "enemy_heads", []) or []),
                "boost": bool(self.pilot.boost),
                "heading_deg": round(math.degrees(angle), 1),
                "nudged": bool(self.life_stats.get("nudged")),
                "learn": self.learn,
            }
            tmp = os.path.join(LIVE_DIR, "meta.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            os.replace(tmp, os.path.join(LIVE_DIR, "meta.json"))
        except OSError:
            pass

    def autopsy(self, t_death, life_s):
        """Classify how the life ended from the last ~2s of play ticks.
        Every death gets a cause, a log line with the evidence, a running
        tally, and (in learn mode) a fitness penalty for that cause."""
        tail = [e for e in self.mode_hist if t_death - e["t"] < 2.0]
        if life_s < 8:
            cause = "spawn-luck"
        elif not tail:
            cause = "unknown"
        elif any(e["boost"] for e in tail[-8:]):
            cause = "boost-suicide"
        elif sorted(e["border"] for e in tail)[len(tail) // 2] > 3000 or \
                sum(e["mode"] == "wall-escape" for e in tail) >= len(tail) // 2:
            cause = "wall"
        elif sum(e["mode"] in ("coil", "graze-run", "hug-break")
                 for e in tail) >= len(tail) * 0.6:
            cause = "boxed-in"
        elif any(e["mode"] == "reflex" for e in tail[-6:]):
            cause = "late-reflex"
        else:
            cause = "head-on"
        modes = "->".join(dict.fromkeys(e["mode"] for e in tail[-8:]))
        min_e = min((e["minE"] for e in tail), default=-1)
        self.log(f"death autopsy: {cause} (life {life_s:.0f}s, last modes "
                 f"{modes or '-'}, minE {min_e:.0f})")
        self.death_db[cause] = self.death_db.get(cause, 0) + 1
        try:
            with open(self.death_db_path, "w", encoding="utf-8") as f:
                json.dump(self.death_db, f, indent=1)
        except OSError:
            pass
        total = sum(self.death_db.values())
        if total % 10 == 0:
            self.log(f"death causes after {total}: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.death_db.items(),
                                              key=lambda kv: -kv[1])))
        return cause

    # ---------- helpers ----------

    def tap_frac(self, fx, fy):
        frame, _ = self.link.frame()
        h, w = frame.shape[:2]
        self.link.tap(int(w * fx), int(h * fy))

    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def dump_replay(self):
        if not self.debug_dir or not self.replay:
            return
        self.deaths += 1
        t_end = self.replay[-1][0]
        manifest = []
        for item in self.replay:
            if len(item) == 5:
                ts, raw, layer, overlay, meta = item
            else:
                ts, raw, layer, overlay = item
                meta = {}
            # filenames sort chronologically and read as time-to-death
            suffix = f"t-{t_end - ts:05.1f}s.png"
            raw_name = f"death{self.deaths:03d}_raw_{suffix}"
            layer_name = f"death{self.deaths:03d}_layer_{suffix}"
            overlay_name = f"death{self.deaths:03d}_overlay_{suffix}"
            cv2.imwrite(os.path.join(self.debug_dir, raw_name), raw)
            cv2.imwrite(os.path.join(self.debug_dir, layer_name), layer)
            cv2.imwrite(os.path.join(self.debug_dir, overlay_name), overlay)
            meta = dict(meta)
            meta.update({"raw": raw_name, "layer": layer_name,
                         "overlay": overlay_name,
                         "time_to_death": round(t_end - ts, 2)})
            manifest.append(meta)
        with open(os.path.join(self.debug_dir,
                               f"death{self.deaths:03d}_meta.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"death": self.deaths, "frames": manifest}, f,
                      indent=2)
        self.log(f"death #{self.deaths}: dumped {len(self.replay)} "
                 f"replay frames")
        self.replay.clear()

    def snap(self, tag, frame, a=None, angle=None):
        if not self.debug_dir:
            return
        vis = vision.overlay_debug(frame, a, angle) if a is not None else frame
        cv2.imwrite(os.path.join(self.debug_dir, f"{tag}_{int(time.time())}.png"), vis)

    # ---------- main ----------

    def run(self, duration=None):
        self.link.start()
        self.link.clear_pointers()
        frame, _ = self.link.frame()
        self.log(f"video up: {frame.shape[1]}x{frame.shape[0]}")
        self.joy = Joystick(self.link, self.cfg)
        self.boost = Booster(self.link, self.cfg)

        period = 1.0 / self.cfg["strategy"]["decision_hz"]
        last_seq = 0
        self.last_play = time.time()
        end_at = None if duration is None else time.time() + duration
        try:
            while True:
                if end_at is not None and time.time() >= end_at:
                    self.log(f"duration {duration:.1f}s reached -> stopping")
                    break
                t = time.time()
                frame, seq = self.link.frame()
                if seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq

                state = vision.classify_state(frame)
                if state != self.last_state:
                    self.log(f"state: {self.last_state} -> {state}")
                    self.snap(f"enter_{state}", frame)
                    if state == vision.STATE_MENU:
                        self.menu_taps = 0
                    if state == vision.STATE_DIALOG:
                        self.dialog_taps = 0
                    if state == vision.STATE_PLAYING:
                        if self.pending_death is None or \
                                t - self.pending_death[0] >= 1.5:
                            self.life_stats = self._fresh_stats()
                        if self.pending_death is not None and \
                                t - self.pending_death[0] < 1.5:
                            # classifier flicker (death-fade band, overlay
                            # flash) — same life continues; a false death
                            # here used to reset grace, poison the tuner
                            # and dump junk replays
                            self.life_t0 = self.pending_death[1]
                        else:
                            self.life_t0 = t
                            vision.reset_own_tracking()
                        self.pending_death = None
                    elif self.last_state == vision.STATE_PLAYING and \
                            self.life_t0 is not None:
                        self.pending_death = (t, self.life_t0)
                    self.last_state = state
                    self.state_since = t

                # commit a pending death only after it survives the debounce
                if self.pending_death is not None and \
                        state != vision.STATE_PLAYING and \
                        t - self.pending_death[0] >= 1.5:
                    life_s = t - self.pending_death[1]
                    cause = self.autopsy(self.pending_death[0], life_s)
                    if self.rl is not None:
                        self.rl.end_life(died=True)
                    if self.tuner is not None:
                        self.tuner.death(life_s, self.pilot.mode,
                                         self.pilot.boost,
                                         self.game_max_score)
                    if self.evolver is not None and \
                            self.life_stats.get("nudged"):
                        self.log("life was nudged by user -> not scored "
                                 "for evolution")
                    elif self.evolver is not None:
                        st = {k: v for k, v in self.life_stats.items()
                              if k not in ("last_mode", "last_tick",
                                           "peak_area")}
                        self.log(f"life: {life_s:.0f}s tier "
                                 f"{self.game_max_score} growth "
                                 f"{st['growth']:.0f} loot "
                                 f"{st['loot_growth']:.0f} dodges "
                                 f"{st['dodges']} crowd {st['crowd_s']:.0f}s")
                        st["cause"] = cause
                        if self.evolver.life_result(life_s,
                                                    self.game_max_score, st):
                            self.weights = self.evolver.current_params()
                            self.log("evolver: next candidate")
                    self.life_t0 = None
                    self.pending_death = None
                    self.dump_replay()

                if state == vision.STATE_PLAYING:
                    self.last_play = t
                    self.step_play(frame, t)
                elif t - self.last_play < 0.8:
                    # classifier blip (wall glare, death-fade flash): KEEP
                    # the joystick held and skip menu/dialog handlers — a
                    # released stick mid-game means zero inputs while the
                    # snake sails straight
                    pass
                else:
                    self.joy.release()
                    self.boost.set(False)
                    # hard recovery: too long without gameplay means the app
                    # crashed or we're on a screen taps can't fix (system
                    # launcher, store page...) — relaunch the game itself
                    if t - self.last_play > 45 and \
                            t - self.state_since > 8:
                        self.log(f"no gameplay for 45s (state={state}) "
                                 f"-> relaunching game app")
                        self.snap("relaunch", frame)
                        self.link.launch_app(self.cfg["package"])
                        self.last_play = t
                        time.sleep(8.0)
                        continue
                    if state == vision.STATE_MENU:
                        self.step_menu(t)
                    elif state == vision.STATE_REVIVE:
                        self.step_revive(t)
                    elif state == vision.STATE_DIALOG:
                        self.step_dialog(frame, t)
                    else:
                        self.step_unknown(frame, t)

                dt = time.time() - t
                if dt < period:
                    time.sleep(period - dt)
        finally:
            self.joy.release()
            self.boost.set(False)
            self.link.stop()

    def step_menu(self, t):
        if self.play_t0 is not None:
            self.log(f"game over. max score >= {self.game_max_score}, "
                     f"revives used {self.revives_used}")
            self.play_t0 = None
            self.game_max_score = 0
        # fresh game: reset counters, tap the orange PLAY wherever it is
        if t - self.state_since > 1.0:
            self.revives_used = 0
            self.score_est = self.cfg["score"]["start_score"]
            self.score_reads = []
            self.score_cand = None
            self.pilot = Pilot(self.cfg, self.weights)
            self.pilot.rl = self.rl
            frame, _ = self.link.frame()
            self.menu_taps = getattr(self, "menu_taps", 0) + 1
            btn = vision.find_button(frame, 10, 30, s_max=215)
            # if the detected button keeps not working, rotate through the
            # known PLAY locations instead of hammering the same spot
            if btn and self.menu_taps <= 2:
                self.link.tap(*btn)
                self.log(f"tapped PLAY at {btn}")
            elif self.menu_taps % 2:
                self.tap_frac(*self.cfg["ui"]["menu_play"])
                self.log("tapped PLAY (fallback coords)")
            else:
                self.tap_frac(*self.cfg["ui"]["start_big"])
                self.log("tapped START (big button coords)")
            if self.menu_taps > 6:
                self.snap("menu_stuck", frame)
                self.menu_taps = 0
            time.sleep(1.5)

    def step_revive(self, t):
        # revive offer has a countdown — act fast
        if t - self.state_since < 0.3:
            return
        frame, _ = self.link.frame()
        btn = vision.find_button(frame, 40, 75)
        self.revive_taps = getattr(self, "revive_taps", 0)
        # if the game still SHOWS a green revive, take it — the game itself
        # decides availability. Cap taps per screen so a dead button can't
        # trap us (that loop idled the bot for minutes).
        if btn and self.revive_taps < 3:
            self.revive_taps += 1
            self.revives_used += 1
            self.log(f"revive #{self.revives_used} at {btn}")
            self.link.tap(*btn)
            time.sleep(1.0)
            if vision.classify_state(self.link.frame()[0]) == \
                    vision.STATE_PLAYING:
                self.revive_taps = 0
            return
        # decline: this screen's dismiss is the blue NEXT button
        self.log("revive unavailable -> NEXT")
        self.snap("revive_decline", frame)
        nxt = vision.find_button(frame, 100, 135, 0.0015,
                                 s_min=50, v_min=150)
        if nxt:
            self.link.tap(*nxt)
        else:
            self.tap_frac(0.34, 0.81)    # NEXT's usual spot
        self.revive_taps = 0
        time.sleep(1.0)

    def step_unknown(self, frame, t):
        # transient screens (countdowns, loading, results fades) usually
        # pass on their own; if truly stuck, snapshot and rotate through
        # the known button spots (reward/skin popups use bottom-row OK/USE,
        # dialogs use mid-right). Never press BACK — it opens Confirm exit.
        if t - self.state_since > 4.0:
            self.snap("stuck_unknown", frame)
            self.unknown_taps = getattr(self, "unknown_taps", 0) + 1
            spots = ((0.36, 0.84),   # OK on reward/skin popups
                     (0.666, 0.643),  # dialog No
                     (0.50, 0.75),
                     (0.50, 0.88))
            self.tap_frac(*spots[(self.unknown_taps - 1) % len(spots)])
            self.state_since = t

    def step_dialog(self, frame, t):
        if t - self.state_since < 0.5:
            return
        self.dialog_taps = getattr(self, "dialog_taps", 0) + 1
        # Try common right/left/center dismiss regions; if this is actually a
        # misclassified death screen, button detection now catches it before
        # dialog, so these should only be true modal/system screens.
        spots = ((0.666, 0.643), (0.50, 0.75), (0.34, 0.64), (0.50, 0.90))
        if self.dialog_taps <= 6:
            fx, fy = spots[(self.dialog_taps - 1) % len(spots)]
            self.log(f"dialog -> dismiss tap {self.dialog_taps} at {fx:.2f},{fy:.2f}")
            self.tap_frac(fx, fy)
            time.sleep(0.55)
            return
        self.log("dialog stuck after dismiss taps -> relaunching game app")
        self.snap("dialog_stuck", frame)
        self.link.launch_app(self.cfg["package"])
        self.dialog_taps = 0
        self.last_play = t
        self.state_since = t
        time.sleep(5.0)

    def step_play(self, frame, t):
        if self.play_t0 is None:
            self.play_t0 = t
            self.games += 1
            self.log(f"game #{self.games} started")

        a = vision.analyze(frame)
        # score lower-bound from the on-screen score digits (drives boost
        # policy; never boost while the real score is still small).
        # Median of the last 5 reads: single-frame OCR glitches must not
        # flip the boost gate.
        self.score_reads.append(vision.read_score_gate(frame))
        if len(self.score_reads) > 5:
            self.score_reads.pop(0)
        med = sorted(self.score_reads)[len(self.score_reads) // 2]
        # tier promotions must PERSIST ~1.5s before being believed — a snake
        # crossing the score digits can fool even the median for a moment
        if med > self.score_est:
            if self.score_cand != med:
                self.score_cand, self.score_cand_t = med, t
            elif t - self.score_cand_t > 1.5:
                self.score_est = med
        else:
            self.score_est = med
            self.score_cand = None
        self.game_max_score = max(self.game_max_score, self.score_est)

        # spawn invulnerability: brave but never blind — the game's window
        # is shorter than it looks, and ghost enemies can kill regardless
        grace = self.life_t0 is not None and (t - self.life_t0) < 2.5
        # cloned net advises (direction-preference bonus in ray scoring);
        # the heuristic pilot and all its safety logic stay in command
        advice = None
        if self.brain is not None:
            advice, _ = self.brain.steer(frame)
        angle, boost = self.pilot.decide(a, self.score_est, grace=grace,
                                         advice=advice)
        nudge = self.read_nudge()
        if nudge is not None and "angle" in nudge:
            angle = math.radians(float(nudge["angle"]))
            self.pilot.heading = angle     # bot continues from your line
            if nudge.get("boost") is not None:
                boost = bool(nudge["boost"])
            self.life_stats["nudged"] = True
        self.joy.steer(angle)
        self.boost.set(boost)
        self.publish_live(frame, a, angle, t)

        # per-life fitness shaping for the evolver
        ls = self.life_stats
        tick_gain = 0.0
        if a.own_area > ls["peak_area"]:
            gained = a.own_area - ls["peak_area"]
            if ls["peak_area"]:
                # eating grows the body a few px/tick; big jumps are vision
                # noise (envelope grabbing a neighbor) and must not pay out
                gained = min(gained, 60)
                tick_gain = gained
                ls["growth"] += gained
                if self.pilot.mode == "loot":
                    ls["loot_growth"] += gained
            ls["peak_area"] = a.own_area
        # online RL: per-tick reward for the last chosen ray
        if self.rl is not None and self.pilot.last_rl is not None:
            self.rl.record(*self.pilot.last_rl,
                           reward=tick_gain / 60.0 + 0.02)
            self.pilot.last_rl = None
        if self.pilot.mode != ls["last_mode"]:
            if self.pilot.mode in ("reflex", "hug-break", "juke") and \
                    t - ls.get("last_dodge_t", 0) > 1.5:
                ls["dodges"] += 1
                ls["last_dodge_t"] = t
            ls["last_mode"] = self.pilot.mode
        if ls["last_tick"] is not None and a.enemy_area > 15000:
            ls["crowd_s"] += min(0.3, t - ls["last_tick"])
        ls["last_tick"] = t
        # fitness tier: raw median (no persistence gate) — the boost gate
        # needs the conservative one, the fitness signal needs the fast one
        if med >= 80:
            self.game_max_score = max(self.game_max_score, med)
            if str(med) not in ls["tier_t"]:
                ls["tier_t"][str(med)] = round(t - (self.life_t0 or t), 1)
        self.mode_hist.append({
            "t": t, "mode": self.pilot.mode, "boost": bool(self.pilot.boost),
            "minE": float(self.pilot.last_min_d),
            "border": int(a.border_mask.sum())})

        if t - self.last_hb > 2.0:
            fps = (self.frames_done - self.hb_frames) / max(1e-3, t - self.last_hb)
            self.log(f"hb: {fps:.0f} dec/s mode={self.pilot.mode} "
                     f"score~{self.score_est:.0f} minEnemy={self.pilot.last_min_d:.0f}px "
                     f"food={len(a.food_pts)} boost={boost}")
            self.last_hb = t
            self.hb_frames = self.frames_done
        self.frames_done += 1

        if self.debug_dir and t - self.last_debug > 5.0:
            self.last_debug = t
            self.snap("play", frame, a, angle)
        if self.debug_dir and t - self.last_replay > 0.5:
            self.last_replay = t
            hud = (f"t{t - (self.life_t0 or t):5.1f}s {self.pilot.mode:<11s} "
                   f"minE{self.pilot.last_min_d:4.0f} sc{self.score_est:4.0f} "
                   f"area{a.own_area:5d} B{int(self.pilot.boost)}")
            layer = vision.debug_layer(a, angle, danger=self.pilot._danger)
            overlay = vision.overlay_debug(
                frame, a, angle, danger=self.pilot._danger, hud=hud)
            meta = {
                "t": round(t - (self.life_t0 or t), 2),
                "mode": self.pilot.mode,
                "score": int(self.score_est),
                "min_enemy_px": round(float(self.pilot.last_min_d), 1),
                "own_area": int(a.own_area),
                "enemy_area": int(a.enemy_area),
                "food": int(len(a.food_pts)),
                "loot": int(len(a.loot)),
                "enemy_heads": len(getattr(a, "enemy_heads", []) or []),
                "border_px": int(a.border_mask.sum()),
                "boost": bool(self.pilot.boost),
                "heading_deg": round(math.degrees(angle), 1),
            }
            self.replay.append((t, frame.copy(), layer, overlay, meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds; useful for live smoke tests")
    ap.add_argument("--learn", choices=("off", "active", "rl"), default="off",
                    help="active = evolve strategy weights from the score "
                         "itself; off = play the best weights found so far. "
                         "For supervised/imitation recording use record.py.")
    ap.add_argument("--brain", choices=("none", "bc"), default="none",
                    help="bc = behavior-cloned net (bc_policy.pt) steers "
                         "normal play, heuristics keep emergency override")
    args = ap.parse_args()
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    bot = Bot(cfg, args.debug_dir, learn=args.learn, brain=args.brain)
    try:
        bot.run(args.duration)
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
