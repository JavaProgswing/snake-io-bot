"""perception.py — explicit, temporal scene perception for Snake.io.

Ground-up rebuild of the vision layer. Design goals, in priority order:

  1. PRECISION over recall. Every detection carries a confidence in [0,1];
     callers threshold it. We would rather miss a far enemy than mislabel
     loot as an enemy (the failure that kept killing the bot).

  2. EXPLICIT categories — SELF, ENEMY (each with a head + velocity),
     BORDER, FOOD (scattered bonus orbs), LOOT (enemy death drops). No blob
     is left ambiguous between two categories.

  3. TEMPORAL. A Tracker carries state across frames. The camera is
     head-locked to OUR snake, so the whole world scrolls by our own
     velocity every frame. We estimate that global scroll (phase
     correlation on the background) and subtract it, so only WORLD-relative
     motion marks a live enemy. Per-frame heuristics could never do this —
     it is why static loot kept being read as a moving enemy.

Everything runs on a half-scale BGR frame for speed. Masks/points are at
half scale; Scene.scale multiplies back to full-res coordinates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------ config

SCALE = 0.5                    # work at half the incoming resolution
ORB_MAX_AREA = 130             # px^2 at half scale: bigger => snake, not orb
ORB_MIN_AREA = 3
SNAKE_MIN_AREA = 260           # a real snake body component
BG_DEV = 70                    # L1 BGR distance from local background = stuff
                               # (higher => snake CORE only, not its glow halo)
LOOT_CLUSTER_MIN = 4           # orbs within a cell to call it loot
LOOT_CELL = 34                 # px at half scale

# UI widgets to ignore (fractions x0,y0,x1,y1). Tight — every masked pixel
# is a blind spot. Landscape layout; mirrored automatically for portrait.
# Kept TIGHT to the OPAQUE icon cores only — the joystick pad and boost
# button are translucent, so a snake crossing them still shows through and
# should be SEEN. We blank just the solid knob/arrow; the surrounding
# translucent ring stays visible (and its sparse rim is rejected as non-snake
# by the fill/elongation test downstream). The leaderboard is masked
# dynamically (only when its ranking text is actually present).
UI_MASKS = [
    (0.455, 0.00, 0.545, 0.11),  # big score digits (top-center)
    (0.075, 0.58, 0.300, 0.94),  # joystick (base ring + knob, centre ~0.18)
    (0.780, 0.58, 0.970, 0.94),  # boost button hexagon (centre ~0.87)
]


# ------------------------------------------------------------------ types

@dataclass
class Enemy:
    head: Tuple[int, int]              # half-scale px
    vel: Tuple[float, float]           # WORLD-relative px/frame (scroll removed)
    speed: float                       # |vel|
    area: int                          # body area, half-scale px^2
    color: Tuple[int, int, int]        # mean BGR of body
    conf: float                        # detection confidence [0,1]
    head_conf: float                   # confidence the head point is right
    track_id: int


@dataclass
class Scene:
    scale: float
    w: int
    h: int
    # self
    self_mask: np.ndarray
    self_head: Optional[Tuple[int, int]]
    self_heading: Optional[float]
    self_area: int
    self_conf: float
    # enemies
    enemies: List[Enemy]
    enemy_mask: np.ndarray
    # world
    border_mask: np.ndarray
    food: np.ndarray                   # Nx2 half-scale px
    food_w: np.ndarray
    loot: List[Tuple[float, float, float]]   # (x, y, weight)
    scroll: Tuple[float, float]        # camera translation this frame (px)
    scroll_conf: float
    obstacle_mask: np.ndarray          # enemy bodies + border, dilated
    dt: float

    def to_full(self, pt):
        return (int(pt[0] / self.scale), int(pt[1] / self.scale))


@dataclass
class _Track:
    tid: int
    head: Tuple[float, float]
    vel: Tuple[float, float]           # screen-space px/frame
    area: int
    color: Tuple[int, int, int]
    hits: int = 1
    misses: int = 0


# ------------------------------------------------------------ segmentation

def _ui_mask(shape, portrait):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in UI_MASKS:
        if portrait:                    # rotate widget layout 90°
            x0, y0, x1, y1 = y0, x0, y1, x1
        m[int(h * y0):int(h * y1), int(w * x0):int(w * x1)] = 1
    return m


def _segment(small):
    """Return (stuff, border, hsv). stuff excludes border + UI."""
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # local background = heavy blur (tracks vignette + horizontal bands);
    # anything deviating strongly is "stuff"
    bg = cv2.blur(small, (61, 61)).astype(np.int16)
    dev = np.abs(small.astype(np.int16) - bg).sum(axis=2)
    stuff = (dev > BG_DEV).astype(np.uint8)

    # border: bright red line + dark red-brown dead zone. Keep only LARGE
    # red structures (the boundary spans the screen; red food dots do not).
    red_line = (((hue <= 8) | (hue >= 172)) & (sat > 120) & (val > 90))
    dead = (((hue <= 12) | (hue >= 168)) & (sat > 55) & (val < 115))
    red = (red_line | dead).astype(np.uint8)
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(red, 8)
    border = np.zeros((h, w), np.uint8)
    for i in range(1, nb):
        if sb[i, cv2.CC_STAT_AREA] > 350 or \
                max(sb[i, cv2.CC_STAT_WIDTH], sb[i, cv2.CC_STAT_HEIGHT]) > 85:
            border[lb == i] = 1
    if border.any():
        border = cv2.dilate(border, np.ones((9, 9), np.uint8))

    # letterbox / screen-edge black bars: the video can have near-black
    # margins that deviate hard from the teal bg and would read as one giant
    # enemy. Detect FULL-height/width black margins at the edges and drop
    # them (a dark snake never spans a whole edge column, so this is safe).
    dark = val < 24
    bar = np.zeros((h, w), np.uint8)
    cd = dark.mean(axis=0)
    c = 0
    while c < w // 4 and cd[c] > 0.85:
        c += 1
    if c:
        bar[:, :c] = 1
    c = w - 1
    while c > 3 * w // 4 and cd[c] > 0.85:
        c -= 1
    if c < w - 1:
        bar[:, c + 1:] = 1
    rd = dark.mean(axis=1)
    r = 0
    while r < h // 4 and rd[r] > 0.85:
        r += 1
    if r:
        bar[:r, :] = 1
    r = h - 1
    while r > 3 * h // 4 and rd[r] > 0.85:
        r -= 1
    if r < h - 1:
        bar[r + 1:, :] = 1
    stuff[bar > 0] = 0
    border[bar > 0] = 0

    # nameplates: floating white NAME text (ours + enemies') sits over the
    # arena and must never become self/enemy/food. Remove small/thin white
    # text blobs; KEEP large white blobs (white snakes are real).
    white = ((sat < 55) & (val > 205)).astype(np.uint8)
    nw, lw, sw, _ = cv2.connectedComponentsWithStats(white, 8)
    for i in range(1, nw):
        a = sw[i, cv2.CC_STAT_AREA]
        if a < 240 or (sw[i, cv2.CC_STAT_HEIGHT] < 15 and a < 900):
            stuff[lw == i] = 0

    portrait = h > w
    # leaderboard: an OPAQUE panel (top-right; top-left in portrait). Mask it
    # ONLY when its white ranking text is present (expanded) — when the player
    # collapses it via the toggle, leave that area visible so snakes there are
    # seen. This is the "see behind it when needed" behaviour.
    if portrait:
        lx0, ly0, lx1, ly1 = (int(w * 0.02), int(h * 0.74),
                              int(w * 0.46), int(h * 0.99))
    else:
        lx0, ly0, lx1, ly1 = (int(w * 0.74), int(h * 0.02),
                              int(w * 0.99), int(h * 0.46))
    lb_v = val[ly0:ly1, lx0:lx1]
    lb_s = sat[ly0:ly1, lx0:lx1]
    if lb_v.size and float(((lb_v > 190) & (lb_s < 70)).mean()) > 0.03:
        stuff[ly0:ly1, lx0:lx1] = 0

    ui = _ui_mask((h, w), portrait)
    stuff[ui > 0] = 0
    stuff = cv2.morphologyEx(stuff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    stuff = cv2.morphologyEx(stuff, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    stuff[border > 0] = 0
    return stuff, border, hsv


def _estimate_scroll(prev_gray, gray, stuff):
    """Global camera translation via phase correlation on the BACKGROUND
    (stuff removed so moving snakes don't bias it). Returns ((dx,dy), conf).
    Convention: a world-static point moves by (dx,dy) on screen each frame."""
    if prev_gray is None:
        return (0.0, 0.0), 0.0
    bg = gray.astype(np.float32)
    m = (stuff == 0).astype(np.float32)
    if m.mean() < 0.25:                 # too little background visible
        return (0.0, 0.0), 0.0
    a = prev_gray.astype(np.float32) * m
    b = bg * m
    try:
        win = cv2.createHanningWindow((gray.shape[1], gray.shape[0]),
                                      cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    except cv2.error:
        return (0.0, 0.0), 0.0
    if abs(dx) > gray.shape[1] * 0.25 or abs(dy) > gray.shape[0] * 0.25:
        return (0.0, 0.0), 0.0          # implausible jump
    return (float(dx), float(dy)), float(min(1.0, max(0.0, resp)))


# ------------------------------------------------------------ head finding

SELF_COLOR_TOL = 96            # summed |BGR| distance from body color
SELF_MERGE_AREA = 8000         # above this, color-trim the (likely merged) blob


def _find_self(stuff, small, cx, cy, diag):
    """SELF = the connected body under screen center (camera is head-locked
    to us). Normally this is the WHOLE connected snake — taking the raw
    component captures its full length with no color gaps. Only when that
    component is implausibly large (our body is pixel-touching an enemy) do
    we fall back to a head-color trim to cut the enemy off. Best of both:
    long/coiled snakes stay whole; genuine merges get separated."""
    h, w = stuff.shape
    # 7x7 close bridges most food-overlap gaps in our own body without
    # reaching a separate enemy; genuine over-merges are caught by the area
    # cap + color-trim below. (Wide-gap coil fragmentation is a known limit
    # left for the learned segmenter in the model phase.)
    src = cv2.morphologyEx(stuff, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if src[cy, cx]:
        sx, sy = cx, cy
    else:
        ys, xs = np.nonzero(src)
        if len(xs) == 0:
            return np.zeros((h, w), np.uint8), None
        dd = (xs - cx) ** 2 + (ys - cy) ** 2
        j = int(np.argmin(dd))
        if dd[j] > (0.07 * diag) ** 2:   # nothing of ours near center
            return np.zeros((h, w), np.uint8), None
        sx, sy = int(xs[j]), int(ys[j])
    nn, lab = cv2.connectedComponents(src, 8)
    comp = (lab == int(lab[sy, sx])).astype(np.uint8)
    if int(comp.sum()) <= SELF_MERGE_AREA:
        return comp, (sx, sy)
    # merged with a touching enemy -> trim to our body color
    x0, x1 = max(0, sx - 11), min(w, sx + 12)
    y0, y1 = max(0, sy - 11), min(h, sy + 12)
    pm = stuff[y0:y1, x0:x1] > 0
    patch = small[y0:y1, x0:x1]
    seed = (np.median(patch[pm].reshape(-1, 3), axis=0) if int(pm.sum()) >= 20
            else small[sy, sx].astype(np.float32))
    dist = np.abs(small.astype(np.int16) - seed).sum(2)
    trim = ((comp > 0) & (dist < SELF_COLOR_TOL)).astype(np.uint8)
    trim = cv2.morphologyEx(trim, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n2, l2 = cv2.connectedComponents(trim, 8)
    lb2 = int(l2[sy, sx])
    if lb2 == 0:
        ys, xs = np.nonzero(trim)
        if len(xs) == 0:
            return comp, (sx, sy)
        dd = (xs - sx) ** 2 + (ys - sy) ** 2
        lb2 = int(l2[ys[int(np.argmin(dd))], xs[int(np.argmin(dd))]])
    return (l2 == lb2).astype(np.uint8), (sx, sy)


def _blob_ends(mask_u8):
    """Two extreme points of a blob along its principal axis, plus the
    'thickness' (distance-transform value) at each. Head = thicker end
    (the tail tapers to a point)."""
    ys, xs = np.nonzero(mask_u8)
    if len(xs) < 12:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    axis = evecs[:, int(np.argmax(evals))]
    proj = (pts - mean) @ axis
    p_lo = pts[int(np.argmin(proj))]
    p_hi = pts[int(np.argmax(proj))]
    dt = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)

    def thick(p):
        x, y = int(p[0]), int(p[1])
        x0, x1 = max(0, x - 6), min(mask_u8.shape[1], x + 7)
        y0, y1 = max(0, y - 6), min(mask_u8.shape[0], y + 7)
        sub = dt[y0:y1, x0:x1]
        return float(sub.max()) if sub.size else 0.0

    return (tuple(p_lo), thick(p_lo)), (tuple(p_hi), thick(p_hi))


# ------------------------------------------------------------ main tracker

class Perception:
    def __init__(self):
        self._prev_gray = None
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._matched: List = []

    def reset(self):
        self.__init__()

    # ---- public API -------------------------------------------------

    def process(self, frame_bgr, dt=1.0) -> Scene:
        small = cv2.resize(frame_bgr, None, fx=SCALE, fy=SCALE,
                           interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        cx, cy = w // 2, h // 2
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        stuff, border, hsv = _segment(small)
        scroll, scroll_conf = _estimate_scroll(self._prev_gray, gray, stuff)
        sat = hsv[:, :, 1]
        diag = float(np.hypot(w, h))

        # --- SELF first: color-grown from center, then removed from the
        # world so it can never be re-detected as an enemy or its pattern
        # sparkle as food.
        self_mask, self_head = _find_self(stuff, small, cx, cy, diag)
        self_area = int(self_mask.sum())
        if self_mask.any():
            not_self = cv2.bitwise_not(
                cv2.dilate(self_mask, np.ones((7, 7), np.uint8)) * 255)
            world = cv2.bitwise_and(stuff, stuff, mask=(not_self > 0)
                                    .astype(np.uint8))
        else:
            world = stuff

        # (1) orbs: small bright blobs (food + loot), excluding self
        n, lab, stats, cents = cv2.connectedComponentsWithStats(world, 8)
        orb_pts, orb_w = [], []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            px, py = int(cents[i][0]), int(cents[i][1])
            if (area <= ORB_MAX_AREA and max(bw, bh) < 22
                    and area >= ORB_MIN_AREA and sat[py, px] >= 80):
                orb_pts.append((px, py))
                orb_w.append(float(area))

        # (2) enemy bodies: CLOSE first so a snake split by an overlapping orb
        # rejoins into ONE component (no duplicate heads); keep only SOLID or
        # ELONGATED blobs — a sparse round blob is a loot pile, not a snake.
        closed = cv2.morphologyEx(world, cv2.MORPH_CLOSE,
                                  np.ones((9, 9), np.uint8))
        ns, lab2, st2, _ = cv2.connectedComponentsWithStats(closed, 8)
        raw = []                        # (head, area, color, body_mask)
        for i in range(1, ns):
            area = int(st2[i, cv2.CC_STAT_AREA])
            if area < SNAKE_MIN_AREA:
                continue
            bw = st2[i, cv2.CC_STAT_WIDTH]
            bh = st2[i, cv2.CC_STAT_HEIGHT]
            comp = (lab2 == i).astype(np.uint8)
            fill = float((world[comp > 0] > 0).mean())
            elong = max(bw, bh) / max(1, min(bw, bh))
            if fill < 0.5 and elong < 2.6:
                continue                # sparse & round => loot cluster
            ends = _blob_ends(comp)
            if ends is None:
                continue
            (p_lo, t_lo), (p_hi, t_hi) = ends
            col = cv2.mean(small, comp)[:3]
            # keep BOTH ends; _associate picks the head end temporally (the
            # one nearest the track's prediction) so it can't flip end-to-end
            raw.append((np.array(p_lo, np.float32), np.array(p_hi, np.float32),
                        t_lo, t_hi, area, tuple(map(int, col)), comp))

        # --- temporal association (enemies only; self is handled above)
        self._associate(raw, scroll, dt)

        enemy_mask = np.zeros((h, w), np.uint8)
        enemies: List[Enemy] = []
        for tr, (head, area, col, comp) in self._matched:
            enemy_mask = cv2.bitwise_or(enemy_mask, comp)
            wvx, wvy = tr.vel[0] - scroll[0], tr.vel[1] - scroll[1]
            enemies.append(Enemy(
                head=(int(head[0]), int(head[1])),
                vel=(wvx, wvy), speed=float(np.hypot(wvx, wvy)),
                area=area, color=col,
                conf=min(1.0, 0.35 + 0.12 * tr.hits),
                head_conf=min(1.0, 0.4 + 0.15 * tr.hits),
                track_id=tr.tid))

        # self heading = negative of camera scroll (camera follows our head)
        if self_head is not None and scroll_conf > 0.4 and _mag(scroll) > 0.6:
            self_heading = _ang((-scroll[0], -scroll[1]))
        else:
            self_heading = None
        # self confidence: present + plausibly sized (flag merge blobs)
        if self_head is None or self_area < 150:
            self_conf = 0.0
        elif self_area > 14000:
            self_conf = round(max(0.3, 14000.0 / self_area), 3)
        else:
            self_conf = 1.0

        # --- food vs loot on the orb set
        orb_pts_a = np.array(orb_pts, np.float32) if orb_pts \
            else np.zeros((0, 2), np.float32)
        orb_w_a = np.array(orb_w, np.float32) if orb_w \
            else np.zeros((0,), np.float32)
        food, food_w, loot = self._split_food_loot(orb_pts_a, orb_w_a)

        obstacle = cv2.bitwise_or(enemy_mask, border)
        if obstacle.any():
            obstacle = cv2.dilate(obstacle, np.ones((7, 7), np.uint8))

        self._prev_gray = gray
        return Scene(
            scale=SCALE, w=w, h=h,
            self_mask=self_mask, self_head=self_head,
            self_heading=self_heading, self_area=self_area,
            self_conf=round(self_conf, 3),
            enemies=enemies, enemy_mask=enemy_mask,
            border_mask=border, food=food, food_w=food_w, loot=loot,
            scroll=scroll, scroll_conf=round(scroll_conf, 3),
            obstacle_mask=obstacle, dt=dt)

    # ---- internals --------------------------------------------------

    def _associate(self, raw, scroll, dt):
        """Greedy match raw detections to existing tracks, then pick each
        matched track's HEAD as the blob end nearest its predicted position
        (prev head + prev screen velocity). Choosing the head temporally —
        not by per-frame thickness — stops it flipping between the snake's
        two ends, which was spiking enemy velocity. Each raw item is
        (endA, endB, thickA, thickB, area, color, comp). Stores
        self._matched = [(track, (head, area, color, comp)), ...]."""
        def pred_of(tr):
            return (tr.head[0] + tr.vel[0], tr.head[1] + tr.vel[1])

        preds = [(tr, pred_of(tr)) for tr in self._tracks]
        used_tr, used_raw = set(), set()
        pairs = []
        for ri, item in enumerate(raw):
            ea, eb = item[0], item[1]
            for tr, pr in preds:
                da = (ea[0] - pr[0]) ** 2 + (ea[1] - pr[1]) ** 2
                db = (eb[0] - pr[0]) ** 2 + (eb[1] - pr[1]) ** 2
                pairs.append((min(da, db) ** 0.5, ri, tr))
        pairs.sort(key=lambda p: p[0])
        matched = []
        gate = 60.0
        for d, ri, tr in pairs:
            if ri in used_raw or tr.tid in used_tr or d > gate:
                continue
            ea, eb, ta, tb, area, col, comp = raw[ri]
            pr = pred_of(tr)
            head = ea if (np.hypot(ea[0] - pr[0], ea[1] - pr[1]) <=
                          np.hypot(eb[0] - pr[0], eb[1] - pr[1])) else eb
            tr.vel = (0.6 * tr.vel[0] + 0.4 * (head[0] - tr.head[0]),
                      0.6 * tr.vel[1] + 0.4 * (head[1] - tr.head[1]))
            tr.head = (float(head[0]), float(head[1]))
            tr.area, tr.color = area, col
            tr.hits += 1
            tr.misses = 0
            used_tr.add(tr.tid)
            used_raw.add(ri)
            matched.append((tr, (head, area, col, comp)))
        # new tracks for unmatched detections (head = thicker end)
        for ri, item in enumerate(raw):
            if ri in used_raw:
                continue
            ea, eb, ta, tb, area, col, comp = item
            head = ea if ta >= tb else eb
            tr = _Track(self._next_id, (float(head[0]), float(head[1])),
                        (0.0, 0.0), area, col)
            self._next_id += 1
            self._tracks.append(tr)
            matched.append((tr, (head, area, col, comp)))
        # age unmatched tracks
        for tr in self._tracks:
            if tr.tid not in used_tr:
                tr.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= 6]
        self._matched = matched

    def _split_food_loot(self, pts, wts):
        """Scattered orbs = FOOD; dense clusters (a kill's remains) = LOOT.
        Loot filtered so points sitting on a body/border are dropped by the
        caller's obstacle test — here we only cluster by density."""
        if len(pts) == 0:
            return pts, wts, []
        cells = {}
        for (x, y), wt in zip(pts, wts):
            cells.setdefault((int(x) // LOOT_CELL, int(y) // LOOT_CELL),
                             []).append((x, y, wt))
        loot, loot_cells = [], set()
        for key, members in cells.items():
            neigh = [m for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for m in cells.get((key[0] + dx, key[1] + dy), [])]
            if len(neigh) >= LOOT_CLUSTER_MIN:
                tot = sum(m[2] for m in neigh)
                lx = sum(m[0] * m[2] for m in neigh) / tot
                ly = sum(m[1] * m[2] for m in neigh) / tot
                loot.append((float(lx), float(ly), float(tot)))
                loot_cells.add(key)
        # dedup loot points that share neighborhoods
        loot = _dedup_loot(loot)
        keep = [i for i, (x, y) in enumerate(pts)
                if (int(x) // LOOT_CELL, int(y) // LOOT_CELL) not in loot_cells]
        keep = np.array(keep, int)
        if len(keep) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32), loot
        return pts[keep], wts[keep], loot


# ------------------------------------------------------------ small utils

def _ang(v):
    return float(np.arctan2(v[1], v[0]))


def _mag(v):
    return float(np.hypot(v[0], v[1]))


def _dedup_loot(loot, min_sep=30.0):
    loot = sorted(loot, key=lambda c: -c[2])
    out = []
    for c in loot:
        if all(np.hypot(c[0] - o[0], c[1] - o[1]) > min_sep for o in out):
            out.append(c)
    return out[:4]


# ------------------------------------------------------------ overlay

def render(scene: Scene, frame_bgr, hud=""):
    """Debug overlay at half scale. Colors:
       green=self  green dot=self head  red=enemy body  red arrow=enemy vel
       magenta dot=enemy head  dark-red=border  cyan=food  yellow ring=loot"""
    vis = cv2.resize(frame_bgr, (scene.w, scene.h))
    vis[scene.border_mask > 0] = (40, 40, 160)
    vis[scene.enemy_mask > 0] = (0, 0, 220)
    vis[scene.self_mask > 0] = (0, 220, 0)
    for (x, y), wt in zip(scene.food, scene.food_w):
        cv2.circle(vis, (int(x), int(y)), 2, (230, 230, 0), -1)
    for lx, ly, lw in scene.loot:
        cv2.circle(vis, (int(lx), int(ly)), 9, (0, 240, 240), 2)
    for e in scene.enemies:
        cv2.circle(vis, e.head, 4, (255, 0, 255), -1)
        ex = int(e.head[0] + e.vel[0] * 4)
        ey = int(e.head[1] + e.vel[1] * 4)
        cv2.arrowedLine(vis, e.head, (ex, ey), (255, 0, 255), 1,
                        tipLength=0.3)
    if scene.self_head:
        cv2.circle(vis, scene.self_head, 5, (0, 255, 120), -1)
        cv2.circle(vis, scene.self_head, 7, (0, 255, 120), 1)
    line = (hud or (f"self:{scene.self_conf:.2f} a{scene.self_area} "
                    f"enemy:{len(scene.enemies)} food:{len(scene.food)} "
                    f"loot:{len(scene.loot)} "
                    f"scroll({scene.scroll[0]:+.1f},{scene.scroll[1]:+.1f})"
                    f":{scene.scroll_conf:.2f}"))
    cv2.rectangle(vis, (0, 0), (scene.w, 16), (25, 25, 25), -1)
    cv2.putText(vis, line, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ------------------------------------------------------- game-state class

STATE_MENU, STATE_PLAYING, STATE_REVIVE = "menu", "playing", "revive"
STATE_DIALOG, STATE_UNKNOWN, STATE_SPECTATE = "dialog", "unknown", "spectate"


def _player_ui_present(frame_bgr):
    """True when OUR player HUD is on screen — the reliable tell that we are
    alive and playing. After death the game pans a spectator camera over other
    snakes: same teal arena, but NO HUD. Two independent cues, either suffices:
      * score digits — big bright-white, top-centre
      * leaderboard white text rows — top-right
    Measured — play frames: score_wht 0.09-0.16, lb_txt 0.06-0.07; spectator
    frame: both 0.000. (boost button is NOT usable: an orange snake in the
    bottom-right trips it.)"""
    h, w = frame_bgr.shape[:2]
    sc = frame_bgr[0:int(h * 0.12), int(w * 0.44):int(w * 0.56)]
    lb = frame_bgr[0:int(h * 0.42), int(w * 0.78):w]
    score_wht = (cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY) > 210).mean()
    lb_txt = (cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY) > 200).mean()
    return score_wht > 0.03 or lb_txt > 0.03


def classify_state(frame_bgr):
    """Coarse game-state from the full-res frame. Teal arena alone is NOT
    enough — it shows during the post-death spectator pan too — so PLAYING
    additionally requires our HUD (`_player_ui_present`); teal arena without a
    HUD is SPECTATE (we are dead, watching)."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    def teal(x0, x1):
        c = hsv[int(h * 0.25):int(h * 0.75), int(w * x0):int(w * x1)]
        return ((c[:, :, 0] >= 76) & (c[:, :, 0] <= 102) &
                (c[:, :, 1] >= 40) & (c[:, :, 1] <= 115) &
                (c[:, :, 2] > 140)).mean()

    if teal(0.32, 0.48) > 0.35 and teal(0.52, 0.68) > 0.35:
        return STATE_PLAYING if _player_ui_present(frame_bgr) else STATE_SPECTATE
    left = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.32):int(w * 0.48), 2]
    right = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.52):int(w * 0.68), 2]
    if (left < 95).mean() > 0.45 and (right < 95).mean() > 0.45:
        return STATE_DIALOG
    return STATE_UNKNOWN
