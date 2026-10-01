"""Frame analysis for Snake.io (Netflix build).

Everything works on a half-scale BGR frame. Core idea: the arena background is
a desaturated hex grid; food, snakes and the border are saturated/bright, so a
saturation+value gate separates "stuff" from background, then connected
components split food (small blobs) from snakes (big elongated blobs). Own
snake = components near screen center (camera is head-locked).
"""

import cv2
import numpy as np

STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_REVIVE = "revive"
STATE_DIALOG = "dialog"
STATE_UNKNOWN = "unknown"

FOOD_MAX_AREA = 140       # at half scale
_PREV_OWN = [None]        # last frame's own-body mask (temporal tracking)
_PREV_DOTS = [None]       # last frame's small-blob string mask (motion)
_PREV_RAWDOTS = [None]    # last frame's raw small-blob mask (dot motion)
_PREV_ENEMY = [None]      # last frame's enemy mask (head/front detection)
_PREV_HEADS = [None]      # previous enemy head candidates for rough velocity
_PREV_GRAY = [None]       # quarter-scale gray for camera-motion estimation
_HANN = [None]            # cached Hanning window for phaseCorrelate
_OWN_COLOR = [None]       # EMA of our body's median BGR (identity check)

LAYER_OBSTACLE = 1
LAYER_DANGER = 2
LAYER_OWN = 3
LAYER_FOOD = 4
LAYER_LOOT = 5
LAYER_TARGET = 6


def reset_own_tracking():
    """Call at every (re)spawn: the previous life's masks must not leak
    into the new life's temporal state."""
    _PREV_OWN[0] = None
    _PREV_DOTS[0] = None
    _PREV_RAWDOTS[0] = None
    _PREV_ENEMY[0] = None
    _PREV_HEADS[0] = None
    _PREV_GRAY[0] = None
    _OWN_COLOR[0] = None


def _shift_mask(m, dx, dy):
    """Translate a mask by integer (dx, dy), zero-filling exposed edges."""
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return m
    out = np.zeros_like(m)
    h, w = m.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0:ys1, xs0:xs1] = m[max(0, -dy):h - max(0, dy),
                              max(0, -dx):w - max(0, dx)]
    return out


def _camera_shift(small, stuff, bg_est):
    """Per-frame camera translation at half scale via phase correlation on
    a quarter-scale gray. The camera is head-locked, so this is OUR motion;
    subtracting it turns frame differencing into world-relative motion.

    Foreground (snakes/food) is REPLACED by the background estimate before
    correlating: a big snake sweeping across the screen must not out-vote
    the hex grid, or the measured "camera" shift is actually his motion.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bg_gray = bg_est.mean(axis=2).astype(gray.dtype)
    fg = cv2.dilate(stuff, np.ones((5, 5), np.uint8)) > 0
    gray = gray.copy()
    gray[fg] = bg_gray[fg]
    g = gray[::2, ::2].astype(np.float32)
    prev = _PREV_GRAY[0]
    _PREV_GRAY[0] = g
    if prev is None or prev.shape != g.shape:
        return 0.0, 0.0, False
    if _HANN[0] is None or _HANN[0].shape != g.shape:
        _HANN[0] = cv2.createHanningWindow((g.shape[1], g.shape[0]),
                                           cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(prev, g, _HANN[0])
    if resp < 0.2 or abs(sx) > 40 or abs(sy) > 40:
        # scene cut / overlay flash / texture-poor view: report failure so
        # downstream memory decays hard instead of trusting stale positions
        return 0.0, 0.0, False
    return sx * 2.0, sy * 2.0, True  # quarter -> half scale
FOOD_SAT_MIN = 90         # floating name labels are white -> low sat
BG_DIST = 48              # L1 BGR distance from local background = "stuff"

# UI regions to ignore, as (x0, y0, x1, y1) fractions of the frame.
# Keep these TIGHT to the actual widgets: every masked pixel is a blind
# spot where an enemy can hide (a snake under the old oversized joystick
# mask killed us invisibly).
UI_MASKS = [
    (0.71, 0.02, 0.77, 0.12),   # leaderboard toggle arrow
    (0.46, 0.00, 0.56, 0.11),   # big score digits (tight: blind spot!)
    (0.12, 0.62, 0.25, 0.89),   # joystick pad
    (0.80, 0.61, 0.94, 0.89),   # boost button
]
# The leaderboard panel COLLAPSES via its toggle: mask its region only
# when the panel is actually there (lots of white ranking text), else a
# snake crossing that corner is invisible.
LEADERBOARD_BOX = (0.75, 0.00, 1.00, 0.45)


def _apply_ui_masks(mask_img, h, w, sat, val):
    for fx0, fy0, fx1, fy1 in UI_MASKS:
        mask_img[int(h * fy0):int(h * fy1), int(w * fx0):int(w * fx1)] = 0
    fx0, fy0, fx1, fy1 = LEADERBOARD_BOX
    y0, y1 = int(h * fy0), int(h * fy1)
    x0, x1 = int(w * fx0), int(w * fx1)
    text = ((val[y0:y1, x0:x1] > 190) & (sat[y0:y1, x0:x1] < 70)).mean()
    if text > 0.04:
        mask_img[y0:y1, x0:x1] = 0


class Analysis:
    __slots__ = ("w", "h", "center", "obstacle", "food_pts", "food_w",
                 "own_mask", "own_area", "enemy_mask", "enemy_area",
                 "enemy_inside", "loot", "border_mask", "front_mask",
                 "moving_mask", "chain_mask", "ghost_mask", "enemy_heads",
                 "cam_shift", "cam_ok", "head_proj")

    def __init__(self):
        self.food_pts = np.zeros((0, 2), np.float32)
        self.food_w = np.zeros((0,), np.float32)
        self.enemy_inside = False
        self.loot = []
        self.enemy_heads = []


def analyze(frame_bgr):
    """frame_bgr: full-res frame. Returns Analysis at half scale."""
    small = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    a = Analysis()
    a.w, a.h = w, h
    a.center = (w // 2, h // 2)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # local background model: the arena is a smooth teal gradient with a
    # faint hex grid; a heavy blur approximates it per-pixel (handles the
    # vignette), and snakes/food/walls deviate strongly from it
    bg_est = cv2.blur(small, (61, 61)).astype(np.int16)
    dist = np.abs(small.astype(np.int16) - bg_est).sum(axis=2)
    # Keep the primary object mask conservative. Translucent/low-contrast
    # enemies are handled later by the row-normalized ghost detector; letting
    # a broad chroma gate into the base mask makes the hex arena look lethal.
    stuff = (dist > BG_DIST).astype(np.uint8)

    _apply_ui_masks(stuff, h, w, sat, val)

    stuff = cv2.morphologyEx(stuff, cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
    # fill snake interiors that blend with the blurred bg estimate
    stuff = cv2.morphologyEx(stuff, cv2.MORPH_CLOSE,
                             np.ones((7, 7), np.uint8))

    # camera pan this frame (half-scale px), measured on BACKGROUND pixels
    # only. All prev-frame masks below are aligned by this before
    # differencing, so "new pixels" mean an enemy actually moved.
    shift_x, shift_y, cam_ok = _camera_shift(small, stuff, bg_est)
    a.cam_shift = (shift_x, shift_y)
    a.cam_ok = cam_ok

    n, labels, stats, cents = cv2.connectedComponentsWithStats(stuff, 8)

    own = np.zeros((h, w), np.uint8)
    enemy = np.zeros((h, w), np.uint8)
    food_pts, food_w = [], []
    small_cents = []
    cx, cy = a.center
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[i, :4]
        if area <= FOOD_MAX_AREA and max(bw, bh) < 20:
            px, py = int(cents[i][0]), int(cents[i][1])
            small_cents.append(cents[i])
            if sat[py, px] >= FOOD_SAT_MIN:   # skip white name labels
                food_pts.append(cents[i])
                food_w.append(area)
            continue
        # kill-reward orb piles: touching orbs merge into one blob bigger
        # than FOOD_MAX_AREA and used to be classified as a snake — so the
        # bot killed enemies and then AVOIDED the payout. A pile is compact
        # (fills its bbox, small extent); snake bodies are long or hollow.
        px, py = int(cents[i][0]), int(cents[i][1])
        if (area < 1200 and max(bw, bh) < 55 and area > 0.45 * bw * bh
                and abs(px - cx) + abs(py - cy) > 40
                and sat[max(0, py - 1):py + 2,
                        max(0, px - 1):px + 2].max() >= 70):
            food_pts.append(cents[i])
            food_w.append(area)
            continue
        enemy[labels == i] = 1

    # own snake by TEMPORAL TRACKING: our head is always at screen center,
    # and our body can only be where it was one frame ago (plus a little
    # drift). Flood the current pixels only inside the previous mask's
    # dilated envelope — an enemy pressed against us stays enemy BY
    # CONSTRUCTION, with no area heuristics or merge gates to mistune.
    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (cx, cy), 42, 1, -1)
    prev = _PREV_OWN[0]
    if prev is not None and prev.shape != (h, w):
        prev = None     # resolution changed (rotation/overlay reconfig)
    if prev is not None and prev.any():
        prev = _shift_mask(prev, shift_x, shift_y)
        # envelope margin covers residual drift after camera compensation
        allowed = cv2.bitwise_or(
            cv2.dilate(prev, np.ones((15, 15), np.uint8)), disk)
    else:
        allowed = disk
    cand = cv2.bitwise_and(stuff, allowed)
    ncc, lcc, scc, _ = cv2.connectedComponentsWithStats(cand, 8)
    winc = lcc[max(0, cy - 16):cy + 16, max(0, cx - 16):cx + 16]
    head_ids = set(int(i) for i in np.unique(winc[winc > 0]))
    # our body is often SPLIT into several components (name label cuts it,
    # coils cross) — adopt fragments that lie mostly inside the previous
    # body's envelope AND match our body color. Without the identity check
    # a spawn-adjacent enemy got painted "own" once and stayed invisible to
    # avoidance forever (it walked us straight into them).
    prev_env = (cv2.dilate(prev, np.ones((9, 9), np.uint8))
                if prev is not None else None)
    own_col = _OWN_COLOR[0]
    prev_sum = int(prev.sum()) if prev is not None else 0
    for i in range(1, ncc):
        comp = lcc == i
        area = int(scc[i, cv2.CC_STAT_AREA])
        if i in head_ids:
            # an enemy crossing our head merges into the head component:
            # clip sudden oversized head comps back to the trusted envelope
            if prev_sum > 300 and area > prev_sum * 1.5 + 300 and \
                    prev_env is not None:
                own[comp & (prev_env > 0)] = 1
            else:
                own[comp] = 1
            continue
        if prev_env is None or area < 12 or area > max(400, prev_sum * 0.5):
            continue
        if float(prev_env[comp].mean()) < 0.75:
            continue
        if own_col is not None:
            col = small[comp].reshape(-1, 3).astype(np.float32)
            if np.abs(np.median(col, axis=0) - own_col).sum() > 110:
                continue        # right place, wrong colors: not us
        own[comp] = 1
    recovered = False
    if prev is not None and prev.shape == (h, w):
        lost = own.sum() < 500 or (prev.sum() > 900 and
                                   own.sum() < prev.sum() * 0.35)
    else:
        lost = own.sum() < 500
    if lost:
        # Temporal tracking can occasionally latch onto a tiny sparkle after
        # an enemy crosses our head. Re-anchor from the screen-center disk so
        # one bad frame does not make the strategy steer from a ghost body.
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if FOOD_MAX_AREA < area <= 18000:
                hx, hy = cents[i]
                dist_c = float(np.hypot(hx - cx, hy - cy))
                if dist_c < min(w, h) * 0.46:
                    comp = labels == i
                    touch = int(comp[max(0, cy - 34):cy + 34,
                                     max(0, cx - 34):cx + 34].sum())
                    score = -dist_c * 9.0 + min(area, 2500) * 0.12 + touch
                    if best is None or score > best[0]:
                        best = (score, "main", i)
        reacq_src = cv2.bitwise_and(stuff, disk)
        nr, lr, sr, _ = cv2.connectedComponentsWithStats(reacq_src, 8)
        for i in range(1, nr):
            area = sr[i, cv2.CC_STAT_AREA]
            if 90 <= area <= 5000:
                comp = lr == i
                touch = int(comp[max(0, cy - 24):cy + 24,
                                 max(0, cx - 24):cx + 24].sum())
                if touch > 0:
                    score = touch * 5 + area
                    if best is None or (best[1] != "main" and
                                        score > best[0]):
                        best = (score, "disk", i)
        if best is not None:
            if best[1] == "main":
                own = (labels == best[2]).astype(np.uint8)
            else:
                own = (lr == best[2]).astype(np.uint8)
            recovered = True
    fresh = prev is None or not prev.any()
    if fresh and own.sum() > 1200:
        # first frame of a life: we are TINY. A fat "own" means an enemy
        # is crossing our spawn point — adopt only the sliver under a SMALL
        # head disk, the rest of that body stays enemy (the old r42 disk
        # adopted whole neighboring snakes as "ally")
        tight_disk = np.zeros((h, w), np.uint8)
        cv2.circle(tight_disk, (cx, cy), 18, 1, -1)
        own = cv2.bitwise_and(own, tight_disk)
    if own.sum() <= FOOD_MAX_AREA and prev is not None:
        own = prev              # transient loss (overlay flash): hold last
    elif (not recovered and prev is not None and
          own.sum() > prev.sum() * 1.08 + 220):
        # growth cap: eating is gradual; a jump means the envelope crawled
        # onto something wide touching us (border band, parked snake) and
        # is flooding it a stripe per frame — clamp to a tight envelope
        tight = cv2.bitwise_and(
            own, cv2.dilate(prev, np.ones((5, 5), np.uint8)))
        own = tight if tight.sum() > FOOD_MAX_AREA else prev
    _PREV_OWN[0] = own
    # body-color identity: EMA of our median color while the mask is healthy
    if own.sum() > 250:
        col = np.median(small[own > 0].reshape(-1, 3).astype(np.float32),
                        axis=0)
        if _OWN_COLOR[0] is None:
            _OWN_COLOR[0] = col
        else:
            _OWN_COLOR[0] = 0.95 * _OWN_COLOR[0] + 0.05 * col
    enemy[own > 0] = 0
    # pixels of our merged component OUTSIDE the envelope stay enemy —
    # that's either a touching opponent or our own far tail; both kill on
    # head contact, so treating them as obstacles is correct either way.

    a.own_mask = own
    a.own_area = int(own.sum())
    a.enemy_mask = enemy
    a.enemy_area = int(enemy.sum())
    a.food_pts = np.array(food_pts, np.float32) if food_pts else np.zeros((0, 2), np.float32)
    a.food_w = np.array(food_w, np.float32) if food_w else np.zeros((0,), np.float32)

    # obstacle = enemies dilated (bodies are lethal on contact; generous
    # margin — hugging a snake at pixel distance is how the bot dies)
    a.obstacle = cv2.dilate(enemy, np.ones((13, 13), np.uint8))

    # moving small blobs: kept as a diagnostic signal only. The camera is
    # head-locked, so normal world pellets move on screen whenever we move;
    # treating this as lethal made the bot starve and panic around food.
    mvmask = np.zeros((h, w), np.uint8)
    raw = np.zeros((h, w), np.uint8)
    for px, py in small_cents:
        cv2.circle(raw, (int(px), int(py)), 3, 1, -1)
    prev_raw = _PREV_RAWDOTS[0]
    if prev_raw is not None and prev_raw.shape == raw.shape and raw.any():
        prev_raw = _shift_mask(prev_raw, shift_x, shift_y)
        moving = cv2.bitwise_and(
            raw, cv2.bitwise_not(cv2.dilate(prev_raw,
                                            np.ones((5, 5), np.uint8))))
        if moving.any():
            grp = cv2.dilate(moving, np.ones((11, 11), np.uint8))
            nm, lm, sm2, _ = cv2.connectedComponentsWithStats(grp, 8)
            for i in range(1, nm):
                if sm2[i, cv2.CC_STAT_AREA] >= 150:   # 2+ clustered dots
                    mvmask[lm == i] = 1
    _PREV_RAWDOTS[0] = raw
    a.moving_mask = mvmask

    # bead-chain snakes: thin snakes render as a string of SMALL blobs that
    # individually pass for food and never reach the enemy branch above —
    # they killed us invisibly more than once. Merge small-blob centroids
    # and flag elongated strings as obstacles (loot piles stay round).
    chain_out = None
    if len(small_cents) >= 5:
        dots = np.zeros((h, w), np.uint8)
        for px, py in small_cents:
            cv2.circle(dots, (int(px), int(py)), 3, 1, -1)
        dots = cv2.morphologyEx(dots, cv2.MORPH_CLOSE,
                                np.ones((13, 13), np.uint8))
        nd, ld, sd, _ = cv2.connectedComponentsWithStats(dots, 8)
        prev_dots = _PREV_DOTS[0]
        chain = np.zeros_like(dots)
        for i in range(1, nd):
            ar = sd[i, cv2.CC_STAT_AREA]
            bw2 = sd[i, cv2.CC_STAT_WIDTH]
            bh2 = sd[i, cv2.CC_STAT_HEIGHT]
            md, sm = max(bw2, bh2), min(bw2, bh2)
            # a chain is long and thin (axis-aligned) or long and sparse in
            # its bbox (diagonal); a loot pile is compact and dense
            if md > 55 and (md > 3 * sm or ar < 0.35 * bw2 * bh2):
                # motion test: a bead-chain SNAKE slides between frames, a
                # kill-reward orb string sits still. Static string = loot,
                # do NOT wall it off (we used to flee our own kill payout).
                comp = (ld == i)
                if prev_dots is not None and prev_dots.shape == dots.shape:
                    aligned = _shift_mask(prev_dots, shift_x, shift_y)
                    ov = float((aligned[comp] > 0).mean())
                    if ov > 0.65:
                        continue          # static -> loot, stays food
                chain[comp] = 1
        _PREV_DOTS[0] = dots
        if chain.any():
            chain_out = chain
            a.obstacle = cv2.bitwise_or(
                a.obstacle, cv2.dilate(chain, np.ones((11, 11), np.uint8)))
    else:
        _PREV_DOTS[0] = None
    a.chain_mask = chain_out if chain_out is not None else np.zeros((h, w), np.uint8)

    # arena border: bright red boundary line + dark red-brown dead zone
    # beyond it. Treat the line and everything past it as a hard wall.
    hue = hsv[:, :, 0]
    red_line = (((hue <= 8) | (hue >= 172)) & (sat > 130) &
                (val > 100)).astype(np.uint8)
    # the arena edge is a NEON RED LINE with a dark red dead zone beyond.
    # The dark-red gate alone also matches brown/dark snakes — over half of
    # all frames used to "see" a wall mid-arena and the bot spent whole
    # lives fleeing phantom borders. Only trust dark-red areas when the
    # bright line itself is on screen to vouch for them.
    nl, ll, sl_, _ = cv2.connectedComponentsWithStats(red_line, 8)
    line = np.zeros_like(red_line)
    for i in range(1, nl):
        if sl_[i, cv2.CC_STAT_AREA] > 250 or \
                max(sl_[i, cv2.CC_STAT_WIDTH], sl_[i, cv2.CC_STAT_HEIGHT]) > 70:
            line[ll == i] = 1
    big = np.zeros_like(red_line)
    dead_zone = (((hue <= 12) | (hue >= 168)) & (val < 110) &
                 (sat > 60)).astype(np.uint8)
    if line.any():
        big = line.copy()
        nz, lz, sz, _ = cv2.connectedComponentsWithStats(dead_zone, 8)
        line_env = cv2.dilate(line, np.ones((25, 25), np.uint8))
        for i in range(1, nz):
            if sz[i, cv2.CC_STAT_AREA] < 1200:
                continue
            comp = lz == i
            # a true dead zone hugs the line; a dark snake does not
            if (line_env[comp] > 0).any():
                big[comp] = 1
    else:
        # close to the wall the neon line scrolls OFF-SCREEN and only the
        # dead zone is visible — that blindness caused head-on wall deaths.
        # A huge dark-red mass anchored to a frame edge is a wall, no
        # vouching line required.
        nz, lz, sz, _ = cv2.connectedComponentsWithStats(dead_zone, 8)
        for i in range(1, nz):
            x0, y0, bw2, bh2, ar = sz[i]
            touches_edge = (x0 == 0 or y0 == 0 or
                            x0 + bw2 >= w - 1 or y0 + bh2 >= h - 1)
            if ar > 8000 and touches_edge:
                big[lz == i] = 1
    if big.any():
        big = cv2.dilate(big, np.ones((13, 13), np.uint8))
        a.obstacle = cv2.bitwise_or(a.obstacle, big)
    a.border_mask = big     # kept separate: during spawn invulnerability
    # only the border can kill us
    if big.any() and own.any():
        # our body can never BE the border: if the temporal envelope
        # brushed the wall band, cut it back out of "own"
        own[big > 0] = 0
        a.own_mask = own
        a.own_area = int(own.sum())
        _PREV_OWN[0] = own

    # ghost/faint snakes: spawn-invulnerable enemies render translucent
    # (light OR dark tinted), slipping under BG_DIST — yet they can sprint
    # over us and kill on head contact. Catch them with a weaker gate on a
    # ROW-NORMALIZED brightness diff: subtracting each row's median cancels
    # the arena's horizontal shaded bands (previously we blanked those rows
    # entirely — and a shiny ghost inside the band was invisible).
    sdiff = small.astype(np.int16).mean(axis=2) - bg_est.mean(axis=2)
    sdiff = sdiff - np.median(sdiff, axis=1, keepdims=True)
    ghost = ((np.abs(sdiff) > 18) & (stuff == 0)).astype(np.uint8)
    _apply_ui_masks(ghost, h, w, sat, val)
    # the translucent leaderboard panel reads as a huge "ghost" and became
    # a fake wall across the top-right — always blind the ghost detector
    # there (real ghosts under it are rare and short-lived)
    fx0, fy0, fx1, fy1 = LEADERBOARD_BOX
    ghost[int(h * fy0):int(h * fy1), int(w * fx0):int(w * fx1)] = 0
    ghost = cv2.morphologyEx(ghost, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if own.any():
        ghost[cv2.dilate(own, np.ones((9, 9), np.uint8)) > 0] = 0
    ng, lg, sg, _ = cv2.connectedComponentsWithStats(ghost, 8)
    gmask = np.zeros_like(ghost)
    for i in range(1, ng):
        ar = sg[i, cv2.CC_STAT_AREA]
        bw2 = sg[i, cv2.CC_STAT_WIDTH]
        md = max(bw2, sg[i, cv2.CC_STAT_HEIGHT])
        # full-width strips are band/vignette artifacts, never a snake
        if 350 < ar < 25000 and md > 45 and bw2 < 0.85 * w:
            gmask[lg == i] = 1
    if gmask.any():
        a.obstacle = cv2.bitwise_or(
            a.obstacle, cv2.dilate(gmask, np.ones((13, 13), np.uint8)))
    a.ghost_mask = gmask

    # food filter, head-aware: loot lying NEXT to an enemy body is fine to
    # eat — only a nearby HEAD contests it. Heads ≈ the enemy mask's moving
    # leading edge (pixels that are enemy now but weren't last frame).
    # Drop food only ON bodies (small margin), near fronts/moving dots
    # (wide margin), on walls, or on our own pattern sparkle.
    fronts = None
    prev_en = _PREV_ENEMY[0]
    if prev_en is not None and prev_en.shape == enemy.shape:
        prev_en = _shift_mask(prev_en, shift_x, shift_y)
        # 7px envelope absorbs sub-pixel registration residue; the opening
        # drops the thin body-edge slivers that are alignment noise, not
        # heads (they saturated the threat map and pinned minE at 6px)
        fronts = cv2.bitwise_and(
            enemy, cv2.bitwise_not(
                cv2.dilate(prev_en, np.ones((7, 7), np.uint8))))
        fronts = cv2.morphologyEx(fronts, cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8))
    _PREV_ENEMY[0] = enemy.copy()
    a.front_mask = fronts if fronts is not None else np.zeros((h, w), np.uint8)
    head_src = np.zeros((h, w), np.uint8)
    if fronts is not None and fronts.any():
        head_src = cv2.bitwise_or(head_src, cv2.dilate(fronts, np.ones((7, 7), np.uint8)))
    if head_src.any():
        nh, lh, sh, ch = cv2.connectedComponentsWithStats(head_src, 8)
        heads = []
        for i in range(1, nh):
            area = sh[i, cv2.CC_STAT_AREA]
            bwh = sh[i, cv2.CC_STAT_WIDTH]
            bhh = sh[i, cv2.CC_STAT_HEIGHT]
            # heads are compact; long thin front arcs are body-edge noise
            if 40 <= area <= 1800 and max(bwh, bhh) <= 4 * min(bwh, bhh):
                hx, hy = ch[i]
                dist_c = np.hypot(hx - cx, hy - cy)
                if dist_c > 35:
                    heads.append((float(hx), float(hy), float(area), float(dist_c)))
        # align previous head positions to this frame's camera before
        # matching: velocity is then the enemy's TRUE motion, not ours
        prev_heads = [(px + shift_x, py + shift_y)
                      for px, py in (_PREV_HEADS[0] or [])]
        with_v = []
        for hx, hy, area, dist_c in heads:
            matched = False
            vx = vy = 0.0
            if prev_heads:
                px, py = min(prev_heads, key=lambda p: (p[0] - hx) ** 2 + (p[1] - hy) ** 2)[:2]
                if (px - hx) ** 2 + (py - hy) ** 2 < 45 ** 2:
                    vx, vy = hx - px, hy - py
                    matched = True
            # only PERSISTENT candidates (seen 2 frames running) count as
            # heads — one-frame sparkles were flapping the count 0->8->0
            # and saturating the threat map
            if matched:
                with_v.append((hx, hy, area, dist_c, float(vx), float(vy)))
        with_v.sort(key=lambda p: p[3])
        a.enemy_heads = with_v[:8]
        _PREV_HEADS[0] = [(float(h[0]), float(h[1])) for h in heads][:12]
    else:
        _PREV_HEADS[0] = None

    # project confident head velocities forward: the swept corridor a head
    # will occupy over the next ~6 frames is lethal-to-be. Strategy dilates
    # this like the other fast-threat masks.
    proj = np.zeros((h, w), np.uint8)
    for head in a.enemy_heads:
        hx, hy, _, _, vx, vy = head
        speed = np.hypot(vx, vy)
        if speed < 3.0:
            continue
        k = min(6.0, 60.0 / speed)      # cap projection length at 60px
        cv2.line(proj, (int(hx), int(hy)),
                 (int(hx + vx * k), int(hy + vy * k)), 1, 11)
    # NOTE: proj stays OUT of a.obstacle — the strategy's danger memory is
    # built from a.obstacle and would remember a predicted (possibly wrong)
    # corridor for ~0.7s. Strategy adds it as an instantaneous fast threat.
    a.head_proj = proj
    if len(a.food_pts):
        bad = cv2.dilate(enemy, np.ones((5, 5), np.uint8))
        bad = cv2.bitwise_or(bad, cv2.dilate(own, np.ones((7, 7), np.uint8)))
        if big.any():
            bad = cv2.bitwise_or(bad, big)
        if fronts is not None and fronts.any():
            bad = cv2.bitwise_or(
                bad, cv2.dilate(fronts, np.ones((19, 19), np.uint8)))
        if chain_out is not None:
            bad = cv2.bitwise_or(
                bad, cv2.dilate(chain_out, np.ones((11, 11), np.uint8)))
        if gmask.any():
            bad = cv2.bitwise_or(
                bad, cv2.dilate(gmask, np.ones((9, 9), np.uint8)))
        fi = a.food_pts.astype(np.int32)
        keep = bad[np.clip(fi[:, 1], 0, h - 1),
                   np.clip(fi[:, 0], 0, w - 1)] == 0
        a.food_pts = a.food_pts[keep]
        a.food_w = a.food_w[keep]

    # legacy: pitch-black void at frame edges (loading transitions)
    dark = (val < 30).astype(np.uint8)
    edge_band = np.zeros_like(dark)
    m = 6
    edge_band[:m, :] = 1
    edge_band[-m:, :] = 1
    edge_band[:, :m] = 1
    edge_band[:, -m:] = 1
    if (dark & edge_band).sum() > 200:
        a.obstacle = cv2.bitwise_or(a.obstacle, dark)

    a.loot = _loot_clusters(a.food_pts, a.food_w)

    # is there enemy mass "inside" us (between our body pixels and center)?
    if a.enemy_area > 0 and a.own_area > 300:
        ys, xs = np.nonzero(own)
        r_own = np.percentile(np.hypot(xs - cx, ys - cy), 80)
        eys, exs = np.nonzero(enemy)
        er = np.hypot(exs - cx, eys - cy)
        a.enemy_inside = (er < r_own * 0.7).sum() > 80
    return a


LOOT_CELL = 34          # px at half scale
LOOT_MIN_WEIGHT = 95    # summed blob area ≈ early death/boost orb burst


def _loot_clusters(food_pts, food_w, max_clusters=5):
    """Dense food clusters = remains of a killed snake. Returns up to
    max_clusters of (x, y, weight), heaviest first."""
    if len(food_pts) < 4:
        return []
    cells = {}
    for (x, y), wt in zip(food_pts, food_w):
        key = (int(x) // LOOT_CELL, int(y) // LOOT_CELL)
        cells.setdefault(key, []).append((x, y, wt))
    out = []
    used = set()
    # neighborhood-summed weight per cell, greedy pick of local maxima
    def nb_weight(key):
        return sum(wt for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for (_, _, wt) in cells.get((key[0] + dx, key[1] + dy), []))
    ranked = sorted(cells, key=nb_weight, reverse=True)
    for key in ranked:
        if len(out) >= max_clusters:
            break
        if key in used:
            continue
        wsum = nb_weight(key)
        if wsum < LOOT_MIN_WEIGHT:
            break
        pts = [(x, y, wt) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for (x, y, wt) in cells.get((key[0] + dx, key[1] + dy), [])]
        tot = sum(p[2] for p in pts)
        cx = sum(p[0] * p[2] for p in pts) / tot
        cy = sum(p[1] * p[2] for p in pts) / tot
        out.append((cx, cy, tot))
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                used.add((key[0] + dx, key[1] + dy))
    return out


def read_score_gate(frame_bgr):
    """Coarse read of the big white score at top-center: returns a LOWER
    BOUND on the score. We don't need exact OCR — boost policy only needs
    'is it comfortably above ~200'. Digit count + whether the leading digit
    is a skinny '1' is enough."""
    h, w = frame_bgr.shape[:2]
    crop = frame_bgr[0:int(h * 0.15), int(w * 0.40):int(w * 0.60)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 225)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    ch = crop.shape[0]
    # STRICT digit shape gate. White SNAKES crossing the score area also
    # match the white gate and were read as extra digits (reporting 1000+
    # at an actual ~350). Digits in this font: taller than wide
    # (aspect <= 0.8), height mid-band of the crop, solid but not square.
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (0.35 * ch < bh < 0.80 * ch):
            continue
        if area < 60 or bw > 0.8 * bh:
            continue
        cands.append((x, y, bw, bh))
    if not cands:
        return 0
    # digits share a height and a baseline; anything off-row is not a digit
    hs = sorted(c[3] for c in cands)
    med_h = hs[len(hs) // 2]
    ys = sorted(c[1] for c in cands)
    med_y = ys[len(ys) // 2]
    cands = [c for c in cands
             if 0.8 * med_h <= c[3] <= 1.2 * med_h and
             abs(c[1] - med_y) < 0.3 * med_h]
    cands.sort()
    row, prev_end = [], None
    for x, y, bw, bh in cands:
        if prev_end is not None and x - prev_end > 1.2 * med_h:
            break               # gap: whatever follows is not the score
        row.append((x, bw, bh))
        prev_end = x + bw
    if not row:
        return 0
    nd = len(row)
    lead_aspect = row[0][1] / max(1, row[0][2])
    if nd >= 4:
        return 1000
    if nd == 3:
        return 100 if lead_aspect < 0.45 else 220   # 1xx vs 2xx..9xx
    if nd == 2:
        return 80
    return 5


# ---------------- game-state classification ----------------

def find_button(frame_bgr, h_lo, h_hi, min_area_frac=0.0015,
                s_min=130, v_min=140, s_max=255):
    """Largest saturated blob in the hue band -> (cx, cy) full-res, or None.
    Used to locate the orange PLAY / green REVIVE buttons wherever they are."""
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (w // 2, h // 2))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
         (hsv[:, :, 1] > s_min) & (hsv[:, :, 1] <= s_max) &
         (hsv[:, :, 2] > v_min)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area_frac * m.size:
        return None
    return int(cents[i][0] * 2), int(cents[i][1] * 2)


def classify_state(frame_bgr):
    """Full-res frame -> one of STATE_*.

    'playing' = the teal hex arena fills the center (no dark overlay panel).
    Overlays (menu, death results, revive) darken/replace it, so check that
    first, then look for the actionable button.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    def half_fracs(x0, x1):
        c = hsv[int(h * 0.25):int(h * 0.75), int(w * x0):int(w * x1)]
        teal = ((c[:, :, 0] >= 76) & (c[:, :, 0] <= 102) &
                (c[:, :, 1] >= 40) & (c[:, :, 1] <= 115) &
                (c[:, :, 2] > 140)).mean()
        # DARK red only: the wall dead-zone is dim (val<130); bright red UI
        # panels (daily-challenge banner) must not read as a wall
        red = (((c[:, :, 0] <= 12) | (c[:, :, 0] >= 168)) &
               (c[:, :, 1] > 60) & (c[:, :, 2] < 130)).mean()
        return teal, red

    # playing: hex arena in BOTH center halves — OR arena in one half and
    # the red border wall/dead zone filling the other (playing next to the
    # map edge used to flip to "menu", release the joystick and let the
    # snake sail head-on into the wall with no inputs at all)
    tl, rl = half_fracs(0.32, 0.48)
    tr, rr = half_fracs(0.52, 0.68)
    if (tl > 0.35 and tr > 0.35) or \
       (tl > 0.33 and rr > 0.30) or (tr > 0.33 and rl > 0.30):
        return STATE_PLAYING

    # a big orange PLAY blob (main menu, death results, "NEXT/PLAY" summary)
    # Checked before green: the menu can show a green popup PLAY too.
    if find_button(frame_bgr, 10, 30, 0.002, s_max=215) is not None:
        return STATE_MENU

    # revive offer: big green button, no orange play
    if find_button(frame_bgr, 40, 75, 0.0015) is not None:
        return STATE_REVIVE

    # modal system dialog (e.g. "Confirm exit"): near-black rounded panel
    # spanning the WHOLE center — both halves dark. Check this AFTER
    # actionable colored buttons so death/revive/results screens do not get
    # trapped as "dialog" forever.
    left = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.32):int(w * 0.48), 2]
    right = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.52):int(w * 0.68), 2]
    if (left < 95).mean() > 0.45 and (right < 95).mean() > 0.45:
        return STATE_DIALOG

    return STATE_UNKNOWN


def overlay_debug(frame_bgr, a, target_angle=None, danger=None, hud=None):
    """Draw analysis for calibration/replay snapshots (at half scale).
    red=obstacle now, orange=danger memory (where snakes swept recently),
    green=own, cyan=food, magenta arrow=chosen direction, HUD top-left."""
    vis = cv2.resize(frame_bgr, (a.w, a.h))
    layer = debug_layer(a, target_angle, danger)
    colors = {
        LAYER_OBSTACLE: (0, 0, 255),
        LAYER_DANGER: (0, 150, 255),
        LAYER_OWN: (0, 255, 0),
        LAYER_FOOD: (255, 255, 0),
        LAYER_LOOT: (0, 255, 255),
        LAYER_TARGET: (255, 0, 255),
    }
    for code, color in colors.items():
        vis[layer == code] = color
    if hud:
        cv2.rectangle(vis, (0, 0), (min(len(hud) * 8 + 8, a.w), 18),
                      (30, 30, 30), -1)
        cv2.putText(vis, hud, (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def debug_layer(a, target_angle=None, danger=None):
    """Single-channel replay layer for the web viewer.

    Keeping classes as numeric pixels lets the viewer recolor and fade the
    overlay later. This is much more useful for death review than baking red
    and green directly into the replay frames.
    """
    layer = np.zeros((a.h, a.w), np.uint8)
    if danger is not None and danger.shape == a.obstacle.shape:
        mem = (danger > 0.45) & (a.obstacle == 0)
        layer[mem] = LAYER_DANGER
    layer[a.obstacle > 0] = LAYER_OBSTACLE
    layer[a.own_mask > 0] = LAYER_OWN
    for (x, y), wt in zip(a.food_pts, a.food_w):
        cv2.circle(layer, (int(x), int(y)), 2, LAYER_FOOD, -1)
    for lx, ly, lw in a.loot:
        cv2.circle(layer, (int(lx), int(ly)), 8, LAYER_LOOT, 2)
    if target_angle is not None:
        cx, cy = a.center
        ex = int(cx + 80 * np.cos(target_angle))
        ey = int(cy + 80 * np.sin(target_angle))
        cv2.arrowedLine(layer, (cx, cy), (ex, ey), LAYER_TARGET, 2)
    return layer
