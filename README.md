# snake_bot — Snake.io (Netflix) perception + autopilot

Plays Snake.io on an Android phone over ADB (drives the bundled
`scrcpy-server` jar directly — no scrcpy window). Rebooted around an
**explicit, temporal perception layer** after the old per-frame heuristics
proved impossible to make reliable.

## Architecture (v2)

- **Capture** ([scrcpy_link.py](scrcpy_link.py)): pushes scrcpy-server 4.0,
  decodes the H264 video socket with PyAV (~30 fps @ 1184px), injects touches
  on the control socket. Unchanged, solid.
- **Perception** ([perception.py](perception.py)): every frame → a typed
  `Scene`. Priorities: **precision over recall**, **explicit categories**
  (SELF, ENEMY w/ head+velocity, BORDER, FOOD, LOOT), **temporal**.
  - The camera is head-locked to us, so the whole world scrolls by our own
    velocity each frame. We estimate that global **scroll** (phase
    correlation on the background) and subtract it, so only *world-relative*
    motion marks a live enemy. This is why static loot no longer reads as an
    enemy.
  - **Self** = the snake component nearest screen center (our head is pinned
    at center by the camera); head = nearest-center body pixel.
  - **Enemies** = solid/elongated snake bodies (fragments reconnected first);
    each gets a head (geometry + temporal leading edge) and a world-relative
    velocity via a greedy tracker.
  - **Food vs loot** = scattered orbs vs dense clusters; sparse round blobs
    are loot, never enemies.
  - Every detection carries a confidence in [0,1].
- **Validation** ([label_perception.py](label_perception.py)): serves a review
  UI to rate each class per frame and computes live **per-class precision**
  (target 95%). This is how the classifier is certified before any steering.

## Data (kept — the training substrate)

- `sessions/` — human play recorded by [record.py](record.py): frames +
  time-aligned touch events. Two long recordings (~12k + frames).
- `debug_*` — annotated/raw frame dumps from past bot runs.
- The old heuristic + RL/BC steering code is parked in `archive_v1/`
  (reversible; there was no git history to fall back on).

## Run

```powershell
cd snake_bot
python smoke.py out.png                     # connectivity + state classify
python record.py                            # record your own play -> sessions/
# perception offline over a recorded session (tracker needs the sequence):
python run_perception.py sessions\<rec> --start 3000 --count 240 --out perc_out
# certify precision (open http://127.0.0.1:8778):
python label_perception.py sessions\<rec> --start 3000 --count 120 --stride 10
```

Deps: `numpy opencv-python av` (+ `torch` for the upcoming perception model,
+ adb from the scrcpy folder; paths in [config.json](config.json)).

## Roadmap

1. ✅ Reset; keep training data.
2. ✅ Perception core (segment, border, orbs).
3. ✅ Heads, tracking, self-ID, food/loot.
4. ⏳ Certify ≥95% per-class precision via the labeling tool; tune.
5. ⏳ Train a small segmentation/detection net (auto-labels from the classical
   detector + human corrections) for the hard cases, then refine.
6. ⏳ Rebuild steering on top of the trusted perception layer.

## Hard-won gotchas (still true)

- Killing the bot mid-game leaves injected touch pointers "down"; Android's
  InputDispatcher then drops ALL injected input. `clear_pointers()` heals it.
- `scid` must fit a signed 31-bit hex int (first digit 0–7).
- Never send KEYCODE_BACK in the menu — it opens "Confirm exit".
- The arena has a vignette + horizontal shaded bands; global color models
  fail. Use the blur-based local background.
