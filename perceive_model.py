"""Inference with the trained U-Net segmenter (train_seg.py -> seg_model.pt).

WHERE THE MODEL STANDS (measured on held-out frames):
  - self  IoU 0.57 : ragged on coils — classical center color-grow is better.
  - enemy IoU 0.85 : smooth on THICK snakes, but MISSES thin bead-chain
                     snakes that the classical detector catches. Missing a
                     snake is a death, so the model must not gate enemies.
  - border 0.88, food 0.26, loot 0.07 : orbs are noise.

Conclusion: the classical perception is the reliable workhorse (self-
consistent, catches bead snakes). The model is used ONLY to ADD enemy-
obstacle coverage via a UNION — it can thicken/fill the avoidance mask on
big snakes but can never remove a classical detection, so the hybrid is
never less safe than classical. Returns the classical Scene with its
enemy/obstacle masks augmented.

(To make the model primary it needs bead-snake recall — retrain with more
thin-snake examples. Until then, this is the honest, safe configuration.)

CLI compares classical vs hybrid overlays on a frame sequence:
  python perceive_model.py <frames_dir> [--start N] [--count M] [--every E]
"""
import argparse
import glob
import os

import cv2
import numpy as np

import perception as P

IMG_W, IMG_H = 320, 144


def _self_territory(stuff, cx, cy):
    """The whole body connected to screen center (our snake, coil and all).
    Used to keep ANY piece of us out of the model's enemy mask."""
    src = cv2.morphologyEx(stuff, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if not src[cy, cx]:
        return np.zeros_like(stuff)
    nn, lb = cv2.connectedComponents(src, 8)
    return (lb == int(lb[cy, cx])).astype(np.uint8)


class SegModel:
    def __init__(self, path=None):
        import torch
        from train_seg import UNet
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = UNet().to(self.dev)
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "seg_model.pt")
        self.net.load_state_dict(torch.load(path, map_location=self.dev))
        self.net.eval()
        self._classical = P.Perception()

    def reset(self):
        self._classical.reset()

    def _classmap(self, frame_bgr, w, h):
        img = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        x = self.torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with self.torch.no_grad():
            logits = self.net(x.unsqueeze(0).to(self.dev))[0]
        cls = logits.argmax(0).byte().cpu().numpy()
        return cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)

    def process(self, frame_bgr, dt=1.0):
        # classical does the reliable work (self, enemies incl. bead snakes,
        # food, loot, border, heads/velocity)
        sc = self._classical.process(frame_bgr, dt)
        small = cv2.resize(frame_bgr, None, fx=P.SCALE, fy=P.SCALE,
                           interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        cx, cy = w // 2, h // 2

        model_enemy = (self._classmap(frame_bgr, w, h) == 2).astype(np.uint8)
        # never let our OWN body (whole connected coil) read as a model enemy
        stuff, _b, _hsv = P._segment(small)
        terr = cv2.bitwise_or(_self_territory(stuff, cx, cy), sc.self_mask)
        model_enemy[cv2.dilate(terr, np.ones((5, 5), np.uint8)) > 0] = 0

        # UNION only — the model can add obstacle coverage, never remove a
        # classical detection.
        enemy_mask = cv2.bitwise_or(sc.enemy_mask, model_enemy)
        obstacle = cv2.bitwise_or(enemy_mask, sc.border_mask)
        if obstacle.any():
            obstacle = cv2.dilate(obstacle, np.ones((11, 11), np.uint8))
        sc.enemy_mask = enemy_mask
        sc.obstacle_mask = obstacle
        return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--every", type=int, default=9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "*.jpg"))) or \
        sorted(glob.glob(os.path.join(args.src, "*.png")))
    files = files[args.start:args.start + args.count * args.stride:args.stride]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "model_cmp")
    os.makedirs(out, exist_ok=True)
    cl = P.Perception()
    md = SegModel()
    for k, fp in enumerate(files):
        img = cv2.imread(fp)
        if img is None or P.classify_state(img) != P.STATE_PLAYING:
            cl.reset()
            md.reset()
            continue
        sc_c = cl.process(img)
        sc_m = md.process(img)
        if k % args.every == 0:
            comp = cv2.vconcat([
                P.render(sc_c, img, hud="CLASSICAL"),
                P.render(sc_m, img, hud="HYBRID (classical + model obstacle)")])
            cv2.imwrite(os.path.join(out, f"c{k:05d}.png"), comp)
    print("compare overlays ->", out)


if __name__ == "__main__":
    main()
