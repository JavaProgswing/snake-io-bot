"""Behavior cloning from record.py sessions: frame -> (steering angle, boost).

Your drag on the left half of the screen is the joystick: for each frame we
take the active touch's direction relative to its recent anchor as the label.
Small CNN, PyTorch. Outputs bc_policy.pt; the bot can load it later with
--brain bc (wired up once a first model exists).

Usage:
  pip install torch
  python train.py --data sessions/rec_YYYYMMDD_HHMMSS [more sessions...]
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("pip install torch  (CUDA build if you have a GPU)")

IMG_W, IMG_H = 296, 134          # quarter scale of the 1184x538 stream


def load_session(sess, screen_w=2376, screen_h=1080):
    frames = [json.loads(l) for l in open(os.path.join(sess, "frames.jsonl"),
                                          encoding="utf-8")]
    touches = [json.loads(l) for l in open(os.path.join(sess, "touches.jsonl"),
                                           encoding="utf-8")]
    if not frames or not touches:
        return []
    # getevent ABS range differs from screen px; normalize by observed max
    xs = [t["x"] for t in touches]
    ys = [t["y"] for t in touches]
    sx = screen_w / max(1, max(xs))
    sy = screen_h / max(1, max(ys))

    # steering label: Snake.io uses a FIXED joystick — the control signal
    # is the finger's position RELATIVE TO THE JOYSTICK CENTER, not its
    # motion (an earlier version labeled drag deltas: a held finger gave
    # pure noise and the net learned to have a stroke).
    # self-calibrate the joystick center: the finger orbits the base, so
    # the median of all steering-touch positions IS the base (assumed
    # coordinates were off and skewed every label up-right)
    lpts = [(tc["x"] * sx, tc["y"] * sy) for tc in touches
            if tc["down"] and tc["x"] * sx < screen_w * 0.55]
    if len(lpts) < 50:
        return []
    lpts.sort()
    jx = sorted(p[0] for p in lpts)[len(lpts) // 2]
    jy = sorted(p[1] for p in lpts)[len(lpts) // 2]
    deadzone = screen_h * 0.03
    samples = []
    ti = 0
    steer = None                   # (t, x, y) latest steering touch state
    steer_down = False
    boost_down = False
    for fr in frames:
        if fr["state"] != "playing":
            steer, steer_down = None, False
            continue
        while ti < len(touches) and touches[ti]["t"] <= fr["t"]:
            tc = touches[ti]
            px, py = tc["x"] * sx, tc["y"] * sy
            if px < screen_w * 0.55:
                steer_down = tc["down"]
                if tc["down"]:
                    steer = (tc["t"], px, py)
            else:
                boost_down = tc["down"]
            ti += 1
        if steer_down and steer and fr["t"] - steer[0] < 0.5:
            dx, dy = steer[1] - jx, steer[2] - jy
            if math.hypot(dx, dy) > deadzone:
                ang = math.atan2(dy, dx)
                samples.append((os.path.join(sess, fr["file"]), ang,
                                1.0 if boost_down else 0.0))
    return samples


class Frames(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, ang, boost = self.samples[i]
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_W, IMG_H))
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # angle as (sin, cos) — regression on the circle
        y = torch.tensor([math.sin(ang), math.cos(ang), boost],
                         dtype=torch.float32)
        return x, y


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 3, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
            nn.Linear(64 * 8, 128), nn.ReLU(),
            nn.Linear(128, 3),           # sin, cos, boost-logit
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    samples = []
    for pattern in args.data:
        for sess in glob.glob(pattern):
            got = load_session(sess)
            print(f"{sess}: {len(got)} labeled frames")
            samples += got
    if len(samples) < 500:
        raise SystemExit(f"only {len(samples)} samples — record more play")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {dev}, {len(samples)} samples")
    dl = DataLoader(Frames(samples), batch_size=args.batch, shuffle=True,
                    num_workers=2, pin_memory=(dev == "cuda"))
    net = PolicyNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(args.epochs):
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            out = net(x)
            loss = ((out[:, :2] - y[:, :2]) ** 2).mean() + \
                0.3 * bce(out[:, 2], y[:, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / len(samples):.4f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bc_policy.pt")
    torch.save(net.state_dict(), out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
