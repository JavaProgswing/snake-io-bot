"""Behavior-cloned steering brain: loads bc_policy.pt (from train.py) and
maps a raw frame -> (steering angle, boost probability).

Used as a hybrid: the net drives normal play (cruise/eat/loot/grace modes),
the heuristic pilot keeps override authority for emergencies (reflex,
wall-escape, hug-break, coil) — a seatbelt around the learned policy.
"""

import math
import os

import cv2
import numpy as np

IMG_W, IMG_H = 296, 134


class BCBrain:
    def __init__(self, path):
        import torch
        import torch.nn as nn

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
                    nn.Linear(128, 3),
                )

            def forward(self, x):
                return self.net(x)

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = PolicyNet().to(self.dev)
        self.net.load_state_dict(torch.load(path, map_location=self.dev))
        self.net.eval()

    def steer(self, frame_bgr):
        """frame -> (angle_rad, boost_prob). ~1-2ms on GPU."""
        img = cv2.resize(frame_bgr, (IMG_W, IMG_H))
        x = self.torch.from_numpy(img).permute(2, 0, 1).float()
        x = (x / 255.0).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            out = self.net(x)[0]
        s, c = float(out[0]), float(out[1])
        angle = math.atan2(s, c)
        boost_p = 1.0 / (1.0 + math.exp(-float(out[2])))
        return angle, boost_p


def load_if_available(here):
    p = os.path.join(here, "bc_policy.pt")
    if not os.path.isfile(p):
        return None
    try:
        return BCBrain(p)
    except Exception as e:            # torch missing / bad checkpoint
        print(f"brain: failed to load ({e})")
        return None
