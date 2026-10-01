"""Train the Scene-space imitation policy from imit_data.jsonl (imitate.py).

Small MLP: 147 features -> (sin, cos) of your commanded heading. Regresses
on the circle (unit-vector MSE), which handles the wraparound cleanly.
Reports held-out median angular error in degrees. Saves policy.pt.

Usage:  python train_policy.py [--epochs 60]
"""
import argparse
import json
import math
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise SystemExit("pip install torch")

from features import FEAT_DIM

HERE = os.path.dirname(os.path.abspath(__file__))


class PolicyNet(nn.Module):
    def __init__(self, din=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2))          # sin, cos (un-normalized)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "imit_data.jsonl"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(HERE, "policy.pt"))
    args = ap.parse_args()

    X, Y = [], []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            X.append(r["x"])
            Y.append([math.sin(r["y"]), math.cos(r["y"])])
    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)
    if len(X) < 400:
        raise SystemExit(f"only {len(X)} samples — record more manual play")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]
    nval = max(100, len(X) // 10)
    Xv, Yv, Xt, Yt = X[:nval], Y[:nval], X[nval:], Y[nval:]
    print(f"{len(Xt)} train / {len(Xv)} val, dim {X.shape[1]}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyNet(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(Yt)),
                    batch_size=args.batch, shuffle=True)
    xv = torch.from_numpy(Xv).to(dev)
    yv = torch.from_numpy(Yv).to(dev)

    def val_deg():
        net.eval()
        with torch.no_grad():
            o = net(xv)
            o = o / o.norm(dim=1, keepdim=True).clamp_min(1e-6)
            pa = torch.atan2(o[:, 0], o[:, 1])
            ta = torch.atan2(yv[:, 0], yv[:, 1])
            d = torch.atan2(torch.sin(pa - ta), torch.cos(pa - ta)).abs()
            return float(torch.median(d)) * 180 / math.pi

    best = 1e9
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            o = net(xb)
            o = o / o.norm(dim=1, keepdim=True).clamp_min(1e-6)
            loss = ((o - yb) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        sched.step()
        md = val_deg()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep{ep+1:02d} loss{tot/len(Xt):.4f} val_median{md:.1f}deg",
                  flush=True)
        if md < best:
            best = md
            torch.save(net.state_dict(), args.out)
    print(json.dumps({"saved": args.out, "best_val_deg": round(best, 1),
                      "samples": len(X)}))


if __name__ == "__main__":
    main()
