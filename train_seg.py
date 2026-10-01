"""Train a compact U-Net segmenter on the auto-labeled data (autolabel.py).

6 classes: 0 bg, 1 self, 2 enemy, 3 border, 4 food, 5 loot. Class-weighted
cross-entropy (heavy imbalance — food/loot are ~0.2% of pixels). Reports
per-class IoU + precision on a held-out split. Saves seg_model.pt.

Usage:
  python train_seg.py <seg_data_dir> [more_dirs...] [--epochs 24] [--batch 32]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("pip install torch")

NCLASS = 6
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["bg", "self", "enemy", "border", "food", "loot"]


class SegSet(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = cv2.imread(ip)
        lab = cv2.imread(lp, 0)
        if self.train and np.random.rand() < 0.5:      # horizontal flip
            img = img[:, ::-1].copy()
            lab = lab[:, ::-1].copy()
        if self.train:                                  # mild brightness
            f = 0.8 + 0.4 * np.random.rand()
            img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(lab.astype(np.int64))
        return x, y


def conv(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, nc=NCLASS):
        super().__init__()
        self.d1 = conv(3, 32)
        self.d2 = conv(32, 64)
        self.d3 = conv(64, 128)
        self.b = conv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c3 = conv(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = conv(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c1 = conv(64, 32)
        self.out = nn.Conv2d(32, nc, 1)

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.c3(torch.cat([self.u3(b), e3], 1))
        d2 = self.c2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.c1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def evaluate(net, dl, dev):
    net.eval()
    inter = np.zeros(NCLASS)
    union = np.zeros(NCLASS)
    tp = np.zeros(NCLASS)
    pp = np.zeros(NCLASS)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(dev)
            pred = net(x).argmax(1).cpu().numpy()
            y = y.numpy()
            for c in range(NCLASS):
                p = pred == c
                g = y == c
                inter[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()
                tp[c] += np.logical_and(p, g).sum()
                pp[c] += p.sum()
    iou = inter / np.maximum(union, 1)
    prec = tp / np.maximum(pp, 1)
    return iou, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "seg_model.pt"))
    args = ap.parse_args()

    pairs = []
    for d in args.data:
        for ip in sorted(glob.glob(os.path.join(d, "img_*.jpg"))):
            lp = ip.replace("img_", "lab_").replace(".jpg", ".png")
            if os.path.isfile(lp):
                pairs.append((ip, lp))
    if len(pairs) < 200:
        raise SystemExit(f"only {len(pairs)} pairs — run autolabel first")
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    nval = max(50, len(pairs) // 10)
    val, train = pairs[:nval], pairs[nval:]
    print(f"{len(train)} train / {len(val)} val")

    # class weights from a sample of the training labels (inverse freq, capped)
    freq = np.zeros(NCLASS)
    for _, lp in train[:800]:
        lab = cv2.imread(lp, 0)
        for c in range(NCLASS):
            freq[c] += (lab == c).sum()
    freq = freq / freq.sum()
    w = np.clip(1.0 / np.sqrt(freq + 1e-4), 1.0, 30.0)
    print("class freq:", [round(float(f), 4) for f in freq])
    print("weights   :", [round(float(x), 2) for x in w])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    # num_workers=0: on Windows each worker re-imports torch+CUDA and blows
    # the commit/paging limit (WinError 1455). Single-process loading is
    # plenty fast for 320x144 jpgs and keeps GPU fed.
    dtr = DataLoader(SegSet(train, True), batch_size=args.batch, shuffle=True,
                     num_workers=0, pin_memory=True, drop_last=True)
    dva = DataLoader(SegSet(val, False), batch_size=args.batch,
                     num_workers=0, pin_memory=True)

    best = -1.0
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for x, y in dtr:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(net(x), y, weight=wt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        sched.step()
        iou, prec = evaluate(net, dva, dev)
        obj = float(np.mean(iou[1:]))       # mean IoU over non-bg
        line = " ".join(f"{NAMES[c]}={iou[c]:.2f}/{prec[c]:.2f}"
                        for c in range(NCLASS))
        print(f"ep{ep+1:02d} loss{tot/len(train):.3f} mIoU{obj:.3f} "
              f"[IoU/prec] {line}", flush=True)
        if obj > best:
            best = obj
            torch.save(net.state_dict(), args.out)
    print(json.dumps({"saved": args.out, "best_mIoU": round(best, 3)}))


if __name__ == "__main__":
    main()
