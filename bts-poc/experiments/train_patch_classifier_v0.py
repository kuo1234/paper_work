from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from envs.image_binding import ControlledImageBindingBenchmark, COLORS, SHAPES
from experiments.image_binding_corruption_v0 import corrupt_image

COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS.keys())}
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}


def crop_patch(img: np.ndarray, xy, size: int = 32) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = xy
    r = size // 2
    patch = np.full((size, size, 3), 245, dtype=np.uint8)
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    px0, py0 = r - (x - x0), r - (y - y0)
    patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = img[y0:y1, x0:x1]
    return patch


class PatchDataset(Dataset):
    def __init__(self, n_scenes: int, seed: int, split: str = "train", noise_std: float = 0.0, occluder: int = 0):
        rng = random.Random(seed)
        env = ControlledImageBindingBenchmark(seed=seed)
        self.items = []
        for _ in range(n_scenes):
            ex = env.sample(split)
            img = corrupt_image(ex.image, rng, noise_std=noise_std, occluder=occluder)
            for o in ex.objects:
                patch = crop_patch(img, o.xy)
                self.items.append((patch, COLOR_TO_IDX[o.color], SHAPE_TO_IDX[o.shape]))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        patch, c, s = self.items[idx]
        x = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        return x, torch.tensor(c), torch.tensor(s)


class PatchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.color = nn.Linear(96, len(COLOR_TO_IDX))
        self.shape = nn.Linear(96, len(SHAPE_TO_IDX))

    def forward(self, x):
        z = self.cnn(x).flatten(1)
        return self.color(z), self.shape(z)


def evaluate(model, loader, device):
    model.eval()
    n = 0; color_ok = 0; shape_ok = 0; both_ok = 0; loss_sum = 0.0
    with torch.no_grad():
        for x, c, s in loader:
            x, c, s = x.to(device), c.to(device), s.to(device)
            lc, ls = model(x)
            loss = F.cross_entropy(lc, c) + F.cross_entropy(ls, s)
            pc, ps = lc.argmax(-1), ls.argmax(-1)
            color_ok += (pc == c).sum().item()
            shape_ok += (ps == s).sum().item()
            both_ok += ((pc == c) & (ps == s)).sum().item()
            n += x.size(0); loss_sum += loss.item() * x.size(0)
    return {"color_acc": color_ok/n, "shape_acc": shape_ok/n, "both_acc": both_ok/n, "loss": loss_sum/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-scenes", type=int, default=1000)
    ap.add_argument("--n-eval-scenes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/image_binding_patch_classifier_v0/summary.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Mixed corruption training prevents over-specializing to one rendering regime.
    train_ds = ConcatDataset([
        PatchDataset(args.n_train_scenes, args.seed, "train", noise_std=0.0, occluder=0),
        PatchDataset(args.n_train_scenes, args.seed + 10, "train", noise_std=20.0, occluder=0),
        PatchDataset(args.n_train_scenes, args.seed + 20, "train", noise_std=0.0, occluder=8),
        PatchDataset(args.n_train_scenes, args.seed + 30, "train", noise_std=20.0, occluder=8),
    ])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_sets = {
        "clean": PatchDataset(args.n_eval_scenes, args.seed+1, "ood", 0.0, 0),
        "noise20": PatchDataset(args.n_eval_scenes, args.seed+2, "ood", 20.0, 0),
        "noise40": PatchDataset(args.n_eval_scenes, args.seed+3, "ood", 40.0, 0),
        "occ12": PatchDataset(args.n_eval_scenes, args.seed+4, "ood", 0.0, 12),
        "noise20_occ12": PatchDataset(args.n_eval_scenes, args.seed+5, "ood", 20.0, 12),
    }
    eval_loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False) for k,v in eval_sets.items()}
    model = PatchClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train(); total = 0; n = 0
        for x,c,s in train_loader:
            x,c,s = x.to(device), c.to(device), s.to(device)
            lc,ls = model(x)
            loss = F.cross_entropy(lc,c) + F.cross_entropy(ls,s)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*x.size(0); n += x.size(0)
        if epoch in {0,args.epochs-1} or (epoch+1)%4==0:
            print(f"epoch {epoch+1} loss={total/n:.4f}")
    res = {k: evaluate(model,l,device) for k,l in eval_loaders.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k,r in res.items():
        print(f"{k:>14} color={r['color_acc']:.3f} shape={r['shape_acc']:.3f} both={r['both_acc']:.3f} loss={r['loss']:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
