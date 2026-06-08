from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from envs.image_binding import ControlledImageBindingBenchmark, COLORS, SHAPES, diagnose_prediction

COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS.keys())}
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
N_ATTR = len(COLOR_TO_IDX) + len(SHAPE_TO_IDX)


def instr_vec(combo):
    color, shape = combo
    v = torch.zeros(N_ATTR)
    v[COLOR_TO_IDX[color]] = 1.0
    v[len(COLOR_TO_IDX) + SHAPE_TO_IDX[shape]] = 1.0
    return v


class PixelBindingDataset(Dataset):
    def __init__(self, split: str, n: int, seed: int):
        env = ControlledImageBindingBenchmark(seed=seed)
        self.examples = [env.sample(split) for _ in range(n)]
        self.size = env.size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = torch.from_numpy(ex.image).permute(2, 0, 1).float() / 255.0
        target_xy = torch.tensor(ex.objects[ex.target_index].xy, dtype=torch.float32) / self.size
        return {
            "image": img,
            "instr": instr_vec(ex.target_combo),
            "target_xy": target_xy,
            "example": ex,
        }


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "instr": torch.stack([b["instr"] for b in batch]),
        "target_xy": torch.stack([b["target_xy"] for b in batch]),
        "examples": [b["example"] for b in batch],
    }


class PixelXYNet(nn.Module):
    def __init__(self, instr_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(24, 48, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(48, 96, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + instr_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid(),
        )

    def forward(self, image, instr):
        z = self.cnn(image).flatten(1)
        return self.head(torch.cat([z, instr], dim=-1))


def xy_to_pred_index(xy: torch.Tensor, examples: List) -> torch.Tensor:
    preds = []
    xy_np = xy.detach().cpu()
    for b, ex in enumerate(examples):
        dists = []
        for o in ex.objects:
            ox = torch.tensor([o.xy[0] / ex.image.shape[0], o.xy[1] / ex.image.shape[0]])
            dists.append(torch.norm(xy_np[b] - ox).item())
        preds.append(int(torch.tensor(dists).argmin().item()))
    return torch.tensor(preds, dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    counters = {"success": 0, "wrong_object": 0, "wrong_color": 0, "wrong_shape": 0}
    total = 0
    loss_sum = 0.0
    for b in loader:
        image = b["image"].to(device)
        instr = b["instr"].to(device)
        target_xy = b["target_xy"].to(device)
        pred_xy = model(image, instr)
        loss = F.mse_loss(pred_xy, target_xy)
        pred_idx = xy_to_pred_index(pred_xy, b["examples"])
        loss_sum += loss.item() * image.size(0)
        for i, ex in enumerate(b["examples"]):
            diag = diagnose_prediction(ex, int(pred_idx[i]))
            for k in counters:
                counters[k] += int(bool(diag[k]))
        total += image.size(0)
    rates = {k: v / total for k, v in counters.items()}
    rates["loss"] = loss_sum / total
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/image_binding_pixel_v0/summary.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = PixelBindingDataset("train", args.n_train, args.seed)
    id_ds = PixelBindingDataset("id", args.n_eval, args.seed + 1)
    ood_ds = PixelBindingDataset("ood", args.n_eval, args.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    eval_loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        "id": DataLoader(id_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        "ood": DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
    }
    model = PixelXYNet(N_ATTR).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for b in train_loader:
            image = b["image"].to(device)
            instr = b["instr"].to(device)
            target_xy = b["target_xy"].to(device)
            pred_xy = model(image, instr)
            loss = F.mse_loss(pred_xy, target_xy)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * image.size(0); n += image.size(0)
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1} train_mse={total/max(1,n):.5f}")
    res = {split: evaluate(model, loader, device) for split, loader in eval_loaders.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for split, r in res.items():
        print(f"pixel_xy {split:>5} success={r['success']:.3f} wrong_object={r['wrong_object']:.3f} wrong_color={r['wrong_color']:.3f} wrong_shape={r['wrong_shape']:.3f} loss={r['loss']:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
