from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from envs.image_binding import ControlledImageBindingBenchmark, COLORS, SHAPES, diagnose_prediction

COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS.keys())}
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
N_COLORS = len(COLOR_TO_IDX)
N_SHAPES = len(SHAPE_TO_IDX)


def combo_vec(color: str, shape: str) -> torch.Tensor:
    v = torch.zeros(N_COLORS + N_SHAPES)
    v[COLOR_TO_IDX[color]] = 1.0
    v[N_COLORS + SHAPE_TO_IDX[shape]] = 1.0
    return v


def object_features(ex, size: int) -> torch.Tensor:
    feats = []
    for o in ex.objects:
        xy = torch.tensor([o.xy[0] / size, o.xy[1] / size], dtype=torch.float32)
        feats.append(torch.cat([xy, combo_vec(o.color, o.shape)], dim=0))
    return torch.stack(feats)


class BindingDataset(Dataset):
    def __init__(self, split: str, n: int, seed: int):
        self.env = ControlledImageBindingBenchmark(seed=seed)
        self.examples = [self.env.sample(split) for _ in range(n)]
        self.size = self.env.size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        instr = combo_vec(*ex.target_combo)
        obj = object_features(ex, self.size)
        target_xy = torch.tensor(ex.objects[ex.target_index].xy, dtype=torch.float32) / self.size
        return {
            "instr": instr,
            "objects": obj,
            "target_idx": torch.tensor(ex.target_index, dtype=torch.long),
            "target_xy": target_xy,
            "example": ex,
        }


def collate(batch):
    return {
        "instr": torch.stack([b["instr"] for b in batch]),
        "objects": torch.stack([b["objects"] for b in batch]),
        "target_idx": torch.stack([b["target_idx"] for b in batch]),
        "target_xy": torch.stack([b["target_xy"] for b in batch]),
        "examples": [b["example"] for b in batch],
    }


class ShortcutXYNet(nn.Module):
    """Shortcut-prone baseline: instruction -> target xy.

    It cannot inspect objects. With train-time color-location correlation, it learns the
    shortcut and fails when OOD breaks the correlation.
    """

    def __init__(self, instr_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(instr_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, instr, objects=None):
        return self.net(instr)


class StructuredBeliefNet(nn.Module):
    """Structured BTS belief: score objects by decomposed attribute matches.

    Unlike a generic MLP scorer, this cannot memorize seen color-shape pairs as atomic
    labels. It learns weights for color and shape agreement, so held-out compositions
    remain solvable.
    """

    def __init__(self):
        super().__init__()
        self.color_weight = nn.Parameter(torch.tensor(1.0))
        self.shape_weight = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, instr, objects):
        obj_attr = objects[..., 2:]
        instr_attr = instr.unsqueeze(1)
        color_match = (obj_attr[..., :N_COLORS] * instr_attr[..., :N_COLORS]).sum(dim=-1)
        shape_match = (obj_attr[..., N_COLORS:] * instr_attr[..., N_COLORS:]).sum(dim=-1)
        return self.color_weight * color_match + self.shape_weight * shape_match + self.bias


class CandidateBeliefNet(nn.Module):
    """Generic candidate belief baseline, kept for diagnosis.

    This MLP can memorize train-time pair labels and fail on held-out compositions.
    StructuredBeliefNet is the BTS variant used for the main result.
    """

    def __init__(self, instr_dim: int, obj_dim: int):
        super().__init__()
        self.obj_proj = nn.Sequential(nn.Linear(obj_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.instr_proj = nn.Sequential(nn.Linear(instr_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.scorer = nn.Sequential(nn.Linear(64 * 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, instr, objects):
        # instr [B,D], objects [B,N,F]
        B, N, _ = objects.shape
        oi = self.obj_proj(objects)
        qi = self.instr_proj(instr).unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([oi, qi, oi * qi], dim=-1)
        return self.scorer(x).squeeze(-1)


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
def evaluate(model, loader, model_type: str, device):
    model.eval()
    total = 0
    counters = {"success": 0, "wrong_object": 0, "wrong_color": 0, "wrong_shape": 0}
    loss_sum = 0.0
    for b in loader:
        instr = b["instr"].to(device)
        objects = b["objects"].to(device)
        target_idx = b["target_idx"].to(device)
        target_xy = b["target_xy"].to(device)
        if model_type == "shortcut":
            pred_xy = model(instr)
            loss = F.mse_loss(pred_xy, target_xy)
            pred_idx = xy_to_pred_index(pred_xy, b["examples"])
        else:
            logits = model(instr, objects)
            loss = F.cross_entropy(logits, target_idx)
            pred_idx = logits.argmax(dim=-1).cpu()
        loss_sum += loss.item() * instr.size(0)
        for i, ex in enumerate(b["examples"]):
            diag = diagnose_prediction(ex, int(pred_idx[i]))
            for k in counters:
                counters[k] += int(bool(diag[k]))
        total += instr.size(0)
    rates = {k: v / total for k, v in counters.items()}
    rates["loss"] = loss_sum / total
    return rates


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = BindingDataset("train", args.n_train, args.seed)
    id_ds = BindingDataset("id", args.n_eval, args.seed + 1)
    ood_ds = BindingDataset("ood", args.n_eval, args.seed + 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    id_loader = DataLoader(id_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    instr_dim = N_COLORS + N_SHAPES
    obj_dim = 2 + instr_dim
    models = {
        "shortcut": ShortcutXYNet(instr_dim).to(device),
        "generic_belief": CandidateBeliefNet(instr_dim, obj_dim).to(device),
        "bts_structured": StructuredBeliefNet().to(device),
    }
    results: Dict[str, Dict] = {}
    for name, model in models.items():
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(args.epochs):
            model.train()
            for b in train_loader:
                instr = b["instr"].to(device)
                objects = b["objects"].to(device)
                target_idx = b["target_idx"].to(device)
                target_xy = b["target_xy"].to(device)
                if name == "shortcut":
                    pred_xy = model(instr)
                    loss = F.mse_loss(pred_xy, target_xy)
                else:
                    logits = model(instr, objects)
                    loss = F.cross_entropy(logits, target_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
        results[name] = {
            "train": evaluate(model, train_loader, "shortcut" if name == "shortcut" else "belief", device),
            "id": evaluate(model, id_loader, "shortcut" if name == "shortcut" else "belief", device),
            "ood": evaluate(model, ood_loader, "shortcut" if name == "shortcut" else "belief", device),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/image_binding_v0_train/summary.json"))
    args = ap.parse_args()
    res = train(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for model_name, splits in res.items():
        for split, r in splits.items():
            print(f"{model_name:>10} {split:>5} success={r['success']:.3f} wrong_object={r['wrong_object']:.3f} wrong_color={r['wrong_color']:.3f} wrong_shape={r['wrong_shape']:.3f} loss={r['loss']:.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
