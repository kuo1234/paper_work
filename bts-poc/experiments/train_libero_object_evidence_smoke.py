from __future__ import annotations

"""Minimal LIBERO object-evidence classifier smoke.

Purpose: establish the data-loading/training/eval pipeline for future visual/object evidence
modules. This is not the final BTS evidence model. It trains a tiny image classifier to predict
the target object label for collected LIBERO-Object frames, mainly verifying that images,
metadata, labels, and train/eval code all work.

Run on spark after collecting evidence data:

    cd ~/bts-poc/experiments
    ~/openvla-spark/.venv/bin/python train_libero_object_evidence_smoke.py \
      --data-dir runs/libero_object_evidence_v0 --epochs 20 --out runs/libero_object_evidence_smoke.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_rows(data_dir: Path) -> List[Dict]:
    meta = data_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {meta}")
    return rows


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def make_arrays(data_dir: Path, rows: List[Dict], image_size: int):
    labels = sorted({r["target_key_name"] for r in rows})
    label_to_id = {k: i for i, k in enumerate(labels)}
    xs, ys, keys = [], [], []
    for r in rows:
        xs.append(load_image(data_dir / r["image"], image_size))
        ys.append(label_to_id[r["target_key_name"]])
        keys.append((r["task_id"], r["init_id"], r["target_key_name"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64), labels, keys


def split_by_init(keys: List[Tuple[int, int, str]], test_init_mod: int = 4):
    # deterministic split: hold out init_id % 4 == 0 as test, unless that would be empty
    test = np.asarray([init_id % test_init_mod == 0 for _task_id, init_id, _target in keys])
    if test.all() or not test.any():
        n = len(keys)
        test = np.zeros(n, dtype=bool)
        test[::5] = True
    train = ~test
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("runs/libero_object_evidence_v0"))
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("runs/libero_object_evidence_smoke.json"))
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    rows = load_rows(args.data_dir)
    x, y, labels, keys = make_arrays(args.data_dir, rows, args.image_size)
    train_mask, test_mask = split_by_init(keys)

    x_train = torch.tensor(x[train_mask])
    y_train = torch.tensor(y[train_mask])
    x_test = torch.tensor(x[test_mask])
    y_test = torch.tensor(y[test_mask])

    class TinyEvidenceCNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.fc = nn.Linear(64, n_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyEvidenceCNN(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            tr_pred = model(x_train.to(device)).argmax(dim=-1).cpu()
            te_pred = model(x_test.to(device)).argmax(dim=-1).cpu()
        train_acc = float((tr_pred == y_train).float().mean()) if len(y_train) else None
        test_acc = float((te_pred == y_test).float().mean()) if len(y_test) else None
        rec = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "train_acc": train_acc, "test_acc": test_acc}
        history.append(rec)
        print(rec)

    result = {
        "n": len(rows),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "labels": labels,
        "device": device,
        "image_size": args.image_size,
        "history": history,
        "final": history[-1] if history else None,
        "note": "Pipeline smoke only; image->task target classification is not the final BTS evidence model.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
