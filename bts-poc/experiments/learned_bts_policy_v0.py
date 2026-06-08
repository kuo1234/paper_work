from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from envs.image_binding import ControlledImageBindingBenchmark, extract_image_candidates, diagnose_prediction, COLORS, SHAPES
from experiments.image_binding_corruption_v0 import corrupt_image, parse_instruction
from experiments.train_patch_classifier_v0 import PatchClassifier, PatchDataset, crop_patch, COLOR_TO_IDX, SHAPE_TO_IDX

IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_SHAPE = {v: k for k, v in SHAPE_TO_IDX.items()}


def train_classifier(args, device):
    train_ds = ConcatDataset([
        PatchDataset(args.n_train_scenes, args.seed, "train", 0.0, 0),
        PatchDataset(args.n_train_scenes, args.seed + 10, "train", 20.0, 0),
        PatchDataset(args.n_train_scenes, args.seed + 20, "train", 0.0, 8),
        PatchDataset(args.n_train_scenes, args.seed + 30, "train", 20.0, 8),
    ])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = PatchClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train(); total = 0; n = 0
        for x, c, s in loader:
            x, c, s = x.to(device), c.to(device), s.to(device)
            lc, ls = model(x)
            loss = F.cross_entropy(lc, c) + F.cross_entropy(ls, s)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        if epoch in {0, args.epochs-1}:
            print(f"classifier epoch {epoch+1} loss={total/n:.4f}")
    return model


@torch.no_grad()
def learned_bts_policy(ex, image, clf, device):
    target_color, target_shape = parse_instruction(ex.instruction)
    cands = extract_image_candidates(image)
    if not cands:
        return 0, []
    patches = []
    for c in cands:
        patches.append(torch.from_numpy(crop_patch(image, c.xy)).permute(2,0,1).float()/255.0)
    x = torch.stack(patches).to(device)
    color_logits, shape_logits = clf(x)
    color_lp = F.log_softmax(color_logits, dim=-1)
    shape_lp = F.log_softmax(shape_logits, dim=-1)
    score = color_lp[:, COLOR_TO_IDX[target_color]] + shape_lp[:, SHAPE_TO_IDX[target_shape]]
    pred_cand = int(score.argmax().item())
    dists = [np.hypot(cands[pred_cand].xy[0] - o.xy[0], cands[pred_cand].xy[1] - o.xy[1]) for o in ex.objects]
    return int(np.argmin(dists)), cands


def eval_setting(args, clf, device, split, noise_std, occluder, seed):
    env = ControlledImageBindingBenchmark(seed=seed)
    rng = random.Random(seed + 999)
    counters = {"success": 0, "wrong_object": 0, "wrong_color": 0, "wrong_shape": 0}
    cand_total = 0; det_all = 0
    for _ in range(args.n_eval):
        ex = env.sample(split)
        image = corrupt_image(ex.image, rng, noise_std, occluder)
        pred, cands = learned_bts_policy(ex, image, clf, device)
        diag = diagnose_prediction(ex, pred)
        for k in counters:
            counters[k] += int(bool(diag[k]))
        cand_total += len(cands); det_all += int(len(cands) == len(ex.objects))
    out = {k: v / args.n_eval for k, v in counters.items()}
    out["avg_candidates"] = cand_total / args.n_eval
    out["detected_all"] = det_all / args.n_eval
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-scenes", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/image_binding_learned_bts_v0/summary.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    clf = train_classifier(args, device)
    settings = {
        "clean": (0.0, 0),
        "noise20": (20.0, 0),
        "noise40": (40.0, 0),
        "occ12": (0.0, 12),
        "noise20_occ12": (20.0, 12),
    }
    res = {name: eval_setting(args, clf, device, "ood", ns, occ, args.seed + i + 100) for i, (name, (ns, occ)) in enumerate(settings.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for name, r in res.items():
        print(f"{name:>14} success={r['success']:.3f} wrong_object={r['wrong_object']:.3f} wrong_color={r['wrong_color']:.3f} wrong_shape={r['wrong_shape']:.3f} detected_all={r['detected_all']:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
