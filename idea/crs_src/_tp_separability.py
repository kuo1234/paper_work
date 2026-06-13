#!/usr/bin/env python3
"""Target-present subproblem: for each base, what set size is needed to reach
FNR<=alpha (recall guarantee) on target samples? This isolates TP/FP score
separability from the no-target gate. GroundingDINO may need MUCH smaller sets
even though its no-target gate is weak -> a bright operating point on the
target-present problem.

Uses whatever rows exist (GDINO partial OK). ref_id parity calib/test.
Reports CRC lambda + empirical FNR + mean set size, per base, per alpha.
"""
import json, numpy as np

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0., ix1-ix0), max(0., iy1-iy0); inter = iw*ih
    aa = max(0., ax1-ax0)*max(0., ay1-ay0); ab = max(0., bx1-bx0)*max(0., by1-by0)
    u = aa+ab-inter
    return inter/u if u > 0 else 0.

def load_tp(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if r.get("prompt_variant") != "canonical": continue
        if r["no_target"] or not r["gt_boxes_xyxy"]: continue
        r["_sc"] = np.array(r["pred_scores"])
        rows.append(r)
    return rows

def fnr(sel, gt):
    cov = sum(1 for g in gt if any(iou(p, g) >= 0.5 for p in sel))
    return 1.0 - cov/len(gt)

def sel(r, lam):
    bx = r["pred_boxes_xyxy"]; sc = r["_sc"]
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]

def risk(rows, lam):
    return np.mean([fnr(sel(r, lam), r["gt_boxes_xyxy"]) for r in rows])

def size(rows, lam):
    return np.mean([int(np.sum(r["_sc"] >= lam)) for r in rows])

def crc(calib, alpha, grid):
    n = len(calib); B = 1.0
    for lam in sorted(grid, reverse=True):
        if (n*risk(calib, lam)+B)/(n+1) <= alpha:
            return lam
    return min(grid)

def run(path, label):
    rows = load_tp(path)
    calib = [r for r in rows if r["ref_id"] % 2 == 0]
    test  = [r for r in rows if r["ref_id"] % 2 == 1]
    if len(test) < 30:
        print(f"{label}: too few test rows ({len(test)})"); return
    grid = np.quantile(np.concatenate([r["_sc"] for r in rows]), np.linspace(0, 0.999, 100))
    # irreducible floor: FNR at lam=min (select all candidates)
    floor = risk(test, grid.min())
    print(f"\n=== {label}: tp_rows={len(rows)} (calib={len(calib)} test={len(test)}) "
          f"irreducible-FNR(select-all)={floor:.3f} ===")
    print(f"  {'alpha':>5} | {'emp_FNR':>7} {'set_size':>8}")
    for a in [0.1, 0.2, 0.3, 0.4]:
        lam = crc(calib, a, grid)
        print(f"  {a:>5.2f} | {risk(test,lam):>7.4f} {size(test,lam):>8.2f}")

if __name__ == "__main__":
    B = "/home/p76141495/selective-grounding/dump"
    run(f"{B}/owlvit_gref_val.jsonl", "OWL-ViT val (full)")
    run(f"{B}/gdino_gref_val.jsonl", "GroundingDINO val (partial)")
