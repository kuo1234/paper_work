#!/usr/bin/env python3
"""Comprehensive CRS result table across both bases x 3 splits.
- target-present: irreducible FNR floor + set size at FNR<=alpha (CRC)
- multi-risk LTT: joint (R1<=alpha, R2<=beta) feasibility + best operating point
Outputs a compact text table -> for the paper's main results.
Runs whatever dumps exist (skips missing splits gracefully).
"""
import json, os, numpy as np
import sys
sys.path.insert(0, "/home/p76141495/selective-grounding/src")
from ltt import load, empirical_risks, ltt_feasible

B = "/home/p76141495/selective-grounding/dump"

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
        r["_sc"] = np.array(r["pred_scores"]); rows.append(r)
    return rows

def fnr(sel, gt):
    return 1.0 - sum(1 for g in gt if any(iou(p, g) >= 0.5 for p in sel))/len(gt)
def sel(r, lam):
    bx = r["pred_boxes_xyxy"]; sc = r["_sc"]
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]
def risk(rows, lam):
    return np.mean([fnr(sel(r, lam), r["gt_boxes_xyxy"]) for r in rows])
def size(rows, lam):
    return np.mean([int(np.sum(r["_sc"] >= lam)) for r in rows])
def crc(calib, a, grid):
    n = len(calib); B_ = 1.0
    for lam in sorted(grid, reverse=True):
        if (n*risk(calib, lam)+B_)/(n+1) <= a: return lam
    return min(grid)

def tp_block(path, label):
    rows = load_tp(path)
    if len(rows) < 60:
        print(f"  {label}: skip (n={len(rows)})"); return
    calib = [r for r in rows if r["ref_id"] % 2 == 0]
    test  = [r for r in rows if r["ref_id"] % 2 == 1]
    grid = np.quantile(np.concatenate([r["_sc"] for r in rows]), np.linspace(0, 0.999, 100))
    floor = risk(test, grid.min())
    cells = []
    for a in [0.1, 0.2, 0.3]:
        lam = crc(calib, a, grid)
        cells.append(f"a{a}: FNR={risk(test,lam):.3f} sz={size(test,lam):.1f}")
    print(f"  {label:28s} floor={floor:.3f} | " + " | ".join(cells))

def ltt_block(path, label):
    rows = load(path)
    if len(rows) < 200:
        print(f"  {label}: skip (n={len(rows)})"); return
    calib = [r for r in rows if r["ref_id"] % 2 == 0]
    test  = [r for r in rows if r["ref_id"] % 2 == 1]
    g_all = np.concatenate([r["_sc"] for r in rows])
    lams = np.quantile(g_all, np.linspace(0.0, 0.99, 25))
    gv = np.array([r["_g"] for r in rows])
    taus = np.quantile(gv, np.linspace(0.0, 0.95, 25))
    for a, b in [(0.3, 0.2)]:
        valid = ltt_feasible(calib, a, b, taus, lams, delta=0.1)
        if not valid:
            print(f"  {label:28s} (a={a},b={b}): feasible EMPTY"); continue
        tA, L, _ = min(valid, key=lambda c: c[2])
        R1, n1, R2, n2, ab = empirical_risks(test, tA, L)
        sizes = [sum(1 for i in range(len(r["pred_boxes_xyxy"])) if r["_sc"][i] >= L)
                 for r in test if r["_tp"] and r["_g"] > tA]
        sz = np.mean(sizes) if sizes else 0.0
        print(f"  {label:28s} (a={a},b={b}): R1={R1:.3f} R2={R2:.3f} "
              f"tp_abst={ab:.3f} sz={sz:.1f} (#feas={len(valid)})")

def main():
    bases = [("owlvit", "OWL-ViT"), ("gdino", "GroundingDINO")]
    splits = ["val", "testA", "testB"]
    print("="*70)
    print("TARGET-PRESENT subproblem (recall guarantee, set size):")
    for tag, name in bases:
        for sp in splits:
            p = f"{B}/{tag}_gref_{sp}.jsonl" if tag == "gdino" else f"{B}/owlvit_gref_{sp}.jsonl"
            if os.path.exists(p): tp_block(p, f"{name} {sp}")
    print("="*70)
    print("MULTI-RISK LTT (joint recall+abstention, best operating point):")
    for tag, name in bases:
        for sp in splits:
            p = f"{B}/{tag}_gref_{sp}.jsonl" if tag == "gdino" else f"{B}/owlvit_gref_{sp}.jsonl"
            if os.path.exists(p): ltt_block(p, f"{name} {sp}")
    print("="*70)

if __name__ == "__main__":
    main()
