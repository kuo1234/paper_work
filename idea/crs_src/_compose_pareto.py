#!/usr/bin/env python3
"""Inspect the FULL feasible region (not just min-abstention point) for the
composition policy, to find Pareto-optimal (tp_abstention, set_size) operating
points. The earlier degenerate sz=45 was a SELECTION artifact: among many valid
configs we picked min-abstention, which happened to use the loosest lambda.
Here we print the Pareto frontier of the joint feasible region."""
import json, numpy as np
import sys
sys.path.insert(0, "/home/p76141495/selective-grounding/src")
from ltt import hoeffding_p

B = "/home/p76141495/selective-grounding/dump"

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0., ix1-ix0), max(0., iy1-iy0); inter = iw*ih
    aa = max(0., ax1-ax0)*max(0., ay1-ay0); ab = max(0., bx1-bx0)*max(0., by1-by0)
    u = aa+ab-inter
    return inter/u if u > 0 else 0.

def read(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        if r["prompt_variant"] != "canonical": continue
        d[(r["ref_id"], r["sent_id"])] = r
    return d

owl = read(f"{B}/owlvit_gref_val.jsonl")
gd  = read(f"{B}/gdino_gref_val.jsonl")
keys = sorted(set(owl) & set(gd))
recs = []
for k in keys:
    o = owl[k]; g = gd[k]
    recs.append(dict(ref_id=k[0], no_target=o["no_target"],
        tp=(not o["no_target"]) and bool(g["gt_boxes_xyxy"]), gt=g["gt_boxes_xyxy"],
        g_owl=float(o["top1_score"]), boxes=g["pred_boxes_xyxy"],
        scores=np.array(g["pred_scores"])))

calib = [r for r in recs if r["ref_id"] % 2 == 0]
test  = [r for r in recs if r["ref_id"] % 2 == 1]

def fnr(sel, gt):
    return 1.0 - sum(1 for q in gt if any(iou(p, q) >= 0.5 for p in sel))/len(gt)
def decide(r, tau, lam):
    if r["g_owl"] <= tau: return []
    bx = r["boxes"]; sc = r["scores"]
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]
def emp(rows, tau, lam):
    fnrs, nt_fs, tp_ab, szs = [], [], [], []
    for r in rows:
        sel = decide(r, tau, lam)
        if r["tp"]:
            if r["g_owl"] <= tau: tp_ab.append(1.0)
            else:
                tp_ab.append(0.0); fnrs.append(fnr(sel, r["gt"])); szs.append(len(sel))
        elif r["no_target"]:
            nt_fs.append(1.0 if len(sel) > 0 else 0.0)
    return (np.mean(fnrs) if fnrs else 0.0, len(fnrs),
            np.mean(nt_fs) if nt_fs else 0.0, len(nt_fs),
            np.mean(tp_ab) if tp_ab else 0.0, np.mean(szs) if szs else 0.0)

sc_all = np.concatenate([r["scores"] for r in recs])
lams = np.quantile(sc_all, np.linspace(0, 0.99, 30))
gpool = np.array([r["g_owl"] for r in recs])
taus = np.quantile(gpool, np.linspace(0, 0.95, 30))

alpha, beta, delta = 0.3, 0.3, 0.1
m = len(taus)*len(lams); th = delta/m
# collect all feasible configs with their TEST operating points
pts = []
for tA in taus:
    for L in lams:
        R1c, n1, R2c, n2, abc, _ = emp(calib, tA, L)
        if hoeffding_p(R1c, n1, alpha) <= th and hoeffding_p(R2c, n2, beta) <= th:
            R1, _, R2, _, ab, sz = emp(test, tA, L)
            pts.append((ab, sz, R1, R2, tA, L))

print(f"composition feasible configs: {len(pts)} (a={alpha},b={beta})")
if pts:
    # Pareto frontier on (abstention, size) both-minimize
    pts.sort()
    frontier = []
    best_sz = 1e9
    for ab, sz, R1, R2, tA, L in pts:
        if sz < best_sz - 1e-9:
            frontier.append((ab, sz, R1, R2)); best_sz = sz
    print("Pareto frontier (tp_abstention up -> set_size down):")
    print(f"  {'tp_abst':>8} {'set_sz':>7} {'R1':>6} {'R2':>6}")
    for ab, sz, R1, R2 in frontier:
        print(f"  {ab:>8.3f} {sz:>7.2f} {R1:>6.3f} {R2:>6.3f}")
    # the bright point: lowest size with abstention <= 0.5
    cand = [p for p in pts if p[0] <= 0.5]
    if cand:
        ab, sz, R1, R2, tA, L = min(cand, key=lambda p: p[1])
        print(f"\nBRIGHT POINT (abst<=0.5, min size): abst={ab:.3f} sz={sz:.2f} R1={R1:.3f} R2={R2:.3f}")
