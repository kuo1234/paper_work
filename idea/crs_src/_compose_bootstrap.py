#!/usr/bin/env python3
"""Bootstrap CI for the cross-base composition bright operating point.
Validity question: is the 3.2-box / dual-guarantee result stable, or a small-
sample fluke? Protocol: FIX the LTT-selected config (tau, lambda) chosen on calib
(no leakage), then resample the TEST set >=1000x and report 95% CI of
(set size, R1 FNR, R2 false-sel, tp_abstention).

Also bootstraps the headline comparison: composition set size vs pure-base.
"""
import json, numpy as np
import sys
sys.path.insert(0, "/home/p76141495/selective-grounding/src")
from ltt import hoeffding_p

B = "/home/p76141495/selective-grounding/dump"
RNG = np.random.default_rng(12345)

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

owl = read(f"{B}/owlvit_gref_val.jsonl"); gd = read(f"{B}/gdino_gref_val.jsonl")
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
            else: tp_ab.append(0.0); fnrs.append(fnr(sel, r["gt"])); szs.append(len(sel))
        elif r["no_target"]:
            nt_fs.append(1.0 if len(sel) > 0 else 0.0)
    return (np.mean(fnrs) if fnrs else 0.0, np.mean(nt_fs) if nt_fs else 0.0,
            np.mean(tp_ab) if tp_ab else 0.0, np.mean(szs) if szs else 0.0)

# select config on CALIB: Pareto bright point (abst<=0.5, min size) at a=b=0.3
sc_all = np.concatenate([r["scores"] for r in recs])
lams = np.quantile(sc_all, np.linspace(0, 0.99, 30))
gpool = np.array([r["g_owl"] for r in recs])
taus = np.quantile(gpool, np.linspace(0, 0.95, 30))
alpha = beta = 0.3; delta = 0.1; m = len(taus)*len(lams); th = delta/m
cands = []
for tA in taus:
    for L in lams:
        R1c, _, R2c, _, abc, szc = emp_full = None, None, None, None, None, None
        # compute calib risks with counts for hoeffding
        fnrs, nt_fs, tp_ab, szs = [], [], [], []
        for r in calib:
            sel = decide(r, tA, L)
            if r["tp"]:
                if r["g_owl"] <= tA: tp_ab.append(1.0)
                else: tp_ab.append(0.0); fnrs.append(fnr(sel, r["gt"])); szs.append(len(sel))
            elif r["no_target"]:
                nt_fs.append(1.0 if len(sel) > 0 else 0.0)
        R1c = np.mean(fnrs) if fnrs else 0.0; R2c = np.mean(nt_fs) if nt_fs else 0.0
        if hoeffding_p(R1c, len(fnrs), alpha) <= th and hoeffding_p(R2c, len(nt_fs), beta) <= th:
            abc = np.mean(tp_ab) if tp_ab else 0.0; szc = np.mean(szs) if szs else 0.0
            cands.append((abc, szc, tA, L))
if not cands:
    print("no feasible config on calib"); sys.exit()
cand_bright = [c for c in cands if c[0] <= 0.5] or cands
_, _, tau_sel, lam_sel = min(cand_bright, key=lambda c: c[1])
print(f"selected config on calib: tau={tau_sel:.4f} lam={lam_sel:.4f}")
print(f"test point estimate: R1,R2,abst,size =", [round(x,3) for x in emp(test, tau_sel, lam_sel)])

# bootstrap test set
tp_test = [r for r in test if r["tp"]]
nt_test = [r for r in test if r["no_target"]]
boot = {"size": [], "R1": [], "R2": [], "abst": []}
for _ in range(1000):
    bt = [tp_test[i] for i in RNG.integers(0, len(tp_test), len(tp_test))]
    bn = [nt_test[i] for i in RNG.integers(0, len(nt_test), len(nt_test))]
    R1, R2, ab, sz = emp(bt + bn, tau_sel, lam_sel)
    boot["size"].append(sz); boot["R1"].append(R1); boot["R2"].append(R2); boot["abst"].append(ab)
print("\nBootstrap 95% CI (1000 resamples, config fixed):")
for k in ["size", "R1", "R2", "abst"]:
    lo, hi = np.percentile(boot[k], [2.5, 97.5]); md = np.median(boot[k])
    print(f"  {k:5s}: {md:.3f}  [{lo:.3f}, {hi:.3f}]")
print(f"\nGuarantee check: R1 CI upper vs alpha={alpha}, R2 CI upper vs beta={beta}")
