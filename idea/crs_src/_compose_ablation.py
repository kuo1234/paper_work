#!/usr/bin/env python3
"""2x2 ablation (gate-base x box-base) proving the composition mechanism is real
complementarity, not just 'GDINO boxes are good'. Run on val (extend to 3 splits
when dumps land). a=b=0.3 dual guarantee, bright Pareto point.

Expected (and confirmed on val):
  box axis  -> GD box gives small sets, OWL box gives large sets
  gate axis -> OWL gate gives low abstention & holds R2; GD gate fails one
  COMPOSE (OWL gate + GD box) = best of both; reverse = worst of both (diagonal).
"""
import json, os, numpy as np
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

def read(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        if r["prompt_variant"] != "canonical": continue
        d[(r["ref_id"], r["sent_id"])] = r
    return d

def build(sp):
    op = f"{B}/owlvit_gref_{sp}.jsonl"; gp = f"{B}/gdino_gref_{sp}.jsonl"
    if not (os.path.exists(op) and os.path.exists(gp)): return None
    owl = read(op); gd = read(gp); keys = sorted(set(owl) & set(gd))
    if len(keys) < 200: return None
    recs = []
    for k in keys:
        o = owl[k]; g = gd[k]
        recs.append(dict(ref_id=k[0], no_target=o["no_target"],
            tp=(not o["no_target"]) and bool(g["gt_boxes_xyxy"]), gt=g["gt_boxes_xyxy"],
            g_owl=float(o["top1_score"]), g_gd=float(g["top1_score"]),
            gboxes=g["pred_boxes_xyxy"], gscores=np.array(g["pred_scores"]),
            oboxes=o["pred_boxes_xyxy"], oscores=np.array(o["pred_scores"])))
    return recs

def fnr(s, gt):
    return 1.0 - sum(1 for q in gt if any(iou(p, q) >= 0.5 for p in s))/len(gt)

def emp(rows, tau, lam, gate, bxk, sck):
    fn, nt, ab, sz = [], [], [], []
    for r in rows:
        sel = [] if r[gate] <= tau else [r[bxk][i] for i in range(len(r[bxk])) if r[sck][i] >= lam]
        if r["tp"]:
            if r[gate] <= tau: ab.append(1.)
            else: ab.append(0.); fn.append(fnr(sel, r["gt"])); sz.append(len(sel))
        elif r["no_target"]: nt.append(1. if len(sel) > 0 else 0.)
    return (np.mean(fn) if fn else 0., len(fn), np.mean(nt) if nt else 0., len(nt),
            np.mean(ab) if ab else 0., np.mean(sz) if sz else 0.)

def run(recs, gate, bxk, sck, gpk, label):
    calib = [r for r in recs if r["ref_id"] % 2 == 0]
    test = [r for r in recs if r["ref_id"] % 2 == 1]
    sc = np.concatenate([r[sck] for r in recs]); lams = np.quantile(sc, np.linspace(0, 0.99, 30))
    gp = np.array([r[gpk] for r in recs]); taus = np.quantile(gp, np.linspace(0, 0.95, 30))
    th = 0.1/(len(taus)*len(lams)); cands = []
    for tA in taus:
        for L in lams:
            R1, n1, R2, n2, ab, szc = emp(calib, tA, L, gate, bxk, sck)
            if hoeffding_p(R1, n1, 0.3) <= th and hoeffding_p(R2, n2, 0.3) <= th:
                cands.append((ab, szc, tA, L))
    if not cands:
        print(f"  {label}: EMPTY"); return
    br = [c for c in cands if c[0] <= 0.5] or cands
    _, _, tau, lam = min(br, key=lambda c: c[1])
    R1, _, R2, _, ab, sz = emp(test, tau, lam, gate, bxk, sck)
    print(f"  {label}: sz={sz:.2f} R1={R1:.3f} R2={R2:.3f} abst={ab:.3f}")

def main():
    for sp in ["val", "testA", "testB"]:
        recs = build(sp)
        if recs is None:
            print(f"\n[{sp}] dumps not ready"); continue
        print(f"\n[{sp}] 2x2 ablation (gate x box), a=b=0.3 dual guarantee:")
        run(recs, "g_owl", "oboxes", "oscores", "g_owl", "OWLgate+OWLbox (pure OWL)")
        run(recs, "g_gd",  "gboxes", "gscores", "g_gd",  "GDgate +GDbox  (pure GD) ")
        run(recs, "g_owl", "gboxes", "gscores", "g_owl", "OWLgate+GDbox  (COMPOSE) ")
        run(recs, "g_gd",  "oboxes", "oscores", "g_gd",  "GDgate +OWLbox (reverse) ")

if __name__ == "__main__":
    main()
