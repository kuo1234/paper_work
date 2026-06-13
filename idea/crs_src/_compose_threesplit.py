#!/usr/bin/env python3
"""Three-split cross-base composition: run compose bright-point + bootstrap CI
on val/testA/testB wherever both owlvit and gdino dumps exist. Produces the
paper's main composition table. Safe to run repeatedly as dumps land.
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

def read(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        if r["prompt_variant"] != "canonical": continue
        d[(r["ref_id"], r["sent_id"])] = r
    return d

def build(split):
    op = f"{B}/owlvit_gref_{split}.jsonl"; gp = f"{B}/gdino_gref_{split}.jsonl"
    if not (os.path.exists(op) and os.path.exists(gp)): return None
    owl = read(op); gd = read(gp)
    keys = sorted(set(owl) & set(gd))
    if len(keys) < 200: return None
    recs = []
    for k in keys:
        o = owl[k]; g = gd[k]
        recs.append(dict(ref_id=k[0], no_target=o["no_target"],
            tp=(not o["no_target"]) and bool(g["gt_boxes_xyxy"]), gt=g["gt_boxes_xyxy"],
            g_owl=float(o["top1_score"]), boxes=g["pred_boxes_xyxy"],
            scores=np.array(g["pred_scores"]),
            o_boxes=o["pred_boxes_xyxy"], o_scores=np.array(o["pred_scores"])))
    return recs

def fnr(sel, gt):
    return 1.0 - sum(1 for q in gt if any(iou(p, q) >= 0.5 for p in sel))/len(gt)

def emp(rows, tau, lam, gate, bxk, sck):
    fnrs, nt_fs, tp_ab, szs = [], [], [], []
    for r in rows:
        if r[gate] <= tau: sel = []
        else:
            bx = r[bxk]; sc = r[sck]
            sel = [bx[i] for i in range(len(bx)) if sc[i] >= lam]
        if r["tp"]:
            if r[gate] <= tau: tp_ab.append(1.0)
            else: tp_ab.append(0.0); fnrs.append(fnr(sel, r["gt"])); szs.append(len(sel))
        elif r["no_target"]:
            nt_fs.append(1.0 if len(sel) > 0 else 0.0)
    return (np.mean(fnrs) if fnrs else 0.0, len(fnrs),
            np.mean(nt_fs) if nt_fs else 0.0, len(nt_fs),
            np.mean(tp_ab) if tp_ab else 0.0, np.mean(szs) if szs else 0.0)

def select_and_eval(recs, gate, bxk, sck, alpha=0.3, beta=0.3, label=""):
    calib = [r for r in recs if r["ref_id"] % 2 == 0]
    test  = [r for r in recs if r["ref_id"] % 2 == 1]
    sc_all = np.concatenate([r[sck] for r in recs])
    lams = np.quantile(sc_all, np.linspace(0, 0.99, 30))
    gpool = np.array([r[gate] for r in recs])
    taus = np.quantile(gpool, np.linspace(0, 0.95, 30))
    m = len(taus)*len(lams); th = 0.1/m
    cands = []
    for tA in taus:
        for L in lams:
            R1c, n1, R2c, n2, abc, szc = emp(calib, tA, L, gate, bxk, sck)
            if hoeffding_p(R1c, n1, alpha) <= th and hoeffding_p(R2c, n2, beta) <= th:
                cands.append((abc, szc, tA, L))
    if not cands:
        print(f"  {label}: EMPTY feasible"); return
    bright = [c for c in cands if c[0] <= 0.5] or cands
    _, _, tau, lam = min(bright, key=lambda c: c[1])
    # bootstrap test
    tp_t = [r for r in test if r["tp"]]; nt_t = [r for r in test if r["no_target"]]
    rng = np.random.default_rng(7)
    bs = {"sz": [], "R1": [], "R2": [], "ab": []}
    for _ in range(500):
        bt = [tp_t[i] for i in rng.integers(0, len(tp_t), len(tp_t))]
        bn = [nt_t[i] for i in rng.integers(0, len(nt_t), len(nt_t))]
        R1, _, R2, _, ab, sz = emp(bt+bn, tau, lam, gate, bxk, sck)
        bs["sz"].append(sz); bs["R1"].append(R1); bs["R2"].append(R2); bs["ab"].append(ab)
    def ci(k):
        lo, hi = np.percentile(bs[k], [2.5, 97.5]); return np.median(bs[k]), lo, hi
    msz = ci("sz"); m1 = ci("R1"); m2 = ci("R2"); mab = ci("ab")
    print(f"  {label}: sz={msz[0]:.2f}[{msz[1]:.2f},{msz[2]:.2f}] "
          f"R1={m1[0]:.3f}[{m1[1]:.3f},{m1[2]:.3f}] "
          f"R2={m2[0]:.3f}[{m2[1]:.3f},{m2[2]:.3f}] "
          f"abst={mab[0]:.3f}[{mab[1]:.3f},{mab[2]:.3f}]")

def main():
    print("Three-split composition (a=b=0.3, bright Pareto point, 500x bootstrap CI):")
    for sp in ["val", "testA", "testB"]:
        recs = build(sp)
        if recs is None:
            print(f"\n[{sp}] dumps not ready"); continue
        ntp = sum(r["tp"] for r in recs); nnt = sum(r["no_target"] for r in recs)
        print(f"\n[{sp}] shared={len(recs)} tp={ntp} nt={nnt}")
        select_and_eval(recs, "g_owl", "o_boxes", "o_scores", label="OWL-ViT only  ")
        select_and_eval(recs, "g_owl", "boxes",   "scores",   label="COMPOSE owl+gd ")

if __name__ == "__main__":
    main()
