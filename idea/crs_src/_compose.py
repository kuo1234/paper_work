#!/usr/bin/env python3
"""Cross-Base Conformal Composition (the invention): two frozen bases, each used
for what it is best at, jointly conformally calibrated.

Insight (measured): no single base is strong on BOTH risks --
  OWL-ViT: no-target separable (gate AUROC ~0.82) but target sets large
  GDINO:   target sets tiny (~5 boxes) but no-target gate weak (~0.60)

Composition:
  abstain gate  := OWL-ViT score (g_owl = top1_score) -> good no-target detection
  box selection := GDINO candidates (recall floor ~0.004, small sets)
Both frozen. LTT calibrates (tau on g_owl, lambda on GDINO scores) for joint
(R1<=alpha recall, R2<=beta abstention).

Compares 4 policies on the SHARED ref/sent keys (val):
  (1) OWL-ViT only   (2) GDINO only   (3) Composition (OWL gate + GDINO boxes)
Join key = (ref_id, sent_id). Split = ref_id parity.
"""
import json, numpy as np
import sys
sys.path.insert(0, "/home/p76141495/selective-grounding/src")
from ltt import empirical_risks, ltt_feasible, hoeffding_p

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

# build composed records: gate signal from OWL, boxes/scores from GDINO
recs = []
for k in keys:
    o = owl[k]; g = gd[k]
    recs.append(dict(
        ref_id=k[0], no_target=o["no_target"],
        tp=(not o["no_target"]) and bool(g["gt_boxes_xyxy"]),
        gt=g["gt_boxes_xyxy"],
        # composition: gate=OWL top1, boxes=GDINO
        g_owl=float(o["top1_score"]), g_gd=float(g["top1_score"]),
        boxes=g["pred_boxes_xyxy"], scores=np.array(g["pred_scores"]),
        # pure-owlvit fields for baseline 1
        o_boxes=o["pred_boxes_xyxy"], o_scores=np.array(o["pred_scores"]),
    ))

def fnr(sel, gt):
    return 1.0 - sum(1 for q in gt if any(iou(p, q) >= 0.5 for p in sel))/len(gt)

def make_risk_fns(gate_key, box_key, score_key):
    def decide(r, tau, lam):
        if r[gate_key] <= tau: return []
        bx = r[box_key]; sc = r[score_key]
        return [bx[i] for i in range(len(bx)) if sc[i] >= lam]
    def emp(rows, tau, lam):
        fnrs, nt_fs, tp_ab = [], [], []
        for r in rows:
            sel = decide(r, tau, lam)
            if r["tp"]:
                if r[gate_key] <= tau: tp_ab.append(1.0)
                else: tp_ab.append(0.0); fnrs.append(fnr(sel, r["gt"]))
            elif r["no_target"]:
                nt_fs.append(1.0 if len(sel) > 0 else 0.0)
        return (np.mean(fnrs) if fnrs else 0.0, len(fnrs),
                np.mean(nt_fs) if nt_fs else 0.0, len(nt_fs),
                np.mean(tp_ab) if tp_ab else 0.0, decide)
    return emp

def ltt(calib, emp, alpha, beta, taus, lams, delta=0.1):
    m = len(taus)*len(lams); th = delta/m; valid = []
    for tA in taus:
        for L in lams:
            R1, n1, R2, n2, ab, _ = emp(calib, tA, L)
            if hoeffding_p(R1, n1, alpha) <= th and hoeffding_p(R2, n2, beta) <= th:
                valid.append((tA, L, ab))
    return valid

def evaluate(label, rows_all, gate_key, box_key, score_key, gate_pool_key):
    calib = [r for r in rows_all if r["ref_id"] % 2 == 0]
    test  = [r for r in rows_all if r["ref_id"] % 2 == 1]
    emp = make_risk_fns(gate_key, box_key, score_key)
    sc_all = np.concatenate([r[score_key] for r in rows_all])
    lams = np.quantile(sc_all, np.linspace(0, 0.99, 25))
    gpool = np.array([r[gate_pool_key] for r in rows_all])
    taus = np.quantile(gpool, np.linspace(0, 0.95, 25))
    for a, b in [(0.3, 0.2), (0.3, 0.3)]:
        valid = ltt(calib, emp, a, b, taus, lams)
        if not valid:
            print(f"  {label:24s} (a={a},b={b}): EMPTY"); continue
        tA, L, _ = min(valid, key=lambda c: c[2])
        R1, n1, R2, n2, ab, dec = emp(test, tA, L)
        sizes = [len(dec(r, tA, L)) for r in test if r["tp"] and r[gate_key] > tA]
        sz = np.mean(sizes) if sizes else 0.0
        print(f"  {label:24s} (a={a},b={b}): R1={R1:.3f} R2={R2:.3f} "
              f"tp_abst={ab:.3f} sz={sz:.1f} #feas={len(valid)}")

print(f"shared keys={len(keys)} tp={sum(r['tp'] for r in recs)} "
      f"nt={sum(r['no_target'] for r in recs)}")
print("\nMulti-risk LTT joint operating point (lower tp_abst & sz = better):")
evaluate("(1) OWL-ViT only", recs, "g_owl", "o_boxes", "o_scores", "g_owl")
evaluate("(2) GDINO only",   recs, "g_gd",  "boxes",   "scores",   "g_gd")
evaluate("(3) COMPOSE owl-gate+gdino-box", recs, "g_owl", "boxes", "scores", "g_owl")
