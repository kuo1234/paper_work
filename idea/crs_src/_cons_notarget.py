#!/usr/bin/env python3
"""Test whether paraphrase consistency complements score signals for no-target
detection on OWL-ViT val. If a learned gate over (score signals + consistency)
beats score-only ~0.82 AUROC, the abstain gate -- and thus the LTT feasible
region -- can expand. Also reports each signal's marginal AUROC."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        r = json.loads(l); d[r["sample_uid"]] = r
    return d

def auroc(score, y):
    score = np.array(score); y = np.array(y)
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(allv)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

canon = read(f"{B}/owlvit_gref_val.jsonl")
paras = [read(f"{B}/owlvit_gref_val_para{i}.jsonl") for i in [1, 2, 3]]

# consistency features per sample
SCORE_SIGS = ["top1_score", "top2_score", "margin12", "score_entropy",
              "score_mean_topk", "score_std_topk"]
rows = []
for uid, c in canon.items():
    if not c["pred_boxes_xyxy"]:
        continue
    cb = c["pred_boxes_xyxy"][0]
    cons = []
    # also: agreement of full top-k set under paraphrase (set-level stability)
    setcons = []
    for pd in paras:
        ov = pd.get(uid)
        if ov is None or not ov["pred_boxes_xyxy"]:
            cons.append(0.); setcons.append(0.); continue
        cons.append(iou(cb, ov["pred_boxes_xyxy"][0]))
        # set-level: mean best-IoU of canonical top5 boxes to paraphrase boxes
        best = []
        for b in c["pred_boxes_xyxy"][:5]:
            ious = [iou(b, ob) for ob in ov["pred_boxes_xyxy"][:10]]
            best.append(max(ious) if ious else 0.)
        setcons.append(np.mean(best) if best else 0.)
    is_tp = (not c["no_target"]) and bool(c["gt_boxes_xyxy"])
    if not (c["no_target"] or is_tp):
        continue
    rows.append(dict(
        uid=uid, ref_id=c["ref_id"], y=1 if is_tp else 0,
        cons_top1=float(np.mean(cons)), cons_set=float(np.mean(setcons)),
        **{s: c[s] for s in SCORE_SIGS}))

y = np.array([r["y"] for r in rows]); rid = np.array([r["ref_id"] for r in rows])
print(f"n={len(rows)} target={int(y.sum())} no-target={int((y==0).sum())}")

# marginal AUROCs
for f in ["cons_top1", "cons_set"] + SCORE_SIGS:
    a = auroc([r[f] for r in rows], y)
    print(f"  {f:18s} AUROC={a:.4f} (flip {1-a:.4f})")

tr = rid % 2 == 0; te = rid % 2 == 1

def gate_auroc(feats):
    X = np.array([[r[f] for f in feats] for r in rows])
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return auroc(p, y[te])

print(f"\nlearned gate (score-only)      test AUROC = {gate_auroc(SCORE_SIGS):.4f}")
print(f"learned gate (score+cons)      test AUROC = {gate_auroc(SCORE_SIGS+['cons_top1','cons_set']):.4f}")
print(f"learned gate (cons-only)       test AUROC = {gate_auroc(['cons_top1','cons_set']):.4f}")
