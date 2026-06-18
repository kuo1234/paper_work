#!/usr/bin/env python3
"""Matched oracle analysis on the truly-decomposed cases.
Compares full-expr GD pool vs decomp-union pool on the SAME keys,
bypassing LTT to isolate decomposition effect from operating-point drift."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import crs_protocol as P

SP = sys.argv[1] if len(sys.argv) > 1 else "val"
owl = P._read(f"{P.B}/owlvit_gref_{SP}.jsonl")
orig = P._read(f"{P.B}/gdino_gref_{SP}.jsonl")
dec  = P._read(f"{P.B}/gdino_gref_{SP}_decomp.jsonl")
print(f"### split={SP}")

# keys that were truly decomposed AND present in all three
dkeys = [k for k, v in dec.items() if v.get("decomposed") and k in orig and k in owl]
print(f"truly-decomposed keys present in orig+owl+decomp: {len(dkeys)}")

def pool_recall(rec):
    """oracle recall: fraction of GT covered by ANY box in pool (ignore scores)."""
    gt = rec["gt_boxes_xyxy"]
    if not gt:
        return None
    boxes = rec["pred_boxes_xyxy"]
    cov = sum(1 for q in gt if any(P.iou(p, q) >= 0.5 for p in boxes))
    return cov / len(gt)

def at_lambda(rec, lam):
    gt = rec["gt_boxes_xyxy"]
    sc = rec["pred_scores"]; bx = rec["pred_boxes_xyxy"]
    sel = [bx[i] for i in range(len(bx)) if sc[i] >= lam]
    if not gt:
        return None
    rec_cov = sum(1 for q in gt if any(P.iou(p, q) >= 0.5 for p in sel)) / len(gt)
    return rec_cov, len(sel)

# oracle recall (pool ceiling) and pool size
o_rec = []; d_rec = []; o_sz = []; d_sz = []
for k in dkeys:
    ro = orig[k]; rd = dec[k]
    pr_o = pool_recall(ro); pr_d = pool_recall(rd)
    if pr_o is None or pr_d is None:
        continue
    o_rec.append(pr_o); d_rec.append(pr_d)
    o_sz.append(len(ro["pred_boxes_xyxy"])); d_sz.append(len(rd["pred_boxes_xyxy"]))
o_rec = np.array(o_rec); d_rec = np.array(d_rec); o_sz = np.array(o_sz); d_sz = np.array(d_sz)
print(f"\n== ORACLE POOL (ignore scores), n={len(o_rec)} ==")
print(f"  pool recall:  full={o_rec.mean():.3f}  decomp={d_rec.mean():.3f}  delta={d_rec.mean()-o_rec.mean():+.3f}")
print(f"  pool size:    full={o_sz.mean():.2f}   decomp={d_sz.mean():.2f}   delta={d_sz.mean()-o_sz.mean():+.2f}")
print(f"  cases decomp pool strictly more GT-covering: {(d_rec>o_rec).sum()}")
print(f"  cases decomp pool strictly less: {(d_rec<o_rec).sum()}")
print(f"  cases tied: {(d_rec==o_rec).sum()}")

# matched recall-vs-size at shared lambda grid
print(f"\n== RECALL @ matched avg pool size (sweep lambda) ==")
allsc = [np.array(orig[k]["pred_scores"]) for k in dkeys] + [np.array(dec[k]["pred_scores"]) for k in dkeys]
lams = np.quantile(np.concatenate(allsc), np.linspace(0, 0.99, 40))
for tag, src in [("full", orig), ("decomp", dec)]:
    pts = []
    for lam in lams:
        rs = []; ss = []
        for k in dkeys:
            r = at_lambda(src[k], lam)
            if r:
                rs.append(r[0]); ss.append(r[1])
        pts.append((np.mean(ss), np.mean(rs)))
    for tgt in [2.0, 3.0, 4.0]:
        best = min(pts, key=lambda p: abs(p[0] - tgt))
        print(f"  {tag}: @size~{tgt}: avg_size={best[0]:.2f} recall={best[1]:.3f}")
