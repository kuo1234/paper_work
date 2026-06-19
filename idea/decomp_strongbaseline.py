#!/usr/bin/env python3
"""Strong full-expression baselines (red-team P1-D): prove decomp beats not just
threshold=0 raw full pool, but also full+NMS-merge and full+top-K(calib-tuned),
all wrapped by the SAME CRS/LTT. If decomp still wins vs full+NMS -> robust."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import crs_protocol as P
import decomp_maintable as M

GATE = "g_owl"

def nms(boxes, scores, iou_thr=0.5):
    """Greedy NMS. Returns kept indices (high score first)."""
    if len(boxes) == 0:
        return []
    idx = list(np.argsort(scores)[::-1])
    keep = []
    while idx:
        i = idx.pop(0); keep.append(i)
        idx = [j for j in idx if P.iou(boxes[i], boxes[j]) < iou_thr]
    return keep

def apply_pool_transform(recs, mode, topk=None, iou_thr=0.5):
    """Return new recs with gboxes/gscores transformed per mode (does not touch gt/gate)."""
    out = []
    for r in recs:
        bx = r["gboxes"]; sc = r["gscores"]
        if mode == "raw":
            nb, ns = bx, sc
        elif mode == "nms":
            k = nms(bx, np.asarray(sc), iou_thr)
            nb = [bx[i] for i in k]; ns = np.asarray([sc[i] for i in k])
        elif mode == "topk":
            k = list(np.argsort(np.asarray(sc))[::-1][:topk]) if len(sc) else []
            nb = [bx[i] for i in k]; ns = np.asarray([sc[i] for i in k])
        nr = dict(r); nr["gboxes"] = nb; nr["gscores"] = ns
        out.append(nr)
    return out

def ltt_row(recs, label):
    calib, test = P.split_calib_test(recs)
    feas, _ = P.feasible(calib, GATE, "gboxes", "gscores")
    if not feas:
        print(f"    {label:<24}: EMPTY feasible"); return
    sz, tau, lam, *_ = min(feas, key=lambda c: c[0])
    ci = P.bootstrap_test(test, tau, lam, GATE, "gboxes", "gscores")
    s, r1, r2, d = ci["sz"], ci["R1"], ci["R2"], ci["defer"]
    print(f"    {label:<24}: sz={s[0]:.2f} R1={r1[0]:.3f} R2={r2[0]:.3f} defer={d[0]:.3f} n_feas={len(feas)}")

for sp in ["val", "testA", "testB"]:
    orig = P.build(sp); dec = M.build_decomp(sp)
    if orig is None or dec is None:
        print(f"\n[{sp}] not ready"); continue
    print(f"\n========== {sp} ==========")
    print("  -- Strong FULL baselines (full-expr pool, various consolidation) --")
    ltt_row(apply_pool_transform(orig, "raw"), "full raw (threshold=0)")
    ltt_row(apply_pool_transform(orig, "nms", iou_thr=0.5), "full + NMS@0.5")
    ltt_row(apply_pool_transform(orig, "nms", iou_thr=0.7), "full + NMS@0.7")
    ltt_row(apply_pool_transform(orig, "topk", topk=10), "full + top-10")
    ltt_row(apply_pool_transform(orig, "topk", topk=20), "full + top-20")
    print("  -- DECOMP (decomp-union pool) --")
    ltt_row(dec, "decomp")
