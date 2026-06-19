#!/usr/bin/env python3
"""Matched Recall-Size Frontier — four metrics (red-team P0-A).
On truly-decomposed cases, fairly compares full vs decomp candidate pools:
  (1) recall @ matched avg size
  (2) avg size @ matched recall
  (3) AURC = area under recall-size curve
  (4) calib-selected lambda on CALIB half, evaluated on EVAL half  <- key anti-cheat
Renamed from 'decisive evidence' to frontier analysis (mechanism evidence)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import crs_protocol as P

SP = sys.argv[1] if len(sys.argv) > 1 else "val"
owl = P._read(f"{P.B}/owlvit_gref_{SP}.jsonl")
orig = P._read(f"{P.B}/gdino_gref_{SP}.jsonl")
dec = P._read(f"{P.B}/gdino_gref_{SP}_decomp.jsonl")
dkeys = [k for k, v in dec.items() if v.get("decomposed") and k in orig and k in owl]
print(f"### split={SP}  truly-decomposed n={len(dkeys)}")

def at_lambda(rec, lam):
    gt = rec["gt_boxes_xyxy"]
    if not gt:
        return None
    sc = rec["pred_scores"]; bx = rec["pred_boxes_xyxy"]
    sel = [bx[i] for i in range(len(bx)) if sc[i] >= lam]
    cov = sum(1 for q in gt if any(P.iou(p, q) >= 0.5 for p in sel)) / len(gt)
    return cov, len(sel)

def curve(keys, src, lams):
    """return list of (avg_size, avg_recall) over lambda grid."""
    pts = []
    for lam in lams:
        rs = []; ss = []
        for k in keys:
            r = at_lambda(src[k], lam)
            if r:
                rs.append(r[0]); ss.append(r[1])
        if rs:
            pts.append((np.mean(ss), np.mean(rs)))
    return sorted(pts)

def recall_at_size(pts, tgt):
    best = min(pts, key=lambda p: abs(p[0] - tgt)); return best[1], best[0]

def size_at_recall(pts, tgt):
    best = min(pts, key=lambda p: abs(p[1] - tgt)); return best[0], best[1]

def aurc(pts, smin=1.0, smax=6.0):
    """area under recall-vs-size curve, clipped to [smin,smax], trapezoidal / range."""
    p = [(s, r) for s, r in pts if smin <= s <= smax]
    if len(p) < 2:
        return float("nan")
    xs = [s for s, _ in p]; ys = [r for _, r in p]
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(ys, xs) / (xs[-1] - xs[0]))

# shared lambda grid from both pools' scores
allsc = [np.array(orig[k]["pred_scores"]) for k in dkeys] + [np.array(dec[k]["pred_scores"]) for k in dkeys]
lams = np.quantile(np.concatenate(allsc), np.linspace(0, 0.99, 60))

print("\n== (1)(2)(3) Frontier metrics (all truly-decomp cases) ==")
print(f"{'pool':<8} {'rec@sz2':>8} {'rec@sz3':>8} {'rec@sz4':>8} {'sz@rec.8':>9} {'AURC':>7}")
for tag, src in [("full", orig), ("decomp", dec)]:
    pts = curve(dkeys, src, lams)
    r2 = recall_at_size(pts, 2.0)[0]; r3 = recall_at_size(pts, 3.0)[0]; r4 = recall_at_size(pts, 4.0)[0]
    s8 = size_at_recall(pts, 0.8)[0]
    au = aurc(pts)
    print(f"{tag:<8} {r2:>8.3f} {r3:>8.3f} {r4:>8.3f} {s8:>9.2f} {au:>7.3f}")

# (4) calib-selected lambda on calib half, evaluated on eval half (anti-cheat)
print("\n== (4) Calib-select lambda @ calib, evaluate @ eval (anti-cheat) ==")
ck = [k for k in dkeys if k[0] % 2 == 0]   # parity: even=calib
ek = [k for k in dkeys if k[0] % 2 == 1]   # odd=eval
print(f"  calib n={len(ck)}  eval n={len(ek)}")
for tgt_size in [2.0, 3.0, 4.0]:
    line = f"  target size~{tgt_size}: "
    for tag, src in [("full", orig), ("decomp", dec)]:
        cpts = curve(ck, src, lams)
        # find lambda achieving target size on CALIB
        best_lam = None; best_d = 1e9
        for lam in lams:
            ss = [at_lambda(src[k], lam)[1] for k in ck if at_lambda(src[k], lam)]
            if ss and abs(np.mean(ss) - tgt_size) < best_d:
                best_d = abs(np.mean(ss) - tgt_size); best_lam = lam
        # evaluate that lambda on EVAL
        ev = [at_lambda(src[k], best_lam) for k in ek if at_lambda(src[k], best_lam)]
        er = np.mean([x[0] for x in ev]); es = np.mean([x[1] for x in ev])
        line += f"{tag} eval(sz={es:.2f},rec={er:.3f})  "
    print(line)
