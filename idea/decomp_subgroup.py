#!/usr/bin/env python3
"""Subgroup analysis on truly-decomposed cases: which query types benefit most
from decomposition? Reports matched recall @ size~3 (full vs decomp) per subgroup,
bypassing LTT (subgroups too small for stable LTT feasibility)."""
import os, sys, json, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import crs_protocol as P

SP = sys.argv[1] if len(sys.argv) > 1 else "val"
owl = P._read(f"{P.B}/owlvit_gref_{SP}.jsonl")
orig = P._read(f"{P.B}/gdino_gref_{SP}.jsonl")
dec  = P._read(f"{P.B}/gdino_gref_{SP}_decomp.jsonl")
dkeys = [k for k, v in dec.items() if v.get("decomposed") and k in orig and k in owl]

def at_lambda(rec, lam):
    gt = rec["gt_boxes_xyxy"]
    if not gt:
        return None
    sc = rec["pred_scores"]; bx = rec["pred_boxes_xyxy"]
    sel = [bx[i] for i in range(len(bx)) if sc[i] >= lam]
    cov = sum(1 for q in gt if any(P.iou(p, q) >= 0.5 for p in sel)) / len(gt)
    return cov, len(sel)

def recall_at_size(keys, src, target_size=3.0):
    """sweep lambda, return recall at avg pool size closest to target."""
    allsc = np.concatenate([np.array(src[k]["pred_scores"]) for k in keys if src[k]["pred_scores"]])
    lams = np.quantile(allsc, np.linspace(0, 0.99, 40))
    pts = []
    for lam in lams:
        rs = []; ss = []
        for k in keys:
            r = at_lambda(src[k], lam)
            if r:
                rs.append(r[0]); ss.append(r[1])
        if rs:
            pts.append((np.mean(ss), np.mean(rs)))
    best = min(pts, key=lambda p: abs(p[0] - target_size))
    return best[1], best[0]

def is_conjunction(expr):
    return bool(re.search(r"\band\b|,|;|\bwith\b", expr.lower()))

# define subgroups by query characteristics (on the orig record for n_gt; expr from dec)
def subgroups(k):
    r = dec[k]; expr = r.get("expression", "")
    ngt = r.get("n_gt", len(orig[k]["gt_boxes_xyxy"]))
    nparts = r.get("n_parts", 1)
    wlen = len(expr.split())
    tags = []
    tags.append("n_gt>=3" if ngt >= 3 else "n_gt==2" if ngt == 2 else "n_gt==1")
    tags.append("long(>=12w)" if wlen >= 12 else "short(<12w)")
    tags.append("conjunction" if is_conjunction(expr) else "no-conj")
    tags.append(f"parts={min(nparts,3)}" + ("+" if nparts > 3 else ""))
    return tags

print(f"### split={SP}  truly-decomposed n={len(dkeys)}")
print(f"{'subgroup':<16} {'n':>5} {'full@3':>8} {'decomp@3':>9} {'Δrecall':>8}")
print("-" * 50)

# collect all subgroup tags
from collections import defaultdict
groups = defaultdict(list)
for k in dkeys:
    for t in subgroups(k):
        groups[t].append(k)
groups["ALL"] = dkeys

order = ["ALL", "n_gt==1", "n_gt==2", "n_gt>=3", "short(<12w)", "long(>=12w)",
         "no-conj", "conjunction", "parts=2", "parts=3", "parts=3+"]
for g in order:
    if g not in groups or len(groups[g]) < 15:
        if g in groups:
            print(f"{g:<16} {len(groups[g]):>5}   (n<15, skipped)")
        continue
    keys = groups[g]
    rf, szf = recall_at_size(keys, orig, 3.0)
    rd, szd = recall_at_size(keys, dec, 3.0)
    print(f"{g:<16} {len(keys):>5} {rf:>8.3f} {rd:>9.3f} {rd-rf:>+8.3f}")
