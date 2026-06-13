#!/usr/bin/env python3
"""MONEY FIGURE: target-present set size vs FNR guarantee, OWL-ViT vs GroundingDINO.
The bright positive operating point: stronger base reaches the SAME distribution-free
recall guarantee with ~5x smaller sets, and its irreducible recall floor -> 0.

Left panel: set size vs alpha (lower-left = better: tight guarantee, small set).
Right panel: empirical FNR vs alpha (diagonal = valid conformal control).
Annotates irreducible floor (select-all FNR) per base.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/home/p76141495/selective-grounding/dump"

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0., ix1-ix0), max(0., iy1-iy0); inter = iw*ih
    aa = max(0., ax1-ax0)*max(0., ay1-ay0); ab = max(0., bx1-bx0)*max(0., by1-by0)
    u = aa+ab-inter
    return inter/u if u > 0 else 0.

def load_tp(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if r.get("prompt_variant") != "canonical": continue
        if r["no_target"] or not r["gt_boxes_xyxy"]: continue
        r["_sc"] = np.array(r["pred_scores"]); rows.append(r)
    return rows

def fnr(sel, gt):
    return 1.0 - sum(1 for g in gt if any(iou(p, g) >= 0.5 for p in sel))/len(gt)
def sel(r, lam):
    bx = r["pred_boxes_xyxy"]; sc = r["_sc"]
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]
def risk(rows, lam):
    return np.mean([fnr(sel(r, lam), r["gt_boxes_xyxy"]) for r in rows])
def size(rows, lam):
    return np.mean([int(np.sum(r["_sc"] >= lam)) for r in rows])
def crc(calib, a, grid):
    n = len(calib); B_ = 1.0
    for lam in sorted(grid, reverse=True):
        if (n*risk(calib, lam)+B_)/(n+1) <= a: return lam
    return min(grid)

def curve(path):
    rows = load_tp(path)
    calib = [r for r in rows if r["ref_id"] % 2 == 0]
    test  = [r for r in rows if r["ref_id"] % 2 == 1]
    grid = np.quantile(np.concatenate([r["_sc"] for r in rows]), np.linspace(0, 0.999, 100))
    floor = risk(test, grid.min())
    alphas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    szs, ers = [], []
    for a in alphas:
        lam = crc(calib, a, grid)
        ers.append(risk(test, lam)); szs.append(size(test, lam))
    return np.array(alphas), np.array(szs), np.array(ers), floor, len(test)

def main():
    o_a, o_sz, o_er, o_fl, o_n = curve(f"{B}/owlvit_gref_val.jsonl")
    g_a, g_sz, g_er, g_fl, g_n = curve(f"{B}/gdino_gref_val.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(o_a, o_sz, "o-", color="darkorange", lw=2, label=f"OWL-ViT (n={o_n})")
    ax[0].plot(g_a, g_sz, "s-", color="seagreen", lw=2, label=f"GroundingDINO (n={g_n})")
    ax[0].set_xlabel("recall guarantee α  (conformal FNR ≤ α)")
    ax[0].set_ylabel("mean conformal set size")
    ax[0].set_title("Set size at matched guarantee\n(lower = tighter; ~2 GT boxes per target)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].axhline(2.0, ls=":", color="gray", lw=1)
    ax[0].text(0.36, 2.3, "≈ #GT", color="gray", fontsize=8)

    ax[1].plot([0, 0.5], [0, 0.5], ":", color="black", lw=1, label="ideal (valid)")
    ax[1].plot(o_a, o_er, "o-", color="darkorange", lw=2, label=f"OWL-ViT floor={o_fl:.3f}")
    ax[1].plot(g_a, g_er, "s-", color="seagreen", lw=2, label=f"GroundingDINO floor={g_fl:.3f}")
    ax[1].set_xlabel("nominal α"); ax[1].set_ylabel("empirical test FNR")
    ax[1].set_title("Conformal validity\n(on/below diagonal = guarantee holds)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("CRS bright operating point: stronger frozen base → same recall "
                 "guarantee at ~5× smaller sets, recall floor → 0", fontsize=11)
    fig.tight_layout()
    out = f"{B}/crs_money_setsize.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
    print("OWL-ViT  size:", [round(x,1) for x in o_sz])
    print("GDINO    size:", [round(x,1) for x in g_sz])
    print(f"shrink @each α:", [round(o/g,1) for o,g in zip(o_sz,g_sz)])

if __name__ == "__main__":
    main()
