#!/usr/bin/env python3
"""Feasible-region figure: LTT multi-risk operating points, OWL-ViT vs GroundingDINO.

For each base and each (alpha, beta) on a grid, run LTT to find feasible configs,
pick the one minimizing target-present abstention, record (tp_abstention, set_size)
measured on TEST. Plot as two panels (one per base): x=alpha (recall guarantee),
color/annotation = best achievable tp_abstention and set size. The money claim:
stronger base -> useful operating points (low abstention, small sets) become feasible.

Reuses ltt.py machinery. Run after gdino_gref_val.jsonl exists.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/p76141495/selective-grounding/src")
from ltt import load, empirical_risks, ltt_feasible

BASE = "/home/p76141495/selective-grounding/dump"

def best_point(path, alpha, beta):
    rows = load(path)
    calib = [r for r in rows if r["ref_id"] % 2 == 0]
    test  = [r for r in rows if r["ref_id"] % 2 == 1]
    g_all = np.concatenate([r["_sc"] for r in rows])
    lams = np.quantile(g_all, np.linspace(0.0, 0.99, 25))
    gv = np.array([r["_g"] for r in rows])
    taus = np.quantile(gv, np.linspace(0.0, 0.95, 25))
    valid = ltt_feasible(calib, alpha, beta, taus, lams, delta=0.1)
    if not valid:
        return None
    tA, L, _ = min(valid, key=lambda c: c[2])
    R1, n1, R2, n2, ab = empirical_risks(test, tA, L)
    sizes = [sum(1 for i in range(len(r["pred_boxes_xyxy"])) if r["_sc"][i] >= L)
             for r in test if r["_tp"] and r["_g"] > tA]
    sz = np.mean(sizes) if sizes else 0.0
    return dict(R1=R1, R2=R2, abst=ab, size=sz, nfeasible=len(valid))

def panel(ax, path, title, alphas, beta):
    absts, sizes, labels = [], [], []
    for a in alphas:
        p = best_point(path, a, beta)
        if p is None:
            absts.append(np.nan); sizes.append(np.nan)
        else:
            absts.append(p["abst"]); sizes.append(p["size"])
    x = np.array(alphas)
    ax.plot(x, absts, "o-", color="crimson", label="target abstention")
    ax.set_xlabel("recall guarantee α (FNR ≤ α)")
    ax.set_ylabel("target-present abstention rate", color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax.set_ylim(-0.05, 1.05)
    ax2 = ax.twinx()
    ax2.plot(x, sizes, "s--", color="navy", label="set size")
    ax2.set_ylabel("mean set size (non-abstained)", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax.set_title(title)
    return absts, sizes

def main():
    alphas = [0.1, 0.2, 0.3, 0.4]
    beta = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    owl = panel(axes[0], f"{BASE}/owlvit_gref_val.jsonl",
                f"OWL-ViT (zero-shot acc 0.42)\nβ≤{beta}", alphas, beta)
    gd = panel(axes[1], f"{BASE}/gdino_gref_val.jsonl",
               f"GroundingDINO (stronger base)\nβ≤{beta}", alphas, beta)
    fig.suptitle("Multi-risk feasible operating points (LTT joint guarantee): "
                 "stronger base → lower abstention & smaller sets at same guarantee",
                 fontsize=11)
    fig.tight_layout()
    out = f"{BASE}/crs_feasible_region.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)
    print("OWL-ViT abst/size:", [round(v,3) if v==v else None for v in owl[0]],
          [round(v,2) if v==v else None for v in owl[1]])
    print("GDINO   abst/size:", [round(v,3) if v==v else None for v in gd[0]],
          [round(v,2) if v==v else None for v in gd[1]])

if __name__ == "__main__":
    main()
