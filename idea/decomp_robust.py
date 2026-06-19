#!/usr/bin/env python3
"""Decomp CRS robustness across split modes (red-team P0-B) + overall target
failure R_total (P0-C). Reports R1/R2/R3/size/n_feas + R_total for Frozen vs
Decomp under parity / random(5 seeds, mean+/-std) / image-disjoint splits."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import crs_protocol as P
import decomp_maintable as M

GATE, BXK, SCK = "g_owl", "gboxes", "gscores"

def run_one(recs, mode, seed):
    calib, test = P.split_calib_test(recs, mode=mode, seed=seed)
    feas, _ = P.feasible(calib, GATE, BXK, SCK)
    if not feas:
        return None
    sz, tau, lam, *_ = min(feas, key=lambda c: c[0])
    ci = P.bootstrap_test(test, tau, lam, GATE, BXK, SCK)
    # R_total = R3 + (1-R3)*R1  (overall target failure on target-present)
    r1 = ci["R1"][0]; r3 = ci["defer"][0]
    rtot = r3 + (1 - r3) * r1
    return dict(sz=ci["sz"][0], R1=r1, R2=ci["R2"][0], R3=r3, rtot=rtot, n_feas=len(feas))

def fmt(d):
    if d is None:
        return "EMPTY feasible"
    return (f"sz={d['sz']:.2f} R1={d['R1']:.3f} R2={d['R2']:.3f} R3={d['R3']:.3f} "
            f"Rtot={d['rtot']:.3f} n_feas={d['n_feas']}")

for sp in ["val", "testA", "testB"]:
    orig = P.build(sp); dec = M.build_decomp(sp)
    if orig is None or dec is None:
        print(f"\n[{sp}] not ready"); continue
    print(f"\n========== {sp} ==========")
    for label, recs in [("Frozen", orig), ("Decomp", dec)]:
        print(f"  -- {label} --")
        # parity
        print(f"    parity      : {fmt(run_one(recs, 'parity', 0))}")
        # random 5 seeds -> mean +/- std on each metric
        runs = [run_one(recs, 'random', s) for s in range(5)]
        ok = [r for r in runs if r is not None]
        if ok:
            keys = ["sz", "R1", "R2", "R3", "rtot"]
            mp = {k: (np.mean([r[k] for r in ok]), np.std([r[k] for r in ok])) for k in keys}
            nf = [r["n_feas"] for r in ok]
            print(f"    random5     : sz={mp['sz'][0]:.2f}±{mp['sz'][1]:.2f} "
                  f"R1={mp['R1'][0]:.3f}±{mp['R1'][1]:.3f} R2={mp['R2'][0]:.3f}±{mp['R2'][1]:.3f} "
                  f"R3={mp['R3'][0]:.3f}±{mp['R3'][1]:.3f} Rtot={mp['rtot'][0]:.3f}±{mp['rtot'][1]:.3f} "
                  f"n_feas={min(nf)}-{max(nf)} ({len(ok)}/5 feasible)")
        else:
            print(f"    random5     : 0/5 feasible")
        # image-disjoint
        print(f"    image-disj  : {fmt(run_one(recs, 'imagedisj', 0))}")
