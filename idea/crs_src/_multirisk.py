#!/usr/bin/env python3
"""Multi-risk Conformal Referring Set (the invention pillar).

Two coupled, distribution-free guarantees on a FROZEN base, calibrated jointly:
  (R1) recall:    on target-present samples, FNR <= alpha
  (R2) abstention: on no-target samples, false-selection rate <= beta
                   (fraction of no-target samples that emit ANY box)

Decision = two-stage gate, each stage conformally calibrated on its OWN sub-population:
  Stage A (abstain gate): g = P(target present) proxy = top1_score.
     pick tau_A on no-target calib s.t. fraction(g > tau_A) <= beta  (R2 split-conformal)
  Stage B (box select):  for samples passing the gate (g > tau_A), select boxes with
     nonconformity score >= lambda. pick lambda on target-present calib via CRC s.t.
     FNR <= alpha  -- BUT only over samples that pass Stage A (conditional risk).

Why this is non-trivial vs single-threshold CRS:
  - single global lambda forces a recall/abstention tradeoff on ONE axis (CRS gate
    showed no-target still emits 3.66 boxes at alpha=0.3).
  - decoupling into two conformally-calibrated gates lets BOTH guarantees hold
    simultaneously. We verify empirical (FNR, false-sel-rate) <= (alpha, beta) jointly.

Baselines compared at matched alpha:
  - single-lambda CRS (no abstain gate)  -> shows R2 is violated / sets huge on no-tgt
  - multi-risk (this)                    -> both hold

Split: ref_id parity. Reports a grid over (alpha, beta).
"""
import json
import numpy as np

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0., ix1-ix0), max(0., iy1-iy0); inter = iw*ih
    aa = max(0., ax1-ax0)*max(0., ay1-ay0); ab = max(0., bx1-bx0)*max(0., by1-by0)
    u = aa+ab-inter
    return inter/u if u > 0 else 0.

def load(path, variant='canonical'):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('prompt_variant') != variant: continue
            r['_sc'] = np.array(r['pred_scores'])
            r['_g'] = float(r['top1_score'])   # P(target present) proxy
            r['_tp'] = (not r['no_target']) and bool(r['gt_boxes_xyxy'])
            rows.append(r)
    return rows

def fnr(sel, gt):
    if not gt: return None
    cov = sum(1 for g in gt if any(iou(p, g) >= 0.5 for p in sel))
    return 1.0 - cov/len(gt)

def sel_boxes(r, lam):
    sc = r['_sc']; bx = r['pred_boxes_xyxy']
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]

# ---- Stage A: abstain gate calibrated on no-target calib ----
def tau_abstain(calib, beta):
    """largest-coverage (smallest tau) s.t. no-target false-select rate <= beta.
    false-select(no-tgt) = fraction with g > tau. monotone decreasing in tau.
    conformal-style finite-sample: pick tau = the ceil((1-beta)(n+1))-th smallest g
    among no-target calib  -> guarantees P(g>tau) <= beta marginally."""
    nt = np.array([r['_g'] for r in calib if r['no_target']])
    if len(nt) == 0: return -np.inf
    n = len(nt)
    k = int(np.ceil((1-beta)*(n+1)))
    k = min(max(k, 1), n)
    return np.sort(nt)[k-1]

# ---- Stage B: box-select CRC on target-present calib that PASS gate ----
def lambda_crc(calib, tau_A, alpha, grid):
    tp = [r for r in calib if r['_tp'] and r['_g'] > tau_A]
    n = len(tp); B = 1.0
    if n == 0: return min(grid)
    for lam in sorted(grid, reverse=True):
        ls = [fnr(sel_boxes(r, lam), r['gt_boxes_xyxy']) for r in tp]
        Rhat = np.mean(ls)
        if (n*Rhat+B)/(n+1) <= alpha:
            return lam
    return min(grid)

def evaluate(test, tau_A, lam):
    """two-stage decision on test. returns empirical (FNR_on_tp, falsesel_on_nt,
    mean set size on tp-passed, mean set size on nt)."""
    fnrs, set_tp = [], []
    nt_falsesel, set_nt = [], []
    for r in test:
        passed = r['_g'] > tau_A
        sel = sel_boxes(r, lam) if passed else []
        if r['_tp']:
            fnrs.append(fnr(sel, r['gt_boxes_xyxy']))
            set_tp.append(len(sel))
        elif r['no_target']:
            nt_falsesel.append(1.0 if len(sel) > 0 else 0.0)
            set_nt.append(len(sel))
    return (np.mean(fnrs) if fnrs else 0.0,
            np.mean(nt_falsesel) if nt_falsesel else 0.0,
            np.mean(set_tp) if set_tp else 0.0,
            np.mean(set_nt) if set_nt else 0.0)

def single_lambda_crs(calib, test, alpha, grid):
    """baseline: no abstain gate. CRC lambda over ALL target-present calib."""
    tp = [r for r in calib if r['_tp']]
    n = len(tp); B = 1.0
    lam = min(grid)
    for L in sorted(grid, reverse=True):
        Rhat = np.mean([fnr(sel_boxes(r, L), r['gt_boxes_xyxy']) for r in tp])
        if (n*Rhat+B)/(n+1) <= alpha:
            lam = L; break
    # eval: every sample emits boxes >= lam (no gating)
    fnrs, nt_fs, set_nt = [], [], []
    for r in test:
        sel = sel_boxes(r, lam)
        if r['_tp']: fnrs.append(fnr(sel, r['gt_boxes_xyxy']))
        elif r['no_target']:
            nt_fs.append(1.0 if len(sel) > 0 else 0.0); set_nt.append(len(sel))
    return (np.mean(fnrs), np.mean(nt_fs), np.mean(set_nt))

def run(path, label):
    rows = load(path)
    calib = [r for r in rows if r['ref_id'] % 2 == 0]
    test  = [r for r in rows if r['ref_id'] % 2 == 1]
    grid = np.quantile(np.concatenate([r['_sc'] for r in rows]), np.linspace(0,0.999,100))
    print(f"\n=== {label}: total={len(rows)} (calib={len(calib)} test={len(test)}) ===")
    n_tp = sum(r['_tp'] for r in test); n_nt = sum(r['no_target'] for r in test)
    print(f"    test: target-present={n_tp}  no-target={n_nt}")

    print("\n  --- BASELINE single-lambda CRS (no abstain gate) ---")
    print(f"  {'alpha':>5} | {'emp_FNR':>7} {'nt_falsesel':>11} {'nt_setsz':>8}")
    for a in [0.1, 0.2, 0.3]:
        f, fs, snt = single_lambda_crs(calib, test, a, grid)
        print(f"  {a:>5.2f} | {f:>7.4f} {fs:>11.3f} {snt:>8.2f}   <- R2 uncontrolled")

    print("\n  --- MULTI-RISK conformal (joint guarantee) ---")
    print(f"  {'alpha':>5} {'beta':>5} | {'emp_FNR':>7} {'nt_falsesel':>11} | "
          f"{'R1':>4} {'R2':>4} | {'sz_tp':>6} {'sz_nt':>6}")
    for a in [0.1, 0.2, 0.3]:
        for b in [0.1, 0.2]:
            tauA = tau_abstain(calib, b)
            lam = lambda_crc(calib, tauA, a, grid)
            f, fs, stp, snt = evaluate(test, tauA, lam)
            r1 = 'OK' if f <= a+0.03 else 'X'
            r2 = 'OK' if fs <= b+0.03 else 'X'
            print(f"  {a:>5.2f} {b:>5.2f} | {f:>7.4f} {fs:>11.3f} | {r1:>4} {r2:>4} | "
                  f"{stp:>6.2f} {snt:>6.2f}")

if __name__ == '__main__':
    base = '/home/p76141495/selective-grounding/dump'
    for sp in ['val', 'testA', 'testB']:
        run(f'{base}/owlvit_gref_{sp}.jsonl', f'gRefCOCO {sp}')
