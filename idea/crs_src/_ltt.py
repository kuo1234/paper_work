#!/usr/bin/env python3
"""Multi-risk Conformal Referring Set via Learn-then-Test (the invention pillar, v2).

Naive sequential calibration of two gates is INVALID for joint control (the abstain
gate, tuned on no-target, silently destroys target recall because OWL-ViT's
target/no-target score distributions overlap). We instead use Learn-then-Test
(Angelopoulos et al. 2021): treat each (tau_A, lambda) on a 2D grid as a hypothesis,
keep only configs whose BOTH empirical risks pass a finite-sample test (Hoeffding
bound + Bonferroni over the grid). This yields a JOINT, distribution-free guarantee
and -- crucially -- the FEASIBLE REGION of simultaneously achievable (alpha, beta).

Risks (selective framing):
  R1 conditional recall:  among NON-abstained target-present samples, FNR <= alpha
  R2 abstention validity: among no-target samples, false-selection rate <= beta
Reported cost:
  ab_tp = abstention rate on target-present (price of the gate; not a guaranteed risk)

LTT test per config (tau_A, lambda) and risk R with empirical mean Rhat over n samples:
  Hoeffding p-value for H0: R > target  ->  p = exp(-2 n (target - Rhat)^2) if Rhat<target
  reject H0 (config valid for that risk) if p <= delta / |grid|  (Bonferroni FWER<=delta)
A config is RETURNED if it passes BOTH risk tests. Among returned, pick the one that
minimizes target-present abstention (most useful) -- selection is post-hoc valid since
all returned configs satisfy the guarantee.

Split: ref_id parity (calib=even -> LTT calibration; test=odd -> report empirical).
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

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('prompt_variant') != 'canonical': continue
            r['_sc'] = np.array(r['pred_scores'])
            r['_g'] = float(r['top1_score'])
            r['_tp'] = (not r['no_target']) and bool(r['gt_boxes_xyxy'])
            rows.append(r)
    return rows

def fnr(sel, gt):
    cov = sum(1 for g in gt if any(iou(p, g) >= 0.5 for p in sel))
    return 1.0 - cov/len(gt)

def decide(r, tauA, lam):
    """two-stage: abstain if g<=tauA (empty), else select boxes with score>=lam."""
    if r['_g'] <= tauA:
        return []
    sc = r['_sc']; bx = r['pred_boxes_xyxy']
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]

def empirical_risks(rows, tauA, lam):
    """returns (R1 cond-FNR over non-abstained tp, n1, R2 false-sel over nt, n2,
    abstention rate on tp)."""
    fnrs, nt_fs, tp_ab = [], [], []
    for r in rows:
        sel = decide(r, tauA, lam)
        if r['_tp']:
            if r['_g'] <= tauA:
                tp_ab.append(1.0)          # abstained on a true target = cost
            else:
                tp_ab.append(0.0)
                fnrs.append(fnr(sel, r['gt_boxes_xyxy']))
        elif r['no_target']:
            nt_fs.append(1.0 if len(sel) > 0 else 0.0)
    R1 = np.mean(fnrs) if fnrs else 0.0
    R2 = np.mean(nt_fs) if nt_fs else 0.0
    return R1, len(fnrs), R2, len(nt_fs), (np.mean(tp_ab) if tp_ab else 0.0)

def hoeffding_p(Rhat, n, target):
    """p-value for H0: true risk > target. small p -> safe to reject -> config valid."""
    if n == 0: return 1.0
    if Rhat >= target: return 1.0
    return float(np.exp(-2 * n * (target - Rhat) ** 2))

def ltt_feasible(calib, alpha, beta, taus, lams, delta=0.1):
    """return list of (tauA, lam, ab_tp_calib) configs passing BOTH risk tests
    with Bonferroni FWER <= delta over the |taus|*|lams| grid."""
    m = len(taus) * len(lams)
    thresh = delta / m
    valid = []
    for tA in taus:
        for L in lams:
            R1, n1, R2, n2, ab = empirical_risks(calib, tA, L)
            p1 = hoeffding_p(R1, n1, alpha)
            p2 = hoeffding_p(R2, n2, beta)
            if p1 <= thresh and p2 <= thresh:
                valid.append((tA, L, ab))
    return valid

def run(path, label):
    rows = load(path)
    calib = [r for r in rows if r['ref_id'] % 2 == 0]
    test  = [r for r in rows if r['ref_id'] % 2 == 1]
    g_all = np.concatenate([r['_sc'] for r in rows])
    lams = np.quantile(g_all, np.linspace(0.0, 0.99, 25))
    gv = np.array([r['_g'] for r in rows])
    taus = np.quantile(gv, np.linspace(0.0, 0.95, 25))
    print(f"\n=== {label}: calib={len(calib)} test={len(test)} | grid {len(taus)}x{len(lams)} ===")
    print(f"  {'alpha':>5} {'beta':>5} | {'#feasible':>9} | {'test R1':>7} {'test R2':>7} "
          f"{'tp_abst':>7} {'sz_tp':>6} | joint")
    for a in [0.2, 0.3, 0.4]:
        for b in [0.1, 0.2]:
            valid = ltt_feasible(calib, a, b, taus, lams, delta=0.1)
            if not valid:
                print(f"  {a:>5.2f} {b:>5.2f} | {'EMPTY':>9} | feasible region empty on calib")
                continue
            # pick config minimizing target-present abstention (most useful)
            tA, L, _ = min(valid, key=lambda c: c[2])
            R1, n1, R2, n2, ab = empirical_risks(test, tA, L)
            # mean set size on non-abstained tp
            sizes = [len(decide(r, tA, L)) for r in test if r['_tp'] and r['_g'] > tA]
            sz = np.mean(sizes) if sizes else 0.0
            joint = 'BOTH OK' if (R1 <= a+0.03 and R2 <= b+0.03) else \
                    ('R1 X' if R1 > a+0.03 else 'R2 X')
            print(f"  {a:>5.2f} {b:>5.2f} | {len(valid):>9} | {R1:>7.4f} {R2:>7.4f} "
                  f"{ab:>7.3f} {sz:>6.2f} | {joint}")

if __name__ == '__main__':
    base = '/home/p76141495/selective-grounding/dump'
    for sp in ['val', 'testA', 'testB']:
        run(f'{base}/owlvit_gref_{sp}.jsonl', f'gRefCOCO {sp}')
