#!/usr/bin/env python3
"""CRS feasibility gate (validity-focused): does Conformal Risk Control hold on
frozen OWL-ViT gRefCOCO multi-target selection?

Risk = FNR on TARGET-PRESENT samples (fraction of GT boxes not covered, IoU>=0.5).
  -> this is the real recall guarantee; no-target samples excluded from risk (loss
     trivially 0 there and would optimistically dilute the estimate).
Also reports no-target mean set-size (should be small -> abstention quality).

CRC: choose largest lambda (smallest set) on calib s.t. (n*Rhat+B)/(n+1) <= alpha.
Split: ref_id parity (even=calib, odd=test), same leakage protocol as M4.
"""
import json, sys
import numpy as np

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0., ix1 - ix0), max(0., iy1 - iy0)
    inter = iw * ih
    aa = max(0., ax1 - ax0) * max(0., ay1 - ay0)
    ab = max(0., bx1 - bx0) * max(0., by1 - by0)
    u = aa + ab - inter
    return inter / u if u > 0 else 0.

def load(path, variant='canonical'):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('prompt_variant') != variant:
                continue
            r['_sc'] = np.array(r['pred_scores'])
            rows.append(r)
    return rows

def fnr_loss(sel_boxes, gt_boxes):
    if len(gt_boxes) == 0:
        return None  # excluded
    covered = sum(1 for g in gt_boxes if any(iou(p, g) >= 0.5 for p in sel_boxes))
    return 1.0 - covered / len(gt_boxes)

def selected(r, lam):
    sc = r['_sc']; bx = r['pred_boxes_xyxy']
    return [bx[i] for i in range(len(bx)) if sc[i] >= lam]

def risk(rows, lam):
    """mean FNR over target-present rows only."""
    ls = []
    for r in rows:
        if r['no_target'] or len(r['gt_boxes_xyxy']) == 0:
            continue
        ls.append(fnr_loss(selected(r, lam), r['gt_boxes_xyxy']))
    return np.mean(ls) if ls else 0.0

def setsize(rows, lam, only=None):
    sz = []
    for r in rows:
        if only == 'target' and (r['no_target'] or len(r['gt_boxes_xyxy']) == 0):
            continue
        if only == 'notarget' and not r['no_target']:
            continue
        sz.append(int(np.sum(r['_sc'] >= lam)))
    return np.mean(sz) if sz else 0.0

def crc_lambda(calib, alpha, grid):
    """largest lambda (smallest set) with (n*Rhat+B)/(n+1) <= alpha. B=1."""
    tp = [r for r in calib if not r['no_target'] and len(r['gt_boxes_xyxy']) > 0]
    n = len(tp); B = 1.0
    for lam in sorted(grid, reverse=True):
        Rhat = risk(tp, lam)
        if (n * Rhat + B) / (n + 1) <= alpha:
            return lam
    return min(grid)

def run(path, label):
    rows = load(path)
    calib = [r for r in rows if r['ref_id'] % 2 == 0]
    test  = [r for r in rows if r['ref_id'] % 2 == 1]
    tp_test = [r for r in test if not r['no_target'] and len(r['gt_boxes_xyxy']) > 0]
    ntg = np.mean([r['no_target'] for r in rows])
    mng = np.mean([r['n_gt'] for r in rows if not r['no_target'] and r['n_gt'] > 0])
    print(f"\n=== {label}: total={len(rows)} calib={len(calib)} test={len(test)} "
          f"(tp_test={len(tp_test)}) ===")
    print(f"    no_target={ntg:.3f}  mean n_gt(target,>0)={mng:.2f}")

    allsc = np.concatenate([r['_sc'] for r in rows])
    grid = np.quantile(allsc, np.linspace(0.0, 0.999, 80))

    print(f"\n  {'alpha':>6} | {'lambda':>8} {'emp_FNR':>8} {'cov':>6} | "
          f"{'sz_tgt':>7} {'sz_noTgt':>8}")
    res = []
    for a in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        lam = crc_lambda(calib, a, grid)
        er = risk(tp_test, lam)
        sz_t = setsize(test, lam, only='target')
        sz_n = setsize(test, lam, only='notarget')
        print(f"  {a:>6.2f} | {lam:>8.4f} {er:>8.4f} {1-er:>6.3f} | "
              f"{sz_t:>7.2f} {sz_n:>8.2f}")
        res.append((a, er))
    viol = [(a, er) for a, er in res if er > a + 0.03]
    print(f"\n  >>> VALIDITY: {'HOLD (empirical FNR <= alpha+0.03 for all)' if not viol else f'VIOLATED at {viol}'}")
    return res

if __name__ == '__main__':
    base = '/home/p76141495/selective-grounding/dump'
    run(f'{base}/owlvit_gref_val.jsonl', 'gRefCOCO val')
    run(f'{base}/owlvit_gref_testA.jsonl', 'gRefCOCO testA')
    run(f'{base}/owlvit_gref_testB.jsonl', 'gRefCOCO testB')
