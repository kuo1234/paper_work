#!/usr/bin/env python3
"""P1 gate: does consistency-belief nonconformity yield SMALLER conformal sets
than raw OWL-ViT score, at the SAME guaranteed FNR?

belief[box] = score * (1 + mean_over_paraphrases(agreement)), where agreement =
  matched-box score in that paraphrase if best-IoU>=0.5 else 0.
  -> rewards boxes that persist (same location, still scored high) under paraphrase.

CRC picks largest lambda (smallest set) on calib with (n*Rhat+B)/(n+1)<=alpha.
Compare raw vs belief on: empirical test FNR (both must HOLD) and mean target set size.
Split: ref_id parity. Risk on target-present rows only.
"""
import json
import numpy as np

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0., ix1 - ix0), max(0., iy1 - iy0)
    inter = iw * ih
    aa = max(0., ax1-ax0)*max(0., ay1-ay0); ab = max(0., bx1-bx0)*max(0., by1-by0)
    u = aa + ab - inter
    return inter/u if u > 0 else 0.

def load_all(canon_path, para_paths):
    """returns list of dicts with canonical fields + computed belief array."""
    def read(p):
        d = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line); d[r['sample_uid']] = r
        return d
    canon = read(canon_path)
    paras = [read(p) for p in para_paths]
    rows = []
    for uid, c in canon.items():
        cb = c['pred_boxes_xyxy']; cs = np.array(c['pred_scores'])
        agr = np.zeros(len(cb))
        npar = 0
        for pd in paras:
            ov = pd.get(uid)
            if ov is None: continue
            npar += 1
            obx = ov['pred_boxes_xyxy']; osc = np.array(ov['pred_scores'])
            for bi, b in enumerate(cb):
                if not obx: continue
                ious = np.array([iou(b, ob) for ob in obx])
                j = ious.argmax()
                if ious[j] >= 0.5:
                    agr[bi] += osc[j]
        if npar > 0: agr /= npar
        c['_sc'] = cs
        c['_bel'] = cs * (1.0 + agr)
        rows.append(c)
    return rows

def fnr(sel, gt):
    if not gt: return None
    cov = sum(1 for g in gt if any(iou(p, g) >= 0.5 for p in sel))
    return 1.0 - cov/len(gt)

def risk(rows, lam, key):
    ls = []
    for r in rows:
        if r['no_target'] or not r['gt_boxes_xyxy']: continue
        sc = r[key]; bx = r['pred_boxes_xyxy']
        sel = [bx[i] for i in range(len(bx)) if sc[i] >= lam]
        ls.append(fnr(sel, r['gt_boxes_xyxy']))
    return np.mean(ls) if ls else 0.0

def setsize(rows, lam, key, only='target'):
    sz = []
    for r in rows:
        tp = not r['no_target'] and bool(r['gt_boxes_xyxy'])
        if only == 'target' and not tp: continue
        if only == 'notarget' and not r['no_target']: continue
        sz.append(int(np.sum(r[key] >= lam)))
    return np.mean(sz) if sz else 0.0

def crc(calib, alpha, grid, key):
    tp = [r for r in calib if not r['no_target'] and r['gt_boxes_xyxy']]
    n = len(tp); B = 1.0
    for lam in sorted(grid, reverse=True):
        if (n*risk(tp, lam, key)+B)/(n+1) <= alpha:
            return lam
    return min(grid)

def run(canon, paras, label):
    rows = load_all(canon, paras)
    calib = [r for r in rows if r['ref_id'] % 2 == 0]
    test  = [r for r in rows if r['ref_id'] % 2 == 1]
    tp_test = [r for r in test if not r['no_target'] and r['gt_boxes_xyxy']]
    print(f"\n=== {label}: total={len(rows)} tp_test={len(tp_test)} ===")
    g_raw = np.quantile(np.concatenate([r['_sc'] for r in rows]), np.linspace(0,0.999,100))
    g_bel = np.quantile(np.concatenate([r['_bel'] for r in rows]), np.linspace(0,0.999,100))
    print(f"  {'alpha':>5} | {'raw_FNR':>7} {'raw_sz':>6} | {'bel_FNR':>7} {'bel_sz':>6} | {'shrink%':>7}")
    for a in [0.1, 0.2, 0.3, 0.4]:
        lr = crc(calib, a, g_raw, '_sc'); lb = crc(calib, a, g_bel, '_bel')
        er, sr = risk(tp_test, lr, '_sc'), setsize(test, lr, '_sc')
        eb, sb = risk(tp_test, lb, '_bel'), setsize(test, lb, '_bel')
        shrink = 100*(sr-sb)/sr if sr > 0 else 0
        valid = (er <= a+0.03) and (eb <= a+0.03)
        flag = '' if valid else '  <-- validity check'
        print(f"  {a:>5.2f} | {er:>7.4f} {sr:>6.2f} | {eb:>7.4f} {sb:>6.2f} | {shrink:>6.1f}%{flag}")

if __name__ == '__main__':
    b = '/home/p76141495/selective-grounding/dump'
    run(f'{b}/owlvit_gref_val.jsonl',
        [f'{b}/owlvit_gref_val_para1.jsonl', f'{b}/owlvit_gref_val_para2.jsonl',
         f'{b}/owlvit_gref_val_para3.jsonl'], 'gRefCOCO val  P1: belief vs raw')
