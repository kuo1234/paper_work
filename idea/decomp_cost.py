"""
誠實成本表 (red-team #6): Expression-Decomposed CRS 的 inference-time compute 成本。

紅隊裁決: decomp 不能宣稱 cheap。每個被 router 考慮的 query 都呼叫 1 次 VLM
(Qwen2.5-VL-7B) 判斷要不要拆; 真正被拆的 (decomposed=True) query 再做 n_parts 次
GDINO forward; passthrough (decomposed=False) 只做 1 次 GDINO forward。
full-expression frozen baseline 永遠只做 1 次 GDINO forward, 0 次 VLM。

關鍵 routing 事實 (來自 decomp_dump{,2,3}.py, 三版邏輯一致):
    parts = decompose(expr) if is_tp else None
    其中 is_tp = (not no_target) and bool(gt_boxes)
也就是 VLM 在「每個 target-present query」都被呼叫一次 (no-target query 直接 passthrough,
0 次 VLM)。所以 dump 雖然只記了 decomposed/n_parts, 但 VLM 呼叫集合 = target-present 集合,
可以從 no_target 欄位精確還原 (不是只報 decomposed 的下界)。

本腳本因此報三個 VLM 數字:
  - 下界  = decomposed case 數 / 總 query   (只算真正觸發拆解的)
  - 實際  = target-present 數 / 總 query     (router 對所有 target-present 都呼叫 VLM)
  - 上界  = 全體 1.0 / query                 (若 router 改成對 no-target 也呼叫)

GDINO forward 計法:
  - frozen   : 每 query 固定 1 次
  - decomp   : passthrough -> 1 次; decomposed -> n_parts 次
             total = (#passthrough * 1) + Σ_decomposed n_parts
"""
import json, collections, os

DUMP = "dump"
SPLITS = ["val", "testA", "testB"]
# 已知 set size (來自主表; 此腳本只負責 compute 成本, size 直接引用)
SET_SIZE = {
    "frozen": {"val": 3.24, "testA": 2.02, "testB": 3.50},
    "decomp": {"val": 3.24, "testA": 1.96, "testB": 3.34},
}


def load(split):
    rows = []
    with open(os.path.join(DUMP, f"gdino_gref_{split}_decomp.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def analyze(split):
    rows = load(split)
    n_total = len(rows)
    n_no_target = sum(1 for d in rows if d.get("no_target"))
    n_tp = n_total - n_no_target  # target-present == VLM 實際被呼叫集合

    decomposed = [d for d in rows if d.get("decomposed")]
    n_dec = len(decomposed)
    n_pass = n_total - n_dec
    nparts = [int(d["n_parts"]) for d in decomposed]
    sum_nparts = sum(nparts)

    # GDINO forwards
    frozen_fwd_total = n_total * 1
    # decomp: passthrough 各 1, decomposed 各 n_parts
    decomp_fwd_total = n_pass * 1 + sum_nparts
    fwd_ratio = decomp_fwd_total / frozen_fwd_total

    # VLM calls / query
    vlm_lower = n_dec / n_total            # 只算真正觸發拆解的 (保守下界)
    vlm_actual = n_tp / n_total            # router 對所有 target-present 呼叫 (實際)
    vlm_upper = 1.0                        # 若連 no-target 也呼叫 (理論上界)

    # n_parts 直方圖
    hist = collections.Counter(nparts)

    return {
        "split": split,
        "n_total": n_total,
        "n_no_target": n_no_target,
        "n_tp": n_tp,
        "n_dec": n_dec,
        "n_pass": n_pass,
        "sum_nparts": sum_nparts,
        "frozen_fwd_per_q": frozen_fwd_total / n_total,
        "decomp_fwd_per_q": decomp_fwd_total / n_total,
        "frozen_fwd_total": frozen_fwd_total,
        "decomp_fwd_total": decomp_fwd_total,
        "fwd_ratio": fwd_ratio,
        "vlm_lower": vlm_lower,
        "vlm_actual": vlm_actual,
        "vlm_upper": vlm_upper,
        "hist": dict(sorted(hist.items())),
        "avg_nparts_dec": sum_nparts / n_dec if n_dec else float("nan"),
    }


def main():
    results = {s: analyze(s) for s in SPLITS}

    print("=" * 78)
    print("Expression-Decomposed CRS — Honest Inference-Time Compute Cost (red-team #6)")
    print("=" * 78)

    for s in SPLITS:
        r = results[s]
        print(f"\n----- {s} -----")
        print(f"  total queries        = {r['n_total']}")
        print(f"  target-present (VLM) = {r['n_tp']}   "
              f"({r['n_tp']/r['n_total']*100:.1f}%)")
        print(f"  no-target (passthru) = {r['n_no_target']}")
        print(f"  decomposed (split>1) = {r['n_dec']}   "
              f"({r['n_dec']/r['n_total']*100:.1f}% of all, "
              f"{r['n_dec']/r['n_tp']*100:.1f}% of target-present)")
        print(f"  passthrough total    = {r['n_pass']}")
        print(f"  Σ n_parts (decomp)   = {r['sum_nparts']}")
        print()
        print(f"  [1] avg GDINO forwards / query")
        print(f"        frozen (full-expr) = {r['frozen_fwd_per_q']:.4f}  (fixed 1/q)")
        print(f"        decomp             = {r['decomp_fwd_per_q']:.4f}")
        print(f"  [2] avg VLM parser calls / query")
        print(f"        frozen             = 0")
        print(f"        decomp lower bound = {r['vlm_lower']:.4f}  "
              f"(only decomposed-triggered)")
        print(f"        decomp actual      = {r['vlm_actual']:.4f}  "
              f"(VLM called on every target-present query)")
        print(f"        decomp upper bound = {r['vlm_upper']:.4f}  "
              f"(if no-target also routed)")
        print(f"  [3] n_parts histogram (decomposed cases)")
        for k, v in r["hist"].items():
            print(f"        n_parts={k}: {v}")
        print(f"        avg n_parts (decomposed only) = {r['avg_nparts_dec']:.3f}")
        print(f"  [4] total GDINO forward multiplier (decomp / frozen)")
        print(f"        {r['decomp_fwd_total']} / {r['frozen_fwd_total']} "
              f"= {r['fwd_ratio']:.4f}x")

    # ---- markdown cost table ----
    print("\n" + "=" * 78)
    print("MARKDOWN COST TABLE")
    print("=" * 78 + "\n")

    print("| Method | Train cost | Model update | VLM calls / query | "
          "GDINO forwards / query | Avg set size (val/testA/testB) |")
    print("|---|---|---|---|---|---|")

    fr = results
    frozen_fwd = "1.00 / 1.00 / 1.00"
    frozen_size = "{val} / {testA} / {testB}".format(**SET_SIZE["frozen"])
    print(f"| Frozen (full-expr) | none (training-free) | weights frozen | "
          f"0 / 0 / 0 | {frozen_fwd} | {frozen_size} |")

    dec_fwd = " / ".join(f"{fr[s]['decomp_fwd_per_q']:.2f}" for s in SPLITS)
    dec_vlm = " / ".join(f"{fr[s]['vlm_actual']:.2f}" for s in SPLITS)
    dec_size = "{val} / {testA} / {testB}".format(**SET_SIZE["decomp"])
    print(f"| Decomp (VLM-routed) | none (training-free) | weights frozen | "
          f"{dec_vlm} | {dec_fwd} | {dec_size} |")

    print("\nClaim: *training-free and weight-frozen, but with additional "
          "inference-time compute* (1 VLM routing call per target-present query "
          "+ up to n_parts GDINO forwards per decomposed query).")
    print("\nNote: VLM calls/query reported as the ACTUAL routing cost "
          "(VLM invoked on every target-present query). Forward multipliers:")
    for s in SPLITS:
        print(f"  {s}: {fr[s]['fwd_ratio']:.3f}x GDINO forwards vs frozen.")


if __name__ == "__main__":
    main()
