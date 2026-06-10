#!/usr/bin/env python3
"""v7 實驗彙整：讀 runs/exp/<variant>_<obs>_s<seed>_compare/figs/*.json，
算每 (variant, obs) 跨 seeds 的 P1/P2/P3 mean±std，輸出 experiments/summary.json 並印表。
headline = dual_partial return_gap vs dual_full return_gap（P3 恢復判讀）。"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# Windows cp950 console 無法輸出 ± 等字元；強制 UTF-8 stdout（Linux 遠端 no-op）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

RUN_ROOT = Path("runs/exp")
OUT = Path("experiments/summary.json")


def mean_std(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # sample std
    return m, math.sqrt(var)


def collect():
    # v7: tag = <variant>_<obs>_s<seed>_compare
    pat = re.compile(r"^(single|dual)_(full|partial)_s(\d+)_compare$")
    rows = {}  # key: "<variant>_<obs>" -> list of per-seed dicts
    for d in sorted(RUN_ROOT.glob("*_compare")):
        m = pat.match(d.name)
        if not m:
            continue
        variant, obs, seed = m.group(1), m.group(2), int(m.group(3))
        key = f"{variant}_{obs}"
        figs = d / "figs"
        try:
            p1 = json.loads((figs / "phenomenon1_entropy_gap.json").read_text())
            p2 = json.loads((figs / "phenomenon2_entropy_decay.json").read_text())
            p3 = json.loads((figs / "phenomenon3_belief_vs_single.json").read_text())
        except FileNotFoundError:
            continue
        rows.setdefault(key, []).append({
            "seed": seed,
            "p1_gap": p1["ambiguous_mean"] - p1["exact_mean"],
            "p2_slope": p2.get("slope"),
            "p2_drop": p2.get("per_episode_mean_drop"),
            "p3_return_gap": p3.get("return_gap"),
            "p3_success_gap": p3.get("success_gap"),
            "belief_return": p3.get("belief_avg_return"),
            "single_return": p3.get("single_avg_return"),
        })
    return rows


def main():
    rows = collect()
    summary = {}
    for key, recs in rows.items():
        recs = sorted(recs, key=lambda r: r["seed"])
        agg = {}
        for k in ["p1_gap", "p2_slope", "p2_drop", "p3_return_gap", "p3_success_gap"]:
            vals = [r[k] for r in recs if r[k] is not None]
            m, s = mean_std(vals)
            agg[k] = {"mean": m, "std": s, "n": len(vals), "per_seed": vals}
        summary[key] = {"seeds": [r["seed"] for r in recs], "agg": agg, "rows": recs}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    def fmt(a):
        return f"{a['mean']:+.4f} ± {a['std']:.4f} (n={a['n']})" if a["mean"] is not None else "n/a"

    order = ["single_full", "single_partial", "dual_full", "dual_partial"]
    print("=" * 70)
    print("v7 summary ({single,dual} hint × {full,partial} obs)")
    print("=" * 70)
    for key in order:
        if key not in summary:
            continue
        a = summary[key]["agg"]
        print(f"\n[{key}]  seeds={summary[key]['seeds']}")
        print(f"  P1 gap        : {fmt(a['p1_gap'])}")
        print(f"  P2 slope      : {fmt(a['p2_slope'])}")
        print(f"  P2 drop       : {fmt(a['p2_drop'])}")
        print(f"  P3 success_gap: {fmt(a['p3_success_gap'])}")
        print(f"  P3 return_gap : {fmt(a['p3_return_gap'])}   <-- 主判準")

    # headline: partial-obs 是否讓 P3 在 dual 下出現（vs fully-obs control）
    if "dual_partial" in summary and "dual_full" in summary:
        dp = summary["dual_partial"]["agg"]["p3_return_gap"]
        df = summary["dual_full"]["agg"]["p3_return_gap"]
        print("\n" + "-" * 70)
        print("HEADLINE — P3 恢復判讀 (dual)")
        if dp["mean"] is not None and df["mean"] is not None:
            print(f"  dual_partial return_gap : {fmt(dp)}")
            print(f"  dual_full    return_gap : {fmt(df)}   (control，應 ~=0)")
            print(f"  partial − full          : {dp['mean'] - df['mean']:+.4f}")
            ps = dp["per_seed"]
            if ps:
                same_sign = all(x > 0 for x in ps)
                lower = dp["mean"] - dp["std"]
                print(f"  dual_partial 每 seed    : {[round(x,4) for x in ps]}  "
                      f"(同號={'是' if same_sign else '否'})")
                print(f"  pre-reg 判準 (mean−1σ>0) : {lower:+.4f}  "
                      f"→ {'通過' if lower > 0 else '未通過'}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
