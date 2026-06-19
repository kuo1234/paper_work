# Decomp CRS Robustness + R_total 結果（紅隊 P0-B/P0-C，2026-06-18）

腳本 `src/decomp_robust.py`。三 split × {parity / random×5 seeds / image-disjoint} × {Frozen, Decomp}，報 R1/R2/R3/size/R_total/n_feas。R_total = R3 + (1-R3)*R1（overall target failure）。

## 結果摘要

| split | mode | Frozen (sz/R1/Rtot) | Decomp (sz/R1/Rtot) | 判讀 |
|---|---|---|---|---|
| val | parity | 3.24/0.191/0.534 | 3.24/0.165/0.519 | ✅ Pareto |
| val | random5 | 3.24/0.191/0.472 | 3.22/0.164/0.452 | ✅ 穩(5/5 feas) |
| val | image-disj | 3.13/0.205/0.536 | 3.11/0.181/0.522 | ✅ |
| testA | parity | 2.02/0.260/0.588 | 1.96/0.244/0.579 | ✅ 雙贏 |
| testA | random5 | 2.05/0.256/0.571 | 1.96/0.248/0.566 | ✅ 穩 |
| testA | image-disj | 2.06/0.252/0.577 | 1.96/0.247/0.550 | ✅ |
| testB | parity | 3.50/0.168/0.512 | 3.34/0.163/0.509 | ✅ size↓ |
| testB | image-disj | 3.39/0.172/0.575 | 3.29/0.159/0.568 | ✅ |
| **testB** | **random5** | **3.53/0.171/0.531** | **2.55/0.226/0.556** | ⚠️ **trade-off 非 Pareto** |

## 結論（誠實）

- **整體站得住**：7/8 cell decomp 的 R_total ≤ frozen，全部 split-run 都 feasible（駁倒「parity 專屬 operating point」攻擊）。deferral(R3) 每 cell 幾乎持平 frozen → R1 改善非靠偷換 deferral 分母（P0-C 守住）。
- **⚠️ 唯一弱點 testB random5**：decomp size 大降(3.53→2.55)但 R1 升(0.171→0.226)、Rtot 升(0.531→0.556)，std 大(size±0.65, R1±0.053)。這是 size-recall trade-off，非純 Pareto。某些 seed 下 LTT 選到「集合更小但漏檢更多」的 operating point。
- **對 claim 的影響**：「跨 split 純帕累托改善」只在 parity/image-disjoint 穩；testB random 下退化成 trade-off。**這正驗證紅隊 B 的擔憂——必須誠實寫成 limitation，不能宣稱所有 split mode 都 Pareto。** 安全 claim：decomp 在固定 risk budget 下「reduce answered-target FNR or set size」（or 不是 and），testB random5 正是換 size 不換 R1 的案例。
- 下一步排查：testB random5 的高 std 可能來自 testB 樣本結構（n_gt≥3 比例高？）。可選做 per-seed operating point 檢視。
