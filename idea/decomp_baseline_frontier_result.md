# Strong Baseline (P1-D) + Frontier 四指標 (P0-A) 結果（2026-06-18）

## P0-A：Matched Recall-Size Frontier 四指標（`src/decomp_frontier.py`）

oracle pool 層（繞過 LTT），真拆 case 上 full vs decomp pool。

| split | rec@sz3 (full→decomp) | size@rec0.8 (full→decomp) | AURC (full→decomp) |
|---|---|---|---|
| val | 0.686 → 0.913 | 5.48 → **2.08** | 0.690 → 0.875 |
| testA | 0.693 → 0.919 | 4.74 → **2.10** | 0.693 → 0.881 |
| testB | 0.631 → 0.813 | 6.77 → **2.78** | 0.629 → 0.805 |

**size@recall0.8 最關鍵**：達 80% recall，full 需 4.7-6.8 框、decomp 只需 2.1-2.8 框（不到一半成本）。

### 指標(4) anti-cheat：calib 選 λ、eval 評（堵「偷看 eval sweep」）

| split | target | full eval (sz,rec) | decomp eval (sz,rec) |
|---|---|---|---|
| val | size~3 | 2.70, 0.682 | **3.16, 0.912** |
| testA | size~3 | 2.80, 0.673 | **3.09, 0.910** |
| testB | size~3 | 3.09, 0.620 | **2.76, 0.799** |

→ calib 選定 λ 在獨立 eval half 上 decomp recall 優勢完全保持(+22~24pp)。**非偷看 eval sweep。frontier 正當性最強證據。**

## P1-D：Strong full baselines（`src/decomp_strongbaseline.py`）

full-expr pool 加各種 consolidation，全接同套 CRS/LTT，與 decomp 比 size/R1。

| split | baseline | size | R1 | vs decomp |
|---|---|---|---|---|
| val | full raw | 3.24 | 0.191 | decomp R1 0.165 贏 |
| | full+NMS@0.5 | 2.31 | 0.211 | decomp R1 贏 |
| | full+NMS@0.7 | 2.99 | 0.174 | decomp R1 0.165 贏 |
| | **decomp** | 3.24 | **0.165** | — |
| testA | full+NMS@0.7 | 2.27 | **0.214** | ⚠️ full+NMS R1 更低(0.214<0.244) |
| | **decomp** | 1.96 | 0.244 | decomp size 更小 |
| testB | full+NMS@0.7 | 2.29 | 0.222 | decomp R1 0.163 贏 |
| | **decomp** | 3.34 | **0.163** | — |

## 誠實裁決（紅隊 D 驗證）

- **紅隊 D 是對的**：threshold=0 full raw 確實偏弱。NMS 大幅改善 full baseline 的 size（val 3.24→2.31）。論文不能只用 full raw 當 baseline。
- **decomp 的護城河是 R1（漏檢質量）**：val/testB 上 decomp R1 低於所有 full 變體（含 NMS）。NMS 只去重縮集合，不改「pool 有沒有留對框」；decomp 的 per-part 乾淨分數才降 R1。
- **⚠️ 但非單點全勝**：testA full+NMS@0.7 的 R1(0.214) 反而低於 decomp(0.244)，只是 size 大。testA 上 decomp=更小集合、full+NMS=更低漏檢，各擅勝場。
- **frontier(#3) vs LTT單點(#4) 落差**：#3 oracle pool frontier decomp 全面碾壓；#4 經 LTT 選單點，testA 被 LTT 替 full+NMS 選到低 R1 點（用大 size 換）。→ frontier 才是公平全貌，單點比較受 operating-point 選擇干擾。**這正驗證「matched=mechanism evidence、LTT=最終證據，須並陳」。**

## 安全 claim（更新）
> Decomp dominates the recall-size frontier (size@recall0.8 needs <half the boxes; anti-cheat calib/eval split holds). When wrapped by CRS/LTT, it achieves the lowest answered-target FNR on val/testB; on testA a strong full+NMS baseline reaches lower FNR at larger set size, so decomp's advantage is frontier shape, not single-point dominance across all splits.
