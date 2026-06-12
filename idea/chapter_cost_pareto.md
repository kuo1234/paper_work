# Cost–Risk–Coverage Pareto（spec §5.8）— 讓「輕量」成為實驗主張

> 紀律（紅隊）：「輕量」不可當形容詞。本節以**每 query forward 次數**（最誠實的成本軸）對上它換來的 reliability，回答：在接近 1× 推論成本下，凍結 base + post-hoc calibrator 能吃到多少 reliability benefit。文獻方法（HieA2G/VIRO）base 不同，僅作定性座標對照，**不在同一 reliability 軸上直接比較**。

## 成本軸的關鍵區分

| 訊號類別 | forward/query | 為何 | 代表方法 |
|---|---|---|---|
| score-based（OWL-ViT entropy / no-target / margin） | **1×** | 候選分數是**單次 forward 的免費副產品** | C2 score_entropy gate、C3 no-target gate |
| forced-output / 單一 threshold | 1× | 同上 | baseline |
| cross-prompt consistency | **K=4×** | 需 K 個 paraphrase prompt 各跑一次 forward | C1/C2 consistency gate、C4 transfer 用 |

GB10 實測 throughput：OWL-ViT base-patch32 ≈ **36.6 query/s/forward**；CLIP-VG batched ≈ 204 q/s（batch eval，effective）。

## Pareto 表（reliability 為 val split 點估計，CI 見 bootstrap 節）

| 方法 | base | 訓練成本 | forward/query | query/s | reliability |
|---|---|---|---|---|---|
| forced-output base | OWL-ViT | 無 | 1× | 36.6 | no-target AUROC 0.50（無 abstain 能力） |
| score-threshold (top1) | OWL-ViT | 無 | 1× | 36.6 | no-target AUROC 0.822 |
| **score_entropy gate** | OWL-ViT | 無 | **1×** | 36.6 | **C2 AURC vs random −22.6%** |
| **learned no-target gate (C3)** | OWL-ViT | 小（logreg） | **1×** | 36.6 | **no-target AUROC 0.824** |
| consistency gate (C2) | CLIP-VG | 小（logreg） | 4× | 51 | C2 AURC vs random −52.3% |
| consistency gate (C2) | OWL-ViT | 小（logreg） | 4× | 9.2 | C2 AURC vs random −22.9% |
| HieA2G（文獻，**base 不同**） | RN101 | 高（全監督） | — | — | full-GREC N-acc ~0.60 |
| VIRO（文獻，**base 不同**） | GDINO/Qwen | 0 base / 重型 pipeline | — | E2E 12.92 q/s | balanced acc 0.611 |

## 主張（可守版本）

1. **1× 成本即取得主要 reliability**：OWL-ViT 的 score 副產品免費，single-forward 即得 no-target AUROC **0.824**（C3）與 C2 AURC 相對 random 降 **22.6%**。這是「輕量」的硬證據——不需額外 forward、不重訓 base。
2. **K=4× 成本買到的是 cross-base transferability（C4），不是更高的 within-base reliability**：consistency 需 4× forward，但它是 base-agnostic、零參數可跨 base 轉移的訊號（C4 主結果）。亦即成本與「可轉移性」掛鉤，而非單純準度。
3. **相對重型 pipeline**：VIRO 走 LLM program + per-operator verifier（E2E 12.92 q/s 含重型推論），HieA2G 全監督專訓 counting head。本研究佔「訓練成本 ↓、推論成本 ↓」光譜的最輕量端；表中文獻欄僅定性對照，因 base 不同不主張直接比 accuracy（符合 §3.2/§3.4/§3.5 定位）。

**圖**：`idea/figures/cost_pareto.png`（x=forward/query，y=reliability，標註 1× 的核心主張）。腳本 `src/cost_pareto.py`，數據 `dump/cost_pareto.json`。

> **實驗面至此收尾**：C1–C4 + M4 + bootstrap CI + cost-Pareto 全部到位。後續為純寫作（Intro / Related Work / Method / Experimental Setup / C1-C3 results / M4 boundary / C4 transfer），不再新增實驗（紅隊 Month 6 起）。
