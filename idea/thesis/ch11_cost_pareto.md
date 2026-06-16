# 第 11 章　Cost–Risk Pareto：讓「輕量」成為實驗主張

本論文全程把「輕量」當作可量測的實驗主張，而非形容詞。本章以最誠實的成本軸——**每查詢前向傳遞次數（forward/query）**——對上它換來的可靠性（reliability），回答一個具體問題：**在接近 1× 推論成本下，凍結基礎模型 + 事後校準器能吃到多少可靠性 benefit？** 文獻方法（HieA2G／VIRO）基礎模型不同，僅作定性座標對照，不在同一可靠性軸上直接比較。

## 11.1 成本軸的關鍵區分

不同訊號的推論成本差異，全部歸結到「需要幾次前向傳遞」：

**表 11.1　訊號的前向成本**

| 訊號類別 | forward/query | 為何 | 代表方法 |
|---|---|---|---|
| score-based（OWL-ViT entropy／no-target／margin） | **1×** | 候選分數是**單次前向的免費副產品** | C2 score_entropy gate、C3 no-target gate |
| forced-output／單一 threshold | 1× | 同上 | baseline |
| cross-prompt consistency | **K=4×** | 需 K 個改寫提示各跑一次前向 | C1/C2 consistency gate、C4 transfer 用 |

GB10 實測 throughput：OWL-ViT base-patch32 約 **36.6 query/s/forward**；CLIP-VG batched 約 204 q/s（batch eval，effective）。

## 11.2 Pareto 表

reliability 為 val split 點估計（CI 見各章 bootstrap 節）：

**表 11.2　Cost–Risk–Coverage Pareto**

| 方法 | base | 訓練成本 | forward/query | query/s | reliability |
|---|---|---|---|---|---|
| forced-output base | OWL-ViT | 無 | 1× | 36.6 | no-target AUROC 0.50（無棄答能力） |
| score-threshold (top1) | OWL-ViT | 無 | 1× | 36.6 | no-target AUROC 0.822 |
| **score_entropy gate** | OWL-ViT | 無 | **1×** | 36.6 | **C2 AURC vs random −22.6%** |
| **learned no-target gate (C3)** | OWL-ViT | 小（logreg） | **1×** | 36.6 | **no-target AUROC 0.824** |
| consistency gate (C2) | CLIP-VG | 小（logreg） | 4× | 51 | C2 AURC vs random −52.3% |
| consistency gate (C2) | OWL-ViT | 小（logreg） | 4× | 9.2 | C2 AURC vs random −22.9% |
| HieA2G（文獻，**base 不同**） | RN101 | 高（全監督） | — | — | full-GREC N-acc ~0.60 |
| VIRO（文獻，**base 不同**） | GDINO/Qwen | 0 base / 重型 pipeline | — | E2E 12.92 q/s | balanced acc 0.611 |

## 11.3 主張

1. **1× 成本即取得主要可靠性。** OWL-ViT 的 score 副產品免費，single-forward 即得 no-target AUROC **0.824**（C3）與 C2 AURC 相對隨機降 **22.6%**。這是「輕量」的硬證據——不需額外前向、不重訓基礎模型。

2. **K=4× 成本買到的是 cross-base transferability（C4），不是更高的 within-base 可靠性。** consistency 需 4× 前向，但它是 base-agnostic、零參數可跨基礎模型轉移的訊號（C4 主結果）。亦即成本與「可轉移性」掛鉤，而非單純準度。這也呼應第 3 章的「成本↔可轉移性」二分：(A) 擾動訊號貴但可轉移、(B) 分數訊號免費但 base-specific。

3. **相對重型 pipeline。** VIRO 走 LLM program + per-operator verifier（E2E 12.92 q/s 含重型推論），HieA2G 全監督專訓 counting head。本研究佔「訓練成本 ↓、推論成本 ↓」光譜的最輕量端；表中文獻欄僅定性對照，因基礎模型不同不主張直接比 accuracy。

（圖 `cost_pareto.png`：x = forward/query，y = reliability，標註 1× 的核心主張。）

## 11.4 小結

Cost–Risk Pareto 把「輕量」從形容詞變成可量測主張：在 1× 推論成本、零或極小訓練成本下，凍結基礎模型 + 事後校準器即取得 no-target AUROC 0.824 與相對隨機降 22.6% 的選擇性風險改善；額外的 4× 前向成本買到的是跨基礎模型可轉移性，而非更高的 within-base 準度。相對於 HieA2G（全監督）與 VIRO（重型 neuro-symbolic pipeline），本研究穩居成本—可靠性光譜的最輕量端。這條成本軸也是 CRS（第 9 章）的背景：CRS 的跨基礎模型組合需要兩個基礎模型各一次前向，仍在這個輕量光譜內，卻換來三風險的分布無關保證。
