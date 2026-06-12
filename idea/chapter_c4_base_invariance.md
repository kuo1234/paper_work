# C4 深化章節草稿（中文）— Grounding 不確定性的 Base-invariant 結構

> 章節定位：把 C4 從「一個 cross-base transfer 數字」升級成一個有機制、有結構的**發現**。
> 這是全篇最接近「科學發現」的貢獻，也是對抗「只是 calibration / threshold」批評的護城河——
> 因為 per-pipeline 的驗證方法（VIRO、True/False Verification）結構上無法提出 transfer 主張。
> 在論文中建議放在 C4（cross-base / cost）章的核心節，標題：
> **Base-invariant Structure of Grounding Uncertainty**（中：Grounding 不確定性的 base-invariant 結構）

---

## C4.1 動機：transfer 數字背後是什麼？

先前的 cross-base transfer 結果顯示：把 CLIP-VG 上定義的 raw consistency 訊號零參數、無 refit 直接套到 OWL-ViT，risk-coverage AURC（0.488）≈ OWL-ViT 自身 native gate（0.489），比 random 降 15.9%。

但「一個 transfer 數字」不足以支撐 thesis-level 主張。本節追問其**機制**：consistency 為什麼能轉移？轉移的到底是訊號的什麼性質？並誠實界定**哪些性質可轉移、哪些不可**。

兩個 base 結構上極為不同（CLIP-VG 是 single-box regression、OWL-ViT 是候選評分 detector），且 RefCOCO val 上準度差近一倍（**CLIP-VG 0.843 vs OWL-ViT 0.419**）。若同一不確定性訊號在如此不同的兩 base 上仍以相同方式運作，則 grounding 失敗存在 base 共享的結構。分析在兩 base 逐列對齊的 10834 個共同樣本上進行（trailing-idx join，0 表達式錯位）。

---

## C4.2 發現一：訊號的「資訊性 / 排序結構」是 base-invariant 的

對兩 base 共有的訊號（cross_prompt_consistency、prompt_box_dispersion）逐一比較其對 correctness 的資訊性：

| 訊號 | CLIP-VG within AUROC | OWL-ViT within AUROC | corr(訊號, 答對) CLIP-VG | corr OWL-ViT |
|---|---|---|---|---|
| cross_prompt_consistency | 0.722 | 0.641 | 0.288 | 0.271 |
| prompt_box_dispersion | 0.654 | 0.634 | 0.153 | 0.231 |

**判讀**：同一個 grounding-specific 訊號，在準度差近一倍的兩 base 上，都以**相同方向、相近相關性**預測錯誤（consistency corr 0.288 vs 0.271，幾乎相等）。這不是「訊號剛好在各自 base 有用」的巧合，而是訊號所捕捉的「referential 不穩定性」是 grounding 任務本身的性質。

---

## C4.3 發現二：同一難度軸對兩 base 同向有效（hardness 是 base-invariant）

更根本的證據不在訊號統計，而在錯誤本身。以 CLIP-VG 的 consistency 作為**共享難度軸**（分 10 個等量分箱），檢視兩 base 在每個分箱的 error rate：

- 兩條 error-rate 曲線**同向**：consistency 越低（query 越不穩定），CLIP-VG 與 OWL-ViT 的 error rate **都上升**。
- 亦即：同一個訊號標出的 hard sample，對兩個獨立、準度迥異的凍結 base 都更容易答錯。難度排序是 query 的性質，跨 base 共享。

（圖 `c4_hardness.png`：x=共享 consistency 軸，雙曲線分別為兩 base 的 bin error rate。）

---

## C4.4 誠實的邊界：什麼**不**可轉移

base-invariant 是有限定的。三個誠實的限制必須寫明，否則過度宣稱：

1. **判別強度（effect size）是 base-specific 的。** 以最低 vs 最高 consistency 的 20% 尾端比較 error lift：CLIP-VG 達 **13.9×**（0.348 vs 0.025），OWL-ViT 僅 **1.32×**（0.677 vs 0.513）。consistency 對 CLIP-VG 是極強的難度指標，對 OWL-ViT 只是弱訊號——**方向不變，強度大不同**。

2. **逐樣本 correctness 的跨 base 相關性是弱的**（phi = 0.186）。並非「同一批 sample 兩 base 一起對 / 一起錯」，而是「低 consistency 區域兩 base 各自的 error 都統計性偏高」。可轉移的是**統計趨勢**，不是**逐樣本一致性**。

3. **訊號的原始數值刻度不可轉移。** 把 CLIP-VG 的 consistency 原始值直接拿去排 OWL-ViT 的 correctness，AUROC 僅 0.551（近隨機）；多特徵 fitted gate 的最佳組合權重也 base-specific（CLIP-VG consistency:dispersion ≈ 1.58:0.86，OWL-ViT native ≈ 0.66:0.07）。可轉移的是**結構**，不是**校準**。

---

## C4.5 收斂主張（可防守版本）

> **Grounding-specific 不確定性訊號（cross-prompt consistency）所定義的「query 難度排序」是 base-invariant 的**：同一訊號在準度差近一倍的兩個凍結 base 上，都以相同方向、相近相關性預測錯誤，且其標定的 hard sample 對兩 base 同向更難。**但其判別強度（effect size）、逐樣本一致性、與最佳組合權重（校準）是 base-specific 的。**

換句話說：**結構可轉移，校準不可轉移。** 這一句同時解釋了原本 C4 的兩個觀察——為什麼 rank-based 的 risk-coverage 能轉移（只看排序結構），以及為什麼 fitted 多訊號 gate 轉移較差（吃絕對數值與 base-specific 權重）。

**在 thesis 中的角色**：這是本研究最接近「科學發現」的貢獻，也是對 VIRO / True-False Verification 唯一站得住的真區辨——它們是 per-pipeline / per-VLM，結構上無法提出「不確定性有 base 共享結構」這類 transfer 主張。

> 〔自評 — 寫給自己，最終稿移除〕
> 這是中等強度亮點：真實、有機制、誠實，足以當一個 named contribution + 一張主圖（c4_hardness.png）。
> 但 phi 弱、effect size 差異大，主張必帶限定詞，不會是「轟動級」。
> 若需要 invention 級亮點（主動修正的 candidate-contrastive operator B），需另評估風險（見紅隊警告：B 易被打成 TTA、且 True-False 壓線）。

**圖**：`idea/figures/c4_hardness.png`（雙 base error-rate vs 共享 consistency 軸）、`idea/figures/c4_invariance.png`（belief-space 幾何，佐證）。
**腳本**：`src/c4_invariance.py`、`src/c4_hardness.py`。**數據**：`dump/c4_invariance.json`、`dump/c4_hardness.json`。
