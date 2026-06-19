---
title: "Selective Conformal Risk Control (SCRC) 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# Selective Conformal Risk Control（SCRC）詳細導讀

## 0. 論文基本資訊

- **論文標題**：Selective Conformal Risk Control
- **作者**：Yunpeng Xu, Wenge Guo, Zhi Wei
- **單位**：New Jersey Institute of Technology（CS + Math Sciences）
- **arXiv**：[2512.12844v2](https://arxiv.org/abs/2512.12844)（v2 提交於 2026-04-27）
- **任務範疇**：把 conformal risk control 與 selective classification（reject option）統一成兩階段框架

> 註：這是 reading-log「方法骨幹補完」標的中**與 CRS 最直接相關**的一篇——CRS 也是「先棄答 gate、再對接受子集控 recall」的兩階段結構。SCRC 是必引必切割對象，地位類似 [read-seqcrc-summary.md](read-seqcrc-summary.md)（結構同源的最近鄰居）。本篇理論建在 [read-rcps-summary.md](read-rcps-summary.md) / CRC 之上。

---

## 1. 問題與動機

### 1.1 兩個既有方法各有缺口

- **Conformal prediction (CP)**：分布無關有限樣本 coverage 保證，但**集合常常過大**（為達 coverage 把幾乎所有 label 都塞進去），實用性受限。
- **Selective classification（reject option）**：模型對不確定樣本**棄答**，用 rejection threshold 換 coverage vs accuracy——但本身**無分布無關集合保證**。

### 1.2 主張：兩者互補，合成 SCRC

> 與其對**所有**樣本都產生大集合，不如**對不確定樣本棄答、只對接受子集產生緊緻校準集合**——「集合最有價值的地方」正是高信賴子集。

SCRC = 兩階段風險控制：
- **第一階段（selection control）**：決定哪些樣本被接受。
- **第二階段（risk control）**：對被接受樣本建構校準預測集合。
保證是 **selective 的**：只對「被接受的子母體」成立，被拒樣本是刻意 deferred 而非硬給一個無資訊集合。

---

## 2. 問題形式化（§2）

- base classifier `f : X → [0,1]^K`，selection function `g : X → [0,1]`（信賴度）。
- 兩個校準門檻 `λ = (λ1, λ2)`：
  - `λ1` 控**接受/棄答**（`g(X) < 1−λ1` → 輸出 ∅ 棄答）；
  - `λ2` 控**接受樣本的集合大小**（`C_{λ2}(X) = {k : f(X)_k ≥ 1−λ2}`）。
- 目標（Selective Conformal Classification Problem，式 8）：

  在 **選擇覆蓋 ≥ γ** 且 **條件風險 ≤ α** 兩約束下，最小化接受樣本的期望集合大小：

  `min_{(λ1,λ2)} E[|C_{λ2}(X_{n+1})| | g(X_{n+1}) ≥ 1−λ1]`
  `s.t. R(f,g) ≤ α, γ(f,g) ≥ γ`

  其中條件風險 `R = E[l(C_{λ2}, Y) | g ≥ 1−λ1]`，`l` 有界且隨集合增大單調遞減。

---

## 3. 方法核心（§4）——選擇破壞可交換性的難題與修補

### 3.1 關鍵挑戰

**選擇這個動作會破壞 calibration 與 test 之間的 exchangeability**（CRC 的根本前提）。標準兩階段 risk control（指 RCPS/CRC 的 two-stage，[31]）**不能直接套用**。

### 3.2 Conditional Exchangeability（Lemma 1）

若選擇規則 `I` 是 **symmetric selection rule**（對任意 permutation，`σ(i) ∈ I(D) ⟺ i ∈ I(D_σ)`），則**條件在選擇事件 E_I 上，被選子集 {(X_i,Y_i)}_{i∈I} 仍可交換**。

→ 這是全文理論支點：只要 selection rule 對稱，就能在被選子集上恢復 exchangeability，重新套用 CRC。

### 3.3 第一階段控制（§4.2）

- 第一階段損失只依賴 X：`L^(1)(X; λ1) = 1{g(X) < 1−λ1}`。
- **關鍵設計**：門檻 `λ̂1` 用 **calibration + test 一起**算經驗分位數（式 11），使 `λ̂1` 成為 (X_1,…,X_{n+1}) 的對稱函數 → 保住對稱性。（對比標準 CRC 只用 calib，會破壞對稱。）
- 保證 `P(g(X_{n+1}) ≥ 1−λ̂1) ≥ γ`。

### 3.4 第二階段控制（§4.3）+ 主定理

- 在被選子集上套標準 CRC counting rule 選 `λ̂2`（式 15）。
- **Theorem 2（Selective CRC Guarantee）**：資料可交換、`λ̂1` 由式 11、`μ1 ≤ λ̂1` 為對稱函數、`λ̂2` 由式 15 ⇒

  `E[l(C_{λ̂2}(X_{n+1}), Y_{n+1}) | g(X_{n+1}) ≥ 1−μ1] ≤ α`（**條件風險控制**）
  且 `P(g(X_{n+1}) ≥ 1−μ1) ≥ γ`（**選擇覆蓋**）。

- **Feasibility check**：需 `(m+1)α − 1 > 0`，即被選子集大小 `m ≥ 1/α − 1`，否則跳過該候選或提高 μ1。

### 3.5 兩個變體

| 變體 | 機制 | 保證 | 代價 |
|------|------|------|------|
| **SCRC-T**（transductive） | 門檻對 calib+test 對稱重算 | 嚴格 exchangeability、exact 有限樣本 | 每個 test 點要重算 |
| **SCRC-I**（inductive） | 重用 calibration 門檻 | **PAC-style 高機率** | 計算高效、略保守、可部署 |

實驗（兩個 benchmark）：兩者都達到目標 coverage/risk，效能幾乎相同；SCRC-I 略保守但實用得多。相較標準 CP **顯著縮小集合大小**。

---

## 4. 對使用者主線（CRS）的對位與切割

> CRS 主線見 [[crs-pivot-conformal-referring-set]] / [[crs-redteam-p0-fixes]] / [[crs-litreview-15papers]]。

### 4.1 為何是「最近鄰居」（結構同源）

SCRC 與 CRS **架構幾乎平行**：
- SCRC：`λ1` 棄答 gate + `λ2` 集合大小，**兩門檻兩階段，棄答+risk 雙保證**。
- CRS：OWL-ViT gate（控可用性/棄答/no-target）+ GroundingDINO box（控 referring set recall/緊緻），**兩 base 兩參數，recall+abstention 雙保證**。

→ 審稿人極可能拿 SCRC 質疑「你這不就是 SCRC 換到 grounding？」**必須主動引用 + 明確切割**。

### 4.2 三條不可被吸收的差異軸（切割策略）

1. **跨 base factorization vs 單模型雙門檻**：SCRC 的 `(λ1, λ2)` 是**同一個模型 `f`/`g` 的兩個內部門檻**（g 是 f 的 confidence）；CRS 的兩參數分屬**兩個異質凍結模型**（OWL gate 與 GD box 各自獨立），且有 **2×2 ablation 證成 factorization 非 ensemble**（pure GD/reverse 全垮只 COMPOSE 活）。這是 SeqCRC 同款的切割軸，SCRC 也吃這刀。
2. **任務維度**：SCRC 是**封閉 K 類分類**（label set ⊆ {1..K}，softmax 分數）；CRS 是**開放詞彙語言 grounding**（referring expression → box set），分數非 softmax、集合是空間框。SCRC 的 set-construction 規則 `{k : f_k ≥ 1−λ2}` 無法原樣搬到 box。
3. **棄答語意**：SCRC 棄答 = 「對這個輸入的分類不夠有信心」；CRS 棄答 = 「no-target / 影像中無此指稱物」——是 gRefCOCO 任務內建的**正交語意軸**（forced-output N-acc 0.18→0.94），不是單純的低信賴 reject。

### 4.3 可借鏡 / 可正面利用

1. **conditional exchangeability（Lemma 1）的修補意識**：CRS 若 OWL gate 棄答後再對接受子集校準 GD recall，**也會遇到「選擇破壞 exchangeability」問題**。SCRC 的 symmetric-selection-rule 修補（門檻用 calib+test 對稱算）是**現成可移植的技術點**——CRS 的 calib-only grid 是否踩到這個 leakage 值得對照（呼應 [[crs-redteam-p0-fixes]] 已修的 calib-only leakage）。這是 SCRC 最有價值的可借鏡點。
2. **SCRC-T vs SCRC-I 的取捨對位**：CRS 用的 LTT 屬 inductive/PAC 型（≈SCRC-I）；可在 future work 提 transductive 變體（exact 但每測試點重算）作為對照。
3. **feasibility check `m ≥ 1/α−1`**：CRS 在高棄答率時被選子集可能太小導致無法控 recall——SCRC 的可行性下界給了明確的「最小被選樣本數」公式，可直接用於 CRS 的 robustness 討論。

### 4.4 引用注意

- SCRC v2 (2026-04) 比 CRS 主結果晚，屬**同期工作**；framing 要寫成「concurrent，結構同源但任務/組合/棄答語意三軸不同」，類似 BCEA 的「同期同調不同任務」處理（[read-bcea-summary.md](read-bcea-summary.md)）。
- 它自承理論「conceptually related to two-stage risk control [31]」——CRS 引用時可一併指出兩者共同的 two-stage 祖先（RCPS/LTT），把差異收斂到 cross-base 那一軸。

**引用優先級**：高（必引必切割，結構最近鄰居之一；conditional-exchangeability 修補可借鏡）。

---

## 5. 一句話總結

> SCRC 是 **2026 年把 conformal risk control 與 selective classification 合成的兩階段框架**：第一階段對稱門檻 `λ1` 棄答低信賴樣本（用 calib+test 對稱算門檻以保住 exchangeability，Lemma 1），第二階段對被接受子集套 CRC 選 `λ2` 控集合大小，給出「接受子母體上的條件風險 + 選擇覆蓋」雙保證（SCRC-T exact / SCRC-I PAC）。它與 CRS 架構幾乎平行（雙門檻雙保證），是**必引必切割的結構最近鄰居**——切割三軸＝跨 base factorization、開放詞彙 grounding vs 封閉 K 類、no-target 正交棄答語意；而其 conditional-exchangeability 修補是 CRS 可直接借鏡的技術點。

---

## 6. 參考連結

- arXiv：<https://arxiv.org/abs/2512.12844>
- 理論基礎：Conformal Risk Control（Angelopoulos et al., 2208.02814）；RCPS（[read-rcps-summary.md](read-rcps-summary.md)）；two-stage risk control [31]
