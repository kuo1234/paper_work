---
title: "RCPS: Distribution-Free, Risk-Controlling Prediction Sets 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# RCPS — Distribution-Free, Risk-Controlling Prediction Sets（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：Distribution-Free, Risk-Controlling Prediction Sets
- **作者**：Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik, Michael I. Jordan（前三位 equal contribution）
- **單位**：UC Berkeley（Stanford/Berkeley 統計+EECS 圈）
- **arXiv**：[2101.02703v3](https://arxiv.org/abs/2101.02703)（v3 提交於 2021-08-04；發表為 JACM 2021）
- **專案頁**：angelopoulos.ai/blog/posts/rcps/
- **任務範疇**：分布無關、有限樣本的風險控制集合預測（distribution-free risk control）

> 註：本文是 CRS 主線所用 **LTT（Learn then Test）的理論前身**。同一作者群（Bates/Angelopoulos）隔年提出 LTT（2110.01052）把 RCPS 的「單調風險 + 單參數 UCB」推廣到「多參數 + 非單調 + 多風險」的 multiple-testing 框架。讀懂 RCPS = 讀懂 CRS 風險控制機制的根。與本目錄 [read-crc-nonmonotonic-summary.md](read-crc-nonmonotonic-summary.md)（把 LTT 定位成非單調 baseline）、[read-conformal-iseg-summary.md](read-conformal-iseg-summary.md)（同型 set≥1 保證）互為一組。

---

## 1. 問題與動機

### 1.1 核心問題

黑箱預測模型部署到高風險場景（醫療、自駕）時，**很少附帶可靠的不確定性量化**，預測失敗往往是「沉默的失敗」。標準 train/val/test 只給平均準確率，不告訴你「這一筆該不該信」。

### 1.2 本文主張：把任意黑箱改造成「風險受控的集合預測器」

不重訓模型，而是用一個 holdout calibration set，把任意 black-box predictor `f̂` 改造成輸出**集合** `T(X) ⊆ Y` 的預測器，使其在未來測試點上的**期望損失（risk）以高機率不超過使用者指定水準**。

MRI 範例：每張影像分到診斷類別，誤判 stroke 損失=100、誤判 normal=0.1。RCPS 輸出一個「合理診斷集合」，保證未來資料上的平均損失低於門檻——醫生可安全排除集合外的診斷。

---

## 2. 核心定義與方法

### 2.1 (α, δ)-RCPS 定義（Definition 1）

集合預測器 `T : X → 2^Y`（隨機函數，依賴訓練資料）稱為 **(α, δ)-risk-controlling prediction set**，若

> 以至少 `1−δ` 的機率（over calibration data），`R(T) = E[L(Y, T(X))] ≤ α`。

其中 α（風險水準，代表值 10%）與 δ（失敗機率）由使用者預先選定。

**關鍵**：這是 **PAC-style 高機率保證**（P(risk > α) ≤ δ），不同於 vanilla conformal 的 marginal 期望保證。CRS 的 LTT 正是繼承這個 PAC 型態。

### 2.2 兩個結構假設

1. **Nesting（嵌套）**：集合預測器由單一參數 λ ∈ Λ ⊆ ℝ∪{±∞} 索引，λ 越大集合越大：`λ1 < λ2 ⇒ T_{λ1}(x) ⊆ T_{λ2}(x)`。
2. **Loss monotonicity（損失單調）**：集合越大損失越小：`S ⊆ S' ⇒ L(y, S) ≥ L(y, S')`。經典 tolerance region 損失 `L(y,S)=1{y∉S}` 滿足。並假設存在 `λ_max` 使 `R(λ_max)=0`。

> ⚠️ **這正是 CRS 不能用 vanilla RCPS/CRC 的原因**：CRS 的漏框 recall 損失是**多維+非單調**（見 [read-crc-nonmonotonic-summary.md](read-crc-nonmonotonic-summary.md)），不滿足這裡的單調假設 → 必須升級到 LTT。

### 2.3 UCB Calibration 程序（§2.2）

核心招式：對每個 λ 取 risk 的 **pointwise upper confidence bound** `R⁺(λ)`，滿足 `P(R(λ) ≤ R⁺(λ)) ≥ 1−δ`。然後選

`λ̂ = inf{ λ : R⁺(λ') < α, ∀λ' ≥ λ }`

即「整個右側信賴帶都壓在 α 以下」的最小 λ。

**Theorem 1（UCB calibration 有效性）**：損失單調 + 集合嵌套 + pointwise UCB 成立 + R(λ) 連續 ⇒ `λ̂` 給出 (α,δ)-RCPS。
- **精髓**：靠 risk function 的**單調性**，把「逐點收斂」升級成「資料驅動選 λ 仍有效」——**不需要 uniform convergence**。若沒有單調性就得 uniform 收斂（這也是 LTT 用 multiple testing 處理非單調的動機）。
- **Remark 2（重要）**：即使初始模型 `f̂` 訓練資料來自不同分布，RCPS 仍成立——**只要 calibration data 與 test data 同分布**（exchangeable）。這對 CRS「frozen base 在 RefCOCO 預訓，calibration 在目標 split」的設定是直接背書。

### 2.4 具體 Concentration Bounds（§3）

把 pointwise UCB 用各種集中不等式實作：

- **Hoeffding（簡版，§3.1.1）**：`R⁺ = R̂ + √(log(1/δ)/2n)`，僅供說明。
- **Hoeffding–Bentkus（HB，§3.1.2）**：反轉 tail probability，二元損失近乎最緊（Bentkus 不等式讓 Binomial 成最差情形）。
- **Waudby-Smith–Ramdas（WSR，§3.1.3）**：基於 martingale/online inference 的 hedged capital CI，**自適應變異數**，非二元有界損失下最優。**本文總建議：二元損失用 exact binomial、其他有界損失用 WSR。**
- **Unbounded losses（§3.2）**：無界損失需額外正則條件（Pinelis–Utev，限制 coefficient of variation）才有非平凡 UCB。

---

## 3. 實驗（§5）

五大規模任務示範框架普適性：
1. 不同誤分類代價的分類；2. multi-label 分類；3. 階層式標籤分類；4. **影像分割（預測含目標物件的像素集合）**；5. 蛋白質結構預測。
另討論 ranking / metric learning / distributionally robust learning 的延伸。

影像分割那組與 CRS 最相關：把「set of pixels containing object」當 set prediction，控 false-negative 型損失——與 [read-conformal-iseg-summary.md](read-conformal-iseg-summary.md) 同脈絡。

---

## 4. 對使用者主線（CRS）的對位與借鏡

> CRS 主線見 [[crs-pivot-conformal-referring-set]] / [[crs-redteam-p0-fixes]] / [[crs-litreview-15papers]]；索引 [reading-log.md](reading-log.md)。

1. **理論祖先，必引於方法章**：CRS 的 LTT 風險控制直接繼承 RCPS 的 (α,δ)-PAC 型態與「holdout calibration 改造黑箱」哲學。方法章談「為何能在 frozen base 上給有限樣本保證」時，RCPS 是源頭引用（Bates et al. 2021）。
2. **正當化「不重訓 base」**：Remark 2 明說「初始模型可來自不同分布，只要 calib/test 同分布」——這是 CRS frozen-base 設計的理論護身符，可直接寫進 method justification。
3. **反襯「為何要 LTT 而非 RCPS」**：RCPS 的兩個假設（單參數嵌套 + 損失單調）**CRS 都不滿足**（跨兩 base 兩參數、recall 損失非單調）。論文可寫：「RCPS/vanilla CRC 要求單調且單參數；CRS 的 cross-base 雙參數 + 非單調 recall 損失超出其適用範圍，故採 LTT（multiple testing over grid）」——把 RCPS 當「不夠用的起點」引用，與 CRC-Non-Monotonic 的論證合流。
4. **Bound 選擇的工程參考**：CRS 若損失二元（漏框/不漏框）用 exact binomial、若是連續型 IoU/recall 用 WSR——本文已給出明確選擇準則，可省去重新比較。
5. **與 conformal 的關係界定**：RCPS 自陳「靈感來自 conformal 的 nested set 詮釋、算法類似 split conformal，但追求 tolerance region 的高機率保證、用完全不同的證明（concentration 而非 exchangeability rank）」——這段是 CRS 在 background 區分「conformal coverage vs risk control PAC」的好範本（呼應 [read-vlm-calibration-summary.md](read-vlm-calibration-summary.md) 的 ECE vs conformal 區分）。

**引用優先級**：高（方法章理論源頭，必引）。

---

## 5. 一句話總結

> RCPS 是 **2021 年把「分布無關有限樣本風險控制」從 conformal coverage 推廣到任意單調損失集合預測**的奠基作：用 holdout calibration + UCB（選整段右側信賴帶都壓在 α 下的最小 λ）把任意黑箱改造成 (α,δ)-PAC 風險受控的集合預測器。它要求**單參數嵌套 + 損失單調**——而 CRS 的跨 base 雙參數、非單調 recall 損失正好突破這兩個假設，因此 CRS 必須升級到其後繼 LTT；RCPS 是 CRS 風險控制的理論祖先與「不夠用的起點」雙重引用對象。

---

## 6. 參考連結

- arXiv：<https://arxiv.org/abs/2101.02703>
- 專案頁/blog：<https://angelopoulos.ai/blog/posts/rcps/>
- 後繼 LTT：Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*（arXiv 2110.01052）
