# 第 3 章　方法框架

本章定義全篇統一的方法框架：以**凍結基礎模型 + 輕量信心策略**為核心，所有實驗（C1–C4、M4、CRS）皆為其實例。框架的主角不是基礎模型 `g`，而是附加在其決策層的信心策略 `π`——所有貢獻都是「`π` 能從 `g` 的副產品學到多少可靠決策」。本章先給出問題形式化（§3.1）、凍結基礎模型與信心訊號（§3.2–3.3）、選擇性預測與跨基礎模型轉移（§3.4）、無目標閘與廣義 REC 邊界（§3.5），再宣告把框架從「測量可靠度」推進到「建構有保證的集合」的關鍵一步——風險受控指稱集合（§3.5b），最後鎖定評測協定（§3.6）。

## 3.1 問題形式化

我們研究**凍結基礎模型下的選擇性指稱定位**。一個指稱定位模型 `g` 把影像 `I` 與描述 `e` 映射到一或多個邊界框。在標準 REC 設定下 `g` 回傳單一框；在廣義設定（GREC）下，它須回傳一個基數可為零（no-target）、一或多（multi-target）的框集合。

我們把 `g` 視為**凍結黑盒**：其參數從不更新。在 `g` 之上，我們附加一個輕量的**信心策略** `π`，它觀測 `g` 單次（或少數幾次）前向傳遞的廉價副產品，並對每個查詢決定下列動作之一：

- **作答（answer）**——輸出基礎模型的預測；
- **棄答（abstain）**——宣告無目標／拒絕作答；
- **（可選）重排（re-rank）**——施加一個與定位相關的候選操作；
- **建構風險受控集合（construct a risk-controlled set）**——不輸出點預測，而回傳一個經校準的候選框集合（可為空），其風險以有限樣本保證受控（即第 9 章 CRS）。

本論文量化的，是跨基礎模型與資料集，`π` 能從 `g` 的副產品中**多可靠地**被校準、**代價多少**、**跨基礎模型轉移得多好**、**點預測在何處原理性地失效**（oracle gap／邊界，第 8 章 M4），以及**當點預測失效時，集合化的 `π` 能提供什麼保證**（CRS）。

## 3.2 凍結基礎模型

我們在兩個結構迥異的凍結基礎模型上實例化框架，以壓力測試信心訊號是否與基礎模型架構無關：

- **CLIP-VG**（單框迴歸）：經 `bbox_embed(reg_token).sigmoid()` 恰輸出一個框，**無候選清單、無分數**。這迫使信心只能來自**擾動**訊號，而非分數統計——此特性反而有利於跨基礎模型轉移（§3.4）。
- **OWL-ViT**（開放詞彙偵測器）：單次前向傳遞輸出多個帶分數候選框，分數為免費副產品，支援分數型信心與**原生棄答**（最大分數低 ⇒ 傾向無目標）。

兩個基礎模型的訊號家族完全不同（擾動 vs 分數），卻都 informative，且其中一類可跨基礎模型轉移——這是 C4 的科學主張（第 10 章）。在 CRS 章再引入第三個凍結基礎模型 **GroundingDINO** 作為高召回的 box selector。

## 3.3 信心訊號

所有訊號皆為事後（post-hoc，不更新基礎模型），依成本與可轉移性分兩族：

**(A) base-agnostic 擾動訊號**（成本 K× 前向，K 個改寫）：
- `cross_prompt_consistency`：K 個改寫提示下預測框的平均成對 IoU。一致性低 ⇒ 指稱身分在良性改寫下不穩 ⇒ 錯誤風險高。這是**與定位相關**的——量測指稱穩定度，而非泛用信心。
- `prompt_box_dispersion`：改寫提示下預測框中心的離散度（除以 √area 正規化）。

**(B) base-specific 分數訊號**（成本 1× 前向，免費副產品；OWL-ViT）：
- `top1_score`、`margin12`（= top1 − top2）、`score_entropy`（softmax 候選分數的熵）、`score_mean_topk`。

(A) 貴但可轉移、(B) 免費但 base-specific——這個「成本↔可轉移性」的二分本身就是貢獻，直接撐起成本—風險 Pareto（第 11 章）與 C4（第 10 章）。校準器是標準化訊號上的邏輯迴歸，於 held-out 校準 split 擬合；其預測機率即作為排序選擇性預測的信心／棄答分數。我們刻意把校準器保持最小——主張是關於**訊號資訊量**，而非校準器容量。

## 3.4 選擇性預測與跨基礎模型轉移

**選擇性預測（C2）。** 給定 `π` 的信心 `c(x)`，回答信心最高的 coverage 比例查詢、棄答其餘。我們報告**風險—覆蓋曲線**及其面積（AURC，越低越好），對照隨機、單訊號閾值，以及以真實 correctness 排序的 oracle。

**跨基礎模型轉移（C4）。** 這是核心的科學測試：僅用**base-normalized** 訊號在一個 *source* 基礎模型上擬合 `π`，再**不重新擬合**地套用到一個 *target* 基礎模型。若策略可轉移，則定位失敗具有*可重用、基礎模型共享的結構*——這是任何 per-pipeline 驗證方法（VIRO、True/False Verification）無法做出的主張。轉移以「被轉移策略在 target 基礎模型上的 AURC」對照「target 原生策略」與「隨機」量測。

C4 的乾淨版主張是：raw consistency 零參數、無 refit，從 CLIP-VG 套到 OWL-ViT，AURC 接近原生——亦即 grounding 不確定性具有 base-invariant 結構。這比 C1/C2 的審計更像「發現」，也是對抗「只是 calibration／threshold」批評的護城河。

## 3.5 無目標閘與廣義 REC

**無目標閘（C3）。** 一個恆吐 argmax 框的凍結基礎模型，依其建構**毫無棄答能力**（無目標查詢的 N-acc = 0）。在 gRefCOCO 上，我們校準 `π` 以基礎模型副產品輸出 `P(no-target)`，再二分為棄答決策。指標：無目標 AUROC、balanced accuracy、N-acc——全部 leak-safe（閾值固定於 calib）。

**廣義 REC 邊界（M4）。** 把 `π` 推到完整 GREC 需要輸出*正確的框集合*，以官方 **Pr@(F1=1, IoU≥0.5) / N-acc / T-acc**（忠實 greedy-IoU 匹配）評測。這是刻意的**壓力測試**，而非方法主張：它定位出事後校準停止有效之處（多目標 exact-match 是超出單一信心閾值的計數／集合預測問題）。我們以逐樣本的 oracle-τ 上界與 bootstrap CI 量化此邊界。M4 在方法框架中先聲明為 boundary analysis，避免在結果章被讀成失敗——「刻意把方法推到斷裂點並量化斷在哪」是成熟度，不是減分。

## 3.5b 風險受控指稱集合（CRS）：從點預測到有保證的集合

M4 的牆（多目標 exact-match 超出單一信心閾值的動作空間）促成框架的最後一步：讓 `π` 不再輸出**點預測**，而輸出一個 **風險受控指稱集合** `S(I,e) ⊆ {候選框}`，基數可為 0（棄答／無目標）、1 或多。我們以 **Learn-then-Test（LTT）** 在校準 split 上**聯合**校準三個有界風險並給出有限樣本保證：

- **R1 已作答目標漏檢率（answered-target FNR）** ≤ α：在未棄答的 target-present 查詢上的漏檢率（**條件**風險，不稱 recall guarantee）；
- **R2 無目標誤選率（no-target false selection）** ≤ β：no-target 樣本吐出任何框的比例；
- **R3 目標棄答率（target deferral）** ≤ γ：target-present 樣本被吐空集的比例。

進一步地，CRS 把風險控制的代價**因式分解**為 gate（該不該答）與 box（答得準）兩個正交瓶頸，並以 **cross-base conformal composition**（一個凍結基礎模型當 gate、另一個當 box selector）在三保證下達到接近真實基數的緊緻集合。完整方法、可行域與結果見第 9 章。

CRS 把 C1–C4 的 measurement 與 M4 的 boundary 縫進同一條線：訊號可測（C1–C4）→ 點預測撞牆（M4）→ 換成有保證的集合預測（CRS）。本章不放 CRS 數字（數字在第 9 章），避免重複。CRS 不宣稱解決完整 GREC 的 exact-match，而是把問題從 exact-match 點預測 pivot 成 risk-controlled 集合建構——這個 metric pivot 在第 8、9 章都會明寫。

## 3.5c 三正交維度：CRS 系統的可改善軸

CRS（§3.5b）確立 base claim 後，本論文進一步把 CRS 系統拆成三個**正交、可獨立替換**的維度。
關鍵觀察：在 frozen detector 給定下，一個 risk-controlled referring set 的產生鏈是

```
候選池 (pool) → 閘分數 (gate) → LTT 憑證 (certificate) → referring set
```

三個環節各自可在**不碰 detector 權重、不改其他兩環**的前提下替換，因此互相正交，可獨立疊加：

- **維度 1 候選池（pool）**：CRS 用 frozen base 的原生候選池。可換成 decomposition-union pool
  （凍結 VLM router 拆子部件 → 各跑 GDINO → union），只改 box 候選來源，gate 與 certificate 不變。
  主攻 R1（多目標漏檢）。完整方法與結果見第 9b 章。
- **維度 2 閘分數（gate）**：CRS 用單一 OWL top1 分數當棄答 gate。可換成 21 維 frozen 特徵上的
  可解釋 learned gate（EBM），只改 gate decision 層，pool 與 certificate 不變。主攻 R2/R3 與可行性。
  完整方法與結果見第 9c 章。
- **維度 3 憑證（certificate）**：CRS 的 LTT 用 Hoeffding bound 算 p-value。可換成更緊的
  Hoeffding–Bentkus（HB），同切分／grid／Bonferroni、**只換 p-value**，免費擴大可行域。見第 9c 章。

**統一協議的意義**：因為三維度共用同一套 LTT 風險定義（R1/R2/R3）、同一套 calib/test 切分與
防洩漏規則，任意 (pool, gate, bound) 組合都套同一協議，四個版本（Rule / Decomp / WB-Gate /
WB+Decomp）可在同一張表逐欄比較。三維度共同決定 frozen detector 給定下可達的 risk–cost frontier
（統一論述見第 9d 章）。

## 3.6 評測協定

- **切分與洩漏**：所有閾值／標準化／校準器擬合都來自校準 split；test 只評估一次。跨基礎模型轉移不讓 target-test 統計回流到 source 訓練。
- **成本**：推論成本以「每查詢前向傳遞次數」加實測 throughput（GB10）報告，不以形容詞描述（第 11 章 Pareto）。
- **不確定性**：所有 headline 指標附 **bootstrap 95% CI**（≥1000 次 resample test，校準器固定）。
- **保證 vs CI（CRS 專用，必須分清）**：對 CRS，**LTT p-value + Bonferroni** 檢定是*有限樣本風險控制保證*；固定 selected config 上的 bootstrap CI 只是*經驗穩定度*，**不是**保證。兩者分開報、never conflated。CRS 閾值網格只建構於**校準 covariate**（無 evaluation covariate 或 label 進入網格建構、風險檢定或操作點選擇）。
- **Oracle gap**：每張選擇性圖都附 oracle 線，界定還有多少訊號未被利用。

這六條（洩漏、成本、顯著性、保證≠CI、網格 calib-only、oracle 上界）是審查最常攻的點；方法章先把協定講死，結果章便能省去反覆辯護。
