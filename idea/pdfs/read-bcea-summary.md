# 論文深讀筆記：BCEA — Look Again Before You Abstain

## 一、書目資訊

- **標題**：Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models
- **作者**：Jian Xu (1,2)、Delu Zeng (3)、John Paisley (4)、Qibin Zhao (2)
- **arXiv**：2606.16667v1 [cs.CV]，2026-06-15
- **授權**：CC BY 4.0
- **領域定位**：LVLM（大型視覺語言模型）幻覺的**選擇性預測（selective prediction）+ 分布無關保證（distribution-free guarantee）**，並引入第三種動作「主動取得更多視覺證據（evidence acquisition）」。
- **一句話概括**：把「回答 / 棄答」的二元 conformal 過濾器，升級成「回答 / 棄答 / 再看一眼（zoom-crop / 結構化介入）」的三元決策；關鍵發現是**天真地把證據取得插進已校準的過濾器會悄悄破壞統計保證**，正確做法是**把整個 acquisition policy 折進 score function、在 post-acquisition 分數上重新校準**，即可恢復有限樣本保證並提升覆蓋率。

---

## 二、問題定義與動機

### 2.1 背景痛點

LVLM 會流暢地「幻覺」出影像中不存在的物件 / 屬性 / 關係。對於醫療、科學、輔助等高可靠場景，光流暢不夠，需要一個**保證**：無論模型說什麼，未獲影像支持的 claim 比例（幻覺率）必須低於使用者指定的門檻。

Conformal prediction 正好提供這種保證。最強的既有工作是 **ConfLVLM (Li et al. 2025)**：把每個生成細節當成一個假設，用便宜的不確定性啟發式打分，把分數低於 conformal 校準門檻的 claim 過濾掉，於是對「留下來的 claim」的事實錯誤率得到有限樣本、分布無關的上界。

### 2.2 核心張力：保證的代價是「狂棄答」

作者用一個直接的測量點出問題：要把幻覺率壓到 **5% 以下**，在平衡的 object-existence benchmark 上，一個 SOTA conformal 過濾器必須**棄答超過 80% 的 claim**（精確數字：LLaVA-1.5 上 α=0.05 只敢 assert 18%，亦即棄答 82%；即使放寬到 α=0.20 也只 assert 40%）。一個「四題拒答三題」的可靠模型，實用價值有限。

### 2.3 動機觀察

**「當更多證據便宜可得時，棄答是浪費的。」** 人類不確定小物件在不在時會「湊近看」，而不是放棄。LVLM 有對應動作：重新編碼一個放大的 crop、或在針對性視覺介入下重新打分，代價只是幾次額外 forward pass。核心問題是：這個「再看一眼」能否**在不放棄統計保證**的前提下加進來？

### 2.4 兩條既有文獻各解一半

- **Conformal LVLM 過濾器**（ConfLVLM、CAP）：有保證，但只會「丟棄」claim，從不取得證據。
- **證據取得方法**（ReCoVERR）：會收集更多證據來救回被棄答的問題，但靠啟發式 risk-tolerance 門檻、**沒有分布無關保證**；而且它取得的是**文字證據**（LLM 生成的追問），不是重新檢視像素。

**沒有方法同時做到「取得視覺證據 + 保留保證」，而作者證明天真地把兩者合併並不安全。**

---

## 三、方法詳解：BCEA

BCEA 把二元「answer / abstain」過濾器變成有保證、會找證據的三元決策器：**answer / abstain / acquire**。三項貢獻如下。

### 3.1 證據充分性分數（Evidence-Sufficiency Scores）

定義 `ℓ(v) = (1/|c|) Σ_t log p_θ(c_t | v, prompt)`，即 claim c 在視覺輸入 v 下的平均 token log-likelihood。

- **Global ungrounding 分數**（沿用前人直覺）：
  `s_glob(x,c) = ℓ(x) − ℓ((blank))`，其中 (blank) 是全黑（blank）影像。
  幻覺 claim 在拿掉影像後幾乎不變（s_glob ~ 0）；grounded claim 則會大幅下降。

- **Region / Intervention 分數**：s_glob 把整張圖丟掉，無法分辨證據「在哪」也分不出依賴**視覺結構**的 claim。改用依 claim 類型選的針對性介入 T：
  `s_T(x,c) = ℓ(x) − ℓ(T(x))`
  - 存在 / 屬性 claim：region masking（把被詢問物件所在區域塗黑）。
  - 左右關係 claim：horizontal flip（水平翻轉）。**正確**的方向 claim 翻轉後變假（s_T > 0），**錯誤**的 claim 翻轉後變真（s_T < 0）。

### 3.2 Acquisition 動作（再看一眼）

當 s(x,c) 太低不能 assert、但又不是明顯 ungrounded（borderline）時，BCEA 進行 acquisition：形成 B 個放大視圖 {x^(1),…,x^(B)}（重疊 crop 的粗網格，**不需要 ground-truth 定位**），組成 post-acquisition 分數：

`s_acq(x,c) = max( ℓ(x), max_{b≤B} ℓ(x^(b)) ) − ℓ((blank))`  …(式3)

對真物件，某個 crop 放大它會抬高 likelihood；對幻覺物件，沒有任何 crop 能提供支持。這個 **max 同時是好處與危險的來源**。

### 3.3 為什麼天真 acquisition 無效，以及如何修

**天真部署**：在 base 分數 s_glob 上校準門檻 τ_α，部署時卻用 s_acq ≥ τ_α 來判定（取得證據、沿用舊門檻）。這是**錯的**。

- **Proposition 1（天真 acquisition 反保守）**：因為 s_acq ≥ s_glob 逐點成立（式3 的 max），接受集 {s_acq ≥ τ_α} 是 {s_glob ≥ τ_α} 的**超集**。對前者認證的 level-α 保證**不會轉移**到這個更大的集合。新被納入的 claim E' = {s_acq ≥ τ_α, s_glob < τ_α} 正好是「全圖分數拒絕、但某個 crop 把它推過門檻」的 claim——對幻覺 claim，crop 可能虛假地抬高 ℓ，所以 Pr[Y=0 | E'] 通常很大，保證被破壞。實測 realized risk 超標可達 **17 個百分點**（摘要），表1 中 α=0.20 時飆到 0.41（兩倍違規）。

- **Proposition 2（單調 acquisition）**：s_A^(B) 對 B 逐點非遞減，所以固定 τ 時 coverage 與 false-acceptance probability 都隨 B 非遞減。更多預算讓更多真 claim 過關，但也讓更多假 claim 過關，**除非把 τ 重新調高**，否則 risk 會爬升。

**修法（核心）**：把整個 acquisition policy 當成 score function 的一部分，**在校準 claim 的 post-acquisition 分數上校準門檻**，讓校準與測試走過完全相同的 pipeline，恢復 exchangeability。

- **Theorem 1（Acquisition-adaptive validity）**：固定一個只用 (x,c) 與模型、**不讀校準標籤**的 acquisition policy A，映到 post-acquisition 分數 s_A(x,c)。若校準與測試 claim 可交換，則用 Clopper–Pearson fixed-sequence 程序在 {s_A(x_i,c_i)}_cal 上選 τ、並以 {s_A ≥ τ} 判定，即可在測試 claim 上以機率 1−δ 把 selective risk 控制在 level α。
  證明要旨：把固定映射 A 套到每個 claim 保持 (s_A, y) 的可交換性，其餘就是標準 split-conformal selective risk control。**這是把已知原理（把固定、標籤無關的變換折進 conformal pipeline 不損 coverage；Shanmugam 2025、Wang 2026）實例化**；作者強調其價值在於**它排除的失敗模式**（Prop 1）對視覺 acquisition 既容易犯、又嚴重。

- **Proposition 3（claim 內序列式 acquisition）**：policy 可以一次形成一個視圖、邊看邊決定要不要再看、何時停（停止時間 κ≤B），只要每個決策只看當前 claim 的觀測。則 s_A 仍是 (x,c) 的固定可測函數，Theorem 1 原封不動成立。亦即「看、判斷、再看、有信心就停」**無需任何修正**即合法。

- **Proposition 4（Anytime-valid risk control）**：第二個 optional stopping 來源是**校準資料流本身**（一直收標籤、看起來好就停）。此時固定樣本的 Clopper–Pearson 失效；改用 betting confidence sequence（基於 supermartingale + Ville 不等式），可在**所有樣本量同時**有效，因此對校準流的任意 data-dependent 停止仍保證有效。作者強調這條**不是現成實例**，把保證從 fixed-n 升級到對「claim 內 acquisition」與「校準流」雙重 optional stopping 都穩健——這才是部署中的可靠性監控器真正的處境。

**Algorithm 1** 最關鍵的一行是第 7 行：門檻校準在已含 acquisition 的 Score 上，使部署決策（第 9 行）與校準看到同一個映射——這正是 Theorem 1 的條件。實用處方一句話：**「在你真正要部署的分數上校準。」**

### 3.4 Acquisition 何時有幫助？（棄答與證據取得的取捨理論化）

Theorem 1 只保證**安全**，不保證**有用**。作者精確刻畫 coverage 增益：

- **Proposition 5（固定 risk 下的 coverage 是一個 ROC 點）**：令 π = Pr[Y=1]。在 selective risk ≤ α 限制下，最大 coverage 落在 ROC 空間中斜率 ρ(α) = πα / [(1−π)(1−α)] 那條射線上的最大 TPR 點，且 C_S(α) = (π + (1−π)ρ(α))·TPR*_S(α)。

- **Theorem 2（acquisition 有幫助 iff 改善 ROC）**：post-acquisition 分數 S' 對**每個** α 都使 coverage 不降，**當且僅當** ROC_{S'} 逐點支配 ROC_S。充分條件：S' 在 Blackwell 意義下對 Y 比 S 更有資訊（其類條件 likelihood ratio 是 S 的單調精化）。

  這把「acquisition 有沒有用」這個模糊問題變成**可測、可證偽**的判準，並預測了實驗順序：uniform-grid 把 AUROC 從 0.82 抬到 0.86、CLIP-guided 再到 0.88，因此 coverage 在每個 risk level 都是 **no-acq < grid < guided**。

- **為什麼不主張 budget allocation（誠實的負結果）**：在固定總預算下不均分配 budget，理論上應由 greedy water-filling 解；但在 existence claim 上**沒有任何改善**（≤0.4 coverage 點，落在 CI 內）。結構原因：單一全域門檻下，多看一眼同時抬高真假 borderline claim 的分數，而分配時標籤未知，所以任何標籤無關的分配在一階上是 coverage-neutral。**作者刻意不把 allocation 當貢獻**，視為一個澄清性的負結果。

---

## 四、實驗

### 4.1 設定

- **模型**：主用 LLaVA-1.5-7B；Qwen2.5-VL-7B 驗證泛化；POPE 實驗再加 LLaVA-NeXT-7B、InternVL2-8B（共四個開源 VLM）。
- **資料**：COCO val2017 ground-truth instance 標註自構（不需額外標註）。
  - Existence：present 物件（真）vs 抽樣 absent 物件（假）。
  - Spatial：兩個最大、水平分離的物件「A 在 B 左/右」，方向交換為假 claim。
  - Count（附錄）：「恰有 k 個 A」。
- 每個 claim 一次 forward pass 算分；acquisition 加 **B=5** 個 grid-crop pass。
- Conformal：50/50 split，平均 300 次隨機 split，δ=0.1；報告 test coverage 與 **90th-percentile realized risk**（因為保證是 1−δ=0.9，所以該對齊的量是 90 分位風險，應坐落在 α）。

### 4.2 主結果：天真破壞、BCEA 恢復（Table 1，1,440 existence claims）

四個方法共用同一批 acquired evidence，只差在 thresholding：

| α | No-Acq (cov/risk) | Naive (cov/risk) | ReCoVERR (cov/risk) | BCEA (cov/risk) |
|---|---|---|---|---|
| 0.05 | .19 / .06 | .40 / **.19** | .32 / **.08** | .22 / .06 |
| 0.10 | .28 / .11 | .57 / **.30** | .40 / **.13** | .37 / .10 |
| 0.20 | .39 / .21 | .74 / **.41** | .50 / **.24** | .47 / .21 |

（粗體 risk = 超過目標 α 超出 Monte-Carlo 誤差，即**保證被破壞**。）

- **No-Acq**（保證棄答）與 **BCEA** 的 90 分位 risk 都緊貼 α。
- **Naive**（沿用門檻）coverage 跳升，但 risk 大幅超標（α=0.20 時 0.41，兩倍違規）。
- **ReCoVERR-style**（用 empirical 校準 risk、無有限樣本修正）coverage 最高但反保守，每個 level 超標約 0.03–0.04。
- **BCEA** 既守住風險，又把 coverage 從棄答的水準大幅抬高（例如 α=0.10：**0.29 → 0.37**）。

**一句話貢獻**：分布無關保證相對於「無保證的取得器」只付出幾個 coverage 點，而那個無保證取得器是**悄悄違規**的；同時 BCEA 比純棄答多賺 coverage。

### 4.3 證據是局部化的（5.2）

800 個真 existence claim 上，遮住物件 GT 區域使 claim log-likelihood 平均降 0.109，遮住等面積無關區域反而略升（−0.050）；配對差 +0.159、勝率 67%、p << 10⁻¹⁰。Occlusion 熱圖集中在物件上，佐證 region/crop-based acquisition 的前提。

### 4.4 Coverage 隨 budget 上升（5.4）

掃 budget B（每 claim 額外視圖數），每個 B 重新校準：guaranteed coverage 從 B=0 的 0.29 單調升到 B=5 的 0.35（α=0.10），realized risk 釘在 ~0.11。多看買到更多 coverage 且不損保證。**Where 分配（greedy vs uniform）統計上無差別**（皆 ~0.34）。

### 4.5 結構化介入救關係 claim（5.5，996 平衡 spatial claim）

| Score | AUROC（correct vs swapped）|
|---|---|
| Raw claim likelihood | 0.567 |
| s_glob（global ungrounding）| 0.574（近乎瞎猜）|
| s_flip（翻轉介入，本文）| **0.765** |

關係 claim 中兩物件都在，blank 影像對正確與交換關係懲罰相同，所以 global 分數無效。單次水平翻轉把驗證從 ~chance 抬到 0.765（真關係 mean s_flip +0.034，假關係 −0.066）。

**有效介入需滿足兩條件**：必須**翻轉 claim 真值**且**保持影像 in-distribution**。對上下關係用 vertical flip **失效**（AUROC 0.50）——上下顛倒是 OOD，likelihood 變得無資訊而非真值反轉。所以設計介入是一個建模步驟，建立 per-claim-type 介入庫是本框架開啟的重要方向。

### 4.6 模型引導 acquisition（5.6）

用 CLIP 從多尺度候選窗中挑與「a photo of a <object>」嵌入最相似的 5 個 crop（推論時不用 GT 位置）：

| | no acq. | uniform grid | CLIP-guided |
|---|---|---|---|
| AUROC | 0.824 | 0.862 | **0.882** |
| cov @ α=0.05 | 0.19 | 0.18 | **0.22** |
| cov @ α=0.10 | 0.28 | 0.33 | **0.37** |
| cov @ α=0.20 | 0.39 | 0.43 | **0.47** |

機制：guided crop 在**小物件**上把 acquisition 增益幾乎翻倍（0.25 → 0.43），而 grid 常把小物件裁掉。

### 4.7 擴展到 POPE benchmark（5.7）

4 個 VLM × 3 個 split（random/popular/adversarial），約 11.5k claim。**所有 12 個設定一致**：acquisition 對每個模型每個 split 都提升 AUROC（最戲劇性是 LLaVA-NeXT random：0.56 → 0.72），BCEA 的 guaranteed coverage 處處勝過棄答（常 2–4 倍），90 分位 risk 緊貼 α=0.10（誤差 ~0.03）。難度上升（random < popular < adversarial）coverage 隨之下降，acquisition 在 base 分數最弱處幫助最大。

### 4.8 跨模型泛化（5.8，Qwen2.5-VL-7B）

Existence claim：acquisition 把 grounding AUROC 從 0.70 抬到 0.77；α=0.05 時棄答幾乎啥都不敢 assert（1%），BCEA 在同樣認證風險下 assert 15%。Spatial 的 global 分數再次 ~chance（0.50），確認 global 分數在關係 claim 上的失敗非單一模型特有。

### 4.9 棄答率 vs 幻覺率（重點數字，給 CRS 對照用）

- 要把幻覺率壓到 **≤5%**：global-score 純棄答過濾器在平衡 existence 上只能 assert **18%（棄答 82%）**（Figure 1 / 5.1）。
- BCEA 在 α=0.05 仍守住 5% risk 的同時，把 coverage 從 0.19 抬到 0.22（LLaVA），Qwen 上甚至 1% → 15%。
- α=0.10 時 BCEA 0.29 → 0.37；α=0.20 時 0.39 → 0.47。
- 換句話說：**保證越嚴（α 越小），純棄答越接近「全部拒答」，acquisition 的相對救援價值越大，但絕對 coverage 仍偏低。**

### 4.10 範圍邊界與限制（6.）

- **只在 backbone「看得見」的屬性上有訊號**：existence、left/right 可救；counting、relative size、above/below、color、color–object **binding** 全部 ~chance（s_glob AUROC 0.50–0.58），任何介入（instance masking、vertical flip、grayscaling）都救不回，coverage 崩到 0。
- **Binding 結果很關鍵**：分數無法認證「the red cup」是否正確 binding，正是因為 backbone 本身就分不出來——這是**忠實行為**：backbone 真的看不見時，分數回報 ungrounded 而非編造信心，過濾器棄答。
- **從 probed claim 到 free-form 生成有真實落差**：動機是 free-form 描述的幻覺，但評估的是 probed/atomic claim（COCO 自構 + POPE 是非題）。因 BCEA 的分數與保證是 per-claim、與 claim 怎麼產生無關，理論上可接 claim parser 處理 free-form 輸出，但 acquisition 在自生 claim 上是否同樣有用、parser 噪聲如何與 conformal 校準互動，**尚未測試**。作者明說 BCEA 是「受控探測下驗證的機制，尚非端到端 free-form pipeline」。

---

## 五、與 CRS（使用者碩論主線）的逐項 Diff 表

CRS = Cross-Base Conformal Composition：以 **OWL-ViT gate（控可用/棄答/no-target）+ GroundingDINO box（控 compact set）** 兩個**凍結** base 組合，用 LTT conformal risk control **同時保證 recall + abstention**，在 gRefCOCO/RefCOCO 上輸出一個 **conformal referring SET of boxes**（GREC 指標），post-hoc、不重訓。

| 維度 | BCEA | CRS（你的碩論）|
|---|---|---|
| **任務** | object-existence claim verification（VQA 式幻覺判定，是非題：「圖中有 X 嗎」「A 在 B 左邊嗎」）| referring expression comprehension / GREC：給定指稱語句，**定位框**（box localization）|
| **輸出** | 對單一 atomic claim 的 **assert / abstain / acquire** 三元決策 | 一個 **conformal referring SET**（多個 box 的集合）+ 可棄答 / no-target |
| **保證對象** | **幻覺率（hallucination rate）**單一上界：Pr[Y=0 \| asserted] ≤ α | **雙保證**：recall（漏掉真 referent 的風險）+ abstention（no-target/棄答）同時用 LTT 控制（R1/R2/R3 多風險）|
| **單模型 vs 組合** | **單一 LVLM**（LLaVA/Qwen/…）自身 likelihood + 對自己做視覺介入（zoom/crop/flip）| **cross-base composition**：兩個異質凍結 base（OWL-ViT + GroundingDINO）的功能分工組合，ablation 證明是 factorization 非 ensemble |
| **「再看一眼」的角色** | **核心貢獻**：budgeted evidence acquisition 是主機制，靠 crop/intervention 抬分救棄答 | 不是 acquisition；CRS 的「組合」是把不同 base 的互補能力拼起來，無 per-claim 額外 forward 預算概念 |
| **conformal 形式** | split-conformal selective risk control，Clopper–Pearson fixed-sequence；anytime-valid 版用 betting CS | **LTT（Learn-then-Test）**多風險聯合校準（Bonferroni over grid），calib-only grid 修 leakage |
| **資料/指標** | COCO 自構 existence/spatial claim + POPE（是非題）；指標 coverage / 90th-pct realized risk / AUROC | gRefCOCO/RefCOCO；GREC 官方指標（Pr@(F1=1)、N-acc、T-acc）+ set size / recall CI |
| **空間關係處理** | 靠 horizontal-flip 介入把左右關係從 chance 救到 0.77 AUROC | REC 直接定位被指稱物件，空間語意內含在指稱語句中由 base 處理，非以介入驗證 |
| **失敗邊界** | counting/binding/above-below 全 chance → coverage 崩 0（誠實棄答）| 已知 multi-target T-acc 天花板來自 frozen detector exact-match 硬上限（不同瓶頸）|
| **是否凍結 / post-hoc** | 是（inference-only、no fine-tuning）| 是（frozen bases、post-hoc）—**兩者在這點同源，需特別切割措辭**|

**結論：CRS 與 BCEA 在「框架詞彙」上高度重疊（selective prediction、abstain、distribution-free guarantee、grounded、frozen/post-hoc、conformal），但在「做什麼」上正交。** BCEA 是 claim-level **是非驗證 + 單模型證據取得 + 單一幻覺率保證**；CRS 是 referring-level **集合定位 + 跨 base 組合 + recall×abstention 雙保證**。重疊只在外殼，不在內核。

---

## 六、論文該怎麼引用 / 切割（可直接貼進 Related Work）

> **Selective prediction with conformal guarantees for VLMs.** A parallel line of work brings conformal selective prediction to vision-language models for *hallucination control*. ConfLVLM (Li et al., 2025) tests each atomic claim with a global uncertainty heuristic and conformally filters out ungrounded ones, bounding the residual factual-error rate; CAP (Tayebati et al., 2025) learns adaptive abstention thresholds. Most closely related to our framing is **BCEA (Xu et al., 2026)**, which augments such a binary answer/abstain filter with a third *acquire-evidence* action—re-examining the image via zoom crops or claim-type interventions under a compute budget—and shows that the acquisition policy must be folded into the score and re-calibrated on post-acquisition scores to preserve the finite-sample hallucination-rate guarantee. **We share BCEA's vocabulary (distribution-free selective risk control over a frozen, post-hoc model) but differ in three load-bearing ways.** (i) *Task and output*: BCEA verifies object-existence and left/right claims as binary, VQA-style judgments and certifies a single hallucination rate over *asserted claims*; we perform referring-expression comprehension and emit a *conformal referring set of boxes* on gRefCOCO/RefCOCO, evaluated with GREC metrics. (ii) *Guarantee*: BCEA controls one risk (the false-assertion rate); CRS jointly controls *recall and abstention* (and no-target detection) via a multi-risk Learn-then-Test calibration. (iii) *Composition*: BCEA acts on a *single* LVLM's own likelihoods plus self-applied visual interventions; CRS *composes two heterogeneous frozen bases* (an OWL-ViT gate and a GroundingDINO box predictor) whose factorized roles—abstention/no-target vs. compact-set construction—are validated to be a true composition rather than an ensemble. BCEA's budgeted acquisition is orthogonal to and could in principle be layered onto either base in our pipeline, but neither addresses set-level recall, cross-base transfer, nor referring-expression localization.

（精簡版一句話切割，若篇幅吃緊）：

> 與我們框架詞彙最近的是 BCEA (Xu et al., 2026)，但它做的是**單模型、is-非題式 object-existence 驗證 + budgeted 視覺證據取得**，只保證單一幻覺率；我們做的是**跨 frozen base 組合 + referring-set 定位**，並以 LTT 同時保證 recall 與 abstention（gRefCOCO/GREC），兩者任務、輸出、保證對象與組合結構皆正交。

---

## 七、個人評價

**優點**

1. **負結果驅動的敘事漂亮**：「天真 acquisition 悄悄破壞保證、超標 17 點」這個 Proposition 1 + Table 1 的對照是全文最有說服力的賣點，把一個容易犯的部署錯誤量化得很乾淨。
2. **理論誠實**：Theorem 1 自承是已知原理（Shanmugam 2025、Wang 2026）的實例化，價值在於指出視覺 acquisition 的特定失敗模式；Prop 4 的 anytime-valid 升級才是真正新的技術貢獻；連 budget allocation 都明說是負結果——這種自覺難得。
3. **Theorem 2（ROC 判準）有實用價值**：把「acquisition 何時有用」化為可測 ROC 支配，並用 AUROC 數字驗證 no-acq < grid < guided 的 coverage 排序，理論與實驗扣得緊。
4. **失敗邊界誠實**：counting/binding/above-below 全崩到 0 並如實報告，把「忠實棄答」當特性而非 bug。

**弱點 / 限制**

1. **probed claim 與 free-form 的落差是真缺口**：作者自承動機是 free-form 幻覺、評估卻是是非題與 COCO 自構 claim，端到端 pipeline 未驗證。這也是它與真實部署最遠的一塊。
2. **絕對 coverage 仍偏低**：α=0.05 時 BCEA 也才 assert 22%（LLaVA），「再看一眼」救回的是邊際，沒有改變「嚴格保證下大量棄答」的本質——它讓棄答少一點，但沒讓模型變得能放心多答。
3. **介入庫靠人工設計**：horizontal flip 有效、vertical flip 失效，每種 claim type 都要手工找到「翻轉真值且 in-distribution」的介入，泛化性受限；論文也把建庫列為 future work。
4. **參考文獻有小瑕疵**：正文寫 Qwen2.5-VL，bib 卻列 "Qwen3-vl technical report (arXiv:2511.21631)"，版本對不上（可能是預印本的引用筆誤），引用時宜核對。

**對 CRS 的威脅評估**：**低**。雖然框架詞彙撞得兇（這正是必須引用切割的原因），但 BCEA 是 VQA 式是非驗證 + 單模型證據取得 + 單一保證，CRS 是 referring-set 定位 + 跨 base 組合 + 雙保證，內核完全正交。BCEA 反而可當成「同期、同調、不同任務」的最佳對照來凸顯 CRS 的 moat（set-level recall、cross-base factorization、gRefCOCO）。**建議在 related work 用上面那段切割文字明確引用並劃清界線，把它變成襯托而非競品。**
