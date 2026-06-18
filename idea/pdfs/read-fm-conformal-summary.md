# 論文精讀筆記：視覺基礎模型是好的 Conformal Predictor 嗎？

> 本筆記為 CRS 碩論（Cross-Base Conformal Referring Set，凍結 VL grounding base 上的 post-hoc conformal risk control）之外部文獻精讀，重點在抽取「**post-hoc risk control 優於重校準 base 分數**」這條主張的外部支撐與 caveat。

---

## 一、書目、問題定義與動機

### 書目
- **標題**：Are foundation models for computer vision good conformal predictors?（已查證標題完全一致）
- **arXiv**：2412.06082，版本 **v3，2026-02-16**（cs.CV），授權 CC BY 4.0
- **作者**：Leo Fillioux、Julio Silva-Rodríguez、Ismail Ben Ayed、Paul-Henry Cournède、Maria Vakalopoulou、Stergios Christodoulidis、Jose Dolz
- **單位**：MICS, CentraleSupélec, Université Paris-Saclay；LIVIA / ILLS, ETS Montréal
- 注意：Silva-Rodríguez（few-shot CLIP / CLAP 作者）與 Ben Ayed、Dolz 是 VLM few-shot adaptation 與 calibration 領域的核心團隊，本文的 few-shot 與 calibration 視角有強烈的「VLM 適配可靠性」背景。

### 問題定義
基礎模型（DINO、DINOv2、VICReg、CLIP 等）正大量進入高風險、安全敏感場景，但其**不確定性建模能力**鮮少被系統性研究。Calibration（校準，如 Temperature Scaling）雖普及，卻**缺乏理論保證**；Conformal Prediction（CP）能在 exchangeability 假設下提供**邊際覆蓋（marginal coverage）的有限樣本保證**，且直接作用在模型輸出（適合黑箱）。然而 CP 在「視覺基礎模型」上的行為此前幾乎無人探討（既有研究多聚焦傳統監督式模型，或只在 LLM 上做過 CP）。

### 核心研究問題
「**在視覺基礎模型時代，我該用哪個 CP 方法、哪個模型，能期待什麼結果？**」並特別檢視三個實務常見情境對 CP 的衝擊：分布偏移（domain shift）、信心校準（calibration）、few-shot 適配。

### 動機相對於 calibration 的立論（與 CRS 高度相關）
本文明確把 CP 與 calibration 並列為兩種不確定性量化路線，並指出 CP 三大優勢：
1. **直接作用於模型預測**，適合黑箱模型（不需重訓、不需碰模型內部）。
2. 產出**預測集合**而非單一點校正，含最可能的若干類別。
3. 在 exchangeability 下有**邊際覆蓋的理論保證**，calibration 沒有。

這正是 CRS「凍結 base + post-hoc 風險控制」的同源哲學。

---

## 二、方法與實驗設定

### 2.1 測試的基礎模型（共 17 個）
- **DINO**（自監督，ViT）：DINO-S、DINO-B（2 個）
- **DINOv2**：DINOv2-S/B/L/G（4 個）
- **VICReg**（自監督，CNN backbone）：ResNet-50、ResNet-50x2、ResNet-200x2（3 個）
- **VLM**（8 個）：5 個 CLIP、MetaCLIP、LLaVa、Phi
- 對照組：在 ImageNet 上以**標準監督式**訓練的 ViT-B（與基礎模型同架構，做公平比較）

> 適配方式：基礎模型**全部凍結（frozen）**，僅在每個資料集上訓練一個 **Linear Probing（LP）頭**（一層線性 + softmax，cross-entropy）。這點與 CRS 的「凍結 base」設定極為類似——模型本體不動，只在輸出端做事。

### 2.2 Conformal 方法（3 種非一致性分數）
1. **LAC**（Least Ambiguous Classifier, Sadinle 2018）：`S = 1 − π_x(y)`，等價於對真類 softmax 做門檻。集合最小，但**無自適應性**（固定門檻），在罕見類/不確定預測上覆蓋不穩。
2. **APS**（Adaptive Prediction Sets, Romano 2020）：`S = ρ(x,y) + π_x(y)·u`，累加排序後比 y 更可能的類別之信心，含 tie-break 隨機量 u。**自適應**，犧牲效率換取對不確定預測的覆蓋。
3. **RAPS**（Regularized APS, Angelopoulos 2020）：在 APS 上加 `λ(o(x,y) − k_reg)^+` 正則項，超過某集合大小後懲罰加入不可能類別，**壓縮集合大小**。

（注意：本文標題列了 THR/LAC 的脈絡，但實際只實作 LAC / APS / RAPS 三種；THR 即 LAC 這類 thresholding 家族的代表。）

### 2.3 資料集
- **主分析**：CIFAR-10、CIFAR-100、ImageNet
- **Domain shift**：ImageNet-R、ImageNet-A、ImageNet-Sketch、ImageNet-V2
- **few-shot（11 個下游細粒度/通用分類）**：SUN397、FGVCAircraft、EuroSAT、StanfordCars、Food101、OxfordPets、Flowers102、Caltech101、DTD、UCF101（加上 ImageNet）

每個資料集切成：訓練集 / conformal 集；conformal 集再切 **calibration 集（調 CP）** 與 **test 集（評估）**。

### 2.4 評估指標（如何量 set efficiency 與 coverage）
- **Set size（效率，↓）**：測試集上平均預測集合大小（公式 7）。
- **Coverage（覆蓋，↑）**：真類落在集合內的比例（公式 8），目標 = 1−α。
- **CovGap（覆蓋落差，↓）**：各類別 class-conditional coverage 與 (1−α) 之平均絕對差（公式 9）——衡量條件覆蓋是否均勻。
- **MCCC（最小類別條件覆蓋，↑）**：所有類別中最差的 class-conditional coverage（公式 10）。

主實驗目標覆蓋率多設 **1−α = 0.90**（表中 coverage 欄皆 0.900）。

---

## 三、完整關鍵發現

論文摘要列出 6 條 key observation，逐條展開：

### (i) 基礎模型本身就是好的 conformal predictor
高效能模型（LP accuracy 高）→ 一致地產出**更小的預測集合**，與 CP 方法無關。且**自監督/對比學習的基礎模型優於傳統監督式訓練的同架構模型**：
- **表 1（ImageNet, ViT-B 同架構公平比較）**：監督式 `ViT_ImageNet` 雖 accuracy 最高（76.08%），但其 **APS set size 暴增到 38.75**（CLIP 9.50、DINO-S 10.02、MetaCLIP 9.43），class-conditional coverage 也明顯較差。
- 圖 3 顯示這種 set size 差異是**大量樣本的系統性現象**，不是少數離群點。
- 結論：**訓練範式（pre-training scheme）對 conformal 性質的影響比單純 accuracy 更關鍵**。

### (ii) APS 覆蓋最穩、RAPS 集合最小（方法層排序）
- **APS**：empirical coverage / 條件覆蓋最佳，但 set size 最大。
- **RAPS**：set size 表現最好（接近 LAC），但代價是條件覆蓋範圍變大（CovGap 變差）。
- **LAC**：集合最小，但因無自適應機制，各類覆蓋率不一致、CovGap 變異大。
- 機制論證：RAPS 的懲罰項使低 accuracy 模型**無法為困難類別擴張集合**，導致 MCCC 被壓低（圖 2 在 CIFAR-100 上實證）；APS 則靠擴大集合來補償弱模型，犧牲效率換條件覆蓋。

### (iii) Domain shift 下 APS 最穩健（覆蓋面）
- 在 ImageNet-R/A/Sketch/V2 上（calibration 用乾淨 ImageNet、test 用偏移版本，**刻意破壞 exchangeability**）：
  - **APS 幾乎一致地維持目標覆蓋**（圖 5 中段），但 set size 隨域複雜度大幅膨脹。
  - **RAPS 表現接近 LAC**，多個模型/域下邊際覆蓋低於目標。
- 圖 4：ImageNet-A 上 APS 的條件覆蓋分布呈**高斯狀**（偏離 1−α 的類別數遞減），RAPS 則近乎**均勻散布**且有不可忽略比例的類別跌破目標覆蓋——RAPS 在 OOD 下條件覆蓋結構明顯較糟。
- **VICReg（CNN backbone）在 domain shift 下最差**（更大集合、更低覆蓋），呼應 (vi)。

### (iv) 【最關鍵】Calibration（Temperature Scaling）會降低 CP 集合效率
這是全文與 CRS 立論最直接相關的發現：
- 設定：對 §4.2.1 的 ImageNet 結果套用 **TS**，掃 T∈[0.85, 2]（14 個值），找到 **T=1.1** 為典型良好校準點（ECE 下降）。
- **表 2 結論**：模型一旦被（更好地）校準，**預測集合反而變大**，尤其對自適應方法（RAPS，特別是 APS）。例如：
  - DINOv2-B：APS set size **7.50 → 10.46**（校準後變大）；
  - DINOv2-L：APS **6.77 → 9.67**；
  - CLIP(ViT-B)：APS **9.50 → 11.11**。
  - LAC 的 set size 幾乎不變（固定門檻對全域縮放不敏感）。
- **圖 6**：校準前後 set size 差異是**橫跨樣本的一致性退化**，非少數異常樣本造成。
- **機制**：TS 平滑 softmax → 預測更不自信 → 主導類別機率下降 → APS 累加更多類別才達到 1−α 門檻 → set size 單調隨 T 上升。
- **代價/補償**：校準確實**略微改善 class-conditional coverage（MCCC 微升）與 CovGap**（圖 7：APS 在 T=1.1 的 CovGap 0.0567，接近其最佳 0.0561）。但 set efficiency 被犧牲。
- 作者立場：在關鍵決策系統中，「適度增大集合換取更好的覆蓋落差」是可接受的取捨；但**就效率（CP 文獻最看重的指標）而言，calibration 是傷害自適應 CP 的**。

> 一句話總結：**對 frozen FM 做信心校準（重新縮放分數），會降低自適應 conformal 集合的效率。**

### (v) Few-shot 適配的 conformal 分數優於 zero-shot（ID 明顯、OOD 邊際）
- **表 3（CLIP ViT-B）**：相對 zero-shot（ZS），few-shot 方法（Adapters: ZSLP/CLAP；Prompt Learning: CoOp/KgCoOp）在 **ID 情境**一致地產出**更小 set size 與更小 CovGap**：
  - ID APS set size：ZS **6.98** → CoOp **2.87**、ZSLP **3.43**；ID APS CovGap：0.094 → 0.059。
- **OOD 情境**：只有 **APS** 一致改善 ZS 的 set efficiency（APS set size 19.22 → 16~17），CovGap 改善有限。
- **機制連結**：呼應 (iv)——Murugesan 2024a 已證 few-shot CLIP 適配會**惡化信心估計（變得 overconfident）**；依本文邏輯，**更不校準（更尖銳的 softmax）反而對應更小的 conformal 集合**，表 3 正好驗證此因果方向。

### (vi) ViT 系（DINO/CLIP）在 domain shift 下退化幅度小於 CNN 系（VICReg）
- 含 visual transformer 的基礎模型在 conformal 指標上對域偏移更穩健；CNN backbone（VICReg）退化最嚴重。

### 結論（§5）
- 視覺基礎模型的 conformal 指標優於傳統預訓練對應物，ViT 系 > CNN 系，尤其在域偏移下。
- 分布偏移與信心校準對「某些」CP 方法有害；**APS 對這些情境最穩健（特別是條件覆蓋），代價是效率退化。**
- 最終選擇取決於任務需求（如醫療診斷偏好最大化條件覆蓋、容忍大集合）。

---

## 四、對 CRS 的意涵

### 4.1 哪些發現「支撐」CRS「post-hoc risk control 優於重校準 base 分數」？

**支撐點 A（最強）— Finding (iv)：校準傷害自適應 CP 效率。**
本文用 17 個 frozen FM、TS 校準、ImageNet 系統性證明：**對凍結基礎模型的分數做重新縮放（recalibration），會一致性地降低自適應 conformal 集合的效率**（APS/RAPS set size 全面變大，表 2 + 圖 6）。CRS 選擇「不去重校準 base 的原始分數，而是直接在凍結 base 上做 post-hoc 風險控制（LTT/conformal）」——本文正好提供獨立的、跨多模型的反面證據：**「先校準再 conform」這條路會犧牲效率**。可直接作為 CRS「不重校準 base」這一設計選擇的外部背書。

**支撐點 B — CP 直接作用於黑箱輸出的哲學（§1, §2）。**
本文反覆強調 CP 相對 calibration 的優勢：直接作用模型預測、適合黑箱、有理論覆蓋保證。這與 CRS「凍結 base、post-hoc」哲學同源，可引用為 motivation 層的範式立論。

**支撐點 C — Finding (i) + 表 1：凍結基礎模型本身就是好的 conformal 基底。**
LP-on-frozen-FM 的 conformal 指標優於監督式重訓模型。這支撐 CRS「凍結 base 而非重訓/微調」的合理性——**重訓不是 conformal 效率的必要條件，甚至可能更差**。

**支撐點 D — Finding (iii)：APS 在 domain shift 下最穩健（維持覆蓋）。**
CRS 強調「在分布變動下仍要守住召回/覆蓋保證」。本文證明自適應 CP（APS）在破壞 exchangeability 時仍近乎維持邊際覆蓋（靠擴大集合），這支撐 CRS「靠 conformal/risk control 而非靠校準分數來守覆蓋」的策略選擇。

### 4.2 Caveat：從「分類」外推到「grounding」的鴻溝（必須誠實揭露）

這是引用本文時**最重要的限制**，審稿人很可能質疑：

1. **分數型態不同。** 本文的非一致性分數全建立在**多類別 softmax 機率 π_x(y)**（K 個互斥類別、總和為 1 的封閉分布）。CRS 用的是 **grounding/detection 分數**（OWL-ViT 的 query-region 相似度、GroundingDINO 的 box logit），這些**不是封閉的 softmax 分布**，沒有「對所有類別累加」的天然語意，APS/RAPS 的「排序累加機率」機制無法原樣搬過來。因此本文「APS 最穩 / calibration 傷效率」的**具體數值結論不能直接外推**到 grounding 分數。

2. **輸出空間結構不同。** 本文 prediction set 是「類別子集」；CRS 的輸出是「referring set / box 集合」與棄答（abstention/no-target），且 CRS 用 **LTT（Learn-then-Test）做風險控制**，是 risk control（控 FNR/FDR 等損失）而非單純 marginal coverage 的 conformal。本文未觸及 LTT 或多風險 Bonferroni 校準。

3. **校準對象的可比性。** 本文證的是「對 softmax logits 做 TS 會傷 APS 效率」。CRS 的 base 分數**未必是過自信的 softmax**（grounding score 的尺度與分布很不同），「校準會傷效率」的因果鏈（平滑 softmax → 累加更多類）在 grounding 設定下不必然成立。所以本文只能作為「**重校準分數不是免費午餐、且在 frozen FM 上常有反效果**」的**類比性/精神性支撐**，不能宣稱「已在 grounding 上證實」。

4. **可信度層級。** 本文是**實證 survey/benchmark**，非新方法論文；結論是經驗觀察（observation），其機制論證（如 RAPS 懲罰壓低 MCCC）是 plausible argument + 實證圖佐證，非嚴格定理。引用時宜定位為「**經驗證據顯示**」而非「理論證明」。

### 4.3 該怎麼引用、放哪裡？

- **放 Motivation（首選）**：用本文支撐「**為何選 post-hoc conformal/risk control 而非重校準 base 分數**」。一句範例敘述：
  > 「近期對視覺基礎模型的大規模 conformal 研究 [Fillioux et al., 2026] 指出，對凍結基礎模型做信心校準（如 Temperature Scaling）會**一致性地降低自適應 conformal 集合的效率**；few-shot 適配雖惡化校準卻反而改善 conformal 分數。這暗示在凍結基礎模型上，**重新校準原始分數並非取得可靠不確定性的最佳路徑**，從而支持本文採取 post-hoc 風險控制、保持 base 分數不動的設計。」
- **可在 Method Justification 輔助一句**：解釋 CRS 為何「不對 OWL/GroundingDINO 分數做 TS/Platt 重校準，而是直接在其上做 LTT」。但**務必加 caveat 句**，聲明本文結論建立在分類 softmax，CRS 設定為 grounding 分數，故僅作為精神性/類比支撐，CRS 的對應現象由本文自身實驗（2×2 ablation、cross-base composition）證成。
- **不要**用本文宣稱 APS/RAPS 的數值結論在 grounding 上成立，也不要拿來當「CRS 方法正確性」的證據——它只支撐**設計取向**（不重校準），不支撐 CRS 的具體機制。

### 4.4 引用優先級

- **中高優先（建議納入）**：作為 motivation 的關鍵外部佐證之一，特別是 §3.x 解釋「為何 post-hoc、為何不重校準 base」時。它是少數**直接在 frozen FM + CP 上做系統實證**的文獻，新近（v3, 2026-02）、來源權威（CLIP few-shot/calibration 核心團隊），時效性與相關性都好。
- **不必當主引用**：因分類→grounding 的外推鴻溝，不宜作為 CRS 核心主張的唯一支柱，宜與 CRS 自身證據並列。

---

## 五、個人評價

### 可信度
- **高**：17 個模型 × 3 CP 方法 × 多資料集 × 4 種情境（標準/域偏移/校準/few-shot）的廣度紮實；指標定義標準（coverage/set size/CovGap/MCCC）；coverage 恆守 0.90 的保證在表中明確呈現，符合 CP 理論。
- 機制論證（RAPS 懲罰壓低弱模型 MCCC、TS 平滑→APS 集合膨脹）有圖 2/6 的樣本級分布佐證，不只是平均數字，說服力佳。
- 作者群在 VLM few-shot / calibration 領域有深厚 track record（Silva-Rodríguez 的 CLAP、Ben Ayed/Dolz），few-shot 與 calibration 的因果連結（呼應 Murugesan 2024a）有內部一致性。

### 限制
1. **僅分類任務**：無偵測/grounding/分割，對 CRS 是最大外推風險（見 4.2）。
2. **公平比較的內在難題**：作者自承監督式 ViT 與基礎模型的訓練資料規模不同（ImageNet 無法訓出 FM），表 1 的「FM 優於監督式」結論受此 confound 限制。
3. **calibration 只測 TS + histogram binning（附錄）**：未涵蓋訓練期校準、其他 post-hoc 方法，「校準傷效率」的普適性有邊界。
4. **T=1.1 的選取**：無正式 validation set，靠掃格 + 對齊文獻先驗（TinyImageNet）決定，略有主觀。
5. **APS 的「穩健」帶嚴重效率代價**：domain shift 下 APS set size 可膨脹數倍（如 OOD set size 19.22），實務可用性需打折扣——這點對 CRS 也是提醒：靠擴大集合守覆蓋，size CI 會變大。

### 一句話定位（for CRS）
本文是 CRS「**不重校準凍結 base 分數、改採 post-hoc 風險控制**」這一設計取向的**最佳近鄰外部佐證**——它在分類域、跨 17 個 FM 上實證了「校準會傷自適應 CP 效率、few-shot（更不校準）反而改善 conformal 分數」。引用時務必標明「分類 softmax → grounding 分數」的外推 caveat，把它放在 motivation 作精神支撐，而非 CRS 機制正確性的證據。

---

*精讀來源：arXiv:2412.06082v3 全文（firecrawl 抓取 HTML，含 abstract、§1–§5、表 1–3、圖 1–7 文字描述）。*
