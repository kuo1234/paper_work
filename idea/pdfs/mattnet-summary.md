---
title: "MAttNet: Modular Attention Network for REC 詳細導讀"
author: "整理：kuo"
date: "2026-06-19"
---

# MAttNet: Modular Attention Network for Referring Expression Comprehension（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：MAttNet: Modular Attention Network for Referring Expression Comprehension
- **作者**：Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, Tamara L. Berg
- **單位**：UNC Chapel Hill、Adobe Research
- **發表會議**：**CVPR 2018**
- **arXiv**：[1801.08186v3](https://arxiv.org/abs/1801.08186)（v3 提交於 2018-03-27）
- **Code**：<https://github.com/lichengunc/MAttNet>（另有 Mask R-CNN 實作 <https://github.com/lichengunc/mask-faster-rcnn>）
- **Demo**：vision2.cs.unc.edu/refer/comprehension
- **任務範疇**：Referring Expression Comprehension（REC，box-level）+ Referential Segmentation（pixel-level）

> 註：本文是 modular REC 研究線的代表性後繼。前傳是 Hu et al. *Modeling Relationships in Referential Expressions*（CVPR 2016，本目錄 [modeling-relationships-summary.md](modeling-relationships-summary.md)）與 *Compositional Modular Networks*（CVPR 2017）。MAttNet 把「Subj/Loc/Rel 三模組 + 語言解析」全面升級成可端到端學習的 attention 版本，成為其後多年 REC 的強 baseline。

---

## 1. 問題定義與動機

### 1.1 任務

給定影像 `I`、一組候選物件 `O = {o_i}^N_{i=1}`（proposal 或 GT box）、一段 referring expression `r`，從 `O` 中挑出 `r` 所指的物件。本質是 language-conditioned region selection。

### 1.2 動機：兩個既有缺陷

1. **「一鍋燴」特徵 + 單一 LSTM**：先前主流（Speaker/Listener/Reinforcer、visdif 等）把 target 外觀、位置、context 特徵直接 concat，用單一 LSTM 編碼整句，**忽略不同 referring expression 強調的資訊型態不同**。論文舉例：同一顆紅球，在黑球堆裡只需說 "red ball"；在紅球堆裡要說 "red ball on the right"；在 100 顆紅球裡要說 "red ball next to the cat"——不同情境該觸發不同的視覺處理模組。
2. **外接 language parser 會傳遞錯誤**：早期 modular network（NMN、CMN）靠 off-the-shelf parser 把句子拆成元件再組裝模組，parser 的解析錯誤會直接污染模型。

### 1.3 三大新穎點（作者自陳）

1. **為「通用」referring expression 設計的三模組**：subject（類別/顏色/屬性）、location（絕對+相對位置）、relationship（與其他物件的關係），涵蓋 Yu et al. 2016 定義的 7 種 referring attribute。各模組在自己的參數空間學習、互不干擾。
2. **soft attention 學會「解析」句子**，取代外接 parser：對句子自動算出每個詞屬於哪個模組（word attention）+ 每個模組該貢獻多少（module weight）。
3. **subject/relationship 模組各用不同視覺 attention**：subject 用 "in-box" soft attention（句子提到 "man in red shirt" → 注意框內紅色衣物區）；relationship 用 "out-of-box" hard attention（"cat on chair" → 注意框外的 chair 來定位 cat）。

訓練只用 `(o_i, r_i)` 配對監督，word attention / module weight / spatial attention / relative-object attention 全部端到端弱監督學成。

---

## 2. 方法架構

模型 = 語言注意力網路 + 三個視覺模組（subject / location / relationship），最後加權合成總分。

### 2.1 語言注意力網路（Language Attention Network）

- 詞 embedding → **雙向 LSTM**，每個詞的表示 `h_t = [→h_t, ←h_t]`。
- 三個可訓練向量 `f_m`（m ∈ {subj, loc, rel}）對所有詞算 attention：
  `a_{m,t} = softmax_t(f_m^T h_t)`，模組片語 embedding `q_m = Σ_t a_{m,t} e_t`。
- 模組權重：concat 首尾 hidden `[h_0, h_T]` 過 FC + softmax →
  `[w_subj, w_loc, w_rel]`。
- 對比 *Modeling Relationships*（CMN）固定 (Subject, Preposition/Verb, Object) 三元結構：MAttNet 不假設固定句法模板，故能處理 "smiling boy"（只有 subj）、"man on left"（subj+loc）、"cat on the chair"（subj+rel）等各種變體。

### 2.2 視覺骨幹

- **Faster R-CNN（ResNet-101）** 當 backbone。對每個 `o_i` 取 C3 feature（低階：顏色/形狀，利於 proposal 判斷）與 C4 feature（高階：類別語意）。兩者都用。
- 與多數前作只用單張 region CNN feature 不同，這裡用偵測器骨幹更原則化也更快。

### 2.3 Subject Module

兩個子任務：

1. **屬性預測分支（Attribute Prediction）**：concat C3+C4 → 1×1 conv → avg pool → multi-attribute 分類（binary cross-entropy，類別權重 `w_j = 1/√freq` 緩解不平衡）。屬性標籤用 template parser 從訓練句抽 color/generic word（取最常用 50 個）。只有含屬性詞的句子走這分支。
2. **片語引導的注意力池化（Phrase-guided Attentional Pooling，"in-box"）**：把 attribute blob 與 C4 融成 subject blob（14×14 grid），用 `q_subj` 算 grid 上 spatial attention，加權求和成 `v_i^subj`。

**匹配函式** `S(o_i|q_subj) = F(v_i^subj, q_subj)`：兩條 MLP + L2 normalize，內積算相似度。同一匹配函式也用於 loc / rel 分數。

### 2.4 Location Module

- 絕對位置 5 維：`[x_tl/W, y_tl/H, x_br/W, y_br/H, w·h/W·H]`（RefCOCO 41%、RefCOCOg 36% 句子含絕對位置詞）。
- 相對位置（"dog in the middle"、"second left person"）：取同類別最多 5 個鄰居，算 offset 與面積比 `δl_ij`。
- 合成 `l_i^loc = W_l[l_i; δl_i] + b_l`，分數 `S(o_i|q_loc) = F(l_i^loc, q_loc)`。

### 2.5 Relationship Module（"out-of-box"）

- 對候選 `o_i` 取周圍（最多 5 個，**不分類別**）支援物件 `o_ij`，用其 avg-pooled C4 當外觀 `v_ij`，加上相對 offset `m_ij`。
- `v_ij^rel = W_r[v_ij; m_ij] + b_r`，對每個鄰居算分取最大：
  `S(o_i|q_rel) = max_{j≠i} F(v_ij^rel, q_rel)`。
- 取 max 等價於弱監督的 Multiple Instance Learning（MIL）。

### 2.6 損失函式

總分（hard attention 由 module weight 體現）：

`S(o_i|r) = w_subj·S(o_i|q_subj) + w_loc·S(o_i|q_loc) + w_rel·S(o_i|q_rel)`

每個正例 `(o_i, r_i)` 取兩個負例（同物件配別句 `r_j`、同句配別物件 `o_k`），combined hinge / ranking loss：

`L_rank = Σ_i [λ_1 max(0, Δ + S(o_i|r_j) − S(o_i|r_i)) + λ_2 max(0, Δ + S(o_k|r_i) − S(o_i|r_i))]`

總損失 `L = L_subj^attr + L_rank`。

---

## 3. 實驗

### 3.1 數據集

RefCOCO、RefCOCO+、RefCOCOg（皆建於 MS COCO）。

- RefCOCO / RefCOCO+：互動式遊戲收集，句短（平均 3.5 詞）、同類物件多（平均 3.9）。RefCOCO+ **禁用絕對位置詞**，逼模型靠外觀。
- RefCOCOg：非互動收集，句長（平均 8.4 詞）、關係描述多、同類物件少（1.63）。
- RefCOCO/RefCOCO+ 測試分 testA（多人）/ testB（多物）。
- 評估：IoU > 0.5 為正確。

### 3.2 GT box 下的主結果（Table 1，res101-frcn 全模型）

| Split | RefCOCO testA | RefCOCO testB | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | RefCOCOg test |
|---|---|---|---|---|---|---|
| **MAttNet（full）** | **85.26** | **84.57** | **75.13** | **66.17** | **78.10** | **78.12** |

相較前 SOTA（Speaker+Listener+Reinforcer）在 bbox localization 上有約 10% 的提升；vgg16 版（Line 9, testA 79.99）已勝所有 vgg16 前作。

### 3.3 模組 ablation（Table 2，res101-frcn，GT box）

| 配置 | RefCOCO testA | RefCOCO+ testA | RefCOCOg test |
|---|---|---|---|
| Matching: subj+loc（baseline，concat 特徵） | 79.42 | 62.71 | 70.92 |
| MAttN: subj+loc | 80.20 | 64.20 | 72.62 |
| + loc(+dif)（相對位置） | 81.28 | 65.77 | 74.46 |
| + rel（關係模組） | 81.58 | 66.59 | 74.56 |
| + attr（屬性分支） | 82.66 | 69.93 | 75.92 |
| **+ attn（片語引導注意力池化）** | **85.26** | **75.13** | **78.12** |
| parser+MAttN（用 template parser 取代學到的解析） | 79.10 | 62.94 | 73.72 |

關鍵讀法：
1. 每個模組逐步加分，乾淨可加性 ablation。
2. **片語引導注意力池化在 person 類（testA）增益最大**（RefCOCO+ 69.93→75.13），印證 "in-box" attention 對「girl with red hat」這類框內細節有用。
3. **學到的 soft attention 解析 > 外接 parser 約 5%**（RefCOCOg val 78.10 vs 73.82），證成「不用外接 parser」的設計選擇。

### 3.4 全自動偵測（Table 3）

換成 Faster R-CNN / Mask R-CNN 自動偵測 proposal（非 GT box）後，分數下降但**各模組增益順序與 GT box 一致**（穩健），且全面勝過前 SOTA。Mask R-CNN 偵測分支（res101-mrcn）甚至優於 Faster R-CNN（多任務訓練帶來更高 AP）。RefCOCO testA 全自動 80.43–81.14。

### 3.5 Referential Segmentation（Table 4）

不另訓 FCN，而是**解耦**：MAttNet 選出最佳 box → 餵 Mask R-CNN mask 分支出 pixel mask。在 RefCOCO/RefCOCO+ 上 Pr@0.5 與 IoU 大幅勝過前 SOTA（D+RMI+DCRF），例如 RefCOCO val IoU 45.18→56.51、Pr@0.5 約翻倍。附錄甚至顯示 **MAttNet+GrabCut**（劣等分割法）都還勝過前 SOTA → 論證「box 定位與分割解耦」優於 FCN 式前景/背景分類。

### 3.6 效率（附錄）

全自動推論平均 0.33 s/張：Mask R-CNN 0.31 s + MAttNet 僅 0.02 s。訓練約半天（單張 Titan-X）。

---

## 4. 關鍵洞見與貢獻

1. **可端到端學的 modular attention**：把 Subj/Loc/Rel 從「外接 parser 拼裝」升級成「soft word attention + module weight 自動解析」，且勝過 parser。
2. **雙視覺 attention**：subject in-box（框內細節）+ relationship out-of-box（框外支援物件），各司其職。
3. **屬性預測輔助分支**：明確處理 "woman in red" 這類屬性區分。
4. **box 定位與分割解耦**：對 instance-level referential segmentation 比 FCN 式分類更優。
5. **多年強 baseline**：RefCOCO/+/g 三庫 SOTA，成為後續 REC 的標準對照。

---

## 5. 限制（以 2026 年眼光）

1. **single-target 假設**：最終 argmax 只挑一個 region，**無法處理 no-target / multi-target**——與 GREC/gRefCOCO 設定不相容（需外接 abstention/set-prediction head）。
2. **task-specific supervised**：需在 RefCOCO 系列 fine-tune，非 frozen / zero-shot；對比 CLIP/OWL-ViT/GroundingDINO 內建跨模態對齊已過時。
3. **無 calibration / 無可靠性分析**：合成分數只用來 argmax，沒有覆蓋率、棄答、信賴度保證——正是 selective grounding / CRS 主線要補的洞。
4. **依賴鄰居枚舉**：location 相對位置與 relationship 模組都要枚舉周圍物件，大場景下成本高。
5. **關係語意有限**：spatial offset 抓得到左右/上下/距離，semantic relation（holding、belonging）仍弱。

---

## 6. 對使用者主線（selective grounding / CRS）的對位與借鏡

> 主線見記憶 [[crs-pivot-conformal-referring-set]] / [[crs-litreview-15papers]]；本目錄索引 [reading-log.md](reading-log.md)。

1. **SOTA 雙區塊表的「Trained」代表**：MAttNet 是經典 trained REC，恰好放進 reading-log 規劃的「雙區塊 + Guarantee? 欄」——MAttNet 標 **Guarantee ✗（point estimate、無分布無關保證）** 對照 Frozen CRS 標 ✓。它的高絕對分數（如 testA 85.26）正可用來說明「高準確率 ≠ 有限樣本保證」的 framing 守則。
2. **factorization ablation 範式**：MAttNet 的 subj→loc→rel→attr 逐步加法 ablation，是 CRS 2×2（OWL gate × GD box）factorization ablation 的可對照前例；差別在 CRS 證的是「跨兩異質 frozen base 分工非 ensemble」，MAttNet 是「同模型內模組可加性」。
3. **module weight 概念可借**：MAttNet 用 module weight 動態決定各模組貢獻——CRS 在 multi-target referring set 內部一致性評分時，可借 attention-aggregation 思路聚合候選集分數。
4. **解耦 box/segmentation 的精神**：MAttNet「先定位 box 再交 mask 分支」與 CRS「OWL gate 控可用性/棄答、GD box 控集合緊緻」的分工哲學相通——都把單一複雜任務拆成各自最擅長的凍結元件。
5. **Related Work 定位**：MAttNet 與 *Modeling Relationships*（explicit relational REC 起點）、Cops-Ref（bias-control benchmark）、GREC（任務泛化）、Zero-Shot True/False Verification（新威脅）並列，鋪陳「REC 從 modular-supervised → frozen/zero-shot → 需要可靠性保證」的十年演進弧線，襯托 CRS 補上的「分布無關 recall+abstention 雙保證」這一空缺。

**引用優先級**：中（REC modular 經典/強 baseline，SOTA 對照表與演進敘事用；非 CRS 直接競品，無切割壓力）。

---

## 7. 一句話總結

> MAttNet 是 **2018 年把 modular REC 推到可端到端學習的代表作**：用語言 soft attention 自動把句子解析成 subject/location/relationship 三模組（取代外接 parser），各模組配專屬視覺 attention（in-box 看框內細節、out-of-box 看框外關係），在 RefCOCO/+/g 三庫 box 與 pixel 兩層級都大幅刷新 SOTA，成為其後多年 REC 的標準強 baseline——但它 single-target、需訓練、無 calibration，正好是 selective grounding / CRS 主線要超越的「高準確率卻無保證」對照組。

---

## 8. 參考連結

- arXiv：<https://arxiv.org/abs/1801.08186>
- Code：<https://github.com/lichengunc/MAttNet>
- Mask R-CNN 實作：<https://github.com/lichengunc/mask-faster-rcnn>
- 前傳 CVPR 2016：*Modeling Relationships in Referential Expressions* <https://arxiv.org/abs/1607.07702>
- 前傳 CVPR 2017：*Compositional Modular Networks* <https://arxiv.org/abs/1611.09978>
