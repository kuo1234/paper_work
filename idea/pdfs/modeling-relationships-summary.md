---
title: "Modeling Relationships in Referential Expressions 詳細導讀"
author: "整理：kuo"
date: "2026-06-18"
---

# Modeling Relationships in Referential Expressions（詳細導讀）

## 0. 論文基本資訊

- **論文標題**：Modeling Relationships in Referential Expressions
- **作者**：Ronghang Hu, Marcus Rohrbach, Jacob Andreas, Trevor Darrell, Kate Saenko
- **單位**：UC Berkeley（EECS / ICSI）、Boston University
- **發表會議**：**CVPR 2016**（IEEE Conference on Computer Vision and Pattern Recognition）
- **arXiv**：[1607.07702](https://arxiv.org/abs/1607.07702)（提交於 2016-07-26）
- **CVF Open Access**：<https://openaccess.thecvf.com/content_cvpr2016/html/Hu_Modeling_Relationships_in_CVPR_2016_paper.html>
- **Code**：<https://github.com/ronghanghu/text_obj_retrieval>
- **Project page**：<https://ronghanghu.com/text_obj_retrieval/>
- **任務範疇**：Referring Expression Comprehension（REC，指稱表達理解）

> 註：作者群同年另有一篇 *Natural Language Object Retrieval*（arXiv 1511.04164, CVPR 2016），那篇是「SCRC（Spatial Context Recurrent ConvNet）」；本篇則是其後續，重點放在「物件之間的關係」建模。隔年（CVPR 2017）他們再延伸成 *Compositional Modular Networks*（arXiv 1611.09978）。三篇是同一條研究線。

---

## 1. 問題定義與動機

### 1.1 任務：Referring Expression Comprehension（REC）

給定：

- 一張影像 `I`
- 一段自然語言描述 `S`（例如 *"the man next to the woman in red"*）

模型要從影像中一群候選 bounding box `{r_1, r_2, ..., r_N}` 中，挑出 `S` 所指的那一個物件。
本質上是 **language-conditioned object selection**，是 grounding/visual reasoning 的核心子任務之一。

### 1.2 動機：既有方法把 referred object「孤立」處理

CVPR 2016 之前主流的 REC 方法（Mao et al. 2016 的 MMI、Neg Bag、Neg+MMI；Hu et al. 的 SCRC）有一個共同的盲點：

> **它們只看「被指物件本身」的外觀＋位置，沒有顯式建模「它跟其他物件的關係」。**

但人類描述物件時，**關係性描述非常普遍**：

- *"the cat **on the chair**"*
- *"the man **next to the woman in red**"*
- *"the cup **between two books**"*

這些描述如果只看被指物件本身的 CNN feature，會出現兩個問題：

1. **同類物件多時無法區分**：場景裡有 5 隻貓，光看貓的外觀區分不開，必須借助「在椅子上」這個關係。
2. **語言中關係名詞被浪費**：模型可能完全忽略 "on", "next to" 這些 prepositional phrase 的資訊。

### 1.3 本文的主張

作者主張顯式地把 **(subject, relationship, object)** 三元組的表徵融合進 REC 模型中，並進一步擴充到「同時考慮多個 context object」的情境。實驗顯示，加入關係建模能在 Google-Ref（RefCOCOg）上把 P@1 從 0.247（Neg+MMI SOTA）**幾乎翻倍**到 **0.493**。

---

## 2. 方法架構

整體模型可以拆成 **四個 scoring module**，每個 module 對 candidate region `r_i` 算一個分數，最後加總得到該 region 是 referred object 的總分。模型的輸出是 `argmax_i score(r_i | S, I)`。

四個 module 對應論文的 ablation 順序：

| 縮寫 | Module | 訊號來源 |
|------|--------|----------|
| **Subj** | Subject module | 候選 region 的 CNN 外觀特徵 |
| **Loc** | Location module | region 的歸一化空間座標／面積 |
| **Rel** | Relationship module | (subject, context object, spatial encoding) 三元組 |
| **Dist** | Distance / context-distillation module | 將多個 context object 的關係資訊聚合 |

語言端則用一個 **LSTM** 編碼 referring expression `S`，並用 **soft attention** 把不同詞分派到不同 module（subject 詞用於 Subj，關係詞用於 Rel，方位詞用於 Loc 等）。

### 2.1 視覺與語言基本元件

- **影像 backbone**：VGG-16，在每個 candidate region 上 RoI-pool 出 fc7（4096 維）feature `v_i`。
- **候選 region**：使用 **ground-truth bounding box** 作為 candidate（評估時所有方法統一條件，避免被 detector recall 拉低）。
- **語言編碼**：將 expression token 用 word embedding 後丟進 **LSTM**，得到每個 timestep 的 hidden state `h_t`。
- **語言注意力**：對 LSTM 隱狀態做 attention，產生四個 module 各自吃的「語言上下文向量」 `q_subj, q_loc, q_rel`。
  - 直覺：*"the man next to the woman in red"*
    - "man" → 偏向 subject module
    - "next to" → 偏向 relationship module
    - "in red" 描述 context object 的外觀 → 進 Rel module 的 object 端
    - 沒有強烈方位詞 → loc 權重小

### 2.2 Subject Module（Subj）

- 輸入：候選 region 外觀 `v_i`、語言查詢 `q_subj`
- 透過 MLP 把兩者投影到共同空間後算內積（或 cosine），得到 subject-only 分數 `s_subj(r_i)`。
- 等價於一個 *image-text matching* 的基線。

### 2.3 Location Module（Loc）

- 輸入：region 的 5 維幾何特徵 `[x_min/W, y_min/H, x_max/W, y_max/H, area/W·H]`
- 加上 region **相對於其他 region** 的位置差異（可選），透過 MLP 與 `q_loc` 結合
- 解決 *"the leftmost girl"*、*"the small cup"* 這類純位置／尺寸描述

### 2.4 Relationship Module（Rel）— 本文核心

對每個候選 subject region `r_i`，與場景中其他 region `r_j`（context object）配對：

1. **subject 端**：`v_i`（外觀）
2. **object 端**：`v_j`（外觀）
3. **spatial encoding**：兩個 box 的相對幾何（中心距離、相對大小、方向 angle、IoU），編碼成 *visual spatial vector* `p_{ij}`
4. **fusion**：`f_{ij} = MLP([v_i, v_j, p_{ij}])`
5. **語言對齊**：用 `q_rel` 對 `f_{ij}` 算分數 `s_rel(r_i, r_j)`

→ 對應 SVO（subject-verb-object）的三元組表徵，這是論文標題「modeling relationships」的核心構件。

### 2.5 Distance / Multi-context Module（Dist）

單一最強 context object 不夠（例如 *"the cup between two books"* 同時需要兩本書）。Dist module 的做法是：

- 對每個 candidate subject `r_i`，跟所有其他 region `r_j` 都算 `s_rel(r_i, r_j)`
- 用 **soft-max attention / weighted aggregation** 把所有 context 的關係分數聚合成一個值
- 也可看成「distillation」：把整個 scene 對該 subject 的關係證據濃縮成一個 scalar

公式上類似：

$$
s_{\text{rel}}^{\text{agg}}(r_i) = \sum_{j \neq i} \alpha_{ij} \cdot s_{\text{rel}}(r_i, r_j)
$$

其中 `α_{ij}` 是 context object 的注意力權重，可由語言或視覺幾何條件決定。

### 2.6 總分

$$
\text{score}(r_i \mid S, I) = s_{\text{subj}} + s_{\text{loc}} + s_{\text{rel}}^{\text{agg}}
$$

並以 **softmax over candidates** 訓練（讓 ground-truth region 的分數高過其他）：

$$
\mathcal{L} = -\log \frac{\exp(\text{score}(r_{\text{gt}}))}{\sum_i \exp(\text{score}(r_i))}
$$

這是 MMI（Maximum Mutual Information，正例與其他 box 對比）的精神，但施加在更豐富的特徵上。

---

## 3. 實驗設定

### 3.1 數據集：Google-Ref（後來叫 RefCOCOg）

- 由 Mao et al. 2016 提出
- **規模**：104,560 referring expressions / 54,822 objects / 26,711 images
- 來源：MSCOCO 影像 + Mechanical Turk 人工撰寫的長描述
- 特色：句子普遍較長、含較多關係描述（適合測試本文方法）

### 3.2 評估指標

- **Top-1 Precision（P@1）** on validation split
- 在 ground-truth box 候選下，模型 argmax 是否落在 gt region

### 3.3 比較基線

- **MMI**（Mao et al. 2016）：純 LSTM-based, 用 expression likelihood 排序 region
- **Neg Bag**：用負例強化的版本
- **Neg+MMI**：兩者合體，CVPR 2016 當時的 SOTA
- 本文自己的 ablation 也作為 baseline 排比

---

## 4. 主要結果

### 4.1 Google-Ref 驗證集 P@1

| Method | P@1 |
|---|---|
| MMI (Mao et al., 2016) | 0.2336 |
| Neg Bag (Mao et al., 2016) | 0.2342 |
| Neg+MMI (Mao et al., 2016) | 0.2468 |
| Naïve (Ours) | 0.3491 |
| Subj (Ours) | 0.4170 |
| Subj + Loc (Ours) | 0.4488 |
| Subj + Loc + Rel (Ours) | 0.4731 |
| **Subj + Loc + Rel + Dist (Ours)** | **0.4932** |

### 4.2 解讀

1. **Naïve > Neg+MMI（0.349 vs 0.247）**：光是把模型從「生成式 likelihood」改成「discriminative softmax over candidates」就已經大躍進；說明訓練 objective 換成直接比候選 region 的 discriminative loss 是重要的設計選擇。
2. **Loc 加 +3.2%**：證明 location 不是被 CNN feature「順便編碼」進去，加進來有獨立貢獻。
3. **Rel 再加 +2.4%**：核心貢獻 — 顯式關係建模真的有用。
4. **Dist（多 context 聚合）再加 +2.0%**：說明「只看一個最強 context 不夠」，多 object 聚合對長描述有幫助。
5. **整體相對前 SOTA 提升約 100%**（0.247 → 0.493）。

### 4.3 Ablation & 分析

論文中還有以下分析（依文中與 project page 的圖表）：

- **依描述類型分組**：含明顯關係詞（"next to", "on", "with"）的子集，Rel module 帶來的 gain 更大。
- **context object 個數 vs P@1**：場景中物件越多時，Dist module 相對 Rel-single 的優勢越大（聚合多個 context）。
- **attention 視覺化**：定性圖顯示語言 attention 確實會把 "next to" 一類詞投放到 Rel module、把 "man"/"cup" 等實體詞投放到 Subj。
- **失敗案例**：1) 關係詞模糊（"near"）時候選 context 太多會稀釋；2) 同類物件密集（如人群）時細粒度外觀仍不足。

---

## 5. 關鍵洞見與貢獻

### 5.1 貢獻清單

1. **顯式關係建模**：首次把 (subject, relation, object) 三元組整進 REC 的 deep model 中，而不是把關係詞當作 LSTM 內隱信號。
2. **多 context 聚合（Dist module）**：把 REC 從「只看一個最佳 context」拓展到「整個 scene 的證據聚合」。
3. **soft attention 把語言分派到 module**：本質上是 modular reasoning 的雛形，預告了後來 *Compositional Modular Networks*（CVPR 2017）。
4. **discriminative training over candidates**：把訓練 loss 從「expression generation likelihood」改成「region softmax」，本身就是強 baseline。
5. **在 Google-Ref/RefCOCOg 上把 SOTA 從 0.247 推到 0.493**：是 2016 年該 benchmark 的標竿結果。

### 5.2 對後續研究的啟發

- **modular networks**：Hu 等人隔年（CVPR 2017）直接把這套 Subj/Loc/Rel 升級成 Compositional Modular Networks，把 module 連接結構也用 language parser 決定。
- **graph-based grounding**：後續一系列 *Scene Graph + REC* 方法（MAttNet、DGA、CMRIN、LGRAN 等）都建立在「relationship matters」這個認識上。
- **MAttNet（CVPR 2018）**：直接把 Subj / Loc / Rel 升級成 *Modular Attention Network*，是這條線最有代表性的後繼，許多年內仍是 REC 的強 baseline。

---

## 6. 與其他參考論文的關係

使用者目錄下另有四篇相關論文，逐一比較：

### 6.1 Cops-Ref（CVPR 2020）

- **任務**：構造一個「需要 compositional reasoning」的 REC benchmark；強迫模型真的用到關係，而不是靠 dataset bias。
- **與本文關係**：Cops-Ref 是「**評測層**」的後繼——本文證明 Rel module 在 RefCOCOg 上有用，Cops-Ref 則挑出 RefCOCO 系列裡 *bias-prone* 的部分、構造對抗性 distractor，逼模型不能只靠 Subj。如果說本文是「給模型加上關係建模能力」，Cops-Ref 就是「設計題目專門測這個能力是否真的有用」。

### 6.2 GREC（Generalized REC, 2023+）

- **任務**：把 REC 從「假設恰好一個答案」放寬為「可能 zero / one / multiple」，即 **multi-target 與 no-target**。
- **與本文關係**：本文模型架構**預設 single-target**（最終 softmax 只挑一個 region），無法處理 *"no such object"* 或多個答案。GREC 是 REC 任務本身的**問題定義升級**，本文方法需要外接 abstention/multi-pick head 才能適配。對應到使用者主線「selective grounding / CRS conformal referring set」研究，這是直接對位的方向。

### 6.3 HieA2G（Hierarchical Attention for Grounding）

- **任務**：以 *hierarchical attention*（詞→片語→句子，或物件→組合→場景）做 REC。
- **與本文關係**：是本文 attention 思路的**深化**——本文只在 LSTM 上做一層 soft attention 分派到 module；HieA2G 把 attention 變成多層次的，並讓 visual 端也有層級。本文是「平面 modular + 單層 attention」，HieA2G 是「層級 attention + 層級 visual」。

### 6.4 Zero-Shot Referring Expression Comprehension via Visual-Language True/False Verification（arXiv 2509.09958）

- **任務**：用 VLM 直接做 *true/false verification*（給定 region + expression，預測是否匹配），不需要在 REC 資料集 fine-tune，做 zero-shot REC。
- **與本文關係**：方法路線完全不同——本文是 **task-specific supervised**、模組化、靠 VGG+LSTM；該文是 **task-agnostic VLM + verification framing**、靠大模型 zero-shot。但兩者面對的核心難題一致：「如何讓模型不只看 subject 本身、而真的理解 expression 的關係結構」。從本文（顯式模組）到該文（靠 VLM 內在能力 + verification scoring），可以看作 REC 過去 10 年方法論的兩端。對使用者的 selective grounding 研究而言，該文是 2026 年的新威脅模型（B→B 上限被壓縮），必須在 Ch5 中正面回應。

---

## 7. 個人評價

### 7.1 優點

1. **直觀且可解釋**：Subj / Loc / Rel / Dist 四個 module 對應人類描述物件時用到的四類資訊，ablation 結果乾淨地證明每個都重要。
2. **modular 設計開啟一條路線**：直接催生了 Compositional Modular Networks、MAttNet 等後續經典工作。
3. **訓練目標升級**：從 generation-based MMI 換成 discriminative softmax over candidates，本身就是一個極具影響力的設計選擇（後續幾乎所有 REC 方法都採用這個 framing）。
4. **Dist module 的 multi-context 聚合**：在 2016 年是少見的 design，預告了後來 GNN-based grounding 的方向。

### 7.2 限制

1. **依賴 GT box**：評估時用 ground-truth bounding box 當 candidate，沒回答「detector 雜訊下還能不能 work」。
2. **VGG + 單層 LSTM**：以 2026 年的眼光看極端過時；現在 CLIP/OWL-ViT/GroundingDINO 都已內建跨模態對齊。
3. **single-target 假設**：無法處理 no-target / multi-target，與 GREC 設定不相容。
4. **關係的語意空間有限**：spatial encoding 只能抓「左右、上下、大小、距離」，無法處理「holding」、"belonging to" 這類 semantic relation（要等 scene graph based 方法）。
5. **計算開銷**：Rel module 對所有 region pair 配對，是 O(N²)；大 scene 下不划算。
6. **未報告 calibration / abstention**：模型 softmax 出來的分數沒有可靠性分析（這正是使用者 selective grounding 主線要補的洞）。

### 7.3 後續方向（若今天重做）

1. 把 VGG 換成 CLIP image encoder、LSTM 換成 transformer text encoder。
2. Rel module 改用 GAT / cross-attention，避免 O(N²) 全配對。
3. 直接接 GREC 設定，加 abstention head 處理 no-target，加 set-prediction head 處理 multi-target。
4. 在 Cops-Ref 上補測，驗證 Rel module 是否真的扛得住 compositional 對抗樣本。
5. 增加 calibration（temperature scaling / conformal prediction）與 selective prediction 分析——這正是 CRS / 使用者論文主線的方向。
6. 加入 verification-style scoring（呼應 zero-shot true/false verification），讓 modular 框架也能在 zero-shot 條件下運作。

### 7.4 給沒讀過 REC 的人的一句話總結

> 這篇是 **2016 年讓 REC 從「只看物件外觀」進化到「也看物件之間關係」的關鍵 paper**——它提出把 subject、location、relationship、multi-context 拆成四個可加性的 scoring module，用 LSTM attention 把語言詞分派給不同 module，在 RefCOCOg 上把準確率從 25% 提升到近 50%，並直接催生了後來 Compositional Modular Networks 與 MAttNet 一整條 modular grounding 研究路線。

---

## 8. 對使用者主線（selective grounding / CRS）的可借鏡點

1. **modular factorization 的乾淨 ablation**：本文 Subj/Loc/Rel/Dist 逐步加法的 ablation，是 CRS 的 2×2 factorization ablation 可以參考的範式。
2. **Rel module 的 spatial encoding**：CRS 的 OWL-ViT gate 也許可以額外吃 box-pair geometry，作為 multi-target 時 referring set 內部一致性的訊號。
3. **多 context 聚合（Dist）**：對應 CRS 在 multi-target 情境的 set-level scoring，可以借這個 attention-aggregation 思路把候選集內部的關係分數聚合起來，作為集合可靠性指標。
4. **討論章節對位**：在 Ch2/Ch3 文獻回顧中，這篇是「explicit relational REC」的代表，應與 Cops-Ref（bias-control benchmark）、MAttNet（modular SOTA）、GREC（task generalization）、Zero-Shot True/False Verification（新威脅模型）並列討論，鋪陳 REC 任務 10 年演進史。

---

## 9. 參考連結

- arXiv：<https://arxiv.org/abs/1607.07702>
- CVF：<https://openaccess.thecvf.com/content_cvpr2016/html/Hu_Modeling_Relationships_in_CVPR_2016_paper.html>
- Code：<https://github.com/ronghanghu/text_obj_retrieval>
- Project page（含結果表）：<https://ronghanghu.com/text_obj_retrieval/>
- Semantic Scholar：<https://www.semanticscholar.org/paper/Modeling-Relationships-in-Referential-Expressions-Hu-Rohrbach/a396a6febdacb84340d139096a1c5b2e9a733a3f>
- 後繼 CVPR 2017：*Compositional Modular Networks* <https://arxiv.org/abs/1611.09978>
- 前傳 CVPR 2016：*Natural Language Object Retrieval*（SCRC）<https://arxiv.org/abs/1511.04164>
- 經典後繼 CVPR 2018：*MAttNet*（Modular Attention Network for Referring Expression Comprehension）
