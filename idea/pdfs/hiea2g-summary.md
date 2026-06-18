---
title: "HieA2G 詳細導讀（中文）"
author: "整理：Kuo"
date: "2026-06-18"
---

# HieA2G 詳細導讀：階層對齊增強的自適應 Grounding 網路

> 為 GREC（Generalized Referring Expression Comprehension）而生

---

## 0. 書目資訊

- **論文標題**：Hierarchical Alignment-enhanced Adaptive Grounding Network for Generalized Referring Expression Comprehension
- **縮寫**：HieA2G（A2G = **A**lignment-enhanced **A**daptive **G**rounding，不是 Attribute-to-Group）
- **作者**：Yaxian Wang¹², Henghui Ding³（通訊）, Shuting He⁴, Xudong Jiang⁵, Bifan Wei⁶⁷, Jun Liu¹²
  - ¹² 西安交通大學 計算機學院 / 智能網絡與網絡安全教育部重點實驗室
  - ³ 復旦大學 大數據學院
  - ⁴ 上海財經大學
  - ⁵ 南洋理工大學
  - ⁶⁷ 西安交通大學 繼續教育學院 / 陝西省大數據知識工程重點實驗室
- **發表**：AAAI 2025
- **arXiv**：2501.01416v1（2025-01-02）
- **連結**：<https://arxiv.org/abs/2501.01416>

> 註：使用者描述為「Hierarchical Attribute-to-Group」，但實際論文 A2G 是「Alignment-enhanced Adaptive Grounding」。本摘要以實際論文為準。

---

## 1. 任務背景：從 REC 到 GREC

### 1.1 三種視覺接地任務一次看懂

| 任務 | 目標物件數 | 是否容許 no-target | 是否容許 multi-target | 代表資料集 |
|---|---|---|---|---|
| 經典 REC | 恰好 1 | 否 | 否 | RefCOCO/+/g |
| Phrase Grounding | 多個（依名詞片語） | 否 | 是（但只看片語） | Flickr30K Entities |
| **GREC** | **0 ~ 多個** | **是** | **是** | **gRefCOCO** |

- **REC（Referring Expression Comprehension）**：「The man in red shirt」→ 強制輸出唯一 box。
- **Phrase Grounding**：把句子裡每個名詞片語都接到 image region 上，但不一定理解整句語意。
- **GREC（Generalized REC）**：把 REC 推廣到「**任意數量** 0、1、多個目標」。例如：
  - no-target：「the player in blue top and black pants」（圖中根本沒這個人）
  - multi-target：「all people」、「three players」
  - 仍允許單目標 REC

### 1.2 為什麼 GREC 比 REC 難很多

論文點出兩個核心挑戰：

1. **跨模態對齊不夠**：現有 REC/RES 方法（MDETR、CLIP、GroupViT 等）通常只做「單一層次」對齊（word-object 或 text-image 二擇一），對於需要拒答（no-target）或需要計數（multi-target）的句子，**細節捕捉不足**，導致 false positive 或 false negative。
2. **輸出個數不固定**：REC 模型強制 top-1 輸出，GREC 需要動態決定「該輸出 0 個還是 5 個 box」。
   - 純粹用「閾值濾候選 box」（如 GREC baseline）會被一個全域閾值卡死，無法適配各個樣本。
   - 把句子拆成多個子句多輪查詢，又無法處理「all people」這種隱式多目標。

→ 必須**同時解決對齊與計數**這兩件事。

---

## 2. 方法概觀：HieA2G 整體架構

整個流程可拆三段：

```
[影像 I]──Visual Encoder──┐
                          ├──Transformer Encoder──┐
[文字 T]──Text Encoder────┘                       │
                                                  ▼
                                  Transformer Decoder ←── N 個 learnable object queries
                                                  │
                                                  ▼
                                  N 個 object embeddings  Oe
                                                  │
                ┌──────────────────────┬──────────┴──────────────┐
                ▼                      ▼                          ▼
        HMSA：3-level alignment   Box / Mask Head             AGC：Adaptive
        (Word-Obj, Phrase-Obj,    （輸出 N 個 box / mask）    Grounding Counter
         Text-Img)                 候選 proposals             決定該選幾個
```

### 元件清單

- **Visual Encoder**：ResNet-101 或 Swin-B。
- **Text Encoder**：RoBERTa-base。
- **Transformer Encoder + Decoder**：類 DETR 結構，N 個 learnable object queries 對應 N 個候選 box / mask。
- **HMSA（Hierarchical Multi-modal Semantic Alignment）**：本論文第一大貢獻，三層對齊。
- **AGC（Adaptive Grounding Counter）**：本論文第二大貢獻，動態決定輸出數。
- **Box Head / Mask Head**：標準偵測與分割頭。

---

## 3. HMSA：階層式多模態語意對齊（核心貢獻一）

HMSA 同時做三個層次的對齊，每層解決不同粒度的語意鴻溝。

### 3.1 Word-Object Alignment（W2O）：用「Masked Text Recovery」當代理任務

**動機**：缺少「word ↔ region」的細粒度標註，無法直接監督。作者借鏡 BERT 的 masked language modeling 概念。

**做法**：
1. 從句子裡抽出「實體名詞」（entity noun），隨機 mask 成 `[MASK]`。
2. 把 masked 句子送 text encoder 得到 $\tilde{T}_w$。
3. 拿 N 個 object embeddings $O_e$ 與 $\tilde{T}_w$ 透過一個 Transformer layer 重建出 $\hat{T}_w$。
4. 損失：

$$
L_{w2o} = \lambda \cdot (1 - \cos(T_w, \hat{T}_w))
$$

- $T_w$ 是原始未 mask 的完整文字特徵；
- $\lambda = 0$ 當 **no-target sample**（因為視覺與文字毫無對應，硬重建只會干擾訓練）；其他情況 $\lambda = 1$。

**效果**：強迫 object embeddings 學會編碼「能還原 masked 實體詞」的視覺語意 → 達到隱式的 word-object 對齊。

### 3.2 Phrase-Object Alignment（P2O）：用 Flickr30K Entities 顯式監督

**動機**：屬性級資訊（顏色、形狀、姿態等）對於區分同類物件至關重要。Flickr30K Entities 提供「片語 → 物件框」的顯式標註，可拿來訓練。

**做法**：
1. 從句子抽 M 個 noun phrases，平均池化得 $T_p \in \mathbb{R}^{M\times C_p}$。
2. 線性投影到同空間：$T_p' = W_1 T_p$，$O_e' = W_2 O_e$。
3. 計算 phrase-query 相似度矩陣：
   $$ S = \sigma(T_p' \cdot O_e'^\top) \in \mathbb{R}^{M\times N} $$
4. 用 bipartite matching（Mask2Former 式）配 ground-truth box，得到二值對應 $Y \in \{0,1\}^{M\times N}$。
5. 損失（BCE）：

$$
L_{p2o} = -\sum_{i,j} Y_{i,j}\log S_{i,j} + (1-Y_{i,j})\log(1-S_{i,j})
$$

**效果**：object embeddings 被推向「能匹配描述片語的屬性表徵」，提升細粒度區辨力。

### 3.3 Text-Image Alignment（T2I）：全域對比

**動機**：除了 word/phrase 級，整句與整圖的整體語意也要對齊。

**做法**：定義 image-text 全域匹配分數
$$
S_T(I,T) = \frac{1}{N}\sum_{j=1}^N \sum_{k=1}^K a_{j,k}\,\langle \hat{o}_j, \hat{w}_k\rangle, \quad
a_{j,k} = \frac{\exp\langle \hat{o}_j, \hat{w}_k\rangle}{\sum_l \exp\langle \hat{o}_j, \hat{w}_l\rangle}
$$
（沿文字維 normalize；類似地沿影像維可得 $S_I(I,T)$）。

對 batch 內 image-text 對做 InfoNCE 對比，分四個方向計算（text-anchor / image-anchor × T 維 / I 維 normalize），總損失：

$$
L_{t2i} = L^{t2i}_{T\to T}(I) + L^{t2i}_{I\to T}(I) + L^{t2i}_{T\to I}(T) + L^{t2i}_{I\to I}(T)
$$

**效果**：全域語意一致性，幫助 no-target 判斷（整圖根本不匹配）。

### 3.4 HMSA 總損失

$$
L_{align} = L_{w2o} + L_{p2o} + L_{t2i}
$$

三層協同 → object queries 從多個粒度 refine，提升 box regression 與 mask segmentation。

---

## 4. AGC：自適應 Grounding 計數器（核心貢獻二）

### 4.1 為什麼需要 AGC

- **閾值法**：He et al.（GREC baseline）對所有候選 box 的 class score 設一個全域閾值，過高漏抓、過低多抓，無法樣本自適應。
- **拆句多輪查詢**：對「all kids」這種隱式多目標無能為力。
- **觀察**：gRefCOCO 的目標數呈長尾分布，0~3 個目標佔絕大多數，>3 屬少數。

### 4.2 設計：把「目標數」當分類任務

1. 取 word feature $T_w$ 與 object embeddings $O_e$，做 average pooling 再串接：
   $$ M_g = [\mathrm{AP}(T_w);\ \mathrm{AP}(O_e)] $$
2. 用 2-layer MLP 預測 5 類標籤：
   $$ y_c = \mathrm{MLP}(M_g) \in \{0, 1, 2, 3, 3^+\} $$
3. 輸出策略：
   - 若 $y_c \in \{0,1,2,3\}$：直接取 top-$y_c$ 分數的候選 box。
   - 若 $y_c = 3^+$：退回閾值法（因為樣本太少，分類不準）。

### 4.3 對比學習強化計數能力

只靠 batch 內樣本對比，負樣本不夠多 → 引入 MoCo 風格 **memory bank** $M$，存歷史 multi-modal features。

採用 supervised contrastive loss（Khosla 2020）：
$$
L_{con} = -\frac{1}{|P(i)|}\sum_{p\in P(i)} \log\frac{\exp(M_g^i\cdot M_g^p/\tau)}{\sum_{a\in A(i)} \exp(M_g^i\cdot M_g^a/\tau)}
$$
- $P(i)$：與 anchor 同計數的正樣本集合；$A(i)$：所有正負樣本。
- 把「同計數」拉近、「不同計數」推遠 → 顯式塑造可計數的多模態特徵空間。

**AGC 總損失**：
$$
L_{agc} = L_{cls} + L_{con}
$$

### 4.4 直觀理解

- HMSA 解決「**看懂語意**」；
- AGC 解決「**算清楚幾個**」；
- 兩者解耦但互補：HMSA 提供區辨性表徵 → AGC 在這個空間裡分類個數比較容易。

---

## 5. 訓練流程

### 5.1 兩階段

1. **Pretrain（聯合）**：在 RefCOCO/+/g + Flickr30K Entities + gRefCOCO 訓練集合併上訓練：
   $$ L_{pretrain} = L_{align} + L_{det} $$
   $$ L_{det} = \lambda_{bbox} L_{bbox} + \lambda_{giou} L_{giou} + \lambda_{class} L_{class} $$
2. **Finetune（下游）**：對各下游任務分別微調：
   - REC / Phrase Grounding：用 $L_{det}$
   - RES：加上 $L_{seg} = \lambda_{mask} L_{mask} + \lambda_{dice} L_{dice}$
   - GREC / GRES：再加上 $L_{agc}$

### 5.2 細節

- Visual backbone：ResNet-101 / Swin-B
- Text encoder：RoBERTa-base
- Box loss：L1 + GIoU + cross-entropy class
- Mask loss：Focal loss + Dice loss

---

## 6. 與既有方法的差異對比

| 維度 | 傳統 two-stage REC | 傳統 one-stage REC | Phrase Grounding | ReLA (GRES) | **HieA2G** |
|---|---|---|---|---|---|
| 輸出數 | 1 | 1 | 多（依片語） | 多（二元判存） | **0~多（顯式計數）** |
| 對齊層次 | word/region | text-image | phrase-region | region-image | **word + phrase + text 三層** |
| no-target 處理 | 無 | 無 | 無 | 二元分類 | **AGC=0 類 + T2I 全域對齊** |
| multi-target 處理 | 不支援 | 不支援 | 限名詞片語 | 全部 candidate 過閾值 | **AGC 動態選 top-k 或閾值** |
| 計數能力 | 無 | 無 | 無 | 隱式 | **顯式分類 + supervised contrastive** |

---

## 7. 實驗

### 7.1 資料集

- **gRefCOCO**（主場，GREC / GRES）
- **RefCOCO / RefCOCO+ / RefCOCOg**（REC / RES）
- **Flickr30K Entities**（Phrase Grounding）

### 7.2 評估指標

- **GREC**：Pr@(F1=1, IoU≥0.5)、N-acc.（no-target 判定正確率）
- **REC**：Pr@(IoU≥0.5)
- **RES**：cIoU、gIoU
- **GRES**：cIoU、gIoU、N-acc.、T-acc.
- **Phrase Grounding**：Recall@1/5/10（ANY-BOX protocol）

### 7.3 主要結果

#### Table 1：gRefCOCO（GREC 主戰場）

| Method | val Pr / N-acc | testA Pr / N-acc | testB Pr / N-acc |
|---|---|---|---|
| MCN†   | 28.0 / 30.6 | 32.3 / 32.0 | 26.8 / 30.3 |
| VLT†   | 36.6 / 35.2 | 40.2 / 34.1 | 30.2 / 32.5 |
| MDETR† | 42.7 / 36.3 | 50.0 / 34.5 | 36.5 / 31.0 |
| UNITEXT† | 58.2 / 50.6 | 46.4 / 49.3 | 42.9 / 48.2 |
| Ferret‡ | 54.8 / 48.9 | 49.5 / 45.2 | 43.5 / 43.8 |
| **HieA2G (R101)** | **67.8 / 60.3** | **66.0 / 60.1** | **56.5 / 56.0** |

> 對比 MLLM-based Ferret（>7B 參數），HieA2G 用 ResNet-101 backbone 平均 **+14.2% Pr@(F1=1)**、**+9.4% N-acc**（vs UNITEXT）。

#### Table 2：RefCOCO/+/g（REC）

HieA2G-R101 在所有 split 都超過 MDETR、SeqTR、TransVG++ 等專用 REC 方法，甚至超過 7B 級 MLLM（GSVA-7B、LISA-7B）。

#### Table 3：Flickr30K Entities（Phrase Grounding）

R@1 / R@5 / R@10 全面超 MDETR、Shrika-7B、Ferret-7B。

#### Table 4：RefCOCO/+/g（RES）

HieA2G-Swin-B 與 ReLA（同 backbone）相比有穩定提升，與 GSVA-7B、LISA-7B 競爭。

#### Table 5：gRefCOCO（GRES）

Swin-B backbone 下：
- val：cIoU 68.4 / gIoU 62.8 / N-acc 70.4
- 超過 ReLA、LISA-7B，cIoU 與 gIoU 也略勝 GSVA-7B。

### 7.4 Ablation（Table 6）

| # | W2O | P2O | T2I | Classifier | $L_{con}$ | Pr | N-acc |
|---|---|---|---|---|---|---|---|
| 1 | ✗ | ✗ | ✗ | ✓ | ✓ | 65.2 | 54.9 |
| 2 | ✗ | ✓ | ✓ | ✓ | ✓ | 67.5 | 58.1 |
| 3 | ✓ | ✗ | ✓ | ✓ | ✓ | 67.1 | 57.3 |
| 4 | ✓ | ✓ | ✗ | ✓ | ✓ | 67.0 | 56.4 |
| 5 | ✓ | ✓ | ✓ | ✗ | ✗ | 53.9 | 48.0 |
| 6 | ✓ | ✓ | ✓ | ✓ | ✗ | 66.5 | 57.3 |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | **67.8** | **60.3** |

**讀法**：
- 拿掉整個 HMSA（#1 vs #7）：Pr -2.6, N-acc -5.4
- 拿掉任一單層對齊（#2/#3/#4 vs #7）：都退步，證明三層互補
- 拿掉整個 AGC 改回閾值法（#5 vs #7）：Pr -13.9, N-acc -12.3 → **AGC 是巨大關鍵**
- 加 classifier 不加 contrastive（#6 vs #7）：Pr -1.3, N-acc -3.0 → $L_{con}$ 提升計數穩定性

### 7.5 定性分析

**成功案例**：
- 多目標：「two airplanes」精準算對 2 個；「the 2nd bus」能辨識序數。
- no-target：「the board in man hand」（圖中沒這東西）能正確拒答。

**失敗案例**：
- 視覺線索模糊：難以分辨所有目標、誤判 no-target。
- 關鍵物件被遮擋：漏抓或誤抓。

---

## 8. 關鍵洞見與貢獻總結

1. **「對齊不該是單層」**：作者明確主張 REC/GREC 過去靠單一層次對齊（word-object 或 text-image），對 GREC 這種需要同時做精細區辨 + 全域拒答的任務不夠用，**三層次階層對齊**才是正解。
2. **「計數能力應顯式建模」**：把目標數當分類問題，加 supervised contrastive 顯式塑造特徵空間，比閾值法穩健太多（ablation 顯示這是最大貢獻來源）。
3. **「Masked Text Recovery 是巧妙代理任務」**：在缺乏 word-region 標註下，用文字重建作為自監督，幾乎不增加成本就獲得 word-object 對齊。
4. **「無視 no-target 樣本的不一致性」**：在 $L_{w2o}$ 設 $\lambda=0$ 處理 no-target，避免硬重建反而干擾收斂——細節上很體貼。
5. **泛化性強**：同一架構橫掃 5 個任務（GREC、REC、Phrase Grounding、RES、GRES），不是只在 GREC 過擬合。

---

## 9. 與相關論文的關係

### 9.1 vs. COPS-Ref（Compositional Phrase Grounding）

- **COPS-Ref**（CVPR 2020 系列）強調 **composition**：把表達式拆成可組合的「主體 + 屬性 + 關係」三元組，再做 modular reasoning。
- **HieA2G** 不顯式做組合推理，而是用 **三層對齊** 把細到粗的語意統統壓進 object embedding。
- 兩者對「屬性區辨」都很重視，但路徑不同：COPS-Ref 走 modular network；HieA2G 走 alignment+contrastive。
- HieA2G 的 **P2O alignment** 其實隱式達到了 COPS-Ref 想要的「屬性匹配」效果。

### 9.2 vs. GREC（He et al. 2023, baseline benchmark）

- **GREC** paper 本身是定義任務、提出 baseline（用閾值濾候選 box）；它把 MDETR/UNITER 等改成多輸出版本。
- **HieA2G** 是首批在 gRefCOCO 上做 GREC 的**專用方法**之一，把 baseline 的閾值法升級成 **AGC 分類 + 對比學習**。
- ablation Table 6 第 5 row 直接量化「換掉閾值法的代價」：-13.9 Pr / -12.3 N-acc。

### 9.3 vs. Modeling Relationships in Referential Expressions（Hu et al. 2017, CMN）

- **CMN（Compositional Modular Networks）**：早期 REC 經典，把表達式拆成 (subject, relationship, object) 三模組，分別在影像上做 attention。
- 與 HieA2G 對比：
  - CMN 顯式建模 **relationship**；HieA2G 沒有顯式關係模組，但 phrase-object alignment 隱式抓到「修飾關係」。
  - CMN 是 two-stage 架構（先有 proposal 再選），HieA2G 是 DETR-like one-stage（learnable queries）。
  - HieA2G 在多目標、no-target 上完勝 CMN（CMN 沒這設計）。

### 9.4 對 selective grounding 研究的啟示（個人觀點）

對於你正在做的 **post-hoc reliability / calibration** 研究：

- HieA2G 的 **AGC 分類器**等於把「該輸出幾個」做成一個 **可學習的、可校準的** 模組，比 ReLA 的二元判存更接近「概率化拒答」。
- 它與你的 CRS（Conformal Referring Set）思路有對話空間：
  - HieA2G 用「分類 + contrastive」軟性決定 cardinality；
  - CRS 用「conformal LTT」硬性保證 risk。
  - 兩條路其實互補：AGC 可以當 score function，CRS 在它的輸出上加 conformal 包裝以獲得分布無關的保證。
- 對你的 N-acc / T-acc 量測：HieA2G 在 N-acc 推到 60 左右，但 T-acc 提升幅度沒講得很明確（gRefCOCO 仍有 multi-target T-acc 天花板問題，這跟你 M4 評估上限觀察一致）。

---

## 10. 個人評價

### 10.1 優點

- **方法清晰、貢獻可獨立 ablation**：HMSA 三層 + AGC 兩元件每塊都有對應 ablation row。
- **不用 7B+ MLLM**：用 ResNet-101 + RoBERTa-base 就贏過 Ferret-7B、GSVA-7B 這類大模型，**參數效率高**。
- **泛化性強**：5 個任務通吃，不像許多 GREC 方法只在 gRefCOCO 上炫技。
- **AGC 是被低估的真正主角**：ablation 顯示其貢獻 > HMSA。
- **細節品質高**：no-target 樣本特別處理（$\lambda=0$）、memory bank 增加負樣本、$3^+$ 退回閾值——都顯示作者了解任務細節。

### 10.2 限制與缺點

- **計數類別硬切 5 類（0/1/2/3/3+）**：對於 >3 的多目標仍依賴閾值，沒有徹底解決長尾。實務上 >3 的訓練樣本少，分類退化是必然，但這也是潛在改進點。
- **缺乏不確定性量化 / 校準分析**：論文只報 Pr@(F1=1) 與 N-acc，沒有提供 reliability diagram、ECE、coverage 等校準指標。對你的研究而言這是一個明顯的「沒做」的縫隙。
- **對 occlusion / ambiguous case 沒設計專屬機制**：失敗案例自己揭露了這兩種，但方法層沒對應對策。
- **HMSA 的 W2O 太依賴 entity noun 抽取品質**：用 spaCy/NLTK 抽 noun 會把抽象詞、複合名詞處理不好，論文沒分析這層失敗。
- **Phrase-Object alignment 強依賴 Flickr30K Entities**：對沒有此類標註的資料集（如純 RefCOCOg）這一支訓不到，泛化到其他 domain 時 P2O 等於關掉。
- **沒做 efficiency 比較**：N=N queries、memory bank、三層對齊都增加訓練開銷，論文沒給 FLOPs / latency。
- **AGC 的 $3^+$ 邊界值靠工程拍腦袋**：缺少 hyperparameter sensitivity 分析。

### 10.3 後續可探方向

1. **連續性計數**：把 AGC 從分類改成回歸或 Poisson regression，配 quantile head，更平滑且能延伸到 >3。
2. **conformal wrapping**：對 AGC + box selection 整體做 split conformal 或 LTT，提供 PAC 級的覆蓋保證（這就直接走向你的 CRS 路線）。
3. **更強的 no-target 機制**：用 OOD detection（如 score entropy、energy-based）強化 N-acc，不只靠 AGC 0 類。
4. **多語言 / open-vocabulary 擴展**：把 text encoder 換 CLIP-text、把 visual encoder 換 OWL-ViT，看是否仍 work。
5. **與 LLM agent 結合**：把 HieA2G 包成工具，讓 LLM 在 visual reasoning 鏈條中呼叫，處理複雜 instruction following。

---

## 11. 一頁速記（給趕時間的人）

- **誰**：Yaxian Wang 等，AAAI 2025，arXiv 2501.01416。
- **要解決什麼**：GREC——文字表達式對應 **0、1、多個**目標的視覺接地。
- **怎麼做**：
  1. **HMSA**：三層對齊（word-object 用 masked text recovery；phrase-object 用 BCE 監督；text-image 用對比學習）。
  2. **AGC**：把「該輸出幾個」當 5 類分類，加 memory-bank supervised contrastive 拉開不同計數的特徵。
- **成果**：gRefCOCO Pr@(F1=1) 上 testA/testB 平均比 SOTA 提 14%，5 個任務全 SOTA 或競爭力，**用 R101 backbone 贏 7B MLLM**。
- **真正的英雄**：AGC（ablation 顯示移除損失最大）。
- **沒做的事**：校準 / 不確定性 / efficiency / >3 長尾。
- **與你的研究關係**：AGC 是一個天然可被 conformal calibration 包裝的 score function；可以作為 CRS 的另一條 base 線實驗。

---
