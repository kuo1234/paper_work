# MAttNet：Modular Attention Network for Referring Expression Comprehension

> **詳細導讀** ｜ CVPR 2018
> 原文：Yu, Lin, Shen, Yang, Lu, Bansal, Berg. *MAttNet: Modular Attention Network for Referring Expression Comprehension.* arXiv:1801.08186
> 程式碼：https://github.com/lichengunc/MAttNet ｜ Demo：http://vision2.cs.unc.edu/refer/comprehension

---

## 目錄

1. [一句話總結](#一句話總結)
2. [背景：什麼是 Referring Expression Comprehension](#背景什麼是-referring-expression-comprehension)
3. [核心洞見：為什麼要「模組化」](#核心洞見為什麼要模組化)
4. [整體架構](#整體架構)
5. [語言注意力網路（軟性 parser）](#語言注意力網路軟性-parser)
6. [三個視覺模組](#三個視覺模組)
7. [匹配函數與損失](#匹配函數與損失)
8. [實驗結果](#實驗結果)
9. [消融分析：每個設計值多少分](#消融分析每個設計值多少分)
10. [影像分割延伸](#影像分割延伸)
11. [貢獻與侷限](#貢獻與侷限)
12. [名詞速查](#名詞速查)

---

## 一句話總結

> **MAttNet 把一句指代描述「軟性地」拆成三塊——主體外觀 / 絕對位置 / 與鄰居的關係——分別交給三個視覺模組各算一個匹配分數，再用學出來的權重動態合併。** 這種「分而治之」比把整句揉成單一向量去比對好很多（RefCOCO 系列大幅領先約 10%），而且天生具備可解釋性：可以看出每個字、每塊影像區域對哪個模組起了作用。

---

## 背景：什麼是 Referring Expression Comprehension

**任務定義**：給定一張影像與一句自然語言描述（referring expression，指代表達），模型要在影像中**定位出唯一對應的物件區域**（輸出 bounding box，或進一步輸出像素級 mask）。

- 輸入：影像 + 一句話，例如「the woman in red on the left（左邊穿紅衣的女人）」
- 輸出：圖中那個女人的框

它與相關任務的差別：
- **物件偵測（object detection）**：找出「所有」某類物件，不涉及語言上的「哪一個」。
- **影像描述（image captioning）**：產生描述，是「生成」而非「定位」。
- **指代表達生成（referring expression generation）**：反向任務——給定物件，產生能唯一指出它的句子。
- **指代表達理解（comprehension，本文）**：給定句子，定位物件。

評測通常採 **候選框排序**設定：影像中已有一組候選物件（GT 框，或偵測器產生的框），模型對每個候選算分，取最高分者，命中（IoU ≥ 0.5）即正確。

---

## 核心洞見：為什麼要「模組化」

過去主流做法（如 CNN-LSTM、Listener/Speaker 模型）把**整句話編碼成單一向量**，再跟影像特徵比對。問題在於：一句指代描述其實是**異質資訊的混合體**：

| 資訊類型 | 例子 | 該看影像的什麼 |
|---|---|---|
| **主體外觀** | "red cat", "man wearing glasses" | 物件框「內部」的顏色、類別、屬性 |
| **絕對位置** | "on the left", "top corner" | 框在整張圖的座標 |
| **物件關係** | "dog next to the table" | 框「外部」周圍的其他物件 |

把這三種揉成一團，會讓模型難以對齊——描述外觀的字和描述位置的字被混在同一個向量裡。

**MAttNet 的主張**：與其用一個大模型硬吃整句，不如**拆成三個各司其職的模組**，並讓模型自己學會「這句話裡哪些字該餵給哪個模組、每個模組該佔多少權重」。這呼應了 neural module networks 的精神，但 MAttNet 的模組分解是**固定三類、軟性加權**，比依賴外部語法樹去動態組裝模組更穩定。

---

## 整體架構

MAttNet 由兩大協作部分構成：

```
                          ┌─────────────────────────────────────┐
   表達式 r               │     語言注意力網路 (Language Attn)      │
  "woman in    ──詞嵌入──▶│  bi-LSTM → 三組詞注意力 → q^subj/loc/rel │
   red left"             │           → 模組權重 w_subj/loc/rel    │
                          └─────────────────────────────────────┘
                                    │ q^subj   │ q^loc    │ q^rel
                                    ▼          ▼          ▼
   候選物件 o_i           ┌──────────┐ ┌──────────┐ ┌──────────┐
  (影像區域)  ──CNN特徵──▶│ Subject  │ │ Location │ │ Relation │
                          │ 模組      │ │ 模組      │ │ 模組      │
                          └────┬─────┘ └────┬─────┘ └────┬─────┘
                          S(o|q^subj)  S(o|q^loc)  S(o|q^rel)
                                    │          │          │
                                    └──── 加權合併 (w) ─────┘
                                              ▼
                                       S(o_i | r)  總分
```

對候選物件 $o_i$ 與表達式 $r$，最終匹配分數為三模組分數的加權和：

$$
S(o_i \mid r) = w_{subj}\, S(o_i \mid q^{subj}) + w_{loc}\, S(o_i \mid q^{loc}) + w_{rel}\, S(o_i \mid q^{rel})
$$

其中 $q^{subj}, q^{loc}, q^{rel}$ 是語言網路產生的三段 phrase embedding，$w_{subj}, w_{loc}, w_{rel}$ 是學出來的模組權重。

**骨幹網路**：Faster R-CNN + ResNet-101（另有 VGG16 版作對照）。抽取兩個層級的特徵：
- **C3**：較低階線索——顏色、形狀。
- **C4**：較高階線索——物件類別語意。

---

## 語言注意力網路（軟性 parser）

這是 MAttNet 的「大腦」——它不靠外部語法分析器，而是**端到端學會怎麼拆句子**。

**1. 詞編碼**
每個詞 $u_t$ 先嵌入成 $e_t$，再過 **雙向 LSTM**，得到融合前後文的表徵：

$$
h_t = [\overrightarrow{h}_t, \overleftarrow{h}_t]
$$

**2. 詞層級注意力（每個模組一組）**
為三個模組各設一個可訓練向量 $f_m$（$m \in \{subj, loc, rel\}$），對每個詞算 softmax 注意力權重：

$$
a_{m,t} = \frac{\exp(f_m^\top h_t)}{\sum_{k=1}^{T} \exp(f_m^\top h_k)}, \qquad q^m = \sum_{t=1}^{T} a_{m,t}\, e_t
$$

直覺：subject 模組的 $f_{subj}$ 會學會對「red」「cat」這類外觀詞給高權重；location 模組的 $f_{loc}$ 會對「left」「top」給高權重。**這個拆分是學出來的，不是規則寫死的。**

**3. 模組權重**
把 bi-LSTM 的首、尾 hidden state 串接，過一層 FC + softmax，得到三個模組的貢獻權重：

$$
[w_{subj}, w_{loc}, w_{rel}] = \text{softmax}(W \cdot [h_0, h_T] + b)
$$

直覺："red cat" 幾乎只靠 subject（$w_{subj}$ 大）；"woman on the left" 則 subject + location 都重要。

> **關鍵優勢**：論文實驗顯示，用這個「學出來的軟性 parser」比接一個外部模板 parser **高約 5%**，因為外部 parser 會犯錯，而軟注意力能容錯並隨任務調整。

---

## 三個視覺模組

### Subject 模組（in-box attention，框內注意力）

負責主體外觀，做兩件事：

**(a) 屬性預測（attribute prediction）**
把 C3 + C4 串接後過 $1\times1$ 卷積、平均池化，預測物件屬性（顏色、材質等）。用**加權二元交叉熵**訓練，對稀有屬性加權以平衡長尾：

$$
L^{attr}_{subj} = \lambda_{attr} \sum_i \sum_j w_j^{attr}\,\big[ y_{ij}\log p_{ij} + (1-y_{ij})\log(1-p_{ij}) \big], \quad w_j^{attr} = 1/\sqrt{\text{freq}_{attr}}
$$

屬性標籤由一個模板 parser 從訓練句子自動抽取（弱監督）。

**(b) Phrase-guided 注意力池化（核心）**
把屬性特徵與 C4 融成 $14\times14$ 的空間特徵網格 $V \in \mathbb{R}^{d \times G}$（$G=196$），用語言端的 $q^{subj}$ 當 query 算出空間注意力，只池化框內**真正相關的部位**：

$$
H_a = \tanh(W_v V + W_q q^{subj}), \quad a^v = \text{softmax}(w_{h,a}^\top H_a), \quad \tilde{v}_i^{subj} = \sum_{i=1}^{G} a_i^v\, v_i
$$

直覺：描述是「穿紅衣的人」時，注意力會聚焦在人的「衣服」區域，而非整個框平均。

---

### Location 模組

**(a) 絕對位置**——用 5 維向量編碼框的位置與面積：

$$
l_i = \Big[ \tfrac{x_{tl}}{W}, \tfrac{y_{tl}}{H}, \tfrac{x_{br}}{W}, \tfrac{y_{br}}{H}, \tfrac{w \cdot h}{W \cdot H} \Big]
$$

**(b) 相對位置（dif）**——取周圍最多 **5 個同類別**物件，編碼它們相對候選的位移與面積比 $\delta l_i$。這對「左邊那隻貓」「中間的人」這類描述至關重要：

$$
\tilde{l}_i^{loc} = W_l\,[l_i; \delta l_i] + b_l
$$

> 消融顯示，加入相對位置（loc dif）帶來**最大的單項提升**（79.68 → 82.06）。

---

### Relationship 模組（out-of-box attention，框外注意力）

負責「與其他物件的關係」，看候選周圍最多 **5 個任意類別**物件（不限同類）：用它們的 C4 平均池化外觀特徵 $v_{ij}$ 加上位移編碼 $\delta m_{ij}$：

$$
\tilde{v}_{ij}^{rel} = W_r\,[v_{ij}; \delta m_{ij}] + b_r
$$

由於**不知道句子到底指的是哪一個鄰居**，採用**弱監督多實例學習（MIL）**——對所有鄰居取 max，由模型自動挑出最匹配的那個：

$$
S(o_i \mid q^{rel}) = \max_{j \neq i} F(\tilde{v}_{ij}^{rel}, q^{rel})
$$

直覺：描述是「桌子旁邊的狗」時，模型會在候選狗周圍的物件裡，找到「桌子」這個鄰居並給高分。

---

## 匹配函數與損失

**匹配函數 $F$**（三模組共用同一形式）：
兩個 MLP 分別把視覺特徵與語言 phrase 投影到共同語意空間，各自 L2 normalize 後取**內積**作為匹配分數。

$$
F(\tilde{v}, q) = \langle \text{L2}(\text{MLP}_v(\tilde{v})),\ \text{L2}(\text{MLP}_q(q)) \rangle
$$

**Ranking loss**：每個正樣本配兩個負樣本——同一物件配錯句子 $r_j$、同一句子配錯物件 $o_k$（hard negatives，皆取自同一張圖）：

$$
L_{rank} = \sum_i \Big[ \lambda_1 \max(0, \Delta + S(o_i \mid r_j) - S(o_i \mid r_i)) + \lambda_2 \max(0, \Delta + S(o_k \mid r_i) - S(o_i \mid r_i)) \Big]
$$

**總損失**：

$$
L = L^{attr}_{subj} + L_{rank}
$$

> **重點**：整套系統——詞注意力、模組權重、空間注意力、鄰居 MIL——全部**僅靠「物件–表達式配對」做端到端訓練**。沒有人工標註哪個字屬於哪個模組、注意力該看哪裡，這些都是自動湧現的。

**訓練設定**：Adam，初始學習率 0.0004，batch 15 張圖；warm-up 8000 步後每 8000 步學習率減半；詞嵌入與 LSTM hidden 皆 512 維；dropout 0.5（語言層）/ 0.2（匹配函數）；約 30000 步收斂（Titan-X Pascal 約半天）；推論約 0.33 秒/次。

---

## 實驗結果

**三個資料集**（皆建於 MS COCO 影像）：

| 資料集 | 收集方式 | 平均句長 | 同類物件數 | 特點 |
|---|---|---|---|---|
| **RefCOCO** | 互動遊戲 | ~3.5 詞 | 3.9 | 句子短，常用位置詞 |
| **RefCOCO+** | 互動遊戲 | ~3.5 詞 | 3.9 | **禁用絕對位置詞**，逼模型靠外觀 |
| **RefCOCOg** | 非互動 | ~8.4 詞 | 1.63 | 句子長，常描述關係 |

**testA vs testB**（RefCOCO/RefCOCO+ 的 test 依物件類型切分）：
- **testA**：含多個「人」的影像。
- **testB**：含多個「其他類別」物件的影像。

### 主結果（使用 GT 候選框，Table 1）

完整模型（res101-frcn，subj+attr+attn+loc+dif+rel）：

| 資料集 | val | testA | testB |
|---|---|---|---|
| **RefCOCO** | **85.65** | **85.26** | **84.57** |
| **RefCOCO+** | **71.01** | **75.13** | **66.17** |
| **RefCOCOg** | **78.10** | **78.12（test）** | — |

對照當時最強前作 **Speaker+Listener+Reinforcer**（VGG16）在 RefCOCO val 約 79.56——MAttNet **大幅領先約 10%**。連 MAttNet 的 VGG16 版本都贏過所有前作。

### 全自動偵測（使用偵測框，非 GT，Table 3）

換成由偵測器（res101-mrcn，Mask R-CNN）產生候選框的真實場景：

| 資料集 | val | testA | testB |
|---|---|---|---|
| RefCOCO | 76.65 | 81.14 | 69.99 |

模組改善的趨勢與用 GT 框時一致；Mask R-CNN 偵測品質優於 Faster R-CNN。

---

## 消融分析：每個設計值多少分

在 RefCOCO val 上逐步堆疊各模組（res101-frcn）：

| 設定 | 準確率 | 增量 |
|---|---|---|
| Baseline matching (subj+loc) | 79.14 | — |
| MAttNet (subj+loc) | 79.68 | +0.54 |
| + loc(dif) 相對位置 | 82.06 | **+2.38** ← 最大單項 |
| + rel 關係模組 | 82.54 | +0.48 |
| + attr 屬性預測 | 83.54 | +1.00 |
| + attn 注意力池化（完整） | **85.65** | **+2.11** |

**關鍵發現**：
1. **模組化本身就贏 baseline**——同樣 subj+loc，模組化版本更好。
2. **相對位置（loc dif）與框內注意力（attn）貢獻最大**——這兩項正是處理「哪一個同類物件」的核心。
3. **注意力池化對「人」類別（testA）幫助尤其明顯**——人的外觀細節多，聚焦相關部位差別大。
4. **學出來的軟 parser 勝外部模板 parser 約 5%**——後者的解析錯誤會傳遞下去。

---

## 影像分割延伸

MAttNet 不只輸出框，也能輸出**像素級 mask**：把骨幹換成 **Mask R-CNN**，將理解模組選出的框餵給 mask 分支。關鍵設計是**把「框定位（理解）」與「分割」解耦**，而非像 FCN 式方法一步到位。

分割結果（IoU，Table 4）：

| 資料集/split | MAttNet IoU | 前作 (D+RMI+DCRF) |
|---|---|---|
| RefCOCO val | **56.51** | 45.18 |
| RefCOCO testA | **62.37** | 45.69 |
| RefCOCO+ val | **46.67** | 29.86 |
| RefCOCO+ testA | **52.39** | 30.48 |

論文形容像素級精度「**幾乎翻倍**」。

---

## 貢獻與侷限

### 主要貢獻
1. **模組化分解**：把指代表達理解明確拆成 subject / location / relationship 三模組，各司其職。
2. **雙重注意力**：語言端（詞注意力 + 模組權重）+ 視覺端（框內空間注意力 + 框外鄰居 MIL）。
3. **端到端弱監督**：只用物件–表達式配對，注意力與模組分工自動湧現，並提供可解釋性。
4. **SOTA + 通用性**：在 bbox 與 pixel 兩個層級的三個資料集上大幅刷新當時紀錄。

### 侷限（也是後續研究的縫）
1. **預設目標存在且唯一**：強制輸出最高分候選框，**沒有「無目標就棄答」「多目標」的概念**——這正是後來 gRefCOCO / GREC（no-target、multi-target）要補的洞。
2. **依賴候選框品質**：是 two-stage 排序框架，效能受限於偵測器召回；漏檢的物件無法被選中。
3. **固定三類模組**：分解粒度固定，對更複雜的多重關係描述（巢狀、多跳關係）表達力有限。
4. **誤差來源**：論文歸納主要錯誤來自訓練資料稀疏、表達式本身歧義、以及偵測失敗。

---

## 名詞速查

| 名詞 | 說明 |
|---|---|
| **Referring Expression** | 指代表達——能唯一指出某物件的自然語言描述 |
| **Comprehension** | 理解——給句子定位物件（對應 generation 反向任務） |
| **in-box attention** | 框內注意力——subject 模組對候選框內部空間做的注意力池化 |
| **out-of-box attention** | 框外注意力——relationship 模組對候選框外鄰居做的 MIL |
| **C3 / C4** | ResNet 中低階 / 高階特徵層（顏色形狀 / 類別語意）|
| **MIL** | 多實例學習——不知正例是誰時，對一組取 max 弱監督 |
| **testA / testB** | RefCOCO 測試切分：多人 / 多其他類物件 |
| **loc(dif)** | 相對位置差——候選與周圍同類物件的位移編碼 |
| **ranking loss** | 排序損失——讓正配對分數高於錯句/錯物件配對至少一個 margin |

---

*導讀整理完畢。本文件對應的分享簡報見 `MAttNet_分享簡報.pptx`。*
