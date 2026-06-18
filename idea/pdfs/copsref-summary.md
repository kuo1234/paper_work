# Cops-Ref 論文詳細導讀（繁體中文版）

> 寫給完全沒讀過的人，也希望能一篇看懂。本文整合 PDF 全文細節 + 後續 follow-up（FineCops-Ref 2024）+ 與你 vault 內其他 REC 論文（GREC、HieA2G、Modeling Relationships、True/False Verification）的關聯。

---

## 0. 一句話總結

**Cops-Ref 是一個 2020 年 CVPR 提出的「組合性指涉表達理解（Compositional Referring Expression Comprehension, REC）」資料集與任務**。它的核心觀察是：以前的 RefCOCO/RefCOCO+/RefCOCOg 用「最短的詞 + 一兩個物體」就能解出來，根本量不到模型的 reasoning 能力。Cops-Ref 用兩招逼模型必須真的讀懂整句話：

1. **表達式引擎（expression engine）**：六種邏輯型態（chain, and, or, order, same, not）任意組合，產生「華麗 (flowery)、長、組合性高」的 query，例如 `The cat on the left that is sleeping and resting on the white towel.`
2. **干擾圖像（distracting images）**：測試時除目標圖外，加入大量視覺上與 target 相似的其他圖像（含相同物件類別、相同類別+屬性、相同物件但關係不同等四種干擾），逼模型不能用 shortcut。

論文同時提出一個 baseline 改良：**MattNet-Mine**，用 modular hard mining 從 sub/loc/rel 三個模組各自找 hard negative。

---

## 1. 論文基本資訊

| 項目 | 內容 |
|---|---|
| **標題** | Cops-Ref: A new Dataset and Task on Compositional Referring Expression Comprehension |
| **作者** | Zhenfang Chen¹, Peng Wang², Lin Ma³, Kwan-Yee K. Wong¹, Qi Wu⁴ |
| **單位** | ¹HKU、²University of Wollongong、³Tencent AI Lab、⁴University of Adelaide |
| **會議/年份** | CVPR 2020（arXiv:2003.00403） |
| **CVF 連結** | https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Cops-Ref_..._paper.html |
| **官方程式碼** | https://github.com/zfchenUnique/Cops-Ref |
| **延伸工作** | FineCops-Ref (EMNLP 2024，arXiv:2409.14750)；TPAMI 2025 擴展版 |

---

## 2. 問題定義與動機

### 2.1 REC 任務本身

**Referring Expression Comprehension (REC)** = 給一張影像 + 一句自然語言描述，模型要在影像中定位出該描述所指的「特定物件」（output 是 bbox 或 region index）。是 visual grounding 的一支。

### 2.2 既有資料集的兩大缺陷

作者抨擊 RefCOCO / RefCOCO+ / RefCOCOg 等主流資料集：

**缺陷 A：表達式太短、太淺**
- 平均長度只有 3.5（RefCOCO）、8.5（RefCOCOg）詞。
- 通常只描述「類別 + 一個屬性 + 一個簡單空間關係」，例如 `the girl with glasses`、`the man sitting next to a table`。
- 模型只要做「淺層 cross-domain alignment」就能答對，無法評估深度推理。

**缺陷 B：圖像中干擾太少 + dataset bias 嚴重**
- 每張圖通常只有 2–3 個同類物件，distractor 不足。
- Cirik et al. (NAACL 2018) 已指出：**只看圖、不看 query 的「deaf model」**，在 RefCOCOg 上 image-only 都能拿到 40.1%（vs 隨機），證明 shortcut 嚴重。
- 即使把整句話的詞序打亂（shuffle）或只留名詞形容詞，效能只掉 4% / 3%——代表模型根本沒在用 reasoning。

**對手選項 CLEVR-Ref+（CVPR 2019）** 雖然合成、有 clean diagnostic 性質，但只有 3 種物件類別、12 種屬性，太簡單也不夠真實。

### 2.3 Cops-Ref 的設計目標

要同時做到：
- **真實圖像**（不像 CLEVR 合成）：建在 GQA 的 real-world image + scene graph 之上。
- **組合性強的長表達式**：用 expression engine 自動產生，可控、無歧義、語法正確。
- **強干擾測試環境**：每個 query 附帶 4 類 distracting images，逼模型必須讀懂整句、做到細粒度推理。

---

## 3. Cops-Ref 方法架構（資料集構建細節）

整個資料集建構分四個模組：**(A) Expression Engine** → **(B) Distractor Discovery** → **(C) Post-processing** → **(D) Statistics**。

### 3.1 (A) Expression Engine（最關鍵的創新）

#### 三步驟流程
1. 給定一個 target region，從預定義的 **logic form family** 抽一個邏輯型態 + 對應 textual template。
2. 從場景圖（scene graph）以 target object 為 root，展開出對應的 **reasoning tree**。
3. 用 reasoning tree 的內容填回 textual template，產生最終表達式。

#### 六種 Logic Forms（論文 Table 1）

| 編號 | Form | Reasoning tree 結構 | Template 範例 | 表達式範例 |
|---|---|---|---|---|
| 1 | **chain** | `obj0(att0)—rel0→obj1(att1)—rel1→obj2(att2)` | The \<att0\> \<obj0\> that is \<rel0\> \<att1\> \<obj1\> that is \<rel1\> \<obj2\>. | The young girl that is touching the glazed donut that is on the round table. |
| 2 | **and** | obj0 同時與 obj1、obj2 有兩條關係 | The \<att0\> \<obj0\> \<rel0\> the \<att1\> \<obj1\> and \<rel1\> the \<att2\> \<obj2\>. | The white fence near the building and behind the walking woman. |
| 3 | **or** | obj0 與 obj1、obj2 至少滿足一條關係 | ...or... | The green suitcase behind the black suitcase or near the yellow suitcase. |
| 4 | **order** | obj0 帶 index/direction/attribute | The \<idx\> \<obj0\> from the \<dir\> that is \<att0\>. | The first glass from the left that is red. |
| 5 | **same** | obj0 與 obj1 共享某屬性類別（顏色/形狀/材質/性別/紋路） | The \<obj0\> that has the same \<cat\> as the \<obj1\>. | The bag that has the same color as the sweater. |
| 6 | **not** | obj0 不具有某屬性 | The \<obj0\> that is not \<att0\>. | The apple that is not red. |

這六種可以再彼此巢狀組合，產生更複雜的句子。

#### Reasoning Tree 解析細節
- **chain/and/or**：直接從 GQA 的 scene graph 子圖切出。
- **order**：把同類物件依 bbox 中心座標由左到右排序；因為 order 的條件較弱（"the left glass" 在干擾圖也可能成立），再額外加 attribute/relation 來保唯一性。
- **not**：遍歷 scene graph，找出「同類別其他物件全部有、但 target 沒有」的 attribute/relation。
- **same**：找「只有 target 和 related object 共享」的屬性，把該屬性類別當作兩者間的「關係」。

### 3.2 (B) Distracting Images 的發現（第二大創新）

測試階段對每個 expression 提供 4 類干擾圖（每類 3 張，共最多 12 張干擾圖 + 1 張 target 圖）：

| 類型 | 定義 | 考驗什麼能力 |
|---|---|---|
| **DiffCat** | 圖中物件類別 ≠ target 類別 | 基本物件識別 |
| **Cat** | 含 target 同類別的物件 | 屬性/關係的進一步辨別 |
| **Cat&attr** | 含 target 同類別+同屬性的物件 | 必須讀關係 |
| **Cat&cat** | 含 reasoning tree 中所有物件，但關係不同 | 必須讀關係的「方向/語意」 |

→ 找不到足夠 distractor 的 region-expression pair 直接丟棄。

### 3.3 (C) Post-processing 與 Balancing

- 用 WordNet 的 synonym 增加表達式詞彙多樣性。
- 刪掉「無法用 bbox 框出」的類別（如 sky, cloud）。
- 刪掉太小的 region（< 1% 影像面積）。
- 手動 check testing set，去除因 GQA scene graph 不完整而造成的「distractor 其實也符合 query」的雜訊樣本。
- 對 GQA 中過度頻繁的關係（如 `to the left of`）做反比例 downsampling，並丟掉「只含簡單空間關係」的 region-expression。

### 3.4 (D) Dataset Statistics（論文 Table 2）

| Dataset | #ObjCat | #Att | #Rel | Avg Exp Length | Avg #Cand | Avg #Same-cat Cand |
|---|---|---|---|---|---|---|
| RefCOCO | 801 | – | – | 3.5 | 10.6 | 4.9 |
| RefCOCOg | 80 | – | – | 8.5 | 8.2 | 2.6 |
| CLEVR-Ref+ | 3 | 12 | 5 | 22.4 | – | – |
| **Cops-Ref** | **508** | **601** | **299** | **14.4** | **262.5** | **20.3** |

最終規模：
- **148,712 expressions**、**1,307,885 regions**、**75,299 images**
- Vocabulary size: 1,596
- Train / Val / Test = 119,603 / 16,524 / 12,586 expressions
- 採用 GQA 的 val 當 test（因 GQA test scene graph 未公開），再從 GQA train 切出新 val

### 3.5 任務形式化

給定 N 張圖（含 1 張 target + 多張 distractor）和 query q，找：

$$r_{i,j}^* = \arg\max_{r_{i,j},\, i\in[1,N],\, j\in[1,J_i]} s(r_{i,j}|q)$$

其中 $s(\cdot|q)$ 是 region-query 匹配分數。**訓練時不提供 distractor 圖**（理由：真實世界難收集 distractor，且方便沿用既有 REF model 的訓練流程），distractor 只在測試時加入。

---

## 4. 場景上下文（Scene Context）是如何被建模的？

論文沒有提一個新的 grounding model 去顯式建 scene context，而是**透過資料設計把 scene context 的需求 push 到 model 端**。具體機制：

### 4.1 資料層次：表達式內嵌 scene graph
- Reasoning tree 直接從 GQA dense scene graph 切出，所以一個 expression 不只是描述 target 本身，而是把 target **嵌入到一張 sub-graph**（target + related objects + relations + attributes）裡。
- 例如 `The young girl that is touching the glazed donut that is on the round table` 強迫模型同時定位 girl/donut/table 並驗證 touching、on 兩條關係。

### 4.2 測試環境層次：multi-image context
- 把 distracting images 加入候選池，**將 scene context 從「單張圖內」擴展到「圖像集合間」**。模型不能只看單張圖內的少數同類物，還必須把整個 image set 視為一個更大的 visual context 去 disambiguate。

### 4.3 模型層次：MattNet-Mine 透過 modular sampling 學 context
- 用 MattNet 的 sub/loc/rel 三個模組的 phrase embedding 相似度，從整個訓練集找「modular-level」的 hard negative。
- 等於把「context distinguishability」當成一個 learning signal 注入訓練：模型被迫學到 sub/loc/rel 三方面都和 target 接近的 negative 才難。

---

## 5. Baseline 模型與 MattNet-Mine

### 5.1 Baseline = MattNet（Yu et al., CVPR 2018）

MattNet 是當年最強 REF baseline，把 query 拆三模組：subject (sub)、location (loc)、relationship (rel)，總匹配分數：

$$s(r_j|q) = \sum_{md \in \{\text{sub, loc, rel}\}} w_{md} \cdot s(r_j|q_{md})$$

原始訓練用 **ranking loss**（hard negative 只在「同一張圖內」抽）：

$$L_{rank} = \sum_m \Big( [\Delta - s(r_m|q_m) + s(r_m|q_n)]_+ + [\Delta - s(r_m|q_m) + s(r_o|q_m)]_+ \Big)$$

→ 在 Cops-Ref 上問題：**沒辦法挑「跨圖」的 hard negative**，所以 distractor 圖一加入立刻崩盤。

### 5.2 MattNet-Mine（本文新增 baseline）

#### Modular Hard Mining 策略

對第 m 個 region-expression pair，用 modular phrase embedding $q_m^{md}$ 算出與第 n 個 pair 的 cosine 相似度 $s_{m,n}^{md}$，再 softmax 得到取樣機率：

$$p_{m,n}^{md} = \frac{\exp(s_{m,n}^{md})}{\sum_{n'=1, n'\neq m}^{N_C} \exp(s_{m,n'}^{md})}$$

其中 $N_C$ = 訓練集中與 m-th sample 同類別的所有 sample 數。

#### Mining Loss

$$L_{mine} = \sum_m \sum_{md} \Big( [\Delta - s(r_m|q_m) + s(r_m|q_n^{md})]_+ + [\Delta - s(r_m|q_m) + s(r_n^{md}|q_m)]_+ \Big)$$

#### 總 Loss
$$L = L_{rank} + L_{mine}$$

- $L_{rank}$ 處理「同圖內」hard negative；$L_{mine}$ 處理「跨圖」hard negative。
- 訓練流程：先用 $L_{rank}$ pre-train，再用 $L_{rank} + L_{mine}$ fine-tune；每 50 iteration 更新一次 sample 機率。
- 高效：只需 expression embedding，不用 load 圖；29 秒就能掃完整個訓練集所有 expression。

---

## 6. 實驗結果

### 6.1 主表（Table 3，testing accuracy %）

| Method | Full | DiffCat | Cat | Cat&attr | Cat&cat | WithoutDist |
|---|---|---|---|---|---|---|
| Chance | 0.4 | 1.7 | 1.8 | 1.9 | 1.7 | 6.6 |
| **GroundeR** | 19.1 | 60.2 | 38.5 | 35.7 | 38.9 | 75.7 |
| Deaf-GroundeR (image-only) | **2.2** | 7.7 | 7.9 | 8.0 | 8.0 | 27.1 |
| Shuffle-GroundeR | 13.1 | 41.8 | 28.6 | 27.2 | 27.6 | 58.5 |
| Obj-Attr-GroundeR (只留名詞形容詞) | 15.2 | 53.1 | 32.6 | 29.6 | 32.7 | 68.8 |
| MattNet-refCOCO (transfer) | 8.7 | 22.7 | 17.0 | 16.7 | 18.9 | 42.4 |
| **MattNet** | 26.3 | 69.1 | 45.2 | 42.5 | 45.8 | 77.9 |
| **CM-Att-Erase** | 28.0 | 71.3 | 47.1 | 43.4 | 48.4 | **80.4** |
| SCAN+MattNet (retrieve+REF) | 18.8 | – | – | – | – | – |
| **MattNet-Mine（本文）** | **33.8** | 70.5 | **54.4** | **46.8** | **52.0** | 78.4 |

### 6.2 Bias Analysis（最有說服力的結果之一）

- **Deaf-GroundeR**（看圖不看字）在 Full 設定下只有 **2.2%**，而在 RefCOCOg 等其他資料集相同實驗能拿到 **40.1%**。→ Cops-Ref 的 dataset bias 顯著被壓抑。
- **Shuffle 詞序**：RefCOCOg 上只掉 4%，Cops-Ref 上掉 **31%**。
- **Obj-Attr only**：RefCOCOg 上只掉 3%，Cops-Ref 上掉 **20%**。
- → 證明 Cops-Ref 真的需要句法、關係、整體推理才能解。

### 6.3 Transfer Performance（兩個方向皆值得注意）

- MattNet trained on RefCOCO → Cops-Ref Full: **8.7%**（崩潰）→ 證明 Cops-Ref 真的比較難。
- MattNet trained on Cops-Ref → RefCOCO testA/B: **56.5 / 64.5%**（約原始 65.7% / 76.4%）→ 證明合成出來的 expression 仍有實用知識可 transfer 回 RefCOCO。

### 6.4 Logic Form 細分（Fig. 3）

- **order > not > 其他四種**：因為 order 的 reasoning tree 較簡單，且本身就提供同類別物的相對空間位置。
- chain / and / or / same 表現接近。

### 6.5 Expression Length（Fig. 4）

- **middle (10–20 words) > short < long**：太短資訊不足，太長則語意太複雜難 reason。

### 6.6 Ablation — Mining Strategy（Table 4）

| Strategy | Full | DiffCat | Cat | Cat&attr | Cat&cat |
|---|---|---|---|---|---|
| 原 MattNet | 26.3 | 69.1 | 45.2 | 42.5 | 45.8 |
| Random | 27.6 | 71.6 | 47.4 | 43.5 | 47.3 |
| Class-aware | 32.2 | 70.3 | 53.2 | 46.1 | 51.4 |
| Sentence-sim | 32.3 | 70.4 | 53.6 | 46.4 | 51.2 |
| **Module-specific（本文）** | **33.8** | 70.5 | **54.4** | **46.8** | **52.0** |

→ Module-specific（用 sub/loc/rel 各自的 embedding 相似度）顯著勝過用整句平均的 sentence-sim。

### 6.7 Retrieve + REF 策略

「先用 SCAN 做 text-to-image retrieval 挑一張圖再 REF」反而只有 18.8%（遠低於 dense REF），原因是 retrieval 粗略，無法做 region-level fine-grained matching。

---

## 7. 關鍵洞見與貢獻

### 7.1 對社群的四大貢獻
1. **新任務 Cops-Ref**：要求模型在 visually-similar 的 image set 中定位 target。
2. **新資料集**：建在 GQA 真實圖像之上、保留視覺真實感與語意豐富性。
3. **Expression Engine**：6 種 logic form 可任意組合，產生可控的長/複雜 query。
4. **MattNet-Mine baseline**：modular hard mining 顯著優於既有 SOTA。

### 7.2 我認為更值得記住的洞見
- **資料集設計本身就是 contribution**：把「scene context」這件事的負擔從 model design 推回 data design，比硬寫一個新 attention module 更有 lever。
- **distractor 是 reasoning 評估的本質**：沒有 distractor，shortcut model 就能假裝在 reason。Cops-Ref 用 4 種 distractor 把 shortcut 路徑一條條切掉。
- **Modular hard mining 是 cheap and effective**：不需要 load image，只用 phrase embedding 相似度，就能挑出強 hard negative。
- **Cross-dataset transfer 是雙向不對稱的**：RefCOCO → Cops-Ref 崩盤；Cops-Ref → RefCOCO 大致保留 → 證明 Cops-Ref 的訓練信號「包含」RefCOCO 但反之不然。

---

## 8. 與你 vault 中其他 REC 論文的關係

### 8.1 vs. **Modeling Relationships in Referential Expressions with Compositional Modular Networks (Hu et al., CVPR 2017)**
- Hu 的 CMN 是「**模型側**」的 compositional approach：把 query 拆成 subject-relation-object triplet，各自用 module 處理。Cops-Ref 是「**資料側**」的 compositional approach：用 expression engine 生 compositional query。
- 兩者互補：Cops-Ref 創造的測試環境，正好是 CMN/MattNet 這類 modular model 的試金石。論文也直接用 MattNet（CMN 的精神後裔）作為主要 baseline。

### 8.2 vs. **HieA2G（Hierarchical Attention-based Approach to Grounding）**
- HieA2G 強調階層式 attention 處理多層級語意關係。
- Cops-Ref 的 chain/and/or 等 logic form 正好提供 multi-hop reasoning 的測試場景，**HieA2G 這類階層 attention model 可以在 Cops-Ref 上量到比 RefCOCO 更明顯的優勢**（如果它真的有 hierarchical reasoning 能力的話）。

### 8.3 vs. **GREC（Generalized REC）**
- GREC 把 REC 從「假設 query 必有對應物件」推廣到「query 可能對應 0 個、1 個或多個物件」，引入 no-target、multi-target 的 abstention 與多選能力。
- Cops-Ref 的 4 種 distractor 中，**Cat、Cat&attr、Cat&cat 都已經帶有「distractor 圖中也有同類物，但不是 target」**的設定，**邏輯上接近 GREC 的「no-target rejection」前身**——但 Cops-Ref 仍假設「全 image set 中至少有一個正解」，沒到 GREC 的徹底程度。
- 你做 CRS（Conformal Referring Set）時，可以把 Cops-Ref 的 distractor 設定當「半個 no-target benchmark」用，但要小心它的訓練集不含 distractor，所以模型沒學過 abstention。

### 8.4 vs. **Zero-Shot REC via Visual-Language True/False Verification (2509.09958)**
- 該論文用 LLM/VLM 做 zero-shot REC，把問題 reframe 成「verify 一個 candidate-claim 對」的 True/False 判斷。
- Cops-Ref 的設計反而是讓 **fully-supervised model 都很難解**，所以是 True/False verification 框架的最硬考題之一：模型對長 reasoning chain + distractor 必須做多步 verification，不能用單一 contrastive score 一次比完。
- 你筆記提到這篇是新競品威脅 Ch5 ceiling — 在 Cops-Ref 上對 True/False verification framework 的 evaluation，可能會比在 RefCOCO 上更能拉開差距、給你 Ch5 找回優勢空間。

---

## 9. 個人評價

### 9.1 優點
1. **問題抓得準**：當時 REC 領域已被 RefCOCO 主宰太久，shortcut 嚴重，Cops-Ref 直接從「能不能評到 reasoning」的根本動機切入。
2. **資料設計嚴謹**：bias analysis 用 deaf/shuffle/obj-only 三組 ablation 把「沒在 reason」的可能路徑全部 expose 出來，論證強。
3. **規模合理**：~15 萬 expression、~7.5 萬影像，足以訓練；又不像 CLEVR 一樣脫離真實。
4. **MattNet-Mine 是個好用的 baseline trick**：modular-level mining 比 sentence-level 顯著好，且實作便宜。
5. **可重現性高**：GitHub repo、CVPR open access、scene graph 來自公開 GQA。

### 9.2 缺點與限制
1. **Expression 還是「合成」出來的**：雖然建在真實圖像 + 真實 scene graph 上，但句子是 template + slot fill，不像人類自由表達自然。所以 Cops-Ref 對 “自然語言的歧義性、口語性、metaphor” 等問題覆蓋不足。
2. **scene graph 噪音傳播**：GQA scene graph 已知 incomplete/noisy，作者只手動清了 testing set，training set 依舊帶噪。
3. **訓練/測試 distractor 不對稱**：訓練時不放 distractor，造成 model 沒學過跨圖比較；MattNet-Mine 其實是在補這個 gap，但仍是後處理式的補救。
4. **6 種 logic form 仍是封閉集合**：不能涵蓋 negation 的多層 scope、quantifier (「所有的」「至少兩個」)、temporal/causal reasoning 等更複雜邏輯。
5. **不評估 no-target / multi-target**：所有 query 在 image set 中恰好有一個正解，無法測 GREC-style 拒答能力（這後來被 FineCops-Ref / GREC 補上）。
6. **缺乏 generative / open-vocab 設定**：bbox 候選來自 GT，等於假設 detector 完美，與現代 open-set / VLM-based REC 的 setup 落差大。
7. **6 年後已被 follow-up 部分超越**：FineCops-Ref (EMNLP 2024) 名字就是致敬 Cops-Ref，補強了 controllable difficulty、negative text、negative image，並把測試對象從專家 REC model 改到 MLLM。

### 9.3 適合什麼研究情境用 Cops-Ref？
- **想評估 compositional reasoning 而非 grounding accuracy 本身** → 強推。
- **想測 zero-shot VLM 能不能讀懂長 query + 拒絕 distractor** → 是不錯的 stress test。
- **想做 selective grounding / calibration 研究**（如你的 CRS 主線）→ 4 類 distractor 是現成的 controlled negative source，可以拿來做 conformal set 的 size/coverage 對比；但要小心 multi-image setup 與 RefCOCO/gRefCOCO 的 protocol 不同，bbox 候選池不一樣。

---

## 10. 對你（kuo）論文主線的可能用途

> 結合你的 CRS / selective grounding 主線：

1. **Cops-Ref 的 distractor 是天然的 conformal calibration set**：4 類干擾分得很乾淨，可以做「分類別 LTT bound」的 fine-grained study，比 gRefCOCO 的 no-target 更可控。
2. **Cross-Base Conformal Composition (OWL gate + GD box) 在 Cops-Ref 上可能更顯優勢**：因為 distractor 多，gate 的價值會放大；可以把它當成第三個 base 或第三個資料集去佐證 cross-base transferability。
3. **Risk**: Cops-Ref 的 ground-truth bbox protocol 與你 frozen detector 的 detection-from-scratch 設定不同，需要先決定要不要用 GT proposal。如果用 GT，那等於把 detector recall 設成 1，CRS 的可用 abstention 會少一個論述空間。
4. **與 True/False Verification 競品的對比**：在 Cops-Ref 上跑你的 post-hoc reliability layer，若能在 Cat&attr / Cat&cat 兩個最硬 split 上贏 True/False baseline，能幫 Ch5 找回 ceiling。

---

## 11. 參考延伸閱讀

- 原論文：[arXiv:2003.00403](https://arxiv.org/abs/2003.00403) / [CVPR 2020 open access](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Cops-Ref_A_New_Dataset_and_Task_on_Compositional_Referring_Expression_CVPR_2020_paper.pdf)
- 程式碼：https://github.com/zfchenUnique/Cops-Ref
- 後繼工作 FineCops-Ref：[arXiv:2409.14750](https://arxiv.org/abs/2409.14750)（EMNLP 2024 main + TPAMI 2025 擴展）
- 同類診斷資料集：CLEVR-Ref+（CVPR 2019）
- Bias analysis 母題：Cirik et al., NAACL 2018, "Visual Referring Expression Recognition: What Do Systems Actually Learn?"
- MattNet（baseline）：Yu et al., CVPR 2018
- CM-Att-Erase（baseline）：Liu et al., CVPR 2019
