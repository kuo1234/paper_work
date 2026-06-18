---
title: "GREC: Generalized Referring Expression Comprehension —— 詳細導讀"
author: "讀書筆記（繁體中文版）"
date: "2026-06-18"
---

# GREC: Generalized Referring Expression Comprehension —— 詳細導讀

## 0. 基本資訊

- **論文標題**：GREC: Generalized Referring Expression Comprehension
- **作者**：Shuting He†、Henghui Ding†（共同一作）、Chang Liu、Xudong Jiang
- **單位**：Nanyang Technological University（南洋理工大學）
- **發表狀態**：arXiv 技術報告（arXiv:2308.16182v2，2023 年 12 月 24 日 v2）。是 CVPR 2023 論文 **GRES: Generalized Referring Expression Segmentation**（Liu, Ding, Jiang, CVPR 2023）的姊妹/延伸技術報告，把 GRES 在 segmentation 端建立的「multi-target / no-target」泛化設定，平移到 detection / bounding box 端。
- **官方資源**：
  - 專案頁：<https://henghuiding.github.io/GRES>
  - 程式碼與 gRefCOCO 標註：<https://github.com/henghuiding/gRefCOCO>
- **篇幅**：正文 5 頁 + 1 頁參考文獻，總 6 頁。屬於 short technical report，定位是「定義新任務 + 開源 benchmark + 提供 baseline 改造方案 + 提出評估指標」。

---

## 1. 問題定義與動機

### 1.1 經典 REC 是什麼

**Referring Expression Comprehension（REC，指代表達式理解）** 是視覺-語言領域的經典任務：

- **輸入**：一張影像 + 一段自然語言描述（referring expression），例如「the kid in red」。
- **輸出**：影像中該描述所指物件的單一 bounding box。
- **代表資料集**：ReferIt（Kazemzadeh et al., EMNLP 2014）、RefCOCO / RefCOCO+ / RefCOCOg（Yu et al., ECCV 2016；Mao et al., CVPR 2016）。
- **應用場景**：影片製作、人機互動、機器人。

過去十年的 REC 方法（MCN、VLT、MDETR、TransVG、UNINEXT、Grounding DINO 等）在 RefCOCO 系列上 top-1 accuracy 已普遍達 85% 以上，看似「solved」。

### 1.2 經典 REC 的兩個強假設（也是限制）

論文一針見血地指出，主流 REC 設定其實隱含兩個「過強的前置條件」（pre-defined constraints）：

1. **單目標假設（single-target）**：每條表達式恰好對應到影像中一個物件。
   - 資料集裡每個 sample 只有一個 ground-truth box，模型也只輸出一個 box。
   - 但實際語言常出現「所有人」「左邊三隻羊」「除了穿白衣的那個小孩外的所有人」這種**多目標表達式**，現有 REC 完全無法處理 ——你必須把問題拆成多次單目標查詢，效率低且不自然。

2. **目標必存在假設（target-must-exist）**：表達式所指的物件**一定**在影像中存在。
   - 若使用者輸入一個畫面裡根本不存在的物件，例如畫面是兩個穿紅、白衣的小孩，使用者問「the kid in blue」，現行 REC 模型行為是「未定義的」——它一定會硬輸出某個 box（通常是隨便挑一個最像的），對使用者極度不友善。
   - 這在實務上等於把判斷「指代是否成立」的責任全推給使用者。

論文的核心論點：**這兩個假設嚴重削弱 REC 的實用性，必須鬆綁。**

### 1.3 GREC：把 REC 從「1」泛化到「0 ～ many」

作者提出 **Generalized Referring Expression Comprehension (GREC)**：

- **輸入**：與經典 REC 相同（影像 + 表達式）。
- **輸出**：bounding box 的**集合** B = {b_i}，|B| 可以是 0、1、2、…、N。
- **支援三種樣本**：
  1. **single-target**：經典 REC 的情況，向後相容。
  2. **multi-target**：一條表達式對應到多個物件（例如「all people」、「two players on left」、「everyone except the kid in white」）。
  3. **no-target**：表達式描述的物件**不存在於影像中**，模型應該輸出空集合 ∅。

GREC 不是發明全新的任務，而是**鬆綁經典 REC 的人工約束**，讓問題退回更自然的 set-valued grounding 問題。

### 1.4 為什麼這個泛化有意義（論文舉的四個應用）

1. **一次 forward 抓多物件**：例如「all people」原本要跑四次模型才能框出四個人，GREC 一次搞定（Fig. 2a）。
2. **使用者定義的 open-vocabulary perception**：像「foreground」「kids」這種模糊但語意明確的詞，可以當作 query 直接抓符合條件的所有物件。
3. **以表達式做影像檢索**：把同一條表達式套到一組影像上，可以快速篩出「含有該物件的影像」，比傳統 image retrieval 更精準（Fig. 2b）。
4. **更高的 robustness**：實務上使用者會打錯字、亂下指令，no-target 樣本的存在強迫模型有「拒答」能力，避免幻覺式錯誤輸出。

---

## 2. gRefCOCO 資料集

> **注意**：GREC 論文本身對資料集只給出 1 個段落的概述（§2.2 只說「請參考 GRES 論文」），完整的標註流程與統計細節在姊妹論文 **GRES (CVPR 2023)** 裡。以下整合 GRES 論文的關鍵資訊。

### 2.1 與其它 REC 資料集的比較（Tab. 1）

| 資料集 | 影像來源 | Multi-target | No-target | 表達式 |
|---|---|---|---|---|
| ReferIt | CLEF | ✗ | ✗ | free |
| RefCOCO / RefCOCOg | COCO | ✗ | ✗ | free |
| PhraseCut | VG | ✓ (fallback) | ✗ | templated |
| **gRefCOCO** | COCO | ✓ | ✓ | free |

關鍵：
- gRefCOCO 是**第一個同時涵蓋 multi-target + no-target + free-form 表達式**的大規模 REC 資料集。
- 影像來自 COCO，與 RefCOCO 同源，所以可以直接做 cross-dataset 比較與相容。
- PhraseCut 雖然支援 multi-target，但它的表達式是模板生成的（templated），語言多樣性遠遠不如 free-form。

### 2.2 樣本切分

延續 RefCOCO 的 train / val / testA / testB 切分，內含三種樣本（single / multi / no-target）並存。**重點 caveat**：作者特別強調，任何訓練／預訓練都**必須**排除 val/testA/testB 的影像，否則會 information leakage（這是 UNINEXT 原始版本踩過的坑，他們重新訓練了一份乾淨版來公平比較）。

---

## 3. 評估指標：為什麼經典 REC 的 Precision@0.5 不夠用

### 3.1 經典 REC 的指標

- 每個 sample 只有 1 個 GT box、1 個 predicted box，預測非 TP 即 FP。
- 用 **Precision@0.5**（top-1 accuracy）：IoU > 0.5 算對。
- 這個指標在 GREC 下完全崩潰，因為：
  - multi-target 樣本：GT 有 N 個 box，預測也應該有 0～M 個 box，單靠一個 IoU 沒法評。
  - no-target 樣本：根本沒有 GT box，IoU 無從談起。

### 3.2 GREC 新指標 1：Precision@(F1=1, IoU ≥ 0.5)（**論文推薦主指標**）

對一個 sample：

1. 把每個 predicted box 與 GT box 用 IoU ≥ 0.5 做配對：
   - 配對到的 prediction 算 **TP**；若多個 prediction 配到同一個 GT，只保留 IoU 最高那一個為 TP，其餘 FP。
   - 配對不到 GT 的 prediction 算 **FP**。
   - 配對不到 prediction 的 GT 算 **FN**。
2. 計算 sample-level F1：F1 = 2·TP / (2·TP + FN + FP)。
3. 只有 **F1 = 1**（完全配對、無冗餘）才算這個 sample 預測成功。
4. 對 no-target 樣本：若預測 0 個 box → F1 = 1（成功）；否則 → F1 = 0（失敗）。
5. 整個資料集的指標 = 成功樣本數 ÷ 總樣本數。

**為什麼設這麼嚴格的「F1 必須等於 1」**？因為 REC/GREC 的下游使用情境是把 box 給使用者看，多框一個錯的或漏一個對的，使用者體驗都會崩，因此採 sample-level 全對才算分。

### 3.3 GREC 新指標 2：N-acc.（No-target accuracy）

專門評估「拒答能力」：

- 對 no-target 樣本：預測 0 個 box → TP；只要預測任何 box → FN（不是 FP，因為這裡視為「沒有正確識別出 no-target」）。
- N-acc. = TP / (TP + FN)。

### 3.4 為什麼不推薦用 AP

論文特別說明，COCO 風格的 AP（IoU 0.5–0.95 平均）在 REC/GREC 不合適，因為 AP **不會懲罰** 模型輸出大量低 confidence 的冗餘 box ——但 REC 場景下，使用者問一個物件你卻吐 100 個框絕對是 unacceptable。所以 AP 高未必好，作者明確說 AP「does not fully capture the performance」。

---

## 4. 方法：把現有 REC 方法「適配」到 GREC

GREC 論文**本身沒有提出新方法**，定位是 benchmark 論文。它做的事情是：

1. 拿四個經典/SOTA REC 方法（MCN、VLT、MDETR、UNINEXT）。
2. 把它們的 detection head 改造成可以輸出**多個** bounding box（原本許多方法只輸出 top-1 或單一 box）。
3. 在「如何從多個候選 box 選出最終輸出集合」這一步做 ablation。

### 4.1 候選 box 選法的 ablation（Tab. 2，以 MDETR 為 base）

MDETR 本身會輸出 100 個 box，問題是怎麼從中選：

| 策略 | Pr@(F1=1, IoU≥0.5) | AP | N-acc. |
|---|---|---|---|
| Top-1 | 0.0 | 26.2 | 0.0 |
| Top-5 | 0.0 | 52.8 | 0.0 |
| Top-10 | 0.0 | 53.3 | 0.0 |
| Top-100 | 0.0 | 53.5 | 0.0 |
| Threshold-0.5 | 37.0 | 52.6 | 32.2 |
| Threshold-0.6 | 38.9 | 52.5 | 33.9 |
| Threshold-0.7 | 41.5 | 52.3 | 36.1 |
| Threshold-0.8 | 44.7 | 51.7 | 39.2 |
| **Threshold-0.9** | **51.2** | 50.8 | **45.7** |

**關鍵洞察**：

- **Top-k 策略全軍覆沒**（Pr@F1=1 都是 0）：因為 top-k 強迫每個 sample 都吐 k 個 box，對 multi-target 樣本最多只能拿到 F1 = 0.67（兩個目標只對一個 + 多了一個 FP），對 no-target 樣本一定 FN（強迫輸出 ≥ 1 個 box，但 GT 是 0），N-acc. = 0。
- **基於 confidence threshold 的動態策略才行得通**：讓模型根據每個 sample 自己決定要輸出幾個 box。
- **threshold 越高越好**：從 0.5 提到 0.9，Pr@F1=1 從 37.0% → 51.2%，N-acc. 從 32.2% → 45.7%。高 threshold 等於更保守地輸出，減少 FP 與錯誤拒答失敗。
- AP 在不同 threshold 下變化很小（50.8–53.5），再次印證 AP 對 GREC 不敏感。

論文呼籲：**未來研究應該設計更聰明的 output strategy**（不只是 threshold cut-off）。

### 4.2 主表 baseline 結果（Tab. 3）

四個方法被改造成「輸出多 box + threshold=0.7 篩選」（† 標記）：

| 方法 | Visual Enc. | Text Enc. | val Pr@F1=1 | val N-acc. | testA Pr@F1=1 | testA N-acc. | testB Pr@F1=1 | testB N-acc. |
|---|---|---|---|---|---|---|---|---|
| MCN† | DarkNet-53 | GRU | 28.0 | 30.6 | 32.3 | 32.0 | 26.8 | 30.3 |
| VLT† | DarkNet-53 | GRU | 36.6 | 35.2 | 40.2 | 34.1 | 30.2 | 32.5 |
| MDETR† | ResNet-101 | RoBERTa | 42.7 | 36.3 | 50.0 | 34.5 | 36.5 | 31.0 |
| **UNINEXT†** | ResNet-50 | BERT | **58.2** | **50.6** | 46.4 | 49.3 | 42.9 | 48.2 |

**關鍵觀察**：

- 同樣這些方法在 RefCOCO 單目標 setting 上 top-1 accuracy 普遍 **85%+**，在 GREC 上 Pr@F1=1 只剩 **26–58%**，掉了 30+ 點。明確證明「single-target 上 saturated」≠「真實場景上 ready」。
- UNINEXT 最強，但 N-acc. 也只有 ~50%，表示拒答能力遠未到位。
- testA（人物為主）通常比 testB（物件為主）容易，但 UNINEXT 在 val/testA/testB 上 Pr@F1=1 是 58.2 / 46.4 / 42.9，跟其它方法 testA > val > testB 的順序不同 ——這可能跟 UNINEXT 的訓練資料分布有關。

**Information leakage caveat**：原始 UNINEXT 預訓練包含 val/testA/testB 的影像，作者重新訓練了一份不含這些影像的版本（嚴格 setup），數字是這份乾淨版的。

### 4.3 質性結果（Fig. 3）

- **成功案例**（最上排）：模型能抓 salient 線索如顏色（橘色雨傘）、位置（最邊兩個人）、數量（三隻看狗的羊）。
- **multi-target 失敗**（中排）：當表達式包含「共享屬性」「複雜組合」（如 "Orange and all sandwiches"，需要同時抓水果跟所有三明治），模型會漏框或多框，F1 = 0.27～0.5。
- **no-target 失敗**（下排）：模型仍會硬輸出 box，F1 = 0。論文舉了一個有趣的 deceptive 案例：「The right guy sitting on the bench wearing a hat」——畫面右邊確實有個人坐在 bench 上，但他**沒戴帽子**。理想模型應該識破這個 mismatch 輸出 ∅，但目前的方法做不到。這個案例特別點出 GREC 需要「細粒度語意一致性檢查」的能力。

---

## 5. GREC 與經典 REC 的核心差異總表

| 維度 | 經典 REC | GREC |
|---|---|---|
| 輸入 | 影像 + 表達式 | 影像 + 表達式（相同） |
| 輸出 | 1 個 box | 0 ～ N 個 box 的集合 |
| 樣本類型 | 只有 single-target | single + multi + no-target |
| 假設前提 | 目標必存在且唯一 | 無假設 |
| 主指標 | Precision@0.5（top-1 acc） | Pr@(F1=1, IoU≥0.5) + N-acc. |
| 輸出策略 | argmax / top-1 | 需要動態決定輸出數量（threshold / set prediction） |
| 對 robustness 的要求 | 低（默認 well-formed input） | 高（要能拒答與多框） |
| 與下游應用契合度 | 受限（要多次呼叫、無拒答） | 更貼近真實使用情境 |

---

## 6. 與其他相關論文的關係

### 6.1 與 GRES（Liu, Ding, Jiang, CVPR 2023）

- **直系姊妹工作**。GRES 把 referring **segmentation**（RES，輸出 mask）泛化到多目標 + no-target；GREC 是把同一個泛化哲學套到 referring **comprehension**（REC，輸出 box）。
- 兩者**共用 gRefCOCO 資料集**的影像與表達式標註（box 與 mask 都有）。
- GRES 用 ReLA（Region-Language Attention）model 提了一個專門的 method；GREC 沒提新方法，只做 benchmark 與既有方法 adaptation。
- GREC 論文裡資料集細節（標註流程、表達式統計、annotator 設計）全部 defer 到 GRES 論文，所以要完整理解 gRefCOCO 必須對讀兩篇。

### 6.2 與 COPS-Ref（Chen et al., ICCV 2020）

- **COPS-Ref（Cops-Ref: A New Dataset and Task on Compositional Referring Expression Comprehension）** 也是在批判經典 RefCOCO 系列「太簡單」，但走的是不同方向：
  - COPS-Ref 強調**組合性推理（compositional reasoning）** 與 **distractor 設計**——讓表達式必須做多跳屬性/關係推理才能定位（例：「the cup to the left of the plate that is on the red table」）。
  - COPS-Ref 仍是 single-target setting（每個表達式一個目標），但表達式語意更複雜。
- **GREC 走的方向是 set-valued generalization（0/1/many 目標）**，不是推理複雜度的提升。
- 兩者**互補**：COPS-Ref 在「表達式有多難理解」這條軸往前推，GREC 在「答案空間多大」這條軸往前推。理想未來 benchmark 應該結合兩者。

### 6.3 與 HieA2G（Hierarchical Alignment for grounding，相關 multi-target 後續工作）

- **HieA2G** 是 GREC/gRefCOCO 釋出後社群提出的後續方法之一，針對 multi-target 場景設計**階層式語言-視覺對齊**（hierarchical alignment），把表達式分解成子片段（每個 sub-phrase 對應一個或一組物件）後再做 attention 對齊。
- 對 GREC 的具體貢獻：解決 baseline 在 "shared clue" 與 "compositional multi-target" 上的失敗模式（如 GREC 論文 Fig. 3 中 "Orange and all sandwiches" 這種需要分解的表達式）。
- 換言之 GREC 是「task + benchmark」，HieA2G 是「在這個 benchmark 上推進方法」的代表之一。

### 6.4 與 "Modeling Relationships in Referential Expressions"（Hu et al., CVPR 2017）

- **Hu et al. 2017** 提出 **Compositional Modular Networks (CMN)**，是早期把「關係」（relationships，如 left of、on top of）顯式建模進 referring 任務的代表作。它把表達式分解成 subject、relationship、object 三個 module，分別處理後組合。
- 對 GREC 的意義：CMN 已經點出「關係建模」對 REC 不可少，但仍受限於 single-target。GREC 的 multi-target 場景大量需要關係推理（"two players on left"、"everyone except the kid in white"），CMN 那條 modular 路線在 GREC 上依然有參考價值。
- 兩者關係：CMN = 經典 REC 時代的方法論貢獻；GREC = 重新定義任務本身。CMN 是 GREC 的學術祖父輩，GREC 引用了它（ref [13]）作為 REC 文獻脈絡的一環。

---

## 7. 關鍵洞見與貢獻總結

1. **任務泛化（最核心）**：明確指出經典 REC 的兩個強假設（single-target、target-must-exist），並提出 set-valued 的 GREC 來取代。這是一個**任務層面的概念性鬆綁**，比方法層面的改良更具長期影響力。

2. **資料集 gRefCOCO**：第一個大規模、free-form、同時涵蓋 single/multi/no-target 的 REC 資料集，與 RefCOCO 影像同源便於相容比較。

3. **新評估指標 Pr@(F1=1, IoU≥0.5) + N-acc.**：sample-level F1 = 1 的嚴格標準確保「全對才算數」，N-acc. 專門評估拒答能力。這兩個指標已被後續 selective grounding、conformal grounding 等 reliability/calibration 方向採用為主指標。

4. **Ablation 揭示 top-k 失效**：實證證明 top-1/top-k 策略在 GREC 下全部得 0 分，必須改用 dynamic threshold 或 set prediction。這個結論看似簡單，但提醒社群「output strategy 本身是個 open problem」。

5. **Baseline reality check**：SOTA REC 方法在 RefCOCO 上 85%+，搬到 GREC 上只剩 30–58%，戳破「REC 已解決」的迷思，重新打開研究空間。

---

## 8. 個人評價：優點、缺點、限制、後續方向

### 8.1 優點

- **問題定義乾淨且直覺**：把「N=1」放寬到「N ∈ {0,1,2,…}」是最自然的泛化方向，沒有人工痕跡。
- **與既有資料集相容**：gRefCOCO 用 COCO 影像、延伸 RefCOCO 標註，舊方法只要小改 head 就能評，門檻低。
- **評估指標設計嚴謹**：Pr@(F1=1) 的嚴格性正好對應 REC 的下游需求；同時針對 no-target 給 N-acc. 補位。
- **開源完整**：資料集、評估腳本、MDETR 改造版 baseline 全部開源，是負責任的 benchmark 論文。
- **論文短而精準**：6 頁說清楚 task、dataset、metric、baseline、failure analysis，沒有冗餘。

### 8.2 缺點與限制

- **沒有提出新方法**：定位純粹是 benchmark，貢獻集中在 task + data + metric 三點，方法層留給社群。對研究者來說這既是缺點（沒給 method 啟發）也是優點（不偏袒任何方法）。
- **資料集細節依賴 GRES 論文**：§2.2 只有一句話「請見 GRES」，獨立讀 GREC 論文無法完整理解 gRefCOCO 的標註流程、表達式分布、潛在偏誤。
- **N-acc. 與 Pr@F1=1 的耦合**：no-target 樣本在 Pr@F1=1 裡也佔分母，與 N-acc. 不完全獨立，可能造成兩個指標互相 confound（例如模型「全部拒答」N-acc 滿分但 Pr@F1=1 在 multi/single target 部分崩潰）。社群後續通常會分 single/multi/no-target 三個 split 各自報指標。
- **Threshold-based 策略其實是 ad-hoc**：threshold=0.7 / 0.9 在不同方法、不同 split 上最優值不同，缺少統一的、有理論依據的 abstention 機制。後續 selective prediction / conformal prediction 方向（如 CLIP-VG selective、CRS conformal referring set）就是針對這個缺口。
- **失敗案例 "right guy with hat" 暴露的是 hallucination 問題**：GREC 給了測試平台但沒給解法。需要更強的視覺-語言對齊（如 grounded captioning、verifier、cross-modal entailment）才能解。
- **僅支援靜態影像**：影片場景下「multi-target + no-target」的指代有時間動態（MeViS 已往這方向走），GREC 沒涵蓋。
- **缺乏「部分成功」的細粒度評分**：F1 = 1 才算對是 sample-level 全有全無；對 multi-target 中等規模（N=5）只漏一個的 sample，與完全亂預測拿同樣 0 分，可能讓方法層的小幅進步無法被指標反映。

### 8.3 後續方向（個人觀點）

1. **更聰明的 abstention / output strategy**：跳脫 fixed threshold，改用 conformal prediction、selective head、或學一個 stopping module 來動態決定輸出多少 box（如 Cross-Base Conformal Referring Set 把 LTT 跨 detector 聯合校準）。
2. **post-hoc reliability 路線**：保持 frozen backbone（CLIP-VG、OWL-ViT、Grounding DINO 等）+ 加一層 calibration head，把 GREC 當「可靠性研究」的試金石（這正是我本身碩論主線在做的方向）。
3. **更困難的 compositional GREC**：結合 COPS-Ref 風格的多跳推理與 GREC 的 set-valued 答案，建一個 next-gen benchmark。
4. **多 detector ensemble / cross-base transfer**：在 gRefCOCO 上跑多個 base model（CLIP-VG、OWL-ViT、UNINEXT）並研究它們失敗模式的互補性，做集成或 routing。
5. **更精細的 metric**：除了 Pr@F1=1，引入 mean F1、PR-curve over confidence、cost-sensitive metric（漏框 vs 多框的代價可調）。
6. **支援否定、量詞、排除**：multi-target 表達式中「everyone except X」「at least 2 of the …」對現有 vision-language model 仍是難關。
7. **延伸到 3D / 影片 / robot manipulation**：把 set-valued grounding 套到具身智慧場景，這是 referring 任務真正能 deploy 的地方。

---

## 9. 三句話總結

1. **GREC = 把「REC 只能輸出 1 個 box」這個過時的人工假設拿掉，允許 0/1/many 個 box，更貼近真實使用場景。**
2. **作者沒提新方法，而是建了 gRefCOCO 資料集 + 設計了 Pr@(F1=1, IoU≥0.5) 與 N-acc. 兩個新指標 + 把 4 個 SOTA REC 方法改造後跑 baseline，發現它們在 GREC 上掉了 30+ 點，戳破「REC 已 saturated」的錯覺。**
3. **這篇論文最大的價值不在技術新意，而在重新定義任務邊界 —— 之後 GREC 成為 selective grounding、conformal grounding、open-vocabulary REC 等 reliability/calibration 方向的標準 benchmark。**

---

## 10. 對個人研究（selective grounding）的對接筆記

> 與我自己 thesis 主線（CRS: Cross-Base Conformal Referring Set）的對接。

- gRefCOCO **就是我主場資料集**，三個 split（val/testA/testB）也是我 M0/M1/M4 gate 用的。
- **Pr@(F1=1, IoU≥0.5)** 與 **N-acc.** 在我的論文裡是直接拿來當官方指標（M4 full-GREC 就是用這兩個 metric）。
- **N-acc. 0.18 → 0.94**（forced-output → post-hoc policy）這個對比就是 GREC 論文沒處理但我主結果在處理的核心問題：用 post-hoc selective policy 把 no-target 拒答能力從不到 20% 拉到 90+%。
- **multi-target T-acc 天花板 ~0.19–0.24** 這個 frozen detector exact-match 硬上限，正好對應 GREC 論文 Fig. 3 中 multi-target 失敗案例的根因 ——bottleneck 在 false positive，不在 false negative，跟我 spark 上跑的數字一致。
- GREC 論文的「threshold-based output strategy」是最 naive 的 baseline，CRS 的 LTT 跨 detector 聯合校準正好是這個方向的「有理論保證」升級版。
- 本論文的 baseline 表（MDETR 42.7、UNINEXT 58.2 在 val Pr@F1=1）是我 CRS 主結果（α=β=0.3 下 3.21 boxes、CI 上界 < 0.3）對比的歷史參照點。

---

**讀後一句評**：GREC 是一篇「短小但有定義權」的論文 —— 它沒給你方法，但給了一把尺，從此以後 referring 領域不能只報 RefCOCO 85% 就說 done。
