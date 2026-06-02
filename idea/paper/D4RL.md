---
type: paper-note
aliases:
  - "D4RL"
year: 2020
stage: "2-offline-rl基礎"
tags:
  - offline-rl
  - benchmark
  - dataset
  - 2-offline-rl基礎
summary: "offline RL的標準benchmark；資料覆蓋/品質/stitching屬性思維可用於robustness ablation。"
---
> **論文標題**：D4RL: Datasets for Deep Data-Driven Reinforcement Learning
> **作者**：Justin Fu、Aviral Kumar、Ofir Nachum、George Tucker、Sergey Levine（UC Berkeley、Google Brain）
> **出處 / 年份**：arXiv:2004.07219（v4，2021 年 2 月）
> **主題**：為 offline RL（離線強化學習）設計的標準 benchmark 與資料集套件
> **整理目標**：理解為何 offline RL 需要專門 benchmark、D4RL 的資料集設計原則、各任務域的特性、normalized score 的評分方式，並對照本研究主線（robot play data + 極少弱語言標註 + offline meta-RL + 文字→task spec/embedding 做 zero-shot 泛化）思考可借用之處。

---

## 0. 閱讀總覽（白話）

這篇論文不是提出新演算法，而是提出一個專門給 **offline RL（離線強化學習）** 用的標準 benchmark：D4RL。作者觀察到，過去 offline RL 評估幾乎都拿 online RL 訓練過程中產生的資料（部分訓練好的 policy 收集到的 buffer）來當「離線資料集」，這跟真實世界裡會遇到的資料分布完全不像。真實資料可能來自人類示範、手寫控制器、用戶被動記錄、多任務混合，這些通通沒被既有 benchmark 覆蓋。

於是 D4RL 整理了一組涵蓋導航（Maze2D、AntMaze）、運動（Gym-MuJoCo）、靈巧操作（Adroit）、家事多任務（FrankaKitchen）、交通流（Flow）、自駕（CARLA）等領域的資料集，並對每個資料集刻意設計不同「難點屬性」：窄/偏分布、未定向多任務資料、稀疏 reward、次優資料、無法用策略類表達的非馬可夫行為、部分可觀測等。每個任務都附 simulator 以便重現評估，並提出 normalized score 把不同環境的成績拉到 0–100 的尺度，方便跨任務比較。

實驗部分把當時主流的 offline RL 方法（BC、BCQ、BEAR、BRAC、AWR、AlgaeDICE、[[CQL|CQL]] 等）全部跑一遍，結論是：當資料只是 RL 訓練過的策略產生時，多數方法表現都不錯；但只要進入 undirected / 多任務 / 混合 policy / 人類示範 等更真實的設定，幾乎所有方法都出現明顯缺陷。這個 benchmark 因此揭露了既有方法被低估的問題，也指出未來研究的方向。

對研究主線（offline meta-RL + 弱語言 + zero-shot）而言，D4RL 的價值不在演算法，而在於 **資料品質 / 涵蓋度與 benchmark 設計如何決定方法的可比性**——這正是構造 robot play dataset、衡量 meta-RL 在 unseen task 上 zero-shot 表現時的基本框架。

---

## 1. 為什麼 offline RL 需要專門的 benchmark？

過去 offline RL 評估的三個主要問題：

1. **資料來源單一**：絕大多數既有工作都用 online RL 訓練軌跡（隨機初始 policy → 部分訓練 → 近專家）當作離線資料。這種資料分布雖然可以驗證「處理 distribution shift」的能力，但離真實世界的離線資料分布相去甚遠。
2. **缺乏標準評估協定**：每篇論文自選任務、自調超參、自定報告方式，導致方法間難以直接比較。Wu et al.（2019）指出在簡單任務上，最 naive 的基線跟新方法表現幾乎沒差，benchmark 本身分辨不出方法強弱。
3. **真實世界資料無法當公開 benchmark**：真實系統難以重現、政策評估需要重新採資料或仰賴 off-policy evaluation（如 NeurIPS 2017 Criteo 挑戰賽），但 OPE 的變異仍太大、最佳方法常常和基線統計上不可分。

因此一個合格的 offline RL benchmark 必須：(a) 涵蓋反映真實應用的挑戰，(b) 廣為可得且有清楚的評估協定，(c) 難度梯度足以分辨方法強弱。D4RL 同時用高品質模擬器避開「真實世界資料不可重現」的痛點，又在資料收集端引入多種真實世界式的程序（人類示範、手寫控制器、被動記錄）來逼近真實分布。

---

## 2. D4RL 的設計關鍵屬性

作者把要刻意挑戰 offline RL 的資料/任務屬性逐項拆解，這節是 D4RL 對社群最重要的貢獻。

### 2.1 Narrow and biased data distributions（窄而偏的資料分布）
來自確定性策略、人類示範、手寫控制器的資料分布往往很窄，沒有覆蓋整個狀態-動作空間。當演算法在這種資料上想往資料分布外推估價值，Q 容易發散（理論上 Munos 2003、Kumar 2019 已證明）。常見處理是保守策略（限制 policy 接近 behavior policy），但這也代表方法很容易「向中庸靠攏」，難以超越行為策略。

### 2.2 Undirected and multitask data（未定向且多任務的資料）
這是 D4RL 最強調的真實場景。被動記錄的資料（如用戶網路行為、行車紀錄）並不是為了當前任務而蒐集的，但其中**片段**仍對學習有用。論文點出 **stitching**（拼接）能力：若資料含 A→B 與 B→C 軌跡，演算法應能將兩段組合出 A→C 的解。能否 stitching 是評估 offline RL 真正泛化能力的核心指標，Maze2D / AntMaze / FrankaKitchen / CARLA 都針對此性質而設。

### 2.3 Sparse rewards（稀疏 reward）
傳統 RL 在稀疏 reward 下要兼顧 credit assignment 與探索。offline RL 因為不能再探索，反而成為孤立檢驗 **credit assignment** 能力的好場景。AntMaze 用 0/1 sparse reward、Adroit 也是稀疏 reward。

### 2.4 Suboptimal data（次優資料）
真實資料常常並非最佳策略所產生。Imitation learning 在這種資料上會直接學壞。Gym-MuJoCo 的 random / medium / medium-replay / medium-expert 四種變體就是專門量測這種混合資料下的方法強度。

### 2.5 Non-representable / non-Markovian behavior policies、partial observability
當資料來自人類或手寫控制器，行為策略可能根本無法被當前 policy class 表達（non-representable），或本質上記憶了歷史狀態（non-Markovian），又或是部分可觀測。這些情況會讓「重要性加權」等假設 Markovian behavior policy 的方法產生額外偏差。Maze2D / AntMaze 的 waypoint 規劃器、CARLA 的視覺輸入都是這類例子。

### 2.6 Realistic domains（現實感）
為了讓 benchmark 可重現，作者選擇社群已驗證的高品質模擬器：MuJoCo、Flow、CARLA、Adroit、FrankaKitchen。並在多個域引入人類示範或行為模型（如 IDM 交通流模型）來逼近真實資料生成程序。

---

## 3. 涵蓋的任務領域

D4RL 共七個域，每個都對應上述屬性的子集：

### 3.1 Maze2D（非馬可夫 policy、未定向多任務資料）
2D 球體在迷宮中到達固定目標。三種尺寸（umaze / medium / large）。資料由 waypoint planner + PD 控制器產生，目標隨機重設後繼續走，刻意產出未定向軌跡，用來測 stitching。控制器記憶了走過的 waypoint，因此 behavior policy 是非馬可夫的。

### 3.2 AntMaze（非馬可夫 + 稀疏 reward + 未定向多任務）
把 2D 球換成 8-DoF 的 Ant 四足機器人。三種版本：固定起點到固定目標、diverse（隨機起終點）、play（人工挑選的子目標序列）。reward 為 0/1。形態複雜得多，因此即便 Maze2D 容易解，AntMaze 仍可能難。

### 3.3 Gym-MuJoCo（次優資料 + 窄分布）
Hopper / HalfCheetah / Walker2d 三隻動物。資料集分四種：random（隨機策略）、medium（SAC 訓到一半的 policy）、medium-replay（訓到 medium 為止整個 replay buffer）、medium-expert（一半 expert + 一半次優混合）。是過去文獻最常用的設定，D4RL 為一致性也納入。

### 3.4 Adroit（非可表達策略 + 窄分布 + 稀疏 reward + 現實）
24-DoF Shadow Hand 進行四種任務：釘釘子、開門、轉筆、抓球放置。三類資料：human（25 條人類示範）、cloned（用示範訓 imitation policy + 與示範各半混合）、expert（fine-tuned RL policy）。重點考驗：人類示範的 narrow 分布、稀疏 reward、以及高維表徵學習。

### 3.5 FrankaKitchen（未定向多任務 + 現實）
9-DoF Franka 機器人在含微波爐、水壺、櫃子、燈、烤箱的廚房中操作多個物件。目標是達到某種多物件狀態組合。三類資料：complete（依序完成全部目標）、partial（含可解任務的子集）、mixed（沒有任何軌跡完整解任務，必須拼接子軌跡）。這個域對 stitching + 泛化要求最高，因為軌跡是複雜路徑而非簡單導航，方法必須對未見狀態泛化。

### 3.6 Flow（非可表達策略 + 現實）
交通流控制（環狀道路、合流道路），自駕車控制以最大化車流。資料包含 IDM（智慧駕駛模型）模擬人類駕駛產生的 human 資料、與隨機加速度的 random 資料。

### 3.7 Offline CARLA（部分可觀測 + 非可表達策略 + 未定向多任務 + 現實）
高擬真自駕模擬器，輸入是 48×48 RGB 第一人稱影像。兩個任務：八字型車道跟隨、小鎮內導航。資料來自手寫啟發式控制器（避車、保持車道、十字路口隨機轉彎），用來測在視覺輸入 + 部分可觀測下的 stitching。

### 3.8 摘要表
- Maze2D / AntMaze / FrankaKitchen / CARLA：主打 **stitching**、未定向多任務
- Adroit：主打人類示範 + 高維操作
- Gym-MuJoCo：傳統次優資料設定
- Flow / CARLA：現實感與行為模型

---

## 4. Normalized Score 評分方式

D4RL 提出統一的標準化分數：

> normalized score = 100 × (score − random score) / (expert score − random score)

- **0** = 隨機策略平均回報（100 episodes）
- **100** = 該域專家平均回報

各域「專家」定義不同：
- Maze2D / Flow：手寫控制器
- CARLA / AntMaze / FrankaKitchen：作者估計的可能最大分數
- Adroit：BC 預訓 + RL fine-tune 後的 policy
- Gym-MuJoCo：SAC（online）訓練到收斂的 agent

這套標準化讓不同任務的成績可以平均後比較，但也要注意有時實際 raw score 與 normalized score 落差很大（附錄 Table 3）。

**評估協定**還規定：把每個域的任務拆成 training / evaluation 兩組，超參數只能用 training 任務調，evaluation 任務完全不准 tune。這直接對抗了過去 offline RL 普遍依賴 online 超參搜尋的問題，把報告貼近真實離線部署情境。

---

## 5. 主要實驗發現

作者跑了 BC、SAC（online / offline）、BEAR、BRAC-p、BRAC-v、AWR、BCQ、cREM、AlgaeDICE、CQL。發現可歸納為：

1. **RL-trained 資料上 offline 方法多半 OK**：Adroit-expert、Gym-MuJoCo 的多數設定下方法能匹敵或超越行為策略。
2. **稀疏 reward 設定下 offline RL 表現亮眼**：Adroit、AntMaze 多數方法贏過 online SAC，顯示 offline RL 對克服探索難題有潛力。
3. **保守型方法（BEAR、BCQ、AWR、CQL）**在窄/偏分布（Flow、Gym-MuJoCo）上表現好。
4. **未定向資料是真痛點**：Maze2D（尤其 large）、FrankaKitchen、CARLA、AntMaze（medium / large）幾乎只有 CQL 有合理表現，其他方法多半接近 0。
5. **混合策略 (medium-expert) 並沒讓多數方法變得更強**——即便資料中含 expert，方法表現仍逼近 medium，說明它們無法挑出最佳子分布。
6. **小量人類示範**（Adroit-human、Kitchen-human）仍是極大挑戰；提示需要更 sample-efficient 的方法。
7. **carla-town、antmaze-large** 接近全軍覆沒，但作者透過軌跡覆蓋圖（附錄 F、G）驗證資料其實有足夠覆蓋度、任務理論上可解。

---

## 6. 為什麼資料品質 / 涵蓋度對 offline RL 那麼重要？

D4RL 之所以非提不可，是因為 offline RL 的方法表現對資料分布**極度敏感**：

- 資料是 **窄而偏** → 方法必須學會謹慎、不越界，否則 Q 發散
- 資料 **未定向（undirected）** → 方法必須能拼接子軌跡 (stitching)，而非依賴模仿
- 資料 **次優** → 方法必須能挑出好的部分超越行為，又不能假設專家存在
- 資料 **非可表達** → 任何依賴 importance weighting 或重建 behavior policy 的方法都會崩
- 資料 **稀疏 reward** → credit assignment 變得獨立於探索可被測試

換句話說，**benchmark 的資料設計決定了方法之間誰能被分辨開**。若 benchmark 全部使用 online RL 訓練資料，會給社群一種「現有方法都差不多好」的錯覺；D4RL 透過刻意設計多樣資料屬性，揭露了被既有評估隱藏的方法缺陷。

---

## 7. 結論與展望

論文總結：D4RL 是第一個系統性把「真實離線資料挑戰」帶進 offline RL benchmark 的工作；它讓社群可以分辨方法、定位問題、看見進展。作者點出幾個未涵蓋的真實面向：環境隨機性（金融、醫療、廣告）、大動作空間（推薦系統）、其他領域（金融、工業、運籌）。並重申長遠目標是把 offline RL 推往真實資料 + 可靠 OPE 的世界。

---

## 與本研究主線的關聯

本研究主線是：**用 robot play data + 極少弱語言標註 + offline meta-RL，學一個能從文字 task spec 映射到 task embedding、對未見任務做 zero-shot 泛化的 agent**，並關注 meta-learning × zero-shot 並行方向。D4RL 雖然不直接做 meta-RL 也不處理語言，但對本主線在幾個層面非常關鍵：

### (1) Robot play 資料 ≈ D4RL 的 undirected / multitask 資料
所謂 robot play（如 Lynch 等人系列工作中機器人自由探索家中物件）幾乎完美對應 D4RL 提的 **undirected and multitask data**：軌跡不指向特定 evaluation task、但子片段含豐富技能成分。D4RL 的 **FrankaKitchen-mixed** 是最接近 play 資料的公開 benchmark——「沒有任何完整解任務的軌跡，需要 stitching + 泛化」。本研究的 offline meta-RL 想做的事，本質上就是 **在 play data 上學 stitching + 條件化於語言/任務描述**，所以 Kitchen-mixed 是極好的初步測試床。

### (2) Dataset quality / coverage 對 offline meta-RL 更關鍵
D4RL 顯示 offline RL 對資料分布極端敏感。**offline meta-RL** 在資料維度上又多一層：除了 (s, a, r, s') 的覆蓋，還需要 **任務分布 (task distribution)** 的覆蓋。如果 play data 只覆蓋少量技能基元，meta-policy 在 unseen task spec 上就無法 zero-shot。設計自己的 dataset / 用 D4RL 既有資料時，應顯式檢查：
- 子技能（pick、push、open、turn 等）的多樣性
- 任務組合的多樣性（multi-object configurations）
- 軌跡是否足以 stitching 出 unseen task 所需的序列

### (3) FrankaKitchen 是「弱語言標註」的天然候選
Kitchen 任務本質上是**子技能組合**（開微波爐 + 把水壺放上爐 + 開燈），這正好對應自然語言可以描述的細粒度動作序列。本研究的「極少弱語言標註」設定可以直接套用：在 Kitchen mixed dataset 上對少量軌跡片段標註自然語言（"opens the microwave then turns on the light"），訓練文字→task embedding 映射，再用 zero-shot 評估在未見組合上的成功率。

### (4) D4RL 的評估協定可直接借用
- **normalized score**：跨任務取平均，方便 meta-RL 報告
- **train/eval task split**：與 zero-shot 評估天然契合——meta-train 用一組 task spec，meta-test 用另一組從未見過的 spec
- **不准用 online tuning**：強迫超參完全 offline 選定，與「部署到 unseen task」的 zero-shot 精神一致

### (5) D4RL 揭露的方法缺陷正是 meta-RL 的研究機會
D4RL 顯示：保守型方法（CQL、BCQ）在 stitching 任務上仍掙扎；混合資料中無法挑出 expert 子集；小量人類示範極難利用。對應到本主線：
- 若加入語言條件化，模型也許能用語言當「先驗」分辨資料中哪些片段對應目標 task → 解決混合資料挑優問題
- meta-learning 結構若能在 task 之間共享，也許比單任務 offline RL 更能解決 stitching（用同一個策略對多個 task 條件化）

### (6) 限制
D4RL 沒有：(a) 任務間的明確語意關係，(b) 自然語言標註，(c) 真正的 meta-RL split。本研究若要對接 D4RL，須自行對任務做語言/embedding 標註（如把 Kitchen 4 子任務的所有 24 種排列當不同 task），並建構 meta-train/meta-test 切分。

---

## 一句話總結

D4RL 是為 offline RL 量身打造的 benchmark，透過刻意涵蓋 undirected、稀疏、次優、人類示範等真實資料屬性，揭露既有方法在 stitching 與多分布資料上的根本缺陷——對 offline meta-RL + 語言條件化的研究而言，它既是現成的測試床（特別是 FrankaKitchen 與 AntMaze），也是「dataset quality 決定方法可比性」這一核心觀念的最佳教材。
