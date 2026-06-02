---
type: paper-note
aliases:
  - "Open X-Embodiment"
  - "RT-X"
  - "OpenX"
year: 2023
stage: "5-zero-shot與generalist"
tags:
  - dataset
  - cross-embodiment
  - standardization
  - 資料基礎
  - 5-zero-shot與generalist
summary: "整合多機構多embodiment資料、標準化格式，展示跨平台正遷移；大型預訓練資料基礎。"
---
# Open X-Embodiment: Robotic Learning Datasets and RT-X Models — 精讀筆記

> 「Can we instead train generalist X-robot policies that can be adapted efficiently to new robots, tasks, and environments?」
>
> 本文集結 21 個機構、22 種機器人本體、60 個既有資料集，統一為單一 [[RLDS|RLDS]] 格式的 Open X-Embodiment (OXE) Dataset，並訓練 [[RT1|RT-1]]-X 與 [[RT2|RT-2]]-X 兩個跨本體 Transformer 策略，以實證跨本體共訓 (cross-embodiment co-training) 能帶來顯著正遷移。

---

## 0. 閱讀總覽（白話）

這篇是「資料 + 生態系」型論文，而不是新演算法論文。核心訊息可以拆成三層：

1. **資料的問題**：機器人界一直缺少像 ImageNet、Common Crawl 那種大規模、多樣化的資料。每個實驗室自己收的資料都太窄（單一機器人、單一環境、單一任務），單獨訓練模型很難泛化。
2. **解法 — 把全世界既有的機器人資料合起來**：作者群把 21 個機構手上現存的 60 份資料集，全部轉換成「統一的 RLDS 格式 + 對齊到 7-DoF end-effector 動作空間」，得到超過 100 萬條真實機器人軌跡、涵蓋 22 種不同形體的機器人（從單臂到雙臂到四足）。這就是 Open X-Embodiment (OXE) 資料集。
3. **驗證 — RT-X 模型**：拿現成的 RT-1（35M Transformer）與 RT-2（55B VLA）這兩個架構幾乎不改，直接用 OXE 的 9 種機器人資料 co-train，得到 RT-1-X 與 RT-2-X。結果是：(a) 小資料 domain 上 RT-1-X 平均比原作者方法高 50% 成功率；(b) 大資料 domain 上 RT-2-X 比單一 domain RT-2 表現出約 3 倍的「emergent skills」遷移（在 Google Robot 上能做 Bridge/WidowX 才有的技能）。

對主線研究的核心啟示：**這篇證明了「不同機器人形體的資料可以一起餵給同一個模型，並產生正遷移」**，這對任何後續想做大規模機器人預訓練、跨任務泛化、零樣本遷移的人來說，是一個「資料基礎建設」級別的貢獻。它沒解決 meta-learning、沒解決 zero-shot task spec，但它把「資料」這個前提鋪好了。

---

## Abstract

- 點出 NLP / CV 都因為「大模型 + 大而多樣的資料」走向 general-purpose pretrained backbone，但機器人界沒有這個條件：傳統做法是「每個應用、每個機器人、每個環境訓一個模型」。
- 提問：能不能訓練一個「generalist X-robot policy」，可以高效率地適配新機器人、新任務、新環境？
- 給出兩個產物：
  1. 標準化資料格式的多本體機器人資料集；
  2. RT-X 模型 — 高容量模型在這個資料上訓練後，**展現正遷移**，能藉由其他平台的經驗提升多個機器人的能力。
- 規模：22 種機器人、21 個機構合作、527 種技能、160266 個任務。

關鍵詞：positive transfer, X-embodiment training, standardized data format。

---

## I. Introduction

### 動機與類比

- 大規模、多樣化資料是 NLP / CV 成功關鍵 — CLIP、LLMs 都是好例子。
- 機器人的窘境：
  - 單一 domain 的資料太窄；
  - 不像 vision/NLP 可從網路爬巨量資料；
  - 即使最大的機器人資料蒐集也只是 vision (5–18M) / NLP (1.5B–4.5B) 的零頭；
  - 多半還只變化某一個軸（單環境、單物件集、單任務範圍）。

### 核心立場

- 「Goal of training generalizable robot policies requires X-embodiment training」 — 想做泛化策略，必須跨形體訓練。
- 「個別資料集都太窄，但聯集就有覆蓋度」 — 多機構聯集的價值。
- 即便目前合起來仍遠小於 LLM 訓練語料，作者主張 **現在就要建立 X-embodiment 研究的基礎**，因為這是未來大型機器人預訓練的必要起點。

### 兩個明確目標

1. **驗證**：跨多機器人/多環境訓練的策略，是否真的比只用自己 domain 資料訓練的策略更強（正遷移）？
2. **建設**：把大量機器人資料整理成統一格式，讓社群能在此基礎上做 X-embodiment 研究。

### 本文焦點

- 限定在 manipulation；
- 拿 RT-1、RT-2 「幾乎不改」直接套用到 9 個機械臂的資料，得 RT-X；
- 重點不是新架構，而是「把模型 + 資料 + 工具一起釋出，推動 X-embodiment 研究」。

---

## II. Related Work（簡述）

三條主線：

1. **跨形體遷移方法**：模組化策略、共享動作表徵、表徵學習目標、把本體資訊條件化進策略、機器人 / 環境表徵解耦等。也有 transformer-based generalist agent（Gato、RoboCat）等先例。
2. **大規模機器人資料集**：抓取、推、teleop demo 等。RoboNet 是少數跨機器人型號的；本研究與這些是互補的——把多份既有資料 **聚合 + 標準化** 成單一 repository。
3. **語言條件策略**：透過模仿學習做 language-conditioned policy；本研究跟著 RT-1 / RT-2 的 recipe，使用預訓練語言 embedding 與 VLM。

差異化：本研究**不引入任何專門縮小 embodiment gap 的機制**，直接訓 — 仍然觀察到正遷移。

---

## III. The Open X-Embodiment Repository

### A. Open X-Embodiment Dataset — 資料整合與標準化

關鍵數字：
- 1M+ 真實機器人軌跡；
- 22 種機器人本體（單臂、雙臂、四足都有）；
- 來自 21 機構、34 lab 的 60 份既有資料集；
- 統一轉成 **RLDS 格式**（serialized tfrecord），可容納各種不同的觀察模態（多攝影機數量、深度、點雲）與動作空間。
- 支援主流深度學習框架的高效平行載入。

> 「the union of all such datasets provides a better coverage of variations in environments and robots.」 — 個別資料窄、聯集才廣。

### B. 資料集分析

- Franka 是出現在最多資料集中的機器人；對應地，Franka 的「distinct scenes」最多樣。
- xArm 與 Google Robot 因為幾份大型資料集，貢獻最多 trajectories。
- 技能用 PaLM 從語言標註中抽取，大多落在 pick-place 家族，但長尾包含 wiping、assembling 等多樣動作。
- 物件範圍從家電到食物到餐具都有。

### 為什麼資料標準化重要（這篇沒明寫得很白，但實際上是核心貢獻）

- 60 份原本各自為政的資料：不同攝影機數、不同動作空間、不同 control rate、不同檔案格式。
- 統一成 RLDS + 對齊到 7-DoF end-effector：研究者只需要寫一份 dataloader 就能跨所有資料訓練。
- **這是研究可重現性與「擴大規模」的前提**：之後任何人都能直接在 OXE 上嘗試新架構，不必先花半年清資料。

---

## IV. RT-X 模型與跨平台正遷移

### A. Data format consolidation（觀察 / 動作對齊）

「Coarsely aligned」的對齊：
- **觀察**：每個資料集挑一個 canonical 相機視角，resize 到共通解析度；模型輸入：近期影像歷史 + 語言指令。
- **動作**：統一為 7-DoF end-effector 動作（x, y, z, roll, pitch, yaw, gripper 開合或速率），離散化前每個資料集動作各自做 normalize；output 解釋（de-normalization）依使用的本體不同。
- **不對齊**：座標系、絕對 / 相對 / 速度的選擇，皆保留各機器人原樣 — 同一個 action 向量在不同機器人上可能產生不同運動。

關鍵設計理念：**不強行消除本體差異，讓模型自己從資料中學會處理**。這是「不需特殊機制就有正遷移」說法的物理基礎。

### B. Policy architectures

兩個架構（皆直接沿用，幾乎不改）：

1. **RT-1**（35M 參數，Transformer）
   - 輸入：15 張影像歷史 + 自然語言；
   - 影像走 ImageNet-pretrained EfficientNet；語言走 USE embedding；用 FiLM 互相交織成 81 個 vision-language tokens；decoder-only Transformer 輸出離散化動作。
   - 動作離散成 256 個 bin × 8 維（7 維 end-effector + 1 維 termination）。

2. **RT-2**（large VLA，本文用 PaLI-X 變體 55B；亦有 5B 變體）
   - 把動作直接編成 text tokens（例：「1 128 91 241 5 101 127」）；
   - 任何預訓練 VLM 都能 fine-tune 成機器人控制器；
   - 視覺骨幹 ViT、語言骨幹 UL2，預訓於 WebLI。

### C. 訓練與推理細節

- 兩者皆用 categorical cross-entropy。
- 「Robotics data mixture」 = 9 個機械臂資料（RT-1, QT-Opt, Bridge, Task Agnostic Robot Play, Jaco Play, Cable Routing, RoboTurk, NYU VINN, Austin VIOLA, Berkeley Autolab UR5, TOTO, Language Table）— 此為實驗時的 OXE 子集，整個 OXE 已成長到 22 本體。
- RT-1-X：只用上述 robotics mixture；
- RT-2-X：與原 RT-2 一樣做 co-fine-tuning，VLM 原資料 : robotics 約 1:1。
- 推理：RT-1 本地 3–10 Hz；RT-2 透過雲端服務。

---

## V. 實驗結論

要回答三個問題：(1) 跨本體共訓能否帶來正遷移？(2) 共訓能否提升對新任務的泛化？(3) 模型大小、架構、資料組成如何影響？

總計 3600 次評估，6 種機器人。

### A. In-distribution 表現

切成兩種情境：

1. **小規模 domain（Fig. 4）** — 該機器人原本只有少量資料
   - RT-1-X 在 5 個小資料 domain 中有 4 個贏過原作者方法；
   - 平均成功率比 Original Method 高 50%；
   - 結論：**資料少的 domain 從共訓中受益最大**。

2. **大規模 domain（Table I：Bridge、RT-1 paper）** — 該機器人本身就有大量資料
   - RT-1-X **輸給** RT-1 baseline — 35M 模型容量不足，呈現 underfitting；
   - 但 RT-2-X（55B）能贏過 Original Method 與 RT-1。
   - 結論：**大資料 domain 也能從跨本體共訓中得利，但需要足夠大的模型容量**。

### B. 對 out-of-distribution 設定的泛化（用 RT-2-X）

1. **未見物件 / 背景 / 環境（沿用 RT-2 的 generalization 測試）**
   - RT-2 與 RT-2-X 表現相當（Table II 行 (1) vs (2) 最後欄）— RT-2 本身就靠 VLM backbone 泛化得不錯。

2. **Emergent skills（重點實驗）**
   - 在 Google Robot 上測一批技能，這些技能 **不存在於 Google Robot 的原訓練資料中**，但出現在 Bridge（WidowX）資料；
   - RT-2-X 比 RT-2 表現約 3 倍（75.8% vs 27.3%）；
   - 拿掉 Bridge 後（行 (3)）：emergent skills 表現大幅下降（42.8%），印證「是 WidowX 的資料把這些技能「灌」進去 Google Robot 的控制」；
   - **結論**：跨形體共訓真的能讓某個平台「學會」其他平台才有的技能。

### C. 設計選擇消融（Table II）

- **影像歷史**：加短歷史顯著提升泛化（行 (4) vs (5)）。
- **Web pre-training**：對大模型至關重要（行 (4) vs (6)）— 從頭訓練幾乎廢掉。
- **模型容量**：55B 在 emergent skills 上遠勝 5B（行 (2) vs (4)）；更大容量帶來更高遷移能力。
- **Co-fine-tuning vs fine-tuning**：跟 RT-2 原論文不同，這次兩者在 emergent / generalization 表現相當 — 作者歸因為 OXE 的機器人資料比之前用的更多樣，原本 co-fine-tuning 的優勢被攤平。

### 小結

- 共訓能正遷移；
- 容量決定能不能在大資料 domain 也受益；
- Web 預訓練、影像歷史、夠大的模型是 RT-2-X 泛化的關鍵組合；
- 跨形體資料能帶入「emergent skills」。

### 開放問題（作者自陳的限制）

- 沒涵蓋感測 / 致動模態差異很大的機器人；
- 沒研究對「全新機器人」的泛化；
- 沒給出「何時會正遷移、何時不會」的決策準則。

---

## 相關工作（補充已在 II. 提過外的脈絡）

- 跨形體：模組化策略、shared action repr、hardware-conditioned policies、graph-based body 表徵、Polybot、Gato、RoboCat。
- Human-to-robot 轉換、affordance 從人類影片學、視覺表徵預訓（R3M、MVP、VIP、Voltron）。
- 抓取大資料：QT-Opt、Dex-Net、Jacquard、GraspNet-1B 等。
- 語言條件策略：CLIPort、SayCan、[[CALVIN|CALVIN]]、PerceiverActor、VIMA、PaLM-E、VoxPoser。
- RLDS 格式（Ramos et al., 2021）：本研究的格式基礎。

---

## 與本研究主線的關聯

主線方向回顧：robot play 資料 + 極少弱語言（weak language）+ offline meta-RL + 文字 → task spec / embedding 做 zero-shot；關注 meta-learning × zero-shot 的並行軸。OXE / RT-X 對這條主線的具體意義可以分四點：

### 1. 為什麼 cross-embodiment data 對主線是「基礎建設」級別的支撐

- 主線的瓶頸之一是：**meta-training 的任務分佈不夠大、不夠多樣**，導致 meta-test 時對「全新但語意接近」的任務泛化差。
- OXE 把 22 種本體 / 527 種技能放進同一格式 — 等於提供了一個「任務分佈天然非常寬」的池子。任何 offline meta-RL / 多任務預訓練的工作，都可以把 OXE 視為「現成的、跨形體的、含語言標註的大型任務池」。
- 對主線中「robot play」這一塊：OXE 內部就含 Task Agnostic Robot Play、Bridge、Jaco Play 等 play-style 資料 — 主線提案直接可以用這些子集 + 自己採的 play 資料一起 pretrain，省去重新建構跨機構資料的成本。
- RT-X 結果 — 「不必特意處理 embodiment gap，光是混在一起就能正遷移」— 對主線的 implication：**就算未來收集到的 play 資料來自不同機器人，也不必先做複雜的對齊就能 co-train**，只要走類似的 coarse alignment（共通 7-DoF + canonical view）即可起步。

### 2. Dataset standardization 對研究可重現性的意義

- 在 OXE 之前，比較「跨機構資料」的研究幾乎不可能可重現：每個 lab 的資料格式、動作定義、camera 設定都不同。
- OXE 提供：統一 RLDS、統一動作空間、canonical 相機視角、normalize 過的動作分佈。**這意味著主線提案要做的 offline meta-RL 基線比較，可以直接用 OXE 任意切子集**，重現別人的 setting 也只是改一段 dataset config。
- 對「文字 → task spec / embedding 做 zero-shot」這支：OXE 每個 trajectory 都已附帶語言指令（雖然弱、雖然短，但有），這正好對應主線中「極少弱語言」的設定 — 不必自己重標。
- 換句話說：**OXE 把「不同 lab 的資料能互相比較」這件事從根本上解決了**，這是可重現性最關鍵的前提。

### 3. 未來大型資料預訓練的基礎

- RT-X 雖然「只」是直接套用 RT-1 / RT-2 架構，但已展示了 **scale × diversity** 的 payoff（55B 才能吃進大資料、3× emergent skills）。
- 主線提案如果要走「大型 offline meta-RL 預訓練 → meta-test 在新任務 zero-shot」，OXE 幾乎就是首選底料：
  - 資料量級 1M+ trajectories；
  - 多本體 = 強烈的 task / context 變化；
  - 已標語言 = 可以做 language → task embedding 的對齊預訓練；
  - 與 VLM (RT-2) 的整合路徑已被驗證 — 主線想做 「文字 task spec」可以直接接 VLM-based action decoder。

### 4. 對「meta-learning × zero-shot」並行軸的直接啟發

- RT-2-X 的 emergent skills 結果，本質上就是一種 zero-shot：**Google Robot 從未看過某些技能，但靠著 WidowX 在這些技能上的資料 + 共享語言條件，就能執行**。
- 這暗示主線設定中「文字 → task spec」這個橋樑可能比想像中更強：當訓練資料涵蓋夠多元本體 / 任務，**語言指令本身可能就足以充當 task spec / embedding**，模型靠語言去 retrieve 它在其他 embodiment 上學到的技能。
- 對主線要小心的地方：RT-2-X 是「大模型 + 大資料 + 強 web pretrain」三者齊備才有此能力。如果主線目標是用「弱語言 + 少量 play 資料」做到類似事，需要自問——是否能透過 meta-learning 的歸納偏置、補上 scale 不足的劣勢？

### 5. 與 PEARL / VariBAD 等 meta-RL 風格的對比

- [[PEARL|PEARL]] / [[VariBAD|VariBAD]] 都假設 task 分佈是 narrow、parametric（如不同 reward function、不同物件位置），meta-test 仍在類似分佈內；
- OXE 提供的是「broad、非參數化」的任務分佈，更接近真實世界 generalist 的需求；
- 主線若要把 meta-RL 提升到 OXE 規模，可能需要重新設計 task posterior 的形式（從 latent z 改為 language token、或 retrieval-based context），這是值得探索的方向。

---

## 一句話總結

Open X-Embodiment 把 21 機構 60 份機器人資料集統一為 RLDS 格式、跨 22 種本體共百萬軌跡，並用 RT-1/RT-2 直接共訓出 RT-1-X / RT-2-X 證明「不加任何 embodiment gap 機制就能跨形體正遷移」—— 為未來大規模機器人預訓練、可重現的 X-embodiment 研究、以及跨平台 zero-shot 技能轉移鋪下資料基礎建設。
