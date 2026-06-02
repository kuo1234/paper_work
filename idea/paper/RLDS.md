---
type: paper-note
aliases:
  - "RLDS"
year: 2021
stage: "6-benchmark與資料"
tags:
  - dataset
  - data-format
  - ecosystem
  - 工程基礎
  - 6-benchmark與資料
summary: "RL資料標準格式（episode/step/metadata）；接Open X或robot trajectory dataset幾乎必經。"
---
> **論文標題**：RLDS: an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning
> **作者**：Sabela Ramos, Sertan Girgin, Léonard Hussenot, Damien Vincent, Hanna Yakubovich, Daniel Toyama, Anita Gergely, Piotr Stanczyk, Raphael Marinier, Jeremiah Harmsen, Olivier Pietquin, Nikola Momchev（Google Research / DeepMind）
> **發表**：arXiv:2111.02767, 2021
> **類型**：工具與生態系（Tooling / Ecosystem paper）
> **核心貢獻**：提出一個用於序列決策（SDM）資料集的標準化格式 RLDS、配套的記錄工具 EnvLogger（合成資料）與 RLDS Creator（人類資料），以及與 TFDS 的整合，讓資料集可以無損地產生、共享與重複使用。

---

## 0. 閱讀總覽（白話版）

這是一篇「工具/生態系」論文，沒有提出新的 RL 演算法，而是試圖解決一個社群層面的痛點：**RL 資料集格式太雜，每個人都有自己的存法，導致跨資料集的演算法評測非常痛苦**。作者觀察到，過去很多 offline RL / imitation learning 的資料集只存 (state, action) 配對（像監督式學習一樣），把時間結構整個丟掉了；即便有時間資訊，「什麼算一個 step、什麼算一個 episode」也各家不同，導致誤判和重工。

RLDS 的解法很簡單但很實用：
1. **資料結構**：把資料集表示成「episodes 的集合，每個 episode 內部又是 steps 的有序序列」這種雙層 nested dataset。每個 step 統一含有 observation、action、reward、discount，加上 is_first / is_last / is_terminal 三個旗標，並允許在 step / episode / dataset 三層都掛上任意 metadata。
2. **無損保存**：強調不要先預設後面要怎麼用，全部資訊（含時間順序）一律保留下來，所以同一份資料可以餵給任何演算法（單一 transition、n-step、整集 episode 都行）。
3. **工具鏈**：EnvLogger 用來把合成 agent 跑出來的軌跡寫進 Riegeli + protobuf 格式；RLDS Creator 是一個 Web UI，讓人類示範者透過瀏覽器操作各種環境收集人類資料。
4. **共享**：直接整合進 TensorFlow Datasets（TFDS），透過 `tfds.load(name)` 一行就能載入；TFDS 不託管資料本身，作者保留資料所有權。
5. **轉換 pipeline**：提供一系列 RL 專用的 tf.data transformations（batch、條件截斷、padding、對齊轉換等），讓使用者可以用同一套 pipeline 跑遍 [[D4RL|D4RL]]、RL Unplugged、Robosuite 等資料集。

對使用者來說最有感的點：**如果你想串 [[OpenX_Embodiment|Open X-Embodiment]] 之類大型 robot trajectory 集合，RLDS 幾乎是標配格式**——它定義了 robot dataset 在 TFDS 生態中的「規格」。讀這篇的價值不在於學一個新方法，而是學一個「資料怎麼組織才不會把自己卡死」的工程思維。

---

## Abstract

RLDS 是一個面向 SDM（包含 RL、Learning from Demonstrations、Offline RL、Imitation Learning）的資料生態系，涵蓋紀錄、回放、操作、標註與共享。它的設計目標有三：
- **可重現性**：讓既有研究可重做。
- **加速新研究**：用標準且無損的格式，讓新演算法能快速套到更多任務上。
- **共享便利**：不必管原始資料是什麼格式，pipeline 可重用。

工具面則提供合成與人類資料的收集、檢視、修改，並透過 TFDS 把資料推給整個社群。

---

## 1. Introduction

### 痛點

- 大多數 RL 演算法樣本效率差，因此社群轉而引入外部知識（LfD、IL、ORL），這些都需要 logged interactions 的資料集。
- 雖然有少數資料集逐漸變成標準（如 D4RL、RL Unplugged），但 RL 資料的用法是多面向的：
  - 低資料量（從單一子取樣軌跡做模仿）到高資料量（從大規模 ORL 資料集學）
  - 是否有 reward、是否只有 observation
  - 專家資料 vs 帶雜訊的示範
  - 狀態空間覆蓋率高低
  - 人類 vs 合成示範

### 跨資料集比較的工程成本

要比較「同一演算法在不同資料集」或「不同演算法在同一資料集」上的表現，必須一直寫 ad hoc 的資料 pipeline 把資料轉成演算法要的格式（例如 n-step transitions）。每個新資料集都要重做一次。

### 資料格式遺失資訊的問題

很多開源 RL 資料集是以「state/action 配對」形式發佈（像監督學習的 example/label）。雖然對某些演算法（行為克隆）方便，但這把原本資料中的時間資訊丟掉，導致那些需要時間相依性的方法（intrinsic reward from demos、self-attention credit assignment 等）都用不了。

即便保留了時間資訊，「step / transition / episode」的定義也不一致，會造成誤讀。

### 主張

SDM 社群需要一個標準格式：保留 RL 的時間特性，讓每個資料集可以跟各種演算法搭配（無論是消費單一 transition、n-step 序列、整集 episode），並讓 pipeline 可以重用。

### 本文具體貢獻

- 釋出 **EnvLogger** 與 **RLDS Creator**：分別產生合成與人類資料集，並支援標記、註解、檢視。
- 釋出 **RLDS 函式庫**：提供 RL 專用的轉換與 pipeline，可跨資料集重用。
- 釋出將資料集加入 **TFDS** 的工具。
- 把幾個常見 RL 資料集（D4RL、RL Unplugged）以 RLDS 格式加入 TFDS。
- 在 Robosuite 上釋出兩個新資料集（合成 + 人類），示範整個生態系。
- 釋出三個 Notebook 教學。

---

## 2. Dataset Lifecycle（資料集生命週期）

作者把生命週期切成三個階段：

### Producing the data（產生）
使用者透過 agent（人類或合成）與環境互動產生資料。重點：raw data 必須以**無損格式**保存，記錄全部資訊、保留 step 與 episode 的時間順序，**不對後續用途做任何假設**。

### Consuming the data（消費）
研究者用資料做分析、視覺化、訓練。不同演算法吃不同形狀（有的吃整集、有的吃隨機 batch of tuples）。所以要能輕鬆設定 pipeline，並能跨資料集重用同一條 pipeline。視覺化也很重要但 RL 社群常被忽略。

### Sharing the data（共享）
資料集產生成本高，共享有助於可重現性與新演算法的快速驗證。但簡化共享不能犧牲擁有者的所有權與引用權。

---

## 3. Related Work

### Dataset Standardization
OpenML、Kaggle 等提供資料集集中存放，但格式不一定標準。tf.data 與 TFDS 已經為 ML 建立資料 pipeline 規範；HuggingFace Datasets 為 NLP 建立標準。RLDS 是建立在 tf.data 與 TFDS 之上，**特化處理 episodic 結構**。

### Reinforcement Learning Datasets
SDM 領域目前沒有廣泛採用的標準。已釋出資料集多集中在機器人領域（D4RL 等）或遊戲（RL Unplugged、NetHack），人類資料部分有 Mandlekar 等人的工作。Offline Policy Estimation 也會受惠於標準化。

### Data Collection Systems
MechanicalTurk 用於眾包標註；RoboTurk 用於遠端操作機器手臂收資料，但有自己格式，要用到別處還是要寫轉換。

---

## 4. Dataset Structure（資料格式設計，本論文核心）

### 邏輯模型

一個 RL 資料集 = 一組（可能有序的）**episodes**；每個 episode = 一組 **steps**（可變長度有序序列）。Agent 可以是 RL 策略、規則控制器、形式規劃器、人類、動物等。

### 三大設計性質

1. **Lossless（無損）**：不對未來用途做假設，保留時間與 episodic 結構，讓資料集能配最廣的演算法。
2. **Uniform（一致）**：對最常用欄位（observation、action、reward）提供一致的存取方式與語意，使 pipeline 可重用。
3. **Flexible（彈性）**：允許在 dataset / episode / step 三層加自訂欄位（如生成 episode 的策略權重、環境視覺渲染）；資料欄位的 shape / type 不限（觀察可以是 ground-truth 特徵、影像、巢狀字典混合）。

### Step 標準欄位

每個 step 包含：
- `observation`：當前觀察
- `action`：套用到 observation 的動作
- `reward`：套用動作後得到的 reward
- `discount`：與 reward 對應的折扣

並包含三個布林旗標：
- `is_first`：是否為 episode 第一步
- `is_last`：是否為 episode 最後一步
- `is_terminal`：環境是否視此 step 為終止狀態（注意：time limit wrapper 可能讓 `is_last=True` 但 `is_terminal=False`）

在 `is_last=True` 的 step 中，只有 observation（與其 metadata）有定義。

### Metadata 機制

每個 step 與 episode 都可以額外加 metadata（任意欄位）。例如：
- 環境相關（statistics、rendering）
- 模型相關（hyperparameters、neural net logits、gradients）
- 標記用（tag、episode_id、agent_id）

### Step 對齊（alignment）

預設使用 **SAR 對齊**（State-Action-Reward 順序）：第一個 step 含 `observation, action, reward`，最後一個 step 只有 observation。

但其他文獻有不同對齊習慣，所以 RLDS 提供工具可以把欄位 shift 成 **RSA 對齊**（reward、next_observation、next_action）。在 RSA 下，第一個 step 的 reward / discount 是未定義，最後一個 step 反而 observation / reward / discount 都有定義。

也可以一鍵把 RLDS dataset 轉成 `(observation, action, reward, next_observation)` 這種 transition tuple。

### 圖 1（episode/step 結構）

```
Dataset
 ├─ Episode (+ episode metadata)
 │   ├─ Step (observation, action, reward, discount, + step metadata)
 │   ├─ Step ...
 │   └─ Step
 ├─ Episode (+ episode metadata)
 │   └─ ...
 └─ ...
```

關鍵概念：**dataset of episodes，每個 episode 內又是 nested dataset of steps**，這對 tf.data 來說是天然的兩層結構。

---

## 5. Generating Datasets（資料產生工具）

### 5.1 EnvLogger（合成資料）

- 包裝原本的環境，每次互動自動寫入長期儲存（disk）。
- 支援 step / episode / dataset 三層 metadata（神經網路 logits、gradients、環境 statistics 等任意輔助資料）。
- 用法是用 `with envlogger.EnvLogger(env, data_directory=...) as env:` 包住環境，之後 `env.reset()` 與 `env.step(action)` 都會自動 log。
- 設計成 standalone 也可用，並非與 RLDS 綁死。
- 只要環境實作 DM Env API（或被包裝成 DM Env 介面）就能用。

#### EnvLogger 內部技術細節（Appendix B）

資料用 protobuf 存到 **Riegeli** 格式：
1. 跨語言互通（protobuf 多語言可讀）
2. 開放格式，未來不被綁定
3. 二進位高效，內建壓縮
4. 適合 logging：append 快、配合 index 也能高效 seek
5. Serverless：不需獨立資料庫

### 5.2 RLDS Creator（人類資料）

- 一個 Web UI，讓人類示範者透過瀏覽器與 RL 環境互動。
- 支援 Atari、DMLab、NetHack、Procgen、Robodesk、Robosuite。
- 支援鍵盤與標準 Gamepad API 控制器（包括 SpaceMouse）。
- 概念基礎為「study」：研究者用線上編輯器設定環境與指示文，使用者註冊後可進行收集。
- 互動全部記錄（user action、environment transitions、rendered images、metadata），可被 RLDS 讀回。
- 支援 replay、step-by-step 檢視、tag 標記（如「success」、「failure」、「object picked」、「object placed」）。
- Client-Server 架構：server 處理環境模擬、client 顯示視覺與傳送 action，透過 WebSocket / WebChannel 通訊。
- 一般工作站上每個 step 的紀錄 overhead 約 0.5-2.0 ms（取決於影像解析度與環境複雜度）。

---

## 6. Sharing（透過 TFDS 共享）

RLDS 與 **TFDS** 整合，獲得四個關鍵好處：

### Data Ownership & Access Control
TFDS 不託管資料，只提供下載入口，作者保有完全控制權；description 內含 citation。

### Discoverability
資料集被加進 TFDS 後會出現在全域 catalog，任何人可用 `tfds.load(name_of_dataset)` 載入。私人資料集也可用同樣 API（自建 repo）。

### Efficiency & Flexibility
TFDS 為載入優化，支援多種設定（如 train/test split），輸出可為 TensorFlow 或 Numpy。

### Support for pre-existing datasets
TFDS 不依賴原始格式：即便資料集不是用 RLDS 工具產生，只要能組裝成 RLDS 相容結構就能加進來。

### TFDS 對 nested dataset 的支援
作者特地擴充 TFDS 讓它能載入巢狀的 episodic 結構。**目前限制：每個 episode 要能完整放進記憶體**（作者承諾未來移除這個限制）。

### 加入 TFDS 的流程
使用者寫一個 Python class 讀 raw data 並產出 RLDS dataset。如果 raw data 是 EnvLogger 格式，可以直接用 RLDS-TFDS integration 提供的基底，只要填欄位名稱與 shape 即可（詳見 Appendix D 完整範例）。

### 已加入 TFDS 的 RLDS 格式資料集
- D4RL（Gym Mujoco + Adroit 子集）
- RL Unplugged（DMLab、Atari、RWRL）
- 兩個 Robosuite 資料集（本論文新釋出）

### 兩個新釋出的範例資料集（Robosuite PickPlaceCan）
1. **合成資料**：SAC 訓練出的隨機策略生成 200 個 episodes，用 EnvLogger 紀錄。
2. **人類資料**：單一操作者用 RLDS Creator + gamepad 紀錄 50 個 episodes。

兩者都是固定 400 step 的 episode；任務完成時加上 tag（存在 step metadata 中），之後可用轉換把固定長度 episode 改成「截斷到 tag 那一步」的變長 episode。另外也存了 scene rendering 用於視覺化或評估。

episode metadata 包含 `episode_id` 與 `agent_id` 用於唯一識別。

---

## 7. Loading, Transforming & Processing（轉換 pipeline）

### 核心理念

RLDS 提供無損格式，但演算法很少直接吃 nested 結構。RLDS / TFDS 回傳 `tf.data.Dataset`，可直接用 numpy 或 tf.data pipeline 處理。RLDS 另外提供一個**為 RL 場景優化過的轉換 library**，常見功能：

- 跨資料集計算選定欄位的統計量
- 彈性 batch（尊重 episode boundary）
- 對 dataset 中所有 step 套用自訂轉換
- padding 函數與 `tf.zeros_like` 等價物（用於建構空 step）
- **條件式截斷**：碰到滿足某條件的 step 就截斷
- 對齊轉換工具

設計理念：不提供太高階的函數（保留彈性，例如 train split 怎麼分由使用者決定），只提供原語讓你快速搭。

### 已有的 RLDS 相容 agents
- Behavioral Cloning（在 Acme 中）
- ValueDICE

### 7.1 LfD / ORL 範例

典型 pipeline：
1. 用 `tfds.load` 載入 robosuite_panda_pick_place_can，shuffle 後 `take(K)` 做確定性子取樣。
2. 過濾觀察欄位（例如 concat `robot0_proprio-state` 與 `object-state`），用 `rlds.transformations.map_nested_steps` 處理 nested 結構。
3. 把 episode 轉成 transition：用 `rlds.transformations.batch(episode[STEPS], size=2, shift=1)` 拿到連續 2-step，再 map 成 `{s_cur, a, r, s_next}` 形式，最後 `flat_map` 把所有 episode 攤平。

重點：**RLDS 轉換和 tf.data 既有 API 互通**——train split 用標準 tf.data，nested 結構用 RLDS 的 `map_nested_steps`。

### 7.2 IL 中的 Absorbing State 範例

針對 GAIL 類演算法的 terminal state bias：
- 在 observation 加上 absorbing bit（terminal=1，否則=0）
- 用 `rlds.transformations.concat_if_terminal` 在 terminal 後加上一個自身的 transition
- 用 `apply_nested_steps` 與 `map_nested_steps` 完成這兩階段操作

### 7.3 Data Analysis 範例

對人類示範資料集計算「截斷到 placed tag 的 episode return」：
- `rlds.transformations.truncate_after_condition`：找到 `tag:placed` 非零的 step 後截斷
- `rlds.transformations.sum_dataset`：對截斷後 reward 求和

也示範了：載入 D4RL、RL Unplugged、Robosuite 三套不同來源的資料集，用同一條 pipeline `flat_map` 攤平、`map` 取 reward、畫 histogram——這是 RLDS 「跨資料集 pipeline 重用」的最強賣點。

---

## 8. Conclusion

RLDS 是 SDM 領域第一個資料集生態系，提供：
- 無損標準格式，保留 step 與 episode 間時間相依性
- 跨資料集可重用的 pipeline
- 常用 RL 轉換 library
- 與 TFDS 整合便利共享，作者保留所有權

訴求：提升可重現性、降低跨資料集研究的工程成本。

---

## 相關工作小結

- **資料標準化**：OpenML、Kaggle、TFDS、HuggingFace Datasets——RLDS 補上 SDM 領域的空缺。
- **RL 資料集**：D4RL、RL Unplugged、RoboNet、MAGICAL、Mandlekar 人類示範、Simitate、NetHack、Transporter Networks 等。
- **資料收集系統**：MechanicalTurk、RoboTurk（後者格式不通用，需手動轉換）。

---

## A. 與本研究主線的關聯

本研究主線：**robot play data + 極少弱語言標註 + offline meta-RL + 文字→task spec/embedding 做 zero-shot**，並關注 meta-learning × zero-shot 的並行路徑。

RLDS 對這條主線的關聯如下：

### 1. 跨資料集研究的工程基礎建設
主線需要混合多個 robot 資料集（可能含 play data、語言標註、unstructured demonstration 等異質性高的資料）。RLDS 的 lossless、uniform、flexible 三性質意味著：
- **無損保存**：play data 中那些「沒有明確 reward」、「沒有明確 task boundary」的微妙時間結構不會被預先壓縮丟失。
- **flexible metadata**：可以在 step / episode 層級掛上 task description、language instruction、language embedding、play 子片段標籤、demonstrator id 等。這對「文字→task embedding」這類研究非常關鍵——你需要把 language annotation 跟 trajectory 對齊存好。
- **pipeline 重用**：同一條 transformation pipeline 可以同時跑在 Robosuite、D4RL、RL Unplugged，這對 offline meta-RL「在多個資料集／多個任務間建構 meta-distribution」直接適用。

### 2. 接 Open X-Embodiment 幾乎是必經之路
Open X-Embodiment（OXE）是目前最大、最具影響力的 robot trajectory 集合，**OXE 整套就是用 RLDS / TFDS 發佈**（每個 sub-dataset 都是 RLDS-compatible，透過 `tfds.load` 取用）。如果主線想：
- 訓練一個跨 embodiment 的 robot foundation policy / world model
- 把 OXE 當作 offline meta-RL 的 task distribution
- 做語言條件 zero-shot 評測（多數 OXE 資料集都附語言 instruction）

那麼**讀懂 RLDS 的 episode/step/metadata 結構就是進入 OXE 的門票**。OXE 的 step metadata 裡會放 language_instruction、natural language embedding 等欄位，這些都是 RLDS metadata 機制的延伸。

### 3. Episode / Step / Metadata 應如何組織（落到主線實作）

對應主線需求，可以這樣設計：

**Step level**：
- `observation`：影像 + proprio + （若有）語言 embedding
- `action`：低階控制（end-effector pose 或 joint）
- `reward`：play data 多半沒有，可留 0 或填 sparse reward
- `discount`：通常 1.0，terminal 為 0
- `step metadata`：
  - `language_instruction`（string）：弱語言標籤，可能稀疏（只在 episode 開頭 / 子片段邊界出現）
  - `subgoal_tag`（string）：play data 中標出的 subgoal
  - `gripper_state`、`scene_id` 等輔助欄位

**Episode level**：
- `episode_metadata`：
  - `task_id` / `task_embedding`：對 zero-shot meta-RL 是 task spec 的來源
  - `language_full_instruction`：完整指令
  - `episode_id`、`agent_id`、`embodiment`
  - `success`（bool）：訓練 reward learner 或 filter 時用

**Dataset level**：
- 跨資料集的 metadata（語言模型、embedding model 版本、機器人型號等）

### 4. 對使用者實作的實用價值

- **不需自己造輪子**：要做 offline meta-RL 時，每個資料集只要實作成 RLDS 格式，後續的 sampling、shuffling、n-step batching 都用 RLDS transformations 與 tf.data 解決。
- **方便混合多個 task distribution**：meta-RL 需要 sample tasks，用 TFDS 一行 `tfds.load(name)` 就能讀，再用 `flat_map` 或 `interleave` 混合。
- **語言標註可掛 metadata**：弱語言條件不必塞進 observation tensor，可以走 metadata 路線，這樣 episode 才能保有「主要是 observation/action sequence、語言只是 side information」的清晰結構，方便日後切換不同語言 encoder。
- **可重現性**：用 RLDS 發佈自己的 dataset 對未來投稿很有利，reviewer 直接 `tfds.load(your_dataset)` 即可重現。
- **限制需注意**：原論文提到目前 TFDS 要求單一 episode 能放進記憶體；對長 horizon play data（幾萬 step 的一集）這可能是瓶頸，需設計 episode 切割策略，或追蹤後續 RLDS / TFDS 是否解除此限制。

### 5. 對 meta-learning × zero-shot 並行路徑的啟發
這篇本身沒談演算法，但它**強迫你把「task」這個概念實體化**到 metadata 中。當資料格式逼你說清楚「task_id 在哪一層？language 在哪一層？」就會自然把「task spec / task embedding」的工程介面定義出來，而這個介面正好是 zero-shot generalization 的關鍵——你需要一個結構化的方式去描述「新任務」，而 RLDS 的 episode metadata 就是一個自然容器。

---

## B. 一句話總結

RLDS 用「dataset of episodes，episode of steps，三層可自由掛 metadata」這套無損標準格式 + EnvLogger / RLDS Creator / TFDS 工具鏈，把 RL 資料的收集、共享、跨資料集 pipeline 重用一次做到位，是接入 Open X-Embodiment 與大型 robot trajectory 集合的事實標準格式。
