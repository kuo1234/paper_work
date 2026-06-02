---
type: paper-note
aliases:
  - "CALVIN"
year: 2022
stage: "6-benchmark與資料"
tags:
  - benchmark
  - long-horizon
  - language-conditioned
  - play-data
  - 主要評估平台
  - 6-benchmark與資料
summary: "最契合benchmark：play+只標1%語言+長時程+嚴格zero-shot（新指令+新環境）；MCIL baseline 5連鏈僅0.08%。"
---
# CALVIN 論文逐段詳細整理

> 論文：**CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks**
> 作者：Oier Mees, Lukas Hermann, Erick Rosete-Beas, Wolfram Burgard（University of Freiburg / University of Technology Nuremberg）
> 出處：IEEE Robotics and Automation Letters (RA-L) 2022（arXiv:2112.03227v4）
> 專案頁：http://calvin.cs.uni-freiburg.de
> 主題：Language-Conditioned Policy Learning、Long-Horizon Manipulation、Play Data、Hindsight Language Relabeling、Zero-shot 泛化評估、Benchmark
> 整理目標：說明 CALVIN 的環境/資料/挑戰設計、長時程語言控制為何難、評估協定與 metric、baseline（MCIL）結果，並分析為何它是使用者「robot play + 弱語言 + zero-shot 任務規格化」主線最合適的 benchmark。

---

## 0. 閱讀總覽：這篇在做什麼？

CALVIN（Composing Actions from Language and Vision）是一個**開源模擬 benchmark**，目標是訓練 agent「只靠人類語言指令，從機載感測器，連續完成長時程機械臂操作任務」。它的三個賣點正好命中使用者的題目：

1. **資料就是 play + 極少語言標註**：含約 24 小時遙操作的「unstructured play data」，外加約 20K 語言指令，但**只有 1% 的互動資料被語言標註**（刻意模擬真實世界「無法把所有經驗都配上語言」的情境）。
2. **長時程 + 組合泛化**：把 34 個子任務串成 5 連續任務鏈，agent 要連續看懂一串無約束語言指令並逐一完成。
3. **明確的 zero-shot 評估**：最難的設定是「在 3 個環境訓練、在第 4 個沒看過的環境測試」，且測試用的語言指令不在訓練集裡（描述任務的新講法）。

它的 baseline（MCIL，即 Lynch et al. 的 play-based 多情境模仿學習）在 CALVIN 上表現很差——短時程單任務最高 53.9%，但連做 5 個指令的成功率掉到 0.08%。這個「baseline 跑很爛」本身就是論文的論點：長時程語言控制還有巨大改進空間。

---

## 1. Abstract 重點

與人類共存的通用機器人，必須學會把人類語言關連到自身的感知與動作。它們需要習得多樣的通用技能，以便「依無約束語言指令」組合出長時程任務。CALVIN 是一個開源模擬 benchmark，用於學習長時程語言條件任務。相較既有 vision-and-language 任務資料集，CALVIN 在序列長度、動作空間、語言複雜度上都更難，並支援彈性的感測器組合。作者用 zero-shot 方式評估 agent 對新語言指令與新環境的泛化，並顯示一個基於 multi-context imitation learning 的 baseline 在 CALVIN 上表現不佳，意味著有很大的創新空間。

---

## 2. Introduction 重點

長期目標是建立「用自然語言指定任務」的機器人。語言提供人類一種直覺的方式去表示、摘要、抽象多樣知識；透過抽象，像「打開抽屜並把中間的物體推進去」這種概念能延伸到無限多新的、沒見過的實體。人類也用概念把複雜任務描述成「一連串語言指令」。

相對地，現有機器人通常缺乏這種泛化能力、一次只學一個任務；而傳統 multi-task 學習常假設測試時用 **goal image** 或 **one-hot skill selector** 指定任務——這些對「非專家使用者」在真實日常環境下指揮機器人並不實用。隨著機器人普及，**直覺式的 task specification**（用語言）需求增加。

CALVIN 連結語言到機器人運動技能、行為、物件，於互動視覺環境中。單一 agent 必須連續理解一串無約束語言表達來解任務，且要能以任意順序執行任意子任務組合。它含約 24 小時遙操作 play data + 20K 語言指令。play 的好處是 task-agnostic、多樣、相對便宜。為了開啟「把抽象概念泛化到新實體（像人一樣）」的未來研究，論文加入一個有挑戰性的 zero-shot 評估：在大量 play 語料（涵蓋三個環境）上訓練、在沒看過的場景測試，測試語言指令也不在訓練集裡。

---

## 3. CALVIN 的三大組件

CALVIN = **Environment + Dataset + Challenge**。

### 3.1 CALVIN Environment

四個結構相關但不同的環境 A/B/C/D（可做泛化評估）。每個含一支 7-DOF Franka Emika Panda 機械臂、一張有滑門與抽屜的桌子、可切換綠燈的按鈕、控制燈泡的開關、三個不同顏色形狀的方塊。為評估語言 grounding 的泛化能力，**四個環境的紋理不同、靜態元件（滑門、抽屜、按鈕、開關）位置都不同**，但桌子、機器人、靜態相機位置一致。物理用 PyBullet 模擬（支援 GPU 平行渲染、大規模平行收資料）。

**觀察空間**：static + gripper 兩個相機的 RGB-D、視覺式觸覺影像、proprioceptive state（末端位置/姿態、夾爪寬度、關節位置、夾爪動作）。**動作空間**：支援絕對 cartesian pose、相對 cartesian displacement、joint action 三種（鼓勵社群研究彈性的觀察/動作空間組合）。控制頻率 30 Hz、閉環連續控制。

**34 個任務**：環境能自動偵測序列中完成了哪個任務——**任務完成準則定義為「序列首尾環境狀態的變化」**（例如「方塊向右移動超過 10 cm 且首尾都有接觸面」）。這個設計很關鍵：因為可把環境 reset 到任一錄製 frame 的狀態，所以能對**任意長度的離線資料自動偵測任務**，這正是後面 hindsight 語言標註的基礎。

### 3.2 CALVIN Dataset

**Unstructured play demonstrations**：請三位未受訓、且不知道下游任務的使用者，用 HTC Vive VR 頭盔各收約 6 小時，共約 24 小時（約 2.4M 互動步、約 40M 短時程 window，每個 1–2 秒）。唯一指引是「探索環境、不要把物體掉到桌外」。這種 play 是 free-form、無類別標籤、含探索與次優行為——這些對學到可泛化、robust 的表徵（如**重試行為 retrying**）很關鍵；而專家示範通常只展示一種解法。

**Language instructions（極少標註是重點）**：不像多數方法完全仰賴 crowd-sourced 標註，CALVIN 收集 400+ 句 crowd-sourced 指令對應 34 個任務，再**用錄製的環境狀態「程序化地」標註 episode**（procedural labeling，只有展現有意義技能的序列才會被標語言）。為模擬真實世界「無法把所有經驗配語言」，**只標註 1% 的互動資料**。另外提供用 MiniLM 預先算好的語言 embedding（詞表 30,522、把任意長句映成 384 維向量）。

> 附錄 B 細節：自動語言標註是隨機抽 64-frame window，用 task detector 檢查首尾之間是否解了某任務、且前半段沒解任何任務（為了把「動作前的移動行為」也含進去，例如開抽屜前手臂要先移到把手）。每個任務約 11 種同義講法，全部共 389 句獨特指令。

### 3.3 CALVIN Challenge（評估設定）

三種訓練/測試環境組合，難度遞增：

- **Single Environment**：同一環境訓練與測試（即 Lynch et al. 的設定）。
- **Multi Environment**：四環境訓練、其中之一測試（要泛化到多紋理、不同位置的元件）。
- **Zero-Shot Multi Environment（最難）**：三環境訓練、第四個沒看過的環境測試。policy 從沒見過測試環境，但場景元素曾以不同位置出現在訓練環境。對應「服務機器人要在沒看過的日常房間也能用」的期待。

---

## 4. 評估 Metric（read.md 重點）

兩種 metric：

- **Multi-Task Language Control (MTLC)**：最簡單，驗證多任務語言 policy 對 34 個任務的泛化。把模擬器 reset 到一個合法的「未見示範」的初始狀態（確保指令可行），每任務做 10 次 rollout。**測試用的語言指令不在訓練集裡**（任務的新講法）。
- **Long-Horizon MTLC (LH-MTLC)**：驗證 policy 能否「連續完成多個語言指令」。把 34 任務當子目標，組出 5 連續任務的合法序列，過濾掉不可行/循環/相似的序列後得 **1000 條獨特指令鏈**。每完成當前任務（由環境狀態指示器判定成功）才轉到下一個子目標。每條序列後把機器人 reset 到中性位置（避免初始姿態洩漏任務資訊——這個「中性初始化」刻意打斷初始狀態與任務的相關性，逼 agent 完全靠語言推論任務）。

### 為什麼 long-horizon language control 比單步難很多？

1. **誤差累積**：要連做 5 個任務，每一步的小失誤會累積，且只有完成當前任務才能進下一個——這就是為何 baseline 從單任務 ~54% 暴跌到 5 連鏈 0.08%。
2. **子目標轉換**：agent 必須學會在不同子目標間平順切換，而非只解一個。
3. **語言 grounding + 感知 grounding 同時要對**：要把「blue block」對到多模態觀察裡正確的物體（論文發現 baseline 會把 red/blue 方塊搞混，因為語言模型把含 red 與 blue 的句子編得太相似）。
4. **task-agnostic 連續控制**：底層還要解決多樣技能習得、物件偵測、referring expression、action grounding。

---

## 5. Baseline 模型與結果

**MCIL（Multi-Context Imitation Learning）**：基於 play data 的通用 goal-reaching policy。假設有一條未分段的遙操作 play 資料流 `D = {(x_t, a_t)}`，用 hindsight relabel：把每個造訪過的 state 當「達成的 goal」，前面的 state-action 當「達成該 goal 的最優行為」。得到 `D_play = {(τ, x_g)}`，用最大概似 goal-conditioned 模仿目標：

\[
\mathcal{L}_{LfP} = \mathbb{E}_{(\tau, x_g)\sim D_{play}}\Big[\sum_{t=0}^{|\tau|}\log \pi_\theta(a_t \mid x_t, x_g)\Big]
\tag{1}
\]

MCIL 用一個 seq2seq CVAE 的 latent「plan」空間，處理 free-form 模仿資料固有的多模態：decoder 是 policy，conditioned on state、goal、以及 latent plan `z`。**關鍵洞見**：對「goal image 或 language goal」學單一模仿 policy，能讓控制大多從**無標註 play data** 學到，把語言標註負擔降到全資料的 **<1%**。形式上：給多個情境模仿資料集 `D = {D_0,…,D_K}`（每個用不同方式描述任務），MCIL 同時訓練一個 latent-goal-conditioned policy `π_θ(a_t | x_t, z)`，並每個資料集配一個 encoder。

**實作細節**：Adam、lr `10^{-4}`、KL 權重 `β = 0.001`、隨機抽 16–32 長度 window 補到 32、不做影像增強、用絕對 cartesian 動作。

**結果（Figure 8）**：
- 最佳 MTLC（static camera, D→D）成功率 **53.9%**；加入 gripper camera / depth / tactile 後相當（沒大幅提升）。
- LH-MTLC（D→D）連做 5 指令成功率僅 **0.08%**；連第 1 個子任務的成功率 48.9%、第 2 個就掉到 12.9%。
- 換到 Multi 與 Zero-Shot Multi 設定時，整體表現顯著下降——因為違反了「訓練/測試任務獨立同分布」的模仿學習假設。
- 定性觀察：表現高度依賴機器人初始位置（暗示 agent 靠 context 而非真的解耦初始狀態與任務，可能是 proprioceptive 與目標動作之間的 causal confusion）。

**作者建議的改進方向**：domain adaptation 技術、更好的資料增強、更重視 depth（對紋理變化不變）、視覺-語言對齊的 auxiliary loss（如 CLIP 式）、mixture-of-experts / view-invariant contrastive 的多模態融合。並指出 MCIL 是離線方法，naïve 跨域資料共享可能 brittle、加劇 distribution shift。

---

## 6. 與本研究主線的關聯（重點章節）

使用者主線：**robot play data + 極少量弱語言標註 + offline meta-RL + 文字→task spec/embedding 做 zero-shot 任務泛化**，關注 meta-learning × zero-shot。CALVIN 幾乎是為這條線量身打造的 benchmark。

### 6.1 為什麼 CALVIN 是最合適的 benchmark

1. **資料形態完全對上**：CALVIN 本身就是 **play data + 只標 1% 語言**——這正是使用者「robot play + 極少弱語言標註」的設定，不必自己造資料。使用者可直接拿 CALVIN 驗證「在極少語言標註下，文字→task embedding 對齊是否成立」。
2. **內建嚴格 zero-shot 評估**：CALVIN 的 Zero-Shot Multi Environment（3 訓 1 測）+「測試語言不在訓練集」，剛好對應 research_direction_options.md 定義 B 的嚴格 zero-shot（無 target demo / 無 gradient / 無 env interaction），且同時涵蓋「unseen instruction（新講法）」與「unseen environment」兩種泛化軸。
3. **可自動偵測任務 → 可程序化標註/relabel**：CALVIN 能對任意長度離線序列自動判定完成了哪個任務，這讓 hindsight 語言標註與 task latent 對齊度診斷變得可行（呼應 innovation_notes 第 3 節 true latent supervision / 第 7 節 actor-z sensitivity 的診斷思路）。
4. **[[LUMOS|LUMOS]] 等 play+language+world-model 方法也在此類長時程語言控制場景評估**，所以用 CALVIN 能與最近鄰工作公平對比。

### 6.2 CALVIN 的 baseline 反而凸顯使用者的機會

MCIL 在 CALVIN 上的慘烈分數（5 連鏈 0.08%）說明：**純 play-based 模仿學習對長時程語言控制不夠**。這正是使用者方向的賣點空位——把問題從「language-conditioned imitation」升級成「offline meta-RL + 文字當 task specification」，可望改善：MCIL 把語言當 goal embedding 直接 condition，但沒有「對任務做推論 / belief」的機制；使用者的 `g(l) ≈ q(τ)` 對齊（[[VariBAD|VariBAD]]/[[T2DA|T2DA]] 式）正是要補上「語言→task latent→policy」這層結構化中介。

### 6.3 可直接借用的零件 / 注意事項

- **MCIL hindsight relabel + <1% 語言標註的協定**：直接是使用者「極少弱語言」設定的可用範本。
- **MiniLM 預算 language embedding**：現成的文字 encoder 起點（呼應 T2DA「text encoder 可換 CLIP/BERT/T5」的 robustness 發現）。
- **34 任務的狀態式成功準則 + 1000 條指令鏈**：現成、客觀、可重現的評估協定，使用者不必自訂 success metric。
- **警訊**：CALVIN baseline 顯示語言模型會把 red/blue 編得太像而搞混顏色——提醒使用者「弱語言對齊」時，視覺 grounding 與語言 grounding 要一起對，否則 task embedding 會混淆相似指令（呼應 [[CSRO|CSRO]] 的 context shift 與 [[CORRO|CORRO]] 的 robust representation）。

---

## 7. 一句話總結

CALVIN 提供「play data + 只標 1% 語言 + 長時程連續控制 + 嚴格 zero-shot（新指令 + 新環境）」的開源 benchmark，而它的 play-based 模仿 baseline 在 5 連續指令上幾乎全垮（0.08%）——這同時證明了使用者選的 benchmark 對題目高度契合，也凸顯「把 language-conditioned imitation 升級為 offline meta-RL + 文字當 task specification」這條路的明確改進空間。
