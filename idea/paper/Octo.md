---
type: paper-note
aliases:
  - "Octo"
year: 2024
stage: "5-zero-shot與generalist"
tags:
  - generalist-policy
  - open-source
  - diffusion-head
  - backbone候選
  - 5-zero-shot與generalist
summary: "開源模組化generalist policy，diffusion action head、彈性接observation/action space；最適合當backbone。"
---
# Octo: An Open-Source Generalist Robot Policy

> Octo Model Team (UC Berkeley, Stanford, CMU, Google DeepMind)
> Robotics: Science and Systems (RSS) 2024
> 關鍵詞：generalist robot policy、transformer、diffusion action head、[[OpenX_Embodiment|Open X-Embodiment]]、cross-embodiment、開源預訓練、可彈性 finetune
> 個人標籤：可作為「研究 backbone」候選 / 系統型論文 / 直接決定上層 task inference 工程量

---

## 0. 閱讀總覽（白話）

Octo 是一個「開源、通用、可以接到很多種機器人身上」的 manipulation 預訓練 policy。它的賣點不是新的演算法理論，而是把幾個工程選擇做對：

1. **核心是一個 transformer**：把語言 / 目標圖 / 觀察影像都切成 token，丟進 transformer，再用 readout token 抽出來餵 action head。
2. **action head 用 diffusion**：因為 demonstration data 本來就有多模態（同一情境可能有多種合理動作），diffusion 比 MSE 或 discretized 都來得貼合分佈。
3. **block-wise causal attention**：observation 只能往前看自己時間步以前的 token 與 task token；readout token 只「讀不寫」，像 BERT 的 [CLS]。這個設計讓「之後想多加一個 camera、多換一種 action space」變得便宜——加新 token、加新 head，預訓練的權重幾乎都能保留。
4. **資料**：精選 Open X-Embodiment 中 25 個資料集共 800k trajectories，是當時最大的 manipulation 預訓練組合。
5. **開源**：27M 與 93M 兩個 checkpoint、JAX 訓練 / finetune pipeline、data loader 全部釋出。

對研究者來說，Octo 的價值不在於「最強」，而在於它是一個「可預期、可改、可重現的起點」。特別是它把「換 observation / 換 action space」這件事制度化了——這直接影響：若把 task inference / meta-RL 放在它上層，到底還要寫多少底層 adapter 程式。

---

## Abstract

- 主張：在多樣機器人資料上預訓練的大型 policy，有望讓下游 task 只需要少量資料 finetune 就能達到不錯效果。
- 痛點：要當「通用」policy，必須能應付不同 sensor、不同 action space、不同 robot、能在合理算力內 finetune。
- 提供物：Octo，transformer-based policy，預訓練於 Open X-Embodiment 的 800k trajectories；可以用 language 或 goal image 下任務；可以在幾小時內、消費級 GPU 上 finetune 到新觀察 / 新動作空間。
- 實驗範圍：9 個機器人平台。Octo 既可當 zero-shot 的多機器人 controller，也可當 finetune 的初始化。並有大量 design ablation。

## Introduction

- 傳統作法：為某機器人某任務從零訓 policy → 資料貴、泛化窄。
- 借鏡 NLP / CV 的大模型路線：希望有一個「general-purpose robot model」。但機器人多樣性（embodiment、sensor、action space、task spec、環境、算力）使得這比 NLP / CV 更難。
- 既有 generalist robot policy（GRP）的限制：
  - 觀察輸入被預先綁死（例如只能單一 camera）。
  - 不支援有效率的 finetune 到新環境。
  - 最大的模型不開放權重。
- Octo 的設計目標：任意 input token → 任意 output token，可彈性接 task 描述（language / goal image）、可換 camera 設定、可換機器人、可換動作空間，**只要加 token / 加 head，不必重訓 backbone**。
- 三大貢獻：
  1. 把 transformer backbone + 語言 / 目標圖兩種 task spec + diffusion action head 結合到大型 cross-embodied 預訓練上（個別元件不新，但組合與規模新）。
  2. 在 9 個機器人 / 4 個機構的實驗中達到 SOTA 級 out-of-the-box 多機器人控制，並可作為 finetune 初始化。
  3. 完整開源（pipeline、checkpoint、資料 loader），並提供詳盡 ablation。

## 方法與架構

### A. 整體架構

Octo 由三段組成：
1. **Tokenizer**：把語言指令 `ℓ`、目標圖 `g`、觀察序列 `o₁…o_H` 各自打成 token 序列 `T_l, T_g, T_o`。
2. **Transformer backbone**：吃整段 token 序列，吐 embedding `e_l, e_g, e_o`。
3. **Readout heads**：取 readout token 的 embedding 做 action 解碼。

#### Task / Observation tokenizer

- **語言**：先做標準 tokenization，丟到 pretrained T5-base（111M），取輸出當成 language token sequence。
- **影像（觀察與目標圖共用）**：通過一個輕量 CNN「patch 化」，再 flatten 成 patch token（ViT 風格）。
- 全部 token 加上可學的 position embedding，再依 `T_T, T_o,1, T_o,2, …` 順序排好。

#### Transformer backbone 的 attention 結構（關鍵設計）

- 採 **block-wise masked attention**：
  - Observation token 只能 causally attend 到同一或更早時間步的 obs token，以及所有 task token。
  - 不存在的觀察（例如該資料集沒有 wrist camera、沒有 language）會整塊 mask 掉。
  - 加入 learnable **readout token** `T_R,t`：能 attend 到先前的 obs / task token，但**沒有任何 token 會 attend 它**——只讀不寫，像 BERT 的 `[CLS]`。
- Action head 只接 readout token 的 embedding。這個切割帶來兩個好處：
  - **可加新 input**：要多塞一種觀察（例如 force-torque）？多開一塊 token、給它 position embedding 與一個小 encoder，其他預訓練權重不動。
  - **可加新 output**：要換一個 action space（例如 joint position 而非 EE delta）？多開一個 readout head，pretrain 的 backbone 一樣不用動。
- 對比過去把 ResNet 視覺 encoder + transformer fuse 在一起的設計：那種設計「鎖死」了觀察的種類與順序，要改就要重新 init。

### B. 訓練資料

- 從 Open X-Embodiment 約 1.5M episodes 中挑出 25 個資料集、共 **800k trajectories**（比 RT-X 的 350k 還大）。
- 篩選原則：
  - 必須有影像觀察。
  - 必須是 delta end-effector 控制。
  - 排除過於重複、解析度太低、過於小眾的任務。
- 加權策略：手動分成「較多樣」與「較不多樣」兩組；前者權重 ×2；資料筆數多但內容重複的則降權。
- 標準化：
  - 缺失的相機 channel 補零。
  - gripper 動作統一語意：+1 = 開、0 = 關。
- 任務描述：每筆訓練樣本隨機把 language 或 goal image 「歸零」一邊，讓模型能單獨吃 language 或單獨吃 goal image。沒有語言標註的資料集則一律用 goal image conditioning。Goal image 採 hindsight relabeling（從未來軌跡中隨機抽一張當 goal）。

### C. 訓練目標：diffusion action head

- 用 conditional diffusion 直接 decode 連續、多模態的 action 分佈。
- Backbone 在每個 action prediction 只跑一次 forward；diffusion 的多步 denoising 是在一個輕量的 head 內做完。
- 形式（DDPM 標準）：取 Gaussian noise `x_K`，迭代 K 步：
  - `x_{k-1} = α(x_k − γ ε_θ(x_k, e, k) + N(0, σ²I))`，其中 `e` 是 action readout 的 transformer 輸出。
- noise schedule 用 cosine（Nichol & Dhariwal）；訓練時用標準 DDPM loss。
- 預測「action chunk」：一次預測未來若干步 action（沿用 ACT、Diffusion Policy、ALOHA 的 chunking 想法）。
- 為何贏過 MSE 與 discretized cross-entropy：
  - MSE 無法表達多模態。
  - Discretized 失去連續精度。
  - Diffusion 同時兼顧多模態與連續精度。
- Finetune 時：沿用同一 diffusion loss、**整個模型一起更新**（凍結部份參數會表現較差）。Finetune 統一配方：~100 demos、50k steps、cosine decay + linear warmup。

### D. 訓練細節

- 兩個尺寸：
  - Octo-Small ≈ ViT-S（27M）。
  - Octo-Base ≈ ViT-B（93M）。
- AdamW、inverse square root LR schedule、weight decay 0.1、grad clip 1.0。
- Octo-Base：batch 2048、TPU v4-128 pod、300k steps，約 14 小時預訓練。
- 單卡 A5000（24GB）finetune 約 5 小時。
- 觀察 history：用 2 幀（再多收益遞減）。
- 訓練時隨機 dropout language 或 goal image；資料增強為常見影像 aug。

### E. 開源內容

- Octo-Small（27M）與 Octo-Base（93M）checkpoint。
- JAX 的 finetune 與 pretrain pipeline。
- 與 JAX / PyTorch 相容的 Open X-Embodiment data loader。

---

## 實驗與結論

實驗圍繞三個問題：
1. Octo 能否 out-of-the-box 控制多種機器人，並同時支援 language 與 goal 兩種任務描述？
2. Octo 當作 finetune 初始化好不好？比 from-scratch、比 pretrained vision encoder（VC-1）強嗎？
3. 哪些設計選擇對 GRP 最關鍵？

評估涵蓋 9 個 setup / 4 個機構，包含 zero-shot（WidowX BridgeV2、UR5 Tabletop、[[RT1|RT-1]] Robot）與 finetune（Berkeley Insertion 多 force-torque、Berkeley Pick-Up 換 joint position、Berkeley Coke、Berkeley Bimanual、Stanford Coffee、CMU Baking）。每個 finetune setup 約 100 demos、< 5 小時 / A5000、**共用同一組超參數**。

### A. Out-of-the-box 多機器人控制（Fig. 5）

- 對手：RT-1-X（35M，同樣以 Open X 預訓練）、[[RT2|RT-2]]-X（55B，VLM-based）。
- Octo（93M）平均比 RT-1-X 高 29% 成功率（語言任務）。
- 在 WidowX 與 RT-1 Robot 上，Octo 與 55B 的 RT-2-X 相當。
- 用 goal image 比用 language 在 WidowX 上再高 25%——因為 goal image 帶更多狀態資訊。
- BridgeV2 細究：對新物體成功率高、對新場景略降、對新行為（翻轉、精細插入）退化大。

### B. 資料效率 Finetune（Table I）

平均成功率：
- ResNet+Transformer Scratch：20%
- VC-1 pretrained 視覺表徵：15%
- Octo：72%

特別說明：
- Berkeley Insertion 多了 force-torque 輸入（**新觀察 modality**）：Octo 70% vs Scratch 10%、VC-1 5%。
- Berkeley Pick-Up / Bimanual 用 joint position（**新 action space**）：Octo 60% / 80%。
- 同一組超參數跨所有 setup → 對研究者很重要，少了一輪繁複調參。

### C. Design ablation（Table II、Fig. 6）

在 WidowX 上跑：
- **Architecture**：ViT-first（shallow CNN patch + 大 transformer）= 83%；ResNet-50 + transformer = 70%。但在「from scratch / 小資料」場景下，ResNet 反而比 ViT 好——表示大 ViT 是專屬「大資料」的工具。
- **資料規模**：Octo 自選 25 datasets（800k）= 83%；RT-X 的 11 datasets mix = 60%；只用 Bridge 單機器人 = 43%。資料越廣越強。
- **Action 表示**：diffusion = 83%；continuous MSE = 35%；discretized = 18%。
- **Model scale**（Fig. 6）：Tiny（10M）< Small（27M）< Base（93M）。Base 對初始 scene 變動更穩、較不會早抓。

### Discussion 與目前缺點

- Wrist camera 的處理仍弱：finetune 時有時只用第三人稱比加 wrist 還好（資料中只有 27% 含 wrist）。
- Language-conditioned 比 goal-conditioned 落後（資料中只 56% 有語言標註）。
- 目前只覆蓋單臂 / 雙臂 manipulation，沒含 navigation、mobile manipulation。
- 純 imitation、純 optimal demo；未來可擴到 sub-optimal、online interaction（這對 offline meta-RL 來說很有意義）。

---

## 相關工作

- 大資料 + transformer 的 single-embodiment imitation：RT-1、Diffusion Policy、ACT、Perceiver-Actor 等。Octo 從單一 embodiment 跳到 cross-embodiment。
- 借助 VLM / pretrained 視覺：RT-2、PaLM-E、VoxPoser、VC-1 等。Octo 不依賴 huge VLM，而是把 generalisation 賭在大量機器人資料上。
- Cross-embodied generalist：
  - GNM / ViNT（navigation）。
  - RoboCat（goal-conditioned 多 embodiment）。
  - RT-X（語言條件多單臂）。
  - 這幾家共同弱點：把 input modality / action space 鎖死、最大模型不公開。Octo 的差異是「更大資料 + 可改 IO + 全開源」。
- 設計靈感：
  - DDPM 與 Diffusion Policy（action 用 diffusion）。
  - Action chunking（ACT、RoboAgent）。
  - Scaling ViT 的 LR schedule 與 architecture。

---

## 與本研究主線的關聯

我的研究主線：robot play data + 極少弱語言標註 + offline meta-RL + 把「文字 / 弱 instruction」轉成 task spec / task embedding，做 zero-shot 任務泛化；並同時關注 meta-learning × zero-shot 的並行。Octo 對這條線的關係很直接：

### 1. Octo 是不是合適的 backbone？

**很合適，原因有四：**

1. **預訓練資料分佈正好對齊「play / 弱結構 demo」**：Open X-Embodiment 雜亂、多 embodiment、多場景、語言註記稀疏（只 56% 有 language），這跟「robot play + 極少弱語言」的資料形態幾乎一致。也就是說 Octo 在「沒有乾淨語言」這件事上已經吃過苦——它的權重不會強烈預設「每筆軌跡都該有清楚指令」。
2. **可只用 goal image conditioning**：訓練時隨機 drop language 或 goal image。對「沒有語言標註的 play」，只給 goal image 一樣能跑。這對我的 zero-shot 想法極友善：可以把「文字 → goal image」「文字 → readout embedding」當作上層 task inference 的輸出形式之一。
3. **diffusion action head + chunked prediction 對「行為多樣的 play data」很友善**：play 本來就充滿多模態行為，diffusion head 能保留這種分佈，比 MSE / discretized 強。這正是 Table II 想說的事。
4. **全開源**：權重、pipeline、loader、JAX & PyTorch loader 都有，可實驗、可改、可重現。對學術研究而言這比「最強閉源 GRP」重要。

**保留意見**：Octo 對 wrist camera 處理偏弱、語言條件表現比 goal-image 弱、且全部是 optimal demos。我的研究若要強調「弱 / 多義語言 → task spec」這條，可能需要：(a) 在 task spec 端自己疊一層 language-aligned embedding，不完全相信 Octo 原生的 T5-token 條件；(b) 接受 wrist camera 不可靠，先用第三人稱為主。

### 2. Observation / action space adaptation 它到底做了什麼？

Octo 把「換 IO」當作一級設計，具體機制：

- **新觀察**：因為 block-wise attention，可以多開一塊 token block（例如 force-torque）。流程：
  1. 為新觀察設計一個 light encoder（FT 6 維 → 一段 token）。
  2. 加上新的可學 position embedding。
  3. backbone 的權重 **完全不動**，繼續讓 readout token 去 attend 這塊新 token。
  4. 整體 finetune（diffusion loss）；50k steps、~100 demos。
- **新動作空間**：開一個新的 readout head（例如 joint position 7 維、bimanual 14 維）。原 head 可以保留也可以丟棄，backbone 不變。
- **新 embodiment**：把對應的 obs encoder / action head 換掉，token mask 改一下，等於同一個 transformer 同時服務新 robot。

「不必重訓 backbone」這點意義在於：之前像 RT-1 / RT-2 的 ResNet-fusion 架構，要改觀察種類就得拆 visual encoder 重來。Octo 的 token-塊狀化使這個工程成本掉到「寫一個 encoder + 新 position embedding」的程度。

### 3. 若把研究放在 Octo 上層做 task inference，可以省多少工程？

可以省下的部分（佔比是粗估，但能省下整個研究 60-70% 的「底層 perception → control 工程量」）：

- **不用自己搭 cross-embodiment 預訓練**：800k 軌跡的訓練成本（TPU v4-128 × 14 小時）和資料 curation 是不可能在實驗室層級重現的，直接取 checkpoint 就好。
- **不用自己挑 visual encoder + 設計多 modality fuse**：影像 patch embedding、language token、proprio、未來新加的 FT，都有 reference 實作。
- **不用自己處理 action chunking + diffusion head**：直接複用，等於免費獲得「行為多模態建模」。
- **不用自己處理 multi-task action space alignment**：Octo 已經統一 gripper 語意、補零缺失 channel 等繁瑣工程。
- **finetune recipe 已標準化**：100 demos、50k steps、同一組超參 → 不用花時間調 hyper。

**仍須自寫的部分**（這就是我的研究真正的差異化）：

1. **Task inference / task embedding 模組**：給一段（可能殘缺、模糊的）語言描述或一小段 play 軌跡，輸出某種 task spec。Octo 自己沒有 meta-learning、沒有 task posterior，這正是我要做的事。
2. **接口設計**：怎麼把上層 task embedding 注入 Octo？三個可行方案：
   - (a) 把 task embedding 視為新的「task token」拼到 `T_T` 序列中（最便宜）。
   - (b) 完全替換 T5 language embedding（要 finetune Octo 接受新分佈）。
   - (c) 透過 goal image 通道（把 task spec 渲染或 retrieve 成 image）。
3. **Offline meta-RL 的 outer loop**：Octo 自己只是 imitation；要做 meta-RL（[[PEARL|PEARL]] / [[VariBAD|VariBAD]] 風格），meta 訓練的 task 分佈、posterior over z、context encoder 都還是我要設計。
4. **零樣本評估設定**：Octo 的 zero-shot 指「同分佈不同物件」；我要的 zero-shot 是「同 skill 集合、新文字描述、新任務」。要自己設計 benchmark 與 baseline。

**一句話結論**：Octo 把「機器人感知 + 控制」做成樂高底座，meta-RL / task inference 可以蓋在上面而不用碰到 actuator 訊號層級。

---

## 一句話總結

Octo 是一個用 transformer + diffusion action head、在 800k cross-embodiment 軌跡上預訓練且**模組化到可任意加減觀察 / 動作空間**的開源 generalist policy，最適合當作「我把 task inference / offline meta-RL 蓋在上面」的研究 backbone，因為它把感知與控制的工程成本攤平到只剩一個 encoder + 一個 head。
