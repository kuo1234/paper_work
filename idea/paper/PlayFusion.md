---
type: paper-note
aliases:
  - "PlayFusion"
year: 2023
stage: "4-play與弱語言"
tags:
  - play-data
  - diffusion
  - skill-bottleneck
  - hindsight-language
  - 4-play與弱語言
summary: "language-annotated play + diffusion + discrete skill bottleneck；diffusion適合noisy multimodal play。"
---
> **論文標題**：PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play
> **作者**：Lili Chen*, Shikhar Bahl*, Deepak Pathak（Carnegie Mellon University）
> **出處 / 年份**：7th Conference on Robot Learning (CoRL 2023)
> **主題**：以「事後語言標註的 play data」為來源，透過條件式 diffusion model 加上 discrete bottleneck（U-Net latent 與語言 embedding 雙重 VQ）萃取機器人技能，達成多模態、語言條件、可組合的視覺-運動策略學習。
> **整理目標**：精讀並以繁體中文系統化解析 PlayFusion 的核心動機、diffusion + VQ 建模公式、hindsight language annotation 的角色，並對應到我自己的研究主線（robot play + 弱語言 + offline meta-RL + 文字 → task spec/embedding zero-shot），釐清 discrete skill bottleneck 與 task embedding 的關聯。

---

## 0. 閱讀總覽（白話）

PlayFusion 處理的問題很實際：機器人最容易拿到的不是乾淨的示範資料，而是人類隨便操作、想到什麼就玩什麼的 **play data**。這種資料有兩個惡質特性：(1) **嚴重多模態**——同一個狀態下，人可能有許多合理的下一步動作；(2) **不最佳**——大部分時間在亂走，但其中夾雜了真正完成某件事情的片段。傳統 BC 只會學到所有動作的平均（往往是無意義的中間值），VAE 系列又得加一堆 latent plan 模組才能勉強運作。

作者的解法分兩層。第一層：把策略改成 **conditional denoising diffusion**，因為 diffusion 天生擅長表達多模態分佈，可以在「同一個 (s, l)」下取樣出多種合理動作軌跡，不會塌成平均。第二層：在 diffusion 的 U-Net 中插一個 **VQ 離散瓶頸**，並同步離散化 **language embedding**，讓模型被迫把連續行為壓縮成一組「可數的技能字典」，這樣才有可能把學到的 A+B、C+D 重組成沒看過的 A+D。

訓練資料是「先收 play，事後再用語言標註」（hindsight language annotation），這比示範資料便宜得多，標註的語言又能當作 task / goal 的代理（取代 goal image）。在 [[CALVIN|CALVIN]]、Franka Kitchen、Ravens 三個模擬基準與三個真實環境（cooking / dining table / sink）上，PlayFusion 全面贏過 [[LatentPlansFromPlay|Play-LMP]]、[[FromPlayToPolicy|C-BeT]]、GCBC 等基線，特別在長序列與 A+D 這類組合泛化上優勢明顯。

---

## Abstract

論文主張：在語言與視覺生成領域，「從非結構化、未經整理的資料學習」已是主流，而機器人領域中對應的就是 **play data**。play 容易收集，但因高度多模態、noisy、非最佳而難以學。本工作以 **hindsight 語言標籤** 的 play data 為輸入，採用 **conditional diffusion** 在狀態 / 動作空間做去噪，自然處理多模態並產出多樣行為。為了讓 diffusion 真正萃取「技能」而非只是擬合分佈，作者在條件生成流程中加入 **discrete bottleneck**（離散瓶頸），使 agent 可獲得一個「技能字彙」。實驗涵蓋 CALVIN、Franka Kitchen、Ravens 與三個真實世界場景，皆優於現有方法。

---

## 1. Introduction

**核心動機**：人類重複使用過往經驗形成廣泛技能字典；機器人也應該如此。但既有路線各有缺陷：

- **手工 primitives / option framework**：sample-efficient 卻需要 task 先驗，難換新場景。
- **latent variable model 自動發現技能**（DIAYN、DADS 等）：通用但極度資料飢渴，難 scale 到實機。
- **imitation learning**：需要近乎完美的示範與重置流程。
- **offline RL**：需要 reward label，現實中 reward engineering 成本高。

相比之下，**play data**（操作者只被告知「去探索」）的收集成本最低；缺點是：
1. **嚴重多模態**：一個狀態下有許多合理動作；給一條軌跡也有許多 latent goal 能解釋它。
2. **無最佳性保證**。

作者觀察到：大型生成模型（特別是 diffusion）擅長對複雜分佈建模，並可由文字驅動，剛好對應 play 的多模態 + 語言條件需求。再加上人類技能本質是「少量、可重複、可組合」的，因此引入 **discrete bottleneck**，得到 PlayFusion。實驗在 6 個環境（3 模擬 + 3 真實）全面勝出。

---

## 2. Related Work（簡述）

- **Goal / Language Conditioned Skill Learning**：goal-conditioned 多用 final state；語言條件路線（BC-Z、SayCan、CLIPort、Perceiver-Actor 等）使用「結構化、curated」資料，本文則直接吃 play data + hindsight 語言。
- **Learning from Play**：Play-LMP / MCIL 用 VAE，RIL 用階層 IL，C-BeT 用 transformer + 動作離散化處理多模態。LAD 把 diffusion 引入 play 但仍保留 VAE 風格的 latent plan，本文則完全捨棄這些。
- **Generative Behavior Modeling**：Diffuser、Decision Diffuser、Diffusion-QL、IDQL 將 diffusion 應用於 offline RL；Diffusion Policy 應用於示範資料的視覺-動作學習。本文差異在於資料是「帶語意 label 的 play」而非 offline RL 或專家示範。
- **Discrete Control**：VQ-VAE 形式的離散表徵與技能字典的天然連結。

---

## 3. Background（公式定義）

### 3.1 Denoising Diffusion Probabilistic Models (DDPM)

DDPM 將生成視為對 Gaussian 雜訊的迭代去噪。從 $x_K \sim \mathcal{N}(0, I)$ 開始，反覆執行 $K$ 步去噪得到 $x_{K-1}, \dots, x_0$：

$$
x_{k-1} = \alpha \left( x_k - \gamma \, \epsilon_\theta(x_k, k) + \mathcal{N}(0, \sigma^2 I) \right) \tag{1}
$$

其中 $\epsilon_\theta$ 為帶可學參數 $\theta$ 的噪聲預測網路。訓練時隨機抽 $x_0$ 與去噪步 $k$、隨機取雜訊 $\epsilon_k$，最小化：

$$
L = \| \epsilon_k - \epsilon_\theta(x_0 + \epsilon_k, k) \|^2 \tag{2}
$$

**符號**：
- $x_k$：第 $k$ 步雜訊化的樣本（在本文應用中為 chunk 化的動作序列 $a_t^k$）。
- $\alpha, \gamma, \sigma$：依 noise schedule 決定的係數。
- $\epsilon_\theta(\cdot)$：U-Net 噪聲預測網路。

### 3.2 Vector-Quantized 表徵（VQ-VAE）

VQ-VAE 透過 encoder $E$ 把 $x$ 編碼為 $z = E(x)$，再從 codebook $\{e_i\}$ 中選擇最近者 $j = \arg\min_i \| z - e_i \|$，使用 $e_j$ 解碼。損失含 reconstruction、quantization、commitment 三項：

$$
L_{\text{VQVAE}} = L_{\text{recon}}(x, D(e_j)) + \| z - \text{sg}(e_j) \|^2 + \| \text{sg}(z) - e_j \|^2 \tag{3}
$$

其中 $\text{sg}(\cdot)$ 為 stop-gradient。第二項拉動 codebook 向 encoder 輸出靠近；第三項把 encoder 「綁定」到 codebook 中的某個離散碼。

### 3.3 Learning from Play (LfP) 設定

資料集 $\{(s, a)\} \in \mathcal{S} \times \mathcal{A}$，無任務或最佳性假設。目標是學 $\pi: \mathcal{S} \times \mathcal{G} \to \mathcal{A}$。Goal 可以是 $s_T$，也可以（如本文）是 hindsight 標註的語言指令 $l$。

---

## 4. PlayFusion 方法

### 4.1 語言條件 Play Data

資料 $D_{\text{play}} = \{(s_t^{(i)}, a_t^{(i)})\}_{i=1}^N$，作者假設動作來自隱意圖 $z_g$ 的分佈：$a_t \sim F(s_t, z_g)$。對軌跡片段 $\tau = \{s_i, a_i\}_{t=k}^H$ 在事後以指令 $l$ 標註，丟給預訓練語言模型 $g_{\text{lang}}$（如 MiniLM、Sentence-BERT 等）得到 $z_l = g_{\text{lang}}(l)$。

策略架構：
- 視覺編碼 $\phi_v$：ResNet 處理影像序列 $s_t$。
- 語言投影 $\phi_l$：MLP 將 $z_l$ 降維。
- 條件向量：$g = [\phi_l(z_l), \phi_v(s_t)]$，作為動作 decoder $f_{\text{act}}$ 的條件。

關鍵：把 $f_{\text{act}}$ 建成 **diffusion process** 而非 VAE，直接以去噪迭代處理多模態，免去 latent plan 的設計。

### 4.2 多模態動作生成（Conditional Diffusion）

把 DDPM 改寫為條件式，預測未來 $T_a$ 步動作 chunk（action chunking）：

$$
a_t^{k-1} = \alpha \left( a_t^k - \gamma \, \epsilon_\theta(g, s_t, a_t^k, k) + \mathcal{N}(0, \sigma^2 I) \right) \tag{4}
$$

$$
L = \| \epsilon_k - \epsilon_\theta(g, s_t, a_t^0 + \epsilon_k, k) \|^2 \tag{5}
$$

**符號補充**：
- $a_t^k$：在去噪步 $k$ 的動作 chunk（時間 $t \sim t+T_a$）。
- $g$：語言 + 視覺合併的條件。
- 預測 chunk 而非單步動作能提高時序一致性（同 Diffusion Policy）。

### 4.3 Discrete Diffusion for Control（離散瓶頸）

#### 動機

人類技能少而可重複；連續 latent goal + 連續 diffusion 無法形成「技能字典」。但暴力強迫離散會破壞 diffusion 對多模態的表達。折衷方案：**在 U-Net 中插入 VQ 瓶頸**，限制條件 → 動作對應的可用代碼，使單一 goal 下動作仍多模態但只覆蓋少數模式。

#### U-Net 內的 VQ 瓶頸

設 U-Net encoder 輸出 $\psi_\epsilon(x)$（$x = (g, s_t, a_t^0 + \epsilon_k, k)$）。瓶頸層挑選最近碼：

$$
j = \arg\min_i \| \psi_\epsilon(x) - e_{u, i} \|
$$

decoder 接收 $e_{u, j}$ 再產出 $\epsilon_\theta(x) = \gamma_\epsilon(\psi_\epsilon(x))$（實作上會在 forward 中替換為對應的離散碼）。訓練加入 quantization + commitment 兩項損失。

#### 語言端離散化

為了讓技能可組合（學過 A+B、B+C、C+D 後推廣到 A+D），也對 $\phi_l(z_l)$ 做 VQ：

$$
e_{l, j} = \arg\min_{e_{l,i}} \| \phi_l(z_l) - e_{l,i} \|
$$

#### 總損失

$$
\begin{aligned}
L_{\text{PlayFusion}} =\;& \| \epsilon_k - \epsilon_\theta(x_0 + \epsilon_k, k) \|^2 \\
& + \beta_1 \| \text{sg}(\psi_\epsilon(x)) - e_{u,j} \|^2 + \beta_1 \| \psi_\epsilon(x) - \text{sg}(e_{u,j}) \|^2 \\
& + \beta_2 \| \text{sg}(\phi_l(z_l)) - e_{l,j} \|^2 + \beta_2 \| \phi_l(z_l) - \text{sg}(e_{l,j}) \|^2
\end{aligned} \tag{6}
$$

四個 VQ 子項分別為 **U-Net quantization / U-Net commitment / language quantization / language commitment**。$\beta_1, \beta_2$ 控制離散化壓力與行為多樣性的權衡，實驗上 $\beta_1 = \beta_2 = 0.5$ 最佳；過大會壓垮 diffusion 表現力。

#### Sampling

測試時收到新指令 $z'$：
1. $\phi_l(z')$ 經量化得到 $e_{l, j'}$。
2. 結合視覺得 $g'$。
3. 從 $\mathcal{N}(0, I)$ 取雜訊化的動作 chunk，依式 (4) 反覆去噪直到 $a_t^0$，輸出給機器人執行。

---

## 架構流程（端到端）

1. **資料收集**：人類遙操作機器人「亂玩」，產出長串 $(s_t, a_t)$ play 序列。
2. **Hindsight 標註**：事後將片段切段並用自然語言敘述意圖 $l$。
3. **語言嵌入**：$z_l = g_{\text{lang}}(l)$ → $\phi_l(z_l)$ → VQ codebook $\{e_{l,i}\}$ 量化。
4. **視覺嵌入**：$\phi_v(s_t)$ 由 ResNet 得到。
5. **條件融合**：$g = [\phi_l(z_l)_{\text{量化}}, \phi_v(s_t)]$。
6. **去噪 U-Net**：encoder 輸出 latent 經 codebook $\{e_{u,i}\}$ 量化後送入 decoder，預測噪聲 $\epsilon_\theta$。
7. **動作生成**：迭代 50 步去噪生成動作 chunk（CALVIN 中 $T_a = 16$）。
8. **執行**：機器人執行 chunk，循環。

---

## 5. 實驗與結論

### 5.1 環境與基線

- **模擬**：CALVIN (A: D→D / B: ABC→D)、Franka Kitchen (A: 單任務 / B: 兩任務鏈)、Language-Conditioned Ravens（block-in-bowl、stack-pyramid、packing-box-pairs）。
- **真實世界**：cooking、dining table、sink；皆設計 A+D 組合泛化測試（訓練看過 A+B、C+D，測試 A+D）。
- **基線**：Play-LMP（VAE primitives）、C-BeT（transformer + 動作離散化）、GCBC（goal-conditioned BC）。

### 5.2 主要結果

| 環境 | C-BeT | Play-LMP | GCBC | **PlayFusion** |
|---|---|---|---|---|
| CALVIN A | 26.3 | 19.9 | 23.2 | **45.2** |
| CALVIN B | 23.4 | 22.0 | 30.4 | **58.7** |
| Kitchen A | 45.6 | 1.9 | 38.0 | **47.5** |
| Kitchen B | 24.4 | 0.0 | 15.5 | **27.7** |
| Ravens | 13.4 | 0.2 | 1.6 | **35.8** |
| Dining Table | 20.0 | 0.0 | 5.0 | **45.0** |
| Cooking | 0.0 | 0.0 | 0.0 | **30.0** |
| Sink | 10.0 | 0.0 | 5.0 | **20.0** |

PlayFusion 全面勝出；Kitchen A 資料較窄、多模態優勢有限，仍持平或微贏。

### 5.3 長序列任務（Long-Horizon CALVIN）

評估串接 5 個任務的能力：CALVIN A 上 PlayFusion 平均序列長度 0.417（最強基線 C-BeT 0.272）；CALVIN B 上 0.611（C-BeT 0.272）。Diffusion 對「任務間切換」這種更高的多模態資料分佈有顯著優勢。

### 5.4 離散瓶頸的分析

- **消融**（Table 3）：拿掉 U-Net 或 language VQ 都會掉分；語言端離散化在 Ravens 上影響極大（拿掉後 put-block-in-bowl 從 63.6 → 4.1）。
- **codebook 嵌入可視化**（Figure 4）：語意相近的技能（同地點 / 同物件）的 codebook 嵌入聚集在相似區域，確實學到「離散技能」結構。
- **$\beta$ 調節**：$\beta_1 = \beta_2 = 0.5$ 最佳；過大反而傷害 diffusion 表現力。
- **部分離散化**：只量化 25% 的 latent 表現甚至更好（48.7 vs 全量化 45.2），顯示在「鼓勵技能形成」與「保留去噪精度」間需要平衡。
- **codebook size**：2048～8192 都不錯，16384 開始下降（瓶頸消失）。

### 5.5 其他消融

- **語言模型**：MiniLM / Distilroberta / MPNet / BERT 表現相近（47–49%），CLIP 較差（35–44%）——VQ 對語言 embedding 的離散化提供一定 robustness；CLIP 在 play instruction 上不對盤。
- **視覺預訓練**（如 R3M）：未帶來增益，從零學亦可。
- **條件方式**：global conditioning 較佳；把 noise 也條件化反而傷害。
- **未見技能**（Table 10）：移除 lift-red-block-slider 後，PlayFusion 仍能泛化（20.0），優於拿掉任一 VQ 的版本。

### 5.6 結論與限制

PlayFusion 把 diffusion 與雙重 VQ 結合到語言條件 play data 學習，顯著改善多模態建模與技能組合泛化。限制：仍需人類事後語言標註（雖比示範便宜）；真實世界成功率仍有提升空間；可擴展到更複雜的居家場景。

---

## 6. 相關工作（補充歸納）

- **Skill discovery / option framework**：Bacon et al. (option-critic)、Sutton et al. (semi-MDP) 為理論底；DIAYN、DADS 為 unsupervised skill；CompILE、Temporal VI 為 latent skill 學習；皆需 reward 或大量資料。
- **Learning from Play 主線**：Play-LMP、MCIL、Lynch & Sermanet（grounding language in play）、RIL（hierarchical IL）、C-BeT（transformer + 動作離散化）、LAD（diffusion + VAE-style plan）。
- **Diffusion for control**：Diffuser、Decision Diffuser、Diffusion-QL、IDQL（offline RL）；Diffusion Policy（示範式 visuomotor）；StructDiffusion、UniPi（場景結構 / 影片生成而非低階動作）。
- **VQ for control**：VQ-VAE [van den Oord et al.] + planning（Ozair et al.）；C-BeT、BeT 將 VQ 用於動作離散化。

---

## 與本研究主線的關聯

我的研究主線：**robot play data + 極少 / 弱語言標註 + offline meta-RL + 以文字 → task spec/embedding 達成 zero-shot**，並關注 meta-learning 與 zero-shot 的並行設計。PlayFusion 與此目標有多處精準對應，可直接借鑑或差異化：

### 1. Hindsight Language Annotation 作為 task spec 的廉價代理

PlayFusion 的設定本質上就是「play + 事後語言」，與我希望使用「**極少弱語言**」的方向高度同構。要點與啟發：

- **語言是 latent intent $z_g$ 的可觀測替身**：作者明示 $a_t \sim F(s_t, z_g)$，用 $z_l = g_{\text{lang}}(l)$ 估計 $z_g$，等價於「以語言當 task spec」。這正是我把 task embedding 解釋為「文字 → 任務空間」的核心構想。
- **稀疏標註的可行性**：CALVIN 的 5K 真正帶標註的片段被重複放大為 200K 訓練軌跡；Kitchen 也只有 566 demo 切成 ~2.2K 標註片段。這顯示**少量語言標註 + 大量 play** 是可工作的，恰好符合「弱語言」的設定。可借鑑為：少量人工 label + caption model 自動擴充（作者亦在限制段點名）。
- **對我的 offline meta-RL 設計**：可以把 hindsight language annotation 視作 meta-RL 中 task identity 的「弱監督訊號」，取代 task ID 或 reward；在 meta-train 階段以語言對應 task distribution，在 meta-test 直接以新語言指令當 zero-shot task spec。

### 2. Diffusion 為何比 BC 更適合 noisy multi-modal play

這是 read.md 必含重點：

- **BC 的失效模式**：play data 對同一 $(s_t, l)$ 常存在多個合理 $a_t$。BC 以 MSE / cross-entropy 訓練時，會收斂到「條件動作分佈的中位數 / 平均」，在多模態情況下這個平均**可能根本不是任何一個有效動作**（如左右兩條路徑的平均落到中間障礙物上）。
- **VAE/latent plan 的代價**：Play-LMP 等用 VAE 隱式建模多模態，但需要額外 plan encoder/decoder 與 KL 平衡，常出現 posterior collapse；LAD 雖引入 diffusion 卻仍保留 VAE 風格的 plan。PlayFusion 完全捨去這些。
- **Diffusion 的天然契合**：iterative denoising 形式上是對條件分佈 $p(a | s, l)$ 的 score matching，可以**同時覆蓋多個 mode 而不塌陷**；條件 $g$ 不夠 informative 時，diffusion 仍可在去噪中保留隨機分支。對 noisy / suboptimal play 而言，這等於把「多種人類可能行為」都當成 valid sample。
- **Action chunking + temporal consistency**：預測 $T_a$ 步動作 chunk 而非單步動作，等於在動作序列空間做去噪，能避免單步多模態下不同步切換 mode 而崩潰。
- **對我的 offline meta-RL**：傳統 offline meta-RL（如 [[PEARL|PEARL]]-offline、[[MACAW|MACAW]]）大多基於 actor-critic，遇到 play 風格資料容易 over-conservative。把 actor 換成 conditional diffusion（類似 Diffusion-QL / IDQL）可能更適合 noisy play，且可同時吃語言條件。

### 3. Discrete Skill Bottleneck 與 task embedding 的關係

這是另一個 read.md 必含重點，也是我覺得對主線最有啟發的部分：

- **作用 1：壓出「可數技能字典」**：U-Net 內的 VQ 把連續 latent 投影到 $\sim$2K 個離散碼，每碼對應一類「條件 → 動作 mode」的 archetype；語言端的 VQ 把連續語言 embedding 投影到離散 task code。**這兩個 codebook 一起構成可組合的 task / skill 表徵。**
- **作用 2：組合泛化**：明確指出「訓練看過 A+B、C+D，希望 zero-shot 推到 A+D」——這正是我關心的 zero-shot 任務組合。離散化使 task space 變成「可被列舉、可被替換」的代碼集合，比連續 embedding 更容易進行 compositional 推理（連續空間中 A+D 的 embedding 不在訓練支撐集中時，模型可能 OOD；離散空間中 A 與 D 的代碼分別存在，組合即合法）。
- **作用 3：對 text-to-task-embedding 的啟示**：我原本構想 task embedding 是連續向量（如 LoRA-style 或 z-vector），PlayFusion 提供一個替代方案：**讓 text encoder 輸出後接 VQ codebook**，使任務空間有限可數且能與下游策略 codebook 對齊。在 meta-RL 中，meta-train 階段同時更新「語言 → 離散 task code」與「task code → policy」，meta-test 時新指令會被量化為已知 code 的鄰近或新組合，是天然的 zero-shot 入口。
- **作用 4：與 meta-learning 對接**：離散 task code 可被視為 meta-learning 的「discrete context variable」，與 PEARL 連續 $z$ 形成對偶；可以同時保有 PEARL 的 posterior inference（meta-learn）與 PlayFusion 的離散組合（zero-shot generalization），這可能是 meta-learning × zero-shot 並行方向上一個具體可實作的架構雛形。
- **設計細節**：不要 100% 量化（只量化 25% latent 反而較佳）、$\beta = 0.5$、codebook size $\sim$2K—這些是實作時可直接搬過去的超參數參考。

### 4. 與主線的差異 / 可延伸點

- **PlayFusion 是 online policy（執行時直接 rollout）**，沒有 RL 的價值估計。我的主線希望保留 offline meta-RL 的形式（有 Q / advantage），因此可往 **Diffusion-QL + Hindsight Language + VQ** 的方向延伸：把 PlayFusion 當 actor，加上 offline Q-learning 處理 play 的 suboptimality。
- **PlayFusion 不真正做 meta-learning**：它是把所有 task mix 在一起訓練 multi-task 策略；沒有 meta-test 時 fast adaptation 的機制。可以加入 context encoder（如 PEARL 的 $z$）或 in-context demonstration prompt，與 VQ task code 結合。
- **語言來源**：PlayFusion 用 hindsight 人類標註；我若改用 caption model 或 LLM 自動標註，可以走向「**zero-human-language**」極限，這是 PlayFusion 自己也提到的未來方向。

---

## 一句話總結

PlayFusion 用 conditional diffusion 處理 play data 的多模態、再以 U-Net + 語言雙重 VQ 瓶頸強行壓出「可數、可組合的離散技能字典」，把「hindsight 語言 → 離散 task code → 多模態動作 chunk」串成可 zero-shot 組合的視覺-運動策略——這對我「弱語言 + play + offline meta-RL + 文字到 task embedding zero-shot」主線提供了一個極具參考價值的離散 task embedding 設計樣板。
