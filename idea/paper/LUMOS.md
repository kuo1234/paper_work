---
type: paper-note
aliases:
  - "LUMOS"
year: 2025
stage: "4-play與弱語言"
tags:
  - play-data
  - world-model
  - CLIP對齊
  - zero-shot
  - sim2real
  - 最近鄰
  - 主要baseline
  - 4-play與弱語言
summary: "最近鄰之二：play→world model→latent空間on-policy練(DITTO reward)→<1%弱語言CLIP對齊→zero-shot真機。"
---
# LUMOS 論文逐段詳細整理

> 論文：**LUMOS: Language-Conditioned Imitation Learning with World Models**
> 作者：Iman Nematollahi, Branton DeMoss, Akshay L Chandra, Nick Hawes, Wolfram Burgard, Ingmar Posner（University of Freiburg / University of Oxford / University of Technology Nuremberg）
> 出處：arXiv:2503.10370v1（2025-03，ICRA 2025 投稿）
> 專案頁：http://lumos.cs.uni-freiburg.de
> 主題：Language-Conditioned Imitation Learning、World Model（DreamerV2/RSSM）、Latent Planning、Play Data、Hindsight Language Relabeling、Zero-shot Sim→Real、Covariate Shift 緩解
> 整理目標：說明 LUMOS 如何「在學到的 world model latent 空間裡 on-policy 練技能、用 <1% 弱語言標註做 language-conditioned、再 zero-shot 遷到真機」，並**深入比較它與 T2DA、與使用者研究主線的關係與 novelty 空位**（這是使用者文獻地景中「play + 弱語言 + zero-shot robot control」那一塊的代表作）。

---

## 0. 閱讀總覽：這篇在做什麼？

一句話：**LUMOS 先從未結構化 play data 學一個 world model，然後把 imitation learning 整個搬進這個 world model 的 latent 空間裡「on-policy 練習」——讓 agent 能在想像（imagination）中跨多步重複練同一技能、從自己的錯誤中恢復，藉此壓制 covariate shift；同時用 <1% 的 hindsight 弱語言標註讓 policy 可被語言指揮，最後 zero-shot 遷到真機。**

它要解的核心痛點是 **behavior cloning 的 covariate shift**：BC 的 policy 不是在自己的分布下訓練的，小誤差會累積、把 policy 拉到沒看過的 state，長時程任務尤其慘（Ross & Bagnell 證明 BC 誤差 ε 的 regret 是 O(T²ε)，隨 horizon 二次成長）。傳統解法是「在模擬器裡 on-policy 訓練」，但模擬器常缺或有 sim2real gap。LUMOS 的答案是：**用 world model 當「資料驅動的模擬器」,在它的 latent 空間裡 on-policy 學**——既能 on-policy、又能長時程、又避開手寫模擬器的 reality gap。

三個組件（Fig. 2）：
1. **World Model 學習**：用 DreamerV2 的 RSSM 從 play data 學 latent dynamics（image encoder + RSSM + decoder）。
2. **Goal-conditioned Policy 學習（behavior learning）**：在凍結的 world model latent 裡訓練 actor-critic，用 **DITTO 的 latent-matching intrinsic reward**（不回歸 action label，而是獎勵「agent 的 latent 軌跡貼近專家 latent 軌跡」）；加上 **latent planning network**（plan recognition/proposal，seq2seq CVAE）與 **CLIP 式語言-latent 對齊**。
3. **語言引導推論**：測試時 `π_θ(a_t | s_t, l)` 由 world model 從當前觀察推 latent，再被使用者語言指令引導。

結果：在 [[CALVIN|CALVIN]]（長時程語言操作 benchmark）上勝過 GCBC / MCIL / HULC；並首次在「離線 world model 內」學到 language-conditioned 連續視覺運動控制、zero-shot 遷到真機。

---

## 1. Abstract 重點

LUMOS 是 robotics 的 language-conditioned 多任務模仿學習框架。它**在學到的 world model 的 latent 空間裡，透過大量長時程 rollout 反覆練習技能**，再 zero-shot 遷到真機。因為是在 latent 空間 on-policy 學，它緩解了多數離線模仿學習都有的 policy-induced distribution shift。LUMOS 從未結構化 play data 學、語言標註 <1%，但測試時可被語言指令操控。做法是結合 latent planning + image/language 雙重 hindsight goal relabeling + 在 world model latent 空間定義的「跨多步 intrinsic reward」來降 covariate shift。在難的長時程 CALVIN benchmark 上勝過先前學習式方法，並（據作者所知）首次在離線 world model 內學到真機可用的 language-conditioned 連續視覺運動控制。

---

## 2. 與相關工作的定位（Related Work 重點）

- **語言條件模仿學習脈絡**：MCIL（Lynch，用 play + 1% 語言）→ CALVIN（Mees，benchmark）→ HULC（Mees，階層式 + 對比學習對齊語言，勝過 MCIL）→ HULC++（加 affordance + motion planning）→ SPIL（標註全部 action 並機率指派到 base skill）。LUMOS 刻意**不**引入「action 維度與技能語意」的先驗（保持 7-DoF 連續動作空間）。
- **World Models**：Ha & Schmidhuber 開創「在 latent 學 policy」；Dreamer 系列（含 DreamerV2/V3）在 Atari、Minecraft 達 SOTA；DayDreamer 把 Dreamer 用到真機四足。
- **本文核心依賴 DITTO（DeMoss 2023）**：一個 world-model-based 模仿學習演算法，在 world model latent 空間定義「agent rollout 與專家示範的 divergence」作為 intrinsic reward，用 actor-critic 優化。這讓模仿學習對長時程 rollout 的誤差 robust，有效緩解 covariate shift。**LUMOS = DITTO + 語言條件 + latent planning + CLIP 對齊。**

---

## 3. 問題設定（Problem Formulation）

goal-conditioned 模仿學習，在一個 **goal-augmented POMDP** `M = (S, A, R, T, G, γ)`：視覺觀察（非直接 state）、連續動作、目標空間 G 含 free-form 語言 `l` 或 latent goal state。離線學習，用一個大型、未標註、無方向（undirected）的固定 play 資料集 `D = {(s_1,a_1),…,(s_T,a_T)}`。

用 hindsight relabel（HER 式）：把每個造訪過的 state 當「達成的 goal」，得 `D_play = {(τ, s_g)}`。並採 MCIL 的做法用語言標註——把少量隨機 window 配上事後（retrospective）語言指令，就能學出統一的 language-conditioned 視覺運動 policy。

---

## 4. 方法（LUMOS 核心）

訓練兩階段：先從未標註 play data 學 world model；再在這個凍結的 world model 內訓練 actor-critic 得 goal-conditioned policy（讓想像中的 latent 序列去 match 專家的 latent 軌跡）。

### 4.1 World Model 學習（DreamerV2 / RSSM）

backbone 是 DreamerV2 的 **RSSM（Recurrent State-Space Model）**：image encoder + RSSM dynamics + image decoder。static 與 gripper 兩相機各有 CNN encoder/decoder，編碼後串接送進 RSSM。RSSM 各模組：

\[
\begin{aligned}
&\text{Recurrent state:} && h_t = f_\phi(\hat{s}_{t-1}, a_{t-1})\\
&\text{Representation model:} && z_t \sim q_\phi(z_t \mid h_t, x_t)\quad(\text{後驗，看當前觀察})\\
&\text{Dynamics predictor:} && \hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t)\quad(\text{先驗，不看觀察})\\
&\text{Image decoder:} && \hat{x}_t \sim p_\phi(\hat{x}_t \mid \hat{s}_t)
\end{aligned}
\tag{1}
\]

合併 model state `ŝ_t = (h_t, z_t)`（確定性 + 隨機性）。先驗/後驗為 categorical 分布，用 straight-through gradient。聯合最小化負變分下界（ELBO）：

\[
\min_\phi\; \mathbb{E}_{q_\phi(z_{1:T}|a_{1:T},x_{1:T})}\Big[\sum_{t=1}^T -\log p_\phi(x_t\mid \hat{s}_t) + \beta\, D_{KL}\big(q_\phi(z_t\mid \hat{s}_t)\,\|\,p_\phi(\hat{z}_t\mid h_t)\big)\Big]
\tag{2}
\]

訓練後可**只用先驗 `ẑ`（不看觀察）生成無限長的想像軌跡** `{(h_t, ẑ_t, a_t)}`，這就是 on-policy 練習的「資料驅動模擬器」。附錄：用 KL balancing（公式 7，δ=0.8，prior 比 posterior 更快對齊）；latent = 32 categorical × 32 class 攤平成 1024 維、加確定性共 k=2048；約 4000 萬參數。

### 4.2 Behavior Learning（actor-critic in latent space）

在凍結 world model 的 latent 裡學 `π_θ(a_t | s_t, g)` 與 `v_ψ(s_t, g)`，`g` 是語言 `l` 或 latent goal `s_g`。

**Latent Plan Encoding（解決多模態）**：同一 `(s_t, s_g)` 有多條有效軌跡，用 seq2seq CVAE 把上下文 latent 軌跡編進「plan」空間。兩個隨機 encoder：**plan recognition**（訓練時看整段序列、認出做了什麼行為）與 **plan proposal**（推論時只看初始與最終 state、提出可能行為）。最小化兩者 KL（`L_KL`）讓 proposal 準確反映行為。用 multimodal transformer encoder（如 HULC）把 latent 軌跡映成多個 categorical 變數。

**語言-latent 語意對齊（CLIP 式）**：不同於 HULC 對齊「視覺特徵↔語言」，LUMOS 對齊 **world model 的 latent 特徵 ↔ 語言特徵**：最大化配對的 cosine 相似度、最小化與不相關指令的相似度，用 in-batch negatives 的對比損失 `L_contrast`（附錄公式 8，可訓溫度 τ；語言用 paraphrase-MiniLM-L3-v2 → 384 維）。

**Intrinsic Reward（DITTO 式 latent matching，本篇關鍵）**：不靠 action label 監督，而是獎勵「agent 的 latent state 貼近專家 latent state」。每步 reward：

\[
r_t^{int}(s_t^{E}, s_t^{\pi}) = \frac{s_t^{E}\cdot s_t^{\pi}}{\max(\|s_t^{E}\|, \|s_t^{\pi}\|)^2}
\tag{3}
\]

（`E` 為專家、`π` 為 agent；一個修改版內積，鼓勵相似但不要求精確相等，讓 agent 能在 latent 空間探索不同軌跡、同時整段對齊示範者。）

**Actor / Critic**：

\[
\text{Actor: } a_t \sim \pi_\theta(a_t\mid s_t, g),\qquad
\text{Critic: } v_\psi(s_t, g) \approx \mathbb{E}_{\pi_\theta, p_\phi}\Big[\sum_{t=0}^{H}\gamma^t r_t^{int}\Big]
\tag{4}
\]

actor 輸出 tanh-transformed Gaussian（可重參數化）。world model 在 behavior learning 時**凍結**（agent 梯度不改變它的表徵）。critic 回歸 λ-target `V^λ`（TD(λ)，附錄 λ=0.95 偏好長時程）：

\[
L(\psi) \doteq \mathbb{E}_{\pi_\theta, p_\phi}\Big[\sum_{t=1}^{H-1}\tfrac12\big(v_\psi(\hat{s}_t) - \mathrm{sg}(V_t^\lambda)\big)^2\Big]
\tag{5}
\]

**Actor 目標（含三項）**：透過學到的 dynamics 反傳，最大化 value、同時最小化 plan encoder 的 KL 與語言對比損失（沒語言標註的 window 就略過 `L_contrast`）：

\[
L(\theta) \doteq \mathbb{E}_{\pi_\theta, p_\phi}\Big[\sum_{\tau=t}^{t+H}\big(-V_\lambda(s_\tau) + \alpha_1 L_{KL} + \alpha_2 L_{contrast}\big)\Big]
\tag{6}
\]

附錄超參：α1=0.1、α2=3.0、γ=0.995、actor lr 2e-4、critic lr 3e-4、slow critic 更新間隔 100。action decoder 是 8 層 FC（256 寬，~100 萬參數，非 recurrent，輸出 tanh-Gaussian）——比 HULC 的 RNN（~1500 萬參數）小得多，因為 world model 已包含時間結構。

---

## 5. 實驗與結論

### 5.1 模擬（CALVIN 環境 D）

6 小時 play data、7-DoF Franka、34 子任務、1000 條指令鏈、最多連做 5 個語言指令、只用機載感測器、**只有 1% 標語言**。所有影像標準化到 64×64（含 baseline，為公平比較）。

**CALVIN 結果（Table I，平均完成連續任務數 Avg. Len.，3 seeds）**：

| 方法 | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
| --- | --- | --- | --- | --- | --- | --- |
| GCBC | 56.7% | 20.9% | 7.3% | 1.6% | 0.04% | 0.63 |
| MCIL | 70.3% | 40.2% | 21.6% | 11.1% | 5.4% | 1.48 |
| HULC | 77.6% | 58.05% | 41.6% | 29.8% | 20.0% | 2.27 |
| **LUMOS** | **80.7%** | **59.3%** | **42.6%** | **30.7%** | **21.1%** | **2.34** |
| No DITTO | 71.8% | 45.6% | 27.8% | 14.2% | 8.7% | 1.68 |
| No latent plan | 76.5% | 50.5% | 28.7% | 15.9% | 11.0% | 1.81 |
| No alignment | 73.3% | 52.6% | 39.8% | 27.3% | 17.2% | 2.05 |

**三個消融的結論（很重要）**：
- **No DITTO（用 BC + MSE 取代 intrinsic reward）**：掉最多（Avg. Len. 2.34→1.68）→ 證明 latent-matching reward 是降 covariate shift 的主因。
- **No latent plan（拿掉 plan proposal/recognition）**：起步還行但長時程崩（→1.81）→ latent planning 對長時程關鍵。
- **No alignment（拿掉語言-latent 對齊）**：仍維持部分長時程，但每步略差，會偶爾操作錯顏色方塊（語言-場景關聯不佳）→ 對齊主要幫「語言 grounding 的精準度」。

### 5.2 真機（Franka + 3D 桌面）

3 小時遙操作 play（VR controller）、static + gripper 相機、**<1% 標語言（約 2800 個隨機 window）**、22+ 種任務（操作 stove/bowl/cabinet/carrot/eggplant）。world model 長時程預測佳（用前 5 張影像當 context，僅憑 action 預測後 195 步，雖然只用 horizon 50 訓練）。

**真機結果（Table II）**：LUMOS 平均成功率 67.68% vs HULC 63.39%（World Model 上界 75.89%）；平均連續完成 2.05 vs HULC 1.90。LUMOS 在個別與多階段長時程任務都勝 HULC。真機 vs 模擬的差距主要來自 world model 不夠準（資料更多可改善）。

### 5.3 結論與限制

LUMOS 證明「在離線 world model 內學到的 dynamics 與 policy 能 zero-shot 遷到真機」。限制：(1) 難以事先知道 world model 表徵/dynamics 品質是否足以產生強 policy；(2) 提高影像解析度能大幅提升表現，但訓練高解析 world model 的算力成本高。

---

## 6. 與本研究主線的關聯（重點章節）

使用者主線：**robot play data + 極少量弱語言標註 + offline meta-RL + 文字→task spec/embedding 做 zero-shot 任務泛化**，關注 meta-learning × zero-shot。在使用者的文獻地景裡，LUMOS 是「**play data + 弱語言 + zero-shot robot control**」這一塊的代表作（與 [[T2DA|T2DA]] 的「offline meta-RL + 語言 supervision」互補）。

### 6.1 LUMOS 已經做了使用者要的哪幾塊？

LUMOS 命中了三塊中的三塊「機器人面」：
- ✅ **robot play data**（CALVIN 的未結構化 play + 自收 3 小時真機 play）。
- ✅ **極少弱語言標註**（<1% hindsight 語言，且明確用 CLIP 式對齊 `L_contrast`）——這與 T2DA 用「乾淨 template 語言」不同，**LUMOS 的弱語言設定更貼近使用者**。
- ✅ **zero-shot 語言操控 + zero-shot sim→real**（測試只給語言、不 fine-tune）。

它沒做的那塊（也是與使用者最大的分野）：
- ❌ **沒有明確採 offline meta-RL formalism / task belief**。LUMOS 是 language-conditioned imitation（world-model 內的 actor-critic 模仿），**語言當 goal embedding 直接 condition policy，沒有「對任務做 posterior 推論 / task uncertainty」的機制**。它沒有 task distribution 上的 meta-learning、沒有 `q(z|τ)` 的 belief，也沒有「測試任務是 unseen task family」的 meta 泛化框架。

### 6.2 LUMOS vs T2DA：使用者主線的兩個近鄰如何互補

| 面向 | LUMOS | T2DA |
| --- | --- | --- |
| 資料 | robot play（CALVIN + 真機），未結構化 | SAC 各任務收的 Mixed/Medium/Expert |
| 語言 | <1% hindsight 弱語言（貼近使用者） | 每任務一句乾淨 template caption |
| 語言對齊 | CLIP 式：語言 ↔ **world model latent 軌跡** | CLIP 式：語言 ↔ **dynamics-aware decision embedding** |
| 核心機制 | world model + DITTO intrinsic reward 壓 covariate shift | 對比對齊 + 生成式 policy（Diffuser/Transformer） |
| 是否 meta-RL | 否（imitation in world model） | 是（offline meta-RL，task representation） |
| 主打 | 長時程連續控制、sim→real | zero-shot 對 unseen task 泛化 |
| benchmark | CALVIN + 真機 | MuJoCo + Meta-World |

**關鍵觀察**：兩者都用 **CLIP 式對比把語言對齊到一個 latent**——這再次強力驗證使用者的核心假設「`g(l) ≈ q(τ)` 對齊可行」。差別在對齊的目標：LUMOS 對齊到 **world model latent 軌跡**（強調 dynamics + covariate shift），T2DA 對齊到 **task embedding**（強調 task 泛化）。**使用者的空位正落在兩者交集之外**：把 LUMOS 的「play + 弱語言 + world model」資料/表徵設定，接上 T2DA 的「offline meta-RL + task belief」formalism——即「**在 play + 弱語言下，學一個可做 task inference 的 belief latent，並對 unseen task 做 zero-shot**」。LUMOS 沒有 task belief、T2DA 沒有 play+弱語言，這個交集就是 novelty。

### 6.3 可直接借用的零件

- **CLIP 式「語言 ↔ latent 軌跡」對齊（公式 8）+ MiniLM text encoder + 沒標註就略過 L_contrast**：直接是使用者「極少弱語言對齊」的可用實作；「沒語言的 window 就跳過對比項」正是處理「只有 <1% 標註」的乾淨做法。
- **DITTO latent-matching intrinsic reward（公式 3）**：把模仿從「回歸 action」變成「在 latent 空間 match 專家軌跡」——這給使用者一個「offline 卻能 on-policy 練習、壓 covariate shift」的機制，正面回應 research_direction_options.md 定義 D「純 offline 下沒得探索」的難題（LUMOS 用 world model imagination 製造 on-policy 練習）。
- **World model（DreamerV2/RSSM）當 backbone**：若使用者要在 play data 上做「belief + 想像中練習」，RSSM 的 `(h_t, z_t)` 本身就是現成的「latent + 動力學」表徵，可與 [[VariBAD|VariBAD]] 的 belief / innovation_notes 第 10 節「含未來重建 ELBO」結合。
- **latent planning（seq2seq CVAE，plan recognition/proposal）**：處理 play 多模態的現成模組（與 [[LatentPlansFromPlay|Play-LMP]]/HULC 一脈相承）。

### 6.4 可當 baseline / 警訊

- **首要 baseline 之一**：LUMOS 與 HULC、MCIL、GCBC 都應列為使用者在 CALVIN 上的對照組（LUMOS 是目前這條線的 SOTA-ish）。使用者方法若主打「加上 task belief / meta 泛化」，要證明能在 LUMOS 之上改善 **unseen task** 而非只是 unseen instruction。
- **警訊一（novelty 要說清楚）**：LUMOS 已經做到「play + 弱語言 + zero-shot 語言操控」，所以使用者**不能**把這些當賣點——賣點必須是 LUMOS 沒有的 **offline meta-RL / task belief / unseen-task 泛化**。否則會被視為 LUMOS 的增量。
- **警訊二（對齊不穩）**：LUMOS 的 No alignment 消融顯示，拿掉語言對齊會操作錯顏色——再次印證 research_direction_options.md 第 5 節「弱語言對齊穩定性是成敗關鍵」，且視覺/latent grounding 要與語言一起對。
- **警訊三（world model 品質是隱性瓶頸）**：LUMOS 自陳「難以事先知道 world model 是否夠好」「解析度越高越好但算力暴增」。使用者若採 world-model 路線，要把「world model 品質 → policy 品質」當成一個風險點（呼應 VariBAD 的 OOD 與重建誤差診斷，innovation_notes 第 14 節）。

---

## 7. 一句話總結

LUMOS 用「從 play data 學 world model → 在其 latent 空間 on-policy 練習（DITTO latent-matching reward 壓 covariate shift）→ <1% 弱語言做 CLIP 式對齊 → zero-shot 遷真機」做出長時程語言操控的 SOTA-ish 結果，幾乎覆蓋了使用者主線的「play + 弱語言 + zero-shot robot control」三塊；但它**沒有 offline meta-RL 的 task belief 與 unseen-task 泛化機制**——這正是使用者把「LUMOS 的 play+弱語言+world-model」接上「T2DA/VariBAD 的 task inference/belief」之後的 novelty 空位。
