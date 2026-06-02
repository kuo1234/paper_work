---
type: paper-note
aliases:
  - "VariBAD"
  - "variBAD"
year: 2020
stage: "1-meta-rl骨架"
tags:
  - meta-rl
  - bayes-adaptive
  - belief
  - VAE
  - 核心
  - 1-meta-rl骨架
summary: "把task belief當BAMDP的state，VAE線上推論belief、policy對belief行動，近似Bayes-optimal探索。"
---
# VariBAD 論文逐段詳細整理

> 論文：**VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning**
> 作者：Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, Yarin Gal, Katja Hofmann, Shimon Whiteson
> 出處：ICLR 2020（arXiv:1910.08348v2）
> 主題：Meta-Reinforcement Learning、Bayes-Adaptive MDP（BAMDP）、Approximate Variational Inference、Bayes-optimal Exploration
> 程式碼：https://github.com/lmzintgraf/varibad
> 整理目標：逐段說明本文要解決的問題、BAMDP 框架、變分推論方法設計、公式意義、訓練流程、實驗設計與主要結論，並對照 PEARL。

---

## 0. 閱讀總覽：這篇論文到底在做什麼？

VariBAD 全名是 **variational Bayes-Adaptive Deep RL**。它要解決的核心問題是：

> 如何在「未知但相關」的任務分布上，學到一個**接近 Bayes-optimal 的探索策略**，讓 agent 在新任務的「第一個 episode 裡邊學邊賺」就能拿到最高的線上回報（online return）？

關鍵概念是 **Bayes-optimal policy**：一個同時最優地權衡 exploration 與 exploitation 的策略。它的行為不只 condition 在環境 state 上，還 condition 在「agent 對目前是哪個 MDP 的不確定性（task uncertainty）」上。理論上這可以用 **Bayes-Adaptive MDP（BAMDP）** 框架算出來，但對絕大多數問題都 intractable（無法計算）。

VariBAD 的做法是用 meta-learning + 變分推論去**近似** BAMDP 的解：

1. 用一個低維、隨機（stochastic）的 latent 變數 `m` 表示「目前是哪個 MDP」。
2. 訓練一個 **VAE**（encoder + decoder）來推論 posterior `q(m | τ:t)`，也就是看過目前為止的軌跡後，對任務的信念分布。
3. 訓練一個 **policy `π(a | s, q(m|τ:t))`**，它直接 condition 在「task posterior（信念）」上，因此能根據不確定性決定要探索還是利用。

和 posterior sampling（如 Thompson sampling / [[PEARL|PEARL]]）最大的差別：posterior sampling 是「抽一個假設 MDP → 對它走最優路 → 再重抽」，探索效率不佳；VariBAD 的 policy 是直接對整個 posterior 行動，學到的是更接近 Bayes-optimal 的**結構化探索**。測試時**不做任何 gradient adaptation、也不用 decoder**，只要 forward pass encoder + policy。

---

## 1. Abstract 逐段解釋

摘要先點出核心：在未知環境中權衡 exploration / exploitation 是「最大化學習期間期望回報」的關鍵。Bayes-optimal policy 能最優地做這件事——它的 action 不只看 state，還看 agent 自己對環境的不確定性。但計算 Bayes-optimal policy 對所有非極小型任務都是 intractable 的。

本文提出 **variBAD**：一種用 meta-learning 去「學會在未知環境中做近似推論」的方法，並把 task uncertainty 直接納入 action selection。

實驗上：在 grid-world 中展示 variBAD 會根據 task uncertainty 做「結構化的線上探索」；在 meta-RL 常用的 MuJoCo domain 上，它取得比現有方法更高的 online return。

---

## 2. Introduction 逐段解釋

### 2.1 問題設定：unknown MDP 與 exploration/exploitation

標準 RL 是針對「reward 與 transition 未知」的單一 MDP 找最優 policy。如果這兩者已知，理論上不需互動就能算出最優 policy。但在未知環境學習，就必須權衡 exploration（了解環境）與 exploitation（採取看起來好的 action）。

在高風險的真實應用（如醫療、教育）尤其重要——你沒有無限次嘗試的機會，**學習期間的每一步回報都算數**。

### 2.2 BAMDP 框架與其困難

Bayes-optimal policy 的 action 同時 condition 在 state 與「對目前 MDP 的信念（belief）」。原則上可以用 **BAMDP**（Martin 1967；Duff & Barto 2002）框架計算：agent 維護一個對「可能環境」的信念分布，把這個 belief 併進 state space，就得到 BAMDP（belief MDP 的特例）。

Bayes-optimal agent 在 BAMDP 中最大化期望回報的方式是：**有系統地去蒐集能快速降低不確定性的資料，但只在這麼做有助於最大化期望回報時才做**。它的表現上界是「擁有真實 MDP 知識的最優 policy」（那個 policy 不需探索）。

困難在於：在 BAMDP 中規劃（planning），即計算 condition 在 augmented state 上的 Bayes-optimal policy，對所有非極小型任務都 intractable。

### 2.3 常見捷徑：posterior sampling 及其不足

常見替代是 **posterior sampling**（Thompson 1933；Strens 2000；Osband 2013）：agent 週期性地（例如每個 episode 開頭）從 posterior 抽一個假設 MDP，然後走「對那個抽樣 MDP 最優的 policy」直到下次抽樣。好處是 planning 變成在普通 MDP 上做，比 BAMDP 容易很多。壞處是它的探索可能非常沒效率、離 Bayes-optimal 很遠。

### 2.4 Gridworld 直覺（Figure 1）

論文用一個 gridworld 例子說明（agent 從左下角出發，要走到灰色區域裡未知的 goal）：

- **(b) Bayes-optimal**：有策略地搜尋 posterior 認為可能的 goal 格子，直到找到——一次掃過去、不重複。
- **(c) Posterior sampling**：抽一個可能 goal → 走最短路過去 → 沒中就重抽——很沒效率（會重訪 state、不確定性沒被最優地降低）。
- **(d) VariBAD**：學到接近 Bayes-optimal 的結構化探索，灰色背景代表它學到的近似 posterior。
- **(e) 效能**：VariBAD 從第二/第三個 rollout 起就接近 Bayes-optimal；posterior sampling 要六個 rollout 才追上。

關鍵挑戰：**在保有 posterior sampling 的 tractability 的同時，學到近似 Bayes-optimal 的 policy**。而且維護 posterior belief 本身（連 posterior sampling 都需要）也可能 intractable。

### 2.5 本文貢獻

結合 Bayesian RL、approximate variational inference、meta-learning。給定 MDP 分布 `p(M)`，用一個學出來的低維隨機 latent 變數 `m` 表示單一 MDP，並**聯合 meta-train** 兩件事：

1. 一個 **VAE**：能在互動的同時，對新任務推論 posterior `q(m | 經驗)`。
2. 一個 **policy**：condition 在這個「對 MDP embedding 的 posterior belief」上，學會在 task uncertainty 下權衡探索與利用。

相較先前 BAMDP 方法只在小 state/action 空間可行、或訓練時依賴 privileged task 資訊，VariBAD 提供一個 tractable、flexible 的做法，**唯一假設是 meta-training 時有任務分布可抽樣**。它替 deep RL 打開「tractable 近似 Bayes-optimal 探索」的路。

---

## 3. Background（背景）逐段解釋

### 3.1 MDP 與 meta-learning 設定

MDP 定義為 tuple `M = (S, A, R, T, T0, γ, H)`：state 集合 S、action 集合 A、reward function R、transition T、初始 state 分布 T0、折扣 γ、horizon H。標準 RL 目標是最大化期望回報：

\[
J(\pi) = \mathbb{E}_{T_0,T,\pi}\left[\sum_{t=0}^{H-1}\gamma^t R(r_{t+1}\mid s_t,a_t,s_{t+1})\right]
\]

**Meta-learning 設定**：有一個 MDP 分布 `p(M)`，從中抽 `Mi = (S, A, Ri, Ti, Ti,0, γ, H)`。跨任務時 reward 與 transition 會變但共享某種結構。index `i` 代表未知的任務描述（如 goal 位置、自然語言指令）或 task ID。Meta-test 時，agent 是用「對 p 抽出的任務，在學習過程中達到的平均回報」來評估。要做好需要兩件事：(1) 利用相關任務學到的先驗知識；(2) 選 action 時對 task uncertainty 做推理以權衡探索/利用。

### 3.2 Bayesian RL 與 BAMDP（核心公式）

MDP 未知時，最優決策必須權衡探索/利用。Bayesian 做法：假設 transition 與 reward 服從先驗 `b0 = p(R, T)`。Agent 維護信念

\[
b_t(R,T) = p(R,T\mid \tau_{:t})
\]

即看過經驗 `τ:t = {s0, a0, r1, s1, …, st}` 後對 MDP 的 posterior。

把 belief 併進 state，得到 **hyper-state** `s+_t ∈ S+ = S × B`（B 是 belief space）。其 transition：

\[
T^+(s^+_{t+1}\mid s^+_t,a_t,r_t)
= \underbrace{\mathbb{E}_{b_t}[T(s_{t+1}\mid s_t,a_t)]}_{\text{對 posterior 取期望的環境轉移}}\;
\underbrace{\delta\!\big(b_{t+1}=p(R,T\mid\tau_{:t+1})\big)}_{\text{belief 依 Bayes 規則確定性更新}}
\tag{1}
\]

hyper-state 上的 reward 是 posterior 下的期望 reward：

\[
R^+(s^+_t,a_t,s^+_{t+1}) = \mathbb{E}_{b_{t+1}}[R(s_t,a_t,s_{t+1})]
\tag{2}
\]

這就構成 **BAMDP** `M+ = (S+, A, R+, T+, T0+, γ, H+)`。它是 belief MDP 的特例：一般 belief MDP 的 hidden state 會隨時間變，但在 BAMDP 中，belief 是對「transition 與 reward function」的信念，而它們對固定任務是**不變的**。

agent 目標變成最大化 BAMDP 上的期望回報：

\[
J^+(\pi) = \mathbb{E}_{b_0,T_0^+,T^+,\pi}\left[\sum_{t=0}^{H^+-1}\gamma^t R^+(r_{t+1}\mid s^+_t,a_t,s^+_{t+1})\right]
\tag{3}
\]

**重點：H（MDP horizon）與 H+（BAMDP horizon）的區別。** 兩者常一致，但若希望 agent 在「前 N 個 MDP episode」內表現 Bayes-optimal，則 `H+ = N × H`。最優地權衡探索/利用，**強烈取決於還剩多少時間**（剩越多，越值得花步數去蒐集資訊）。

(3) 由 Bayes-optimal policy 達到，它會自動權衡：只在「有助於在 horizon 內最大化期望回報」時才做探索。BAMDP 框架很 powerful，但解它對多數有趣問題 hopelessly intractable。三大困難：

1. 通常不知道真實 reward/transition model 的參數化形式。
2. belief update（算 posterior `p(R,T|τ:t)`）通常 intractable。
3. 即使有正確 posterior，在 belief space 規劃也通常 intractable。

VariBAD 同時 meta-learn：reward/transition function、如何在未知 MDP 做推論、如何用 belief 最大化線上回報。因為 Bayes-adaptive policy 與推論框架是 **end-to-end** 一起學，**測試時不需 planning**，且不需 privileged task 資訊。

---

## 4. Method：Bayes-Adaptive Deep RL via Meta-Learning（核心方法）

### 4.1 用 latent 變數 `m` 取代直接建模 R、T

在 meta-learning 設定下，各 MDP 獨有的 reward/transition 未知但跨任務共享結構。存在一個真實的 `i`（task 描述或 ID），但 agent 沒有存取權。於是用一個學出來的**隨機 latent 變數 `mi`** 來表示。對給定 `Mi`：

\[
R_i(r_{t+1}\mid s_t,a_t,s_{t+1}) \approx R(r_{t+1}\mid s_t,a_t,s_{t+1};\,m_i)
\tag{4}
\]
\[
T_i(s_{t+1}\mid s_t,a_t) \approx T(s_{t+1}\mid s_t,a_t;\,m_i)
\tag{5}
\]

其中 `R`、`T` 跨任務共享。因為沒有真實 task 描述，需要從到 t 為止的經驗推論 `m`：

\[
\tau_{:t} = (s_0,a_0,r_1,s_1,a_1,r_2,\dots,s_{t-1},a_{t-1},r_t,s_t)
\tag{6}
\]

目標即推論 posterior `p(m | τ:t)`。**好處**：reward/transition function 可能有上百萬參數，但只要對一個小向量 `m` 做推論就夠了——對深度學習特別有利。

### 4.2 Approximate Inference（近似推論，VAE 部分）

精確 posterior 算不出來（沒有 MDP 存取權，且對任務 marginalise 不可行）。所以學一個環境模型 `pθ(τ:H+ | a:H+−1)`（參數 θ）以及一個 **amortised inference network** `qφ(m | τ:t)`（參數 φ），讓 runtime 每個 t 都能快速推論。模型學習目標是最大化：

\[
\mathbb{E}_{\rho(M,\tau_{:H^+})}\big[\log p_\theta(\tau_{:H^+}\mid a_{:H^+-1})\big]
\tag{7}
\]

(7) intractable，所以最大化它的 **tractable lower bound（ELBO）**：

\[
\mathbb{E}_{\rho}[\log p_\theta(\tau_{:H^+})]
\;\ge\;
\mathbb{E}_{\rho}\Big[\;\underbrace{\mathbb{E}_{q_\phi(m\mid\tau_{:t})}[\log p_\theta(\tau_{:H^+}\mid m)]}_{\text{reconstruction（解碼器）}}
\;-\;
\underbrace{\mathrm{KL}\big(q_\phi(m\mid\tau_{:t})\,\|\,p_\theta(m)\big)}_{\text{KL：posterior 與 prior}}\Big]
= \mathrm{ELBO}_t
\tag{8}
\]

**關鍵設計（與一般 VAE 不同）：**

- 在 timestep t，只用**過去**軌跡 `τ:t` 編碼出**當前** posterior `q(m|τ:t)`（這是當下能拿來推論的全部資訊）。
- 但 decoder 要**重建整段軌跡 `τ:H+`，包含未來**（訓練時才有未來資訊）。
- 解碼未來很重要：這樣 variBAD 才學會「**從過去推論未見過的 state**」，而不是只記住看過的東西。

**Prior 設計**：把 prior 設成「上一時刻的 posterior」`qφ(m|τ:t−1)`，初始 prior `qφ(m) = N(0, I)`。

**Reconstruction 項的分解**（公式 9）：

\[
\log p(\tau_{:H^+}\mid m, a_{:H^+-1})
= \log p(s_0\mid m) + \sum_{i=0}^{H^+-1}\big[\log p(s_{i+1}\mid s_i,a_i,m) + \log p(r_{i+1}\mid s_i,a_i,s_{i+1},m)\big]
\tag{9}
\]

即拆成「初始 state 分布 + 各步的 transition 解碼 + reward 解碼」。

### 4.3 訓練目標（Training Objective）

用神經網路表示各元件：

1. **Encoder** `qφ(m | τ:t)`（參數 φ）。
2. 近似 **transition** `T' = pθ^T(s_{i+1}|s_i,a_i;m)` 與近似 **reward** `R' = pθ^R(r_{i+1}|s_t,a_t,s_{i+1};m)`（共享參數 θ）。
3. **Policy** `πψ(a_t | s_t, qφ(m|τ:t))`（參數 ψ，且依賴 φ）。

policy 同時 condition 在「環境 state」與「對 `m` 的 posterior」上。和 2.2 的 BAMDP 差別是：這裡學的是「對 MDP embedding 的統一分布」，而不是直接對 transition/reward 建模——參數更少、推論更容易，且能用所有任務的資料學共享的 R、T。posterior 可用分布參數表示（如 Gaussian 的 mean、std）。

**整體目標：**

\[
\mathcal{L}(\phi,\theta,\psi)
= \mathbb{E}_{p(M)}\Big[\,J(\psi,\phi) + \lambda\sum_{t=0}^{H^+}\mathrm{ELBO}_t(\phi,\theta)\,\Big]
\tag{10}
\]

- `J` 是 RL（policy）目標；`λ` 權衡「監督式模型學習目標」與「RL loss」（因為 φ 同時被 model 與 policy 共用）。
- 期望用 Monte Carlo 近似；ELBO 用 reparameterisation trick（Kingma & Welling 2014）優化；t=0 用 prior `N(0,I)`。
- 過去軌跡用 **recurrent network（RNN/GRU）** 編碼（如 [[RL2|RL²]] / Duan 2016、Wang 2016），但也可用 Deep Sets、Neural Processes、PEARL 那類 encoder。
- (10) 的 ELBO 對**所有 context 長度 t** 都出現：讓 variBAD 學會**線上推論**、隨資料增多降低不確定性。實作上若 H+ 很大，可對隨機 t 子抽樣固定數量的 ELBO 項以省算力。

**重要實作觀察（很關鍵的 trick）：**

> 雖然 φ 被 model 與 policy 共享，但作者發現「**把 RL loss 反傳穿過 encoder 通常沒必要**」。不這麼做反而：(a) 大幅加速訓練；(b) 不用去 trade off 兩個 loss；(c) 避免兩個對立 loss 的梯度互相干擾。

因此實驗中 policy 與 VAE 用**不同 optimiser、不同 learning rate**，且用**不同 data buffer**：policy 只用最近資料（on-policy 演算法），VAE 維護一個獨立、較大的軌跡 buffer。

**Meta-test**：在隨機抽的測試任務上 roll out policy（只做 encoder + policy 的 forward pass）。**decoder 測試時不用，也不做 gradient adaptation**——policy 在 meta-training 時就已學會近似 Bayes-optimal 行動。

---

## 5. Related Work（與其他方法的關係，重點對照）

### 5.1 Meta-RL

- **RL²（Wang 2016；Duan 2016）**：用 recurrent network 的動態做快速適應，把上一步 action/reward 當輔助輸入，task 內學習全發生在 RNN 動態中。**若把 variBAD 的 decoder 與 VAE 目標拿掉，就退化成 RL²。** 兩者差別：variBAD 多了 (a) 一個**隨機 latent 變數**（表示不確定性的 inductive bias）、(b) 一個 **decoder 重建過去與未來** transition/reward（當作 auxiliary loss，把 task 編進 latent 並推論未見 state）。
- **[[MAML|MAML]] / Reptile（Finn 2017；Nichol & Schulman 2018）**：學一個初始化，測試時幾步 gradient 就好。但沒有直接處理「初始 policy 需要探索」的問題（E-MAML、ProMP 才補上）。MAML/ProMP 模型輕（多為前饋 policy）；RL²、variBAD 用 recurrent 模組，較重但支援線上適應。
- 這些 gradient-adaptation 方法多半在測試時**把探索（gradient 前）與利用（gradient 後）拆開**，因此較不 sample efficient。

### 5.2 Skill / Task Embeddings

許多方法學（變分）task/skill embedding。VariBAD 的差異在於 **embedding 表示什麼、以及怎麼用**：variBAD 的 policy condition 在「對 MDP 的 posterior 分布」上，能對 task uncertainty 推理並**線上**權衡探索/利用；其目標 (8) 明確為 Bayes-optimal 行為優化。多數同類方法測試時要在 latent space 做適應或學新 embedder；variBAD 測試時不用 model（model-based planning 是未來方向）。

### 5.3 Bayesian RL

Bayesian RL 用不確定性支援 action selection、納入先驗。Bayes-optimal policy 原則上可用 BAMDP 算，但對非極小型任務 intractable，現有方法侷限在小/離散 state-action 或離散任務集。VariBAD 用 meta-learning + 變分推論打開 deep RL 的近似 Bayes-optimal 探索之路，**唯一假設是能在一組相關任務上 meta-train**。代價是因用深度網路而**缺乏部分方法的形式化保證**。

- 與 **Humplik et al. (2019)** 最接近：兩者都把 policy condition 在 posterior over MDP 上。但 Humplik 用 privileged 資訊（如 task 描述）來 meta-train；**variBAD 以非監督方式 meta-learn belief，不需 privileged task 資訊**。
- 與 **posterior sampling（Strens 2000；Osband 2013，PEARL 亦屬此類）** 同樣估 posterior over MDP，但 posterior sampling 是抽單一假設 MDP 走最優路，效率較差、學習期間 return 較低。

### 5.4 與 POMDP 的關係

VariBAD 聚焦 **BAMDP**——POMDP 的特例：hidden state 即 transition/reward function，agent 要對它維護 belief。一般 POMDP 的 hidden state 每步可變；BAMDP 的底層任務（因此 hidden state）對單一任務**固定**。VariBAD 利用此性質，學一個**時間上固定**的 embedding（不像 Igl 2019 用 filtering 追蹤會變的 hidden state）。其他 BAMDP 方法（如 Lee 2019）常離散化 latent 並用 Bayesian filtering 更新 posterior，較準但較不可擴展。

---

## 6. 實驗（Experiments）

### 6.1 Gridworld（didactic）

- 5×5 gridworld，goal 均勻隨機（不在起點附近）。Goal 不可見 → 製造 task uncertainty → 必須探索。
- Actions：上、右、下、左、stay（確定性執行）。MDP horizon `H = 15`；BAMDP horizon `H+ = 4 × H = 45`（要求在 4 個 MDP episode 內最優）。
- Sparse reward：非 goal 格 −0.1，goal 格 +1。最佳策略：探索到找到 goal，之後留在 goal 或被 reset 後回到 goal。latent 維度 = 5。

**結果（Figure 3）：**

- (a) Rollout：藍色背景視覺化 posterior belief（用學到的 reward function）。variBAD 學到正確 prior，並隨時間正確調整 belief——已訪過的格子預測無 reward，並探索其餘格子直到找到 goal。
- (b) Reward 預測：每條線是一格的「得 reward 機率」。隨資料增加，越來越多格被排除（p=0），最終找到 goal。
- (c) Latent space（5 維）：找到 goal 後 posterior **收斂**——variance 掉到接近 0、mean 穩定。

**可解釋性是 variBAD 的好處**：能直接從 decoder 的預測與 latent space 變化，看到 agent 對環境的信念。其行為接近 Figure 1e 的 Bayes-optimal policy。（附錄補充：20 seeds 中 4 次學到完全 Bayes-optimal，RL² 為 0 次；其餘也都很接近。）

### 6.2 MuJoCo 連續控制 meta-RL

- 環境：**AntDir / HalfCheetahDir**（前進或後退，僅兩個任務）、**HalfCheetahVel**（不同目標速度）、**Walker**（隨機化系統參數）。
- 目標是「在**單一 episode 內**邊學邊最大化 reward」，所以**第一個 rollout 之後的表現與本文目標無直接關係**。

**結果（Figure 4）：**

- **只有 variBAD 與 RL² 能在單一 episode 內適應任務。** variBAD 在 HalfCheetahDir 上勝過 RL²，且 RL² 學得較慢、較不穩定。
- 即使第一個 rollout 含探索步，variBAD 仍能逼近「擁有真實任務描述的 oracle policy」（差距很小）。
- PEARL、E-MAML、ProMP 不是為「單一 rollout 內最大化 reward」設計，在此情境表現差，需要明顯更多互動才好。PEARL（類 posterior sampling）要到第三個 episode 才開始表現好。
- 註：PEARL 在 meta-training 的 **frame 數上更 sample efficient**（因為是 off-policy）；on-policy vs off-policy 與本文貢獻正交，把 variBAD 擴到 off-policy 是未來工作。

---

## 7. 結論與未來工作

VariBAD 用 meta-learning 利用相關任務知識、在未知環境做近似推論，來逼近 Bayes-optimal 行為。Gridworld 上接近 Bayes-optimal；MuJoCo 上在「單一 episode 內的 reward」勝過現有方法。它替 deep RL 打開 tractable 近似 Bayes-optimal 探索之路。

**未來方向：**

1. 測試時目前不用 decoder——可改用 decoder 做 model-predictive planning，或用它判斷預測有多錯（指示是否 out-of-distribution、需要再訓練）。
2. **Out-of-distribution（OOD）泛化**：訓練/測試任務分布不同時會出兩個問題——(a) 推論程序會錯（prior 與/或 posterior update 錯）；(b) policy 無法詮釋改變過的 posterior。此時可能需要進一步訓練 encoder/decoder、更新 policy、或顯式 planning。
3. （附錄 B.1 提的小方向）目前 prior 固定為 N(1,0)/N(0,I)，但實驗看到找到 goal 前 variance 還先上升一陣，顯示**學一個 task-specific 的 prior** 是自然的延伸。

---

## 8. 附錄重點（實作細節，對重現很重要）

### 8.1 Gridworld 超參數

- RL 演算法：**A2C**；policy steps 60；parallel processes 16；γ = 0.95。
- ELBO loss 係數 1.0；Policy LR 0.001；VAE LR 0.001；**task embedding size = 5**。
- Policy：2 層、各 32 nodes、TanH。
- **Encoder**：FC 40 nodes → **GRU hidden 64** → 輸出 10（µ 與 σ 各 5），ReLU。
- Reward decoder：2 層、各 32 nodes、**25 個輸出 head**（對應 25 格），ReLU；loss 用 **binary cross entropy**。

### 8.2 MuJoCo 超參數

- RL 演算法：**PPO**；batch 3200；epochs 2；minibatches 4；clip 0.1；RL loss 用 **Huber loss**。
- **ELBO 中 KL 項權重 0.1**；Policy LR 0.0007；VAE LR 0.001；**task embedding size = 5**。
- Policy：2 層、各 128 nodes、TanH。
- Encoder：state/action/reward encoder（FC 32/16/16）→ **GRU hidden 128** → 輸出 5，ReLU。
- Reward decoder：2 層（64、32 nodes），ReLU；loss 用 **MSE**。
- **所有 MuJoCo 環境只用 reward decoder（不解碼 transition）**，連 Walker（dynamics 會變）也是只用 reward decoder 表現較好。

### 8.3 與 RL² 的細部比較 / 穩定性（附錄 B、C）

- variBAD 與 RL² 都訓練在多 rollout（gridworld H+=4×H=60；MuJoCo H+=2×H=400），用「done flag」讓 agent 知道何時被 reset，故可跨多 rollout 評估而不重置 RNN hidden state。
- **RL² 在跨多 rollout 時常不穩**（例如 CheetahVel：reset 後表現掉下來）。推測因為 RL² 的 128 維 hidden state 在 state 突變時會劇烈位移、無法正確代表任務；且 Cheetah 達到正確速度後可從自身速度推任務、停止推論，reset 後速度突變就出問題。**variBAD 較少這問題**：它的 latent 被訓練成**只表示任務**，posterior 收斂後不隨更多資料改變，reset 回起點不需重做推論。

### 8.4 Runtime（HalfCheetahDir 粗估）

- ProMP / E-MAML：5–8 小時（無 recurrent，最快）。
- **variBAD：48 小時**；RL²：60 小時；PEARL：24 小時（off-policy，frame 上最省）。
- variBAD 比 RL² 快的原因：**不把 RL loss 反傳穿過 recurrent encoder**，使 PPO 的 minibatch 更新不必重算 embedding，省下大量 forward/backward。

### 8.5 ELBO 完整推導（附錄 A）

從 `log pθ(τ:H)` 出發，乘除 `qφ(m|τ:t)`、用 Jensen 不等式把 log 換到期望內，再展開即得 (11)：

\[
\mathbb{E}_\rho[\log p_\theta(\tau_{:H})]
\ge
\mathbb{E}_\rho\big[\mathbb{E}_{q_\phi(m\mid\tau_{:t})}[\log p_\theta(\tau_{:H}\mid m)] - \mathrm{KL}(q_\phi(m\mid\tau_{:t})\,\|\,p_\theta(m))\big]
= \mathrm{ELBO}_t
\tag{11}
\]

---

## 9. 與 PEARL 的對照總表（與本研究專案最相關）

| 面向 | PEARL | VariBAD |
| --- | --- | --- |
| Task latent 表示 | `q(z|c)`，context = transition 集合，permutation-invariant（product of Gaussians） | `q(m|τ:t)`，用 RNN/GRU 線上編碼整段過去軌跡 |
| latent 學習訊號 | critic（Bellman）loss + KL，latent 只要「對控制有用」 | **VAE reconstruction（解碼過去+未來 transition/reward）+ KL**，當 auxiliary loss |
| 探索機制 | **posterior sampling**（抽 z → 走最優 → 重抽） | **policy 直接 condition 在 posterior 上**，學近似 **Bayes-optimal** 結構化探索 |
| 訓練 RL | **off-policy（SAC）**，meta-training frame 上很省 | **on-policy（A2C / PPO）**；off-policy 列為未來工作 |
| 測試時適應 | 收 context → 重推 posterior（無 gradient） | 純 forward（encoder+policy），**無 gradient、不用 decoder** |
| 主打指標 | meta-training sample efficiency（20–100×） | **單一 episode 內的 online return / Bayes-optimality** |
| 何時表現好 | PEARL 約第 3 episode 起才好 | 第 1 個 rollout 就接近 oracle |
| 理論框架 | probabilistic context + posterior sampling | **BAMDP（belief 併入 state，Bayes-optimal policy）** |
| 是否用 decoder | 不用 generative decoder | 用 decoder 當 auxiliary（但測試時不用） |

---

## 10. 一句話總結

VariBAD 把「對任務的不確定性（task belief）」當成 BAMDP 的 state 的一部分，用一個 VAE 線上推論這個 belief、再讓 policy 直接對 belief 行動，於是在「邊學邊賺」的單一 episode 設定下，學到比 RL²、MAML、PEARL 都更接近 **Bayes-optimal** 的結構化探索——代價是 on-policy、訓練較慢、缺形式化保證，且對 out-of-distribution 任務尚需額外處理。
