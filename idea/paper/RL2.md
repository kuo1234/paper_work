---
type: paper-note
aliases:
  - "RL2"
  - "RL²"
  - "RL^2"
year: 2016
stage: "1-meta-rl骨架"
tags:
  - meta-rl
  - recurrent
  - fast-learner
  - baseline
  - 1-meta-rl骨架
summary: "把fast learner塞進RNN hidden state，forward-pass即適應；是forward-adaptation baseline家族最樸素者。"
---
> **論文標題**：RL$^2$: Fast Reinforcement Learning via Slow Reinforcement Learning
> **作者**：Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, Pieter Abbeel
> **出處 / 年份**：arXiv:1611.02779（ICLR 2017 投稿版本），2016
> **主題**：meta-RL via recurrent policy；以 RNN 的 hidden state 編碼一個「快速 RL 學習器」，外層用標準（慢）RL 訓練。
> **整理目標**：細讀 RL$^2$ 的問題建構、policy 表示、optimisation pipeline，並抽出與我的主線研究（robot play data + 弱語言標註 + offline meta-RL + 文字→task embedding 做 zero-shot）相關的可借用零件、可當 baseline / ablation 的設計，以及與 MAML 式 gradient-based meta-RL 的本質差異。

---

## 0. 閱讀總覽（白話）

RL$^2$ 的核心想法可以用一句話講清楚：**不要再手刻「快」的 RL 演算法，乾脆用一般 RL 去訓練一顆能夠在 inference time 自己學新任務的 RNN policy**。

- 你有一個任務分佈 $\rho_M$（例如不同的 multi-armed bandit、不同的 tabular MDP、不同的 maze）。每次 trial 都從中 sample 一個 MDP，agent 跟它互動 $n$ 集（episodes）。
- Policy 是一顆 GRU，輸入除了 state 之外還包含上一步的 action、reward、termination flag。**最關鍵的設計：hidden state 在同一個 trial 的多個 episode 之間是不重置的**，跨 trial 才重置。
- 訓練目標是「整個 trial 的折扣累積 reward」（而不是單一 episode），這等價於最小化 cumulative pseudo-regret。
- 訓練演算法用 TRPO（first-order 版）+ GAE，相當於把 meta-learning 問題變成一個 outer-loop RL 問題。

訓練完成之後，**RNN 的 weight 是「慢」演算法的產物，RNN 的 activation 才是「快」演算法的內部狀態**——也就是說，weights 編碼的是「如何從 trajectory history 推論 task 並決策」的 prior 與策略，而 activations 則是針對手上這個未知 MDP 的 belief / 學習進度。所以 hidden state 可以被解讀成一個 fast learner。

實驗在 (1) 隨機 bandit、(2) 隨機 tabular MDP、(3) ViZDoom 視覺迷宮三個尺度上都驗證 RL$^2$ 能逼近甚至超過手刻最優演算法（Gittins index、PSRL、UCRL2、BEB 等）。

---

## 1. Abstract 逐段精讀

**第一段（問題與動機）**：Deep RL 學習過程動輒要 tens of thousands episodes，動物卻能在很少 trial 內學新任務，差距來自 prior knowledge。本文不再「設計一個快的 RL 演算法」，而是把快的演算法 **表示成一顆 RNN，用一般 RL 去學它的權重**。

**第二段（方法輪廓）**：方法稱為 RL$^2$。RNN 收到的輸入跟一般 RL 演算法會收到的差不多——observation、action、reward、termination flag——而且 hidden state 跨 episode 保留（在同一個 MDP/trial 內）。RNN 的 activations 就是「快」RL 演算法針對當前未知 MDP 的內部狀態。

**第三段（實驗）**：在小尺度的隨機 MAB 與 tabular MDP 上，RL$^2$ 的表現逼近具有 optimality guarantees 的 human-designed 演算法；在大尺度的 ViZDoom vision-based navigation 上也展現可擴展性。

---

## 2. Introduction 重點

- **痛點**：state-of-the-art Atari agent 需要相當於 40 天不眠不休的遊玩經驗，而人類玩家 2 小時就上手。差距在 prior。
- **既有方向**：Bayesian RL 理論上完美，但 Bayesian update 在簡單情況以外都不可解；實務做法（guided policy search、PILCO）對環境要假設多、或在高維下計算爆炸。
- **本文做法**：把「agent 的學習過程本身」當成可被 RL 優化的目標。
  - Objective 是在一個 task 分佈 $\rho_M$ 上平均；這個分佈就是要 distill 進 agent 的 prior。
  - Agent = RNN，輸入除了 observation，還有歷史 reward、action、termination flag。
  - Hidden state 跨 episode 保留 ⇒ agent 能用 activations 來「在心裡學」。
  - Learned agent **本身就是 learning algorithm**，部署時對手上任務自我適應。
- **三組實驗**：MAB、tabular MDP（兩者有理論最優演算法可比）、ViZDoom 視覺導航（高維）。

---

## 3. 方法與核心公式

### 3.1 標準 MDP 與 policy gradient（Preliminaries）

定義一個有限 horizon、折扣的 MDP $M = (\mathcal S, \mathcal A, P, r, \rho_0, \gamma, T)$。對隨機 policy $\pi_\theta$，目標是最大化期望折扣回報：

\[
\eta(\pi_\theta) = \mathbb E_\tau\!\left[\sum_{t=0}^{T} \gamma^{t}\, r(s_t, a_t)\right],
\quad s_0 \sim \rho_0,\ a_t \sim \pi_\theta(\cdot|s_t),\ s_{t+1} \sim P(\cdot|s_t,a_t).
\]

符號說明：
- $\tau$：整條 trajectory $(s_0, a_0, s_1, a_1, \dots)$。
- $\gamma \in [0,1]$：折扣因子。
- $T$：episode horizon。

到這裡為止是一般 RL。

### 3.2 RL$^2$ 的核心 formulation：把「學 RL 演算法」當成一個 outer-loop RL 問題

**新增物件**：
- $\mathcal M$：一群 MDP 的集合。
- $\rho_M : \mathcal M \to \mathbb R_+$：任務分佈，agent 只需可從中 sample。
- $n$：每個 MDP 允許互動的 episode 數。
- **Trial**：與**同一個** MDP 連續互動 $n$ 個 episode 的整段過程。

**Agent–environment 互動流程**（Figure 1）：
- 每個 trial 開始時抽一個新 MDP $M \sim \rho_M$。
- 每個 episode 開始時，對應 MDP 的 $\rho_0$ 抽 $s_0$。
- Agent 觀察 $s_t$，輸出 $a_t$；環境回 $r_t$ 與下一狀態 $s_{t+1}$；若該 episode 結束則 termination flag $d_t=1$，否則 $0$。
- 下一步 policy 的輸入為四元組 $(s_{t+1}, a_t, r_t, d_t)$，並以 $h_{t+1}$ 為 hidden state，產生 $h_{t+2}$ 與 $a_{t+1}$。
- **同一 trial 內的多個 episode 之間，hidden state $h$ 是保留的**；跨 trial 才重置。

**目標函數**：在 RL$^2$ 中，要最大化的是整個 trial 的折扣累積 reward（而非單一 episode）：

\[
\eta_{\mathrm{RL}^2}(\pi_\theta)
= \mathbb E_{M \sim \rho_M}\!\left[
   \mathbb E_{\tau^{(1:n)}\sim \pi_\theta,\, M}\!\left[
     \sum_{i=1}^{n}\sum_{t=0}^{T-1} \gamma^{t}\, r^{(i)}_t
   \right]
\right].
\]

其中 $\tau^{(1:n)}$ 是該 trial 內 $n$ 個 episode 串接的軌跡。論文指出**最大化此目標等價於最小化 cumulative pseudo-regret**（Bubeck & Cesa-Bianchi, 2012）。

**為什麼這樣 setup 會自然逼出 exploration/exploitation 與 belief-tracking**：
- 既然不同 MDP 需要不同最優策略，agent 只能根據 history 推論「我現在在哪個 MDP」並據此調整行為。
- 因此 agent 被迫整合 $(\text{past actions}, \text{rewards}, \text{termination flags})$，且要持續更新策略。
- 這個強制條件等於在 end-to-end 的 outer-loop 訓練中，把「fast RL algorithm」逼到 RNN 的權重裡。

**Inner problem 是 POMDP 也適用**：論文是用 MDP 講解清楚，但只要把 $s_t$ 換成 observation $o_t$，整套 formulation 對 POMDP 一樣 work；ViZDoom 視覺導航就是 POMDP 版本。

### 3.3 Policy representation

- RNN 用 GRU（Cho et al., 2014），解決 vanishing/exploding gradient。
- 輸入 $(s, a, r, d)$ 經 embedding 函數 $\phi(s,a,r,d)$ 後送入 GRU。
- GRU 輸出接一層 fully connected + softmax，得到 discrete action 分佈。
- 作者實驗過「每個 sampled MDP 的 episode 開始時 reset 部分 hidden state」的變體，但沒帶來改善，最後用最單純的版本。

### 3.4 Policy optimization

- 把 RL$^2$ 視為一個普通 RL 問題，直接用 off-the-shelf 演算法。
- 採用 **TRPO（first-order）**，理由是經驗表現好、不太需要超參調整。
- Value baseline 用另一顆同樣以 GRU 為主幹的 RNN。
- Optional **GAE** 進一步降低 variance。

---

## 4. 演算法流程（pseudo-code 視角）

雖然原文沒有完整 pseudo-code，可整理成：

1. 初始化 RNN policy $\pi_\theta$（與 baseline RNN $V_\phi$）。
2. **重複**直到收斂：
   1. **Sample batch of trials**：對每個 trial，
      - 從 $\rho_M$ 抽一個 MDP $M$。
      - 重置 hidden state $h_0$。
      - 重複 $n$ 個 episode：每個 episode 開始時從 $M$ 抽 $s_0$；hidden state 沿用上一 episode 結束時的值；按 $(s_t, a_{t-1}, r_{t-1}, d_{t-1}) \to h_t \to a_t$ rollout 到 episode 結束。
   2. 收集所有 trial 的軌跡，用整個 trial 的 reward 作為訓練訊號。
   3. 計算 advantage（GAE on top of $V_\phi$）。
   4. 用 TRPO 一步更新 $\theta$（first-order 變體）。
   5. 用回歸更新 baseline $V_\phi$。

---

## 5. 實驗與結論

### 5.1 Multi-armed bandits（章節 3.1）

- Setup：每個 arm 是 Bernoulli$(p_i)$，$p_i \sim \mathrm{Uniform}[0,1]$；測 $k \in \{5,10,50\}$、$n \in \{10, 100, 500\}$。
- Baselines：Random、**Gittins index**（折扣無限 horizon 下的 Bayes 最優）、UCB1、Thompson sampling (TS)、optimistic TS、$\epsilon$-greedy、greedy。
- 結果（Table 1）：RL$^2$ 在大多數設置與 Gittins index 等手刻最優演算法 **統計上不顯著差異**。在 $n=10$、$k=50$ 這種短 horizon 還小贏 Gittins。
- 唯一明顯落後：$n=500, k=50$（最難的場景）。作者再做了個 control experiment：用 Gittins 跑出來的 trajectory 對 RL$^2$ 做 supervised learning，發現可逼近 Gittins 水準——**結論：bottleneck 在 outer-loop 的 RL 算法，不在 RNN 架構**。

### 5.2 Tabular MDPs（章節 3.2）

- Setup：$|\mathcal S|=10, |\mathcal A|=5$；reward $\mathcal N(\mu,1)$、$\mu \sim \mathcal N(1,1)$；transition 用 flat Dirichlet 抽，episode horizon $T=10$。
- Baselines：PSRL（posterior sampling RL）、OPSRL、UCRL2、BEB、$\epsilon$-greedy、greedy。
- 結果（Table 2）：
  - 在 $n=10, 25, 50$ 時，RL$^2$ **大贏** 所有 baseline。
  - 在 $n=75, 100$ 時，RL$^2$ 開始被 OPSRL/PSRL 反超。
- 解讀：小 $n$ 時，要估的 transition 參數有 140 個 d.o.f.，10 個 episode 根本估不準；RL$^2$ 學到了「該何時放棄完整估計、提早 exploit」的 prior，這是手刻演算法不會做的事。隨 $n$ 增加，outer-loop RL 變難（要 credit assignment 跨更長的 horizon），優勢就被吃掉。

### 5.3 Visual navigation（章節 3.3，ViZDoom）

- Setup：5x5 maze；agent 要找紅色 target；reward `+1` 到達、`-0.001` 撞牆、`-0.04` 每步。一個 trial 內 maze 與 target 固定，agent 互動多 episode。
- 訓練時用 1000 個 maze configurations，測試另抽 1000 個；同時測 9x9 大 maze 與 episode 數延長到 5 的 extrapolation。
- 結果（Table 3）：
  - 第二 episode 的成功軌跡長度明顯短於第一 episode（小 maze 由 52.4 降到 39.1，大 maze 180.1 降到 151.8），代表 agent 有用上前一 episode 學到的 maze 知識。
  - 成功率在 5 個 episode 內都維持高水準（小 99%、大 95%+）。
  - 失敗模式：偶爾 agent 在第二 episode 「忘了」target 在哪，繼續探索（Figure 6c, 6d）。

### 5.4 Discussion 與結論

- RL$^2$ 證實「把 fast RL 編進 RNN，用 slow RL 訓練」這條路線可行。
- 兩個明確 bottleneck：(i) outer-loop RL 演算法、(ii) 長 horizon 的架構（可能要更好的 memory）。
- 一般 RL 演算法 + generic RNN 已能逼近理論最優——若把 episodic 結構放進 architecture / algorithm，預期可進一步提升。

---

## 6. 相關工作（章節 4）重點

論文把 RL$^2$ 放在三條譜系的交叉點：

1. **Meta-RL 早期**：Ishii et al. (2002)、Schweighofer & Doya (2003) 自動 tune learning rate、temperature；Wilson et al. (2007) 用 hierarchical Bayesian 維持 dynamics posterior。
2. **Meta-learning as optimisation（gradient-descent 視角）**：Younger et al., 2001；Santoro et al., 2016（MANN）；Vinyals et al., 2016（Matching Networks）。這些是 supervised one-shot learning，RL$^2$ 把它推廣到 RL，因此 agent 還要學「探索」。
3. **Learning to optimize**：Hochreiter et al., 2001；Andrychowicz et al., 2016；Li & Malik, 2016。這些 meta-learner 對 model 做 explicit gradient update；RL$^2$ 則**沒有 explicit parametric policy 被更新**——RNN 同時是 meta-learner 也是 fast policy。
4. **POMDP / Dual control theory**：RL$^2$ 的 inner problem 本質上是把「未知 MDP」reduce 成 POMDP（task identity 為 hidden state），可追溯到 Feldbaum (1960) 的 dual control。

---

## 7. 附錄與實作細節

- 在 $t=0$ 沒有「上一個 action / reward / termination」時，用 placeholder：action 用 action 0 的 embedding，reward / termination flag 用 0。
- ReLU activations + weight normalization（無 data-dependent init）；hidden-to-hidden 用 orthogonal init；其他用 Xavier；bias 全 0。
- TensorFlow + rllab 實作；classical baselines 用 TabulaRL package。
- 三組任務的 TRPO 超參（共通：discount 0.99、256 GRU units、mean KL 0.01）：
  - **MAB**：GAE λ=0.3、batch size 250k、最多 1000 iters；state 用常數 0、action 用 one-hot。
  - **Tabular MDP**：GAE λ=0.3、batch size 250k、最多 10000 iters；state/action one-hot concat。
  - **ViZDoom**：GAE λ=0.99、batch size 50k、最多 5000 iters；影像 40x30、2 層 conv（16 filters, 5x5, stride 2）；action 256 維 embedding；policy 與 baseline **共享 backbone**，作者觀察到這在視覺任務下穩定且效果較佳。

---

## 8. 重點問題深挖（read.md 必答）

### 8.1 為什麼 RNN 的 hidden state 可以被解讀成「fast learner」？

要從訓練目標 + 架構結合來看：

- **訓練目標逼出 belief**：目標是整個 trial 的累積 reward。要在 trial 中愈來愈會 act，agent 必須在 hidden state 裡 **維護某種「我目前認為自己在哪個 MDP」的 belief**。理論上，最優策略是在 belief MDP 上 act。
- **架構提供 belief 載體**：hidden state 跨 episode 保留、輸入又包含 reward / termination flag，於是 GRU 能把「歷史 (s, a, r, d) 累計成內部統計量」。這就是 Bayesian RL 裡 belief update 的 functional approximation。
- **權重凍結後仍能適應新 MDP**：deployment 時 $\theta$ 不變、只是 hidden state 在 rollout 中變化——也就是說「learning 發生在 activations，而不是 weights」。Weights 是 prior（慢學），activations 是 posterior / fast policy state（快學）。

簡言之：**hidden state 同時扮演了「task belief / sufficient statistic」與「policy state」**，因此可以說「RNN 在執行一個被學出來的 RL 演算法」。

### 8.2 為什麼 meta-RL 會自然碰到 exploration / exploitation 問題？

- 在標準 supervised meta-learning 裡，每個 task 的資料是給定的，learner 只要「在這份固定資料上 generalise 好」即可。沒有「我要不要主動收集更多資料」的選擇。
- 但 meta-RL 中，agent 自己 generate trajectory：**每一步 action 決定下一步收到什麼 reward / state**。
- 因此 agent 在 trial 早期面對未知 MDP 時，必須衡量 **「現在做動作 a 是為了學 MDP 結構（exploration）」 vs.「為了賺更多 reward（exploitation）」**。
- RL$^2$ 的訓練目標是整個 trial 的累積 reward，所以 **explore 太少→無法 exploit 到好的 arm/state→trial 累積 reward 差**；**explore 太多→浪費 budget**。outer-loop RL 在這個 trade-off 上會自動找到最優平衡。
- 也因為這個原因，RL$^2$ 的學出來行為會自然展現類 Gittins index、類 Thompson sampling 的 exploration 模式（Table 1 已直接證實），這在 supervised meta-learning 文獻裡是看不到的。

### 8.3 與 MAML 式 gradient-based adaptation 的差別

[[MAML|MAML]]（Finn et al., 2017，本文發表時點正同期）採取另一條路線。對比如下：

| 面向 | RL$^2$（本論文） | MAML 式 gradient-based meta-RL |
| --- | --- | --- |
| **Fast learner 的形式** | RNN 的 forward pass / hidden state | 對 policy 參數 $\theta$ 做幾步 gradient descent |
| **Adapt 在哪一層發生** | activations（weights 凍結） | weights（每個 task 都會被更新） |
| **是否需要 task reward 來 adapt** | 不需要 explicit gradient；只要把 reward 餵進 RNN 輸入即可 | 需要 task 上的 reward 來算 inner-loop gradient |
| **Adapt 機制是否「可學」** | 完全 learned；機制本身是任意函數 | Adapt 機制固定為 SGD，只有 init $\theta_0$ 可學 |
| **Inner-loop 步數限制** | 任意長序列；想 adapt 多久就跑多久 RNN | 通常 1–5 步 gradient（受 second-order 計算限制） |
| **記憶 / belief 表達能力** | 強：可以用 hidden state 維護任意 sufficient statistic | 弱：兩步 gradient 不容易維護 belief tracker |
| **Exploration 的表達** | 直接從序列 policy 中學出來（如 Gittins-like）| 需要額外設計（如 MAESN、E-MAML），否則 inner-loop 不會自發 explore |
| **計算成本** | 訓練長序列 RNN + TRPO，記憶體開銷大 | inner-loop second-order gradient 計算複雜 |
| **Inductive bias** | 幾乎沒有結構先驗，全靠 RNN 自學 | 強結構先驗：「fast adapt = 幾步 SGD」 |
| **Out-of-distribution 行為** | RNN extrapolation 不可控；hidden state 行為難解釋 | 仍是 SGD，較有理論保證 |

**直觀對比**：MAML 是「我已經知道適應的機制（SGD），只需要找一個好起點」；RL$^2$ 是「連適應機制都讓網路自己學」。前者結構偏多、樣本效率高但表達上限受限；後者結構偏少、容量大但需要大量 meta-training 資料、且行為不一定可解釋。

---

## 9. 與本研究主線的關聯

我的主線：**robot play data + 極少弱語言標註 + offline meta-RL + 文字 → task specification / task embedding 做 zero-shot 泛化**，特別關心 meta-learning $\times$ zero-shot。

### 9.1 啟發

1. **「Activations 是 fast learner」這個觀念是核心**：在 offline meta-RL + zero-shot 的設定下，我們不太可能在 deployment 時做 inner-loop gradient（沒有充足 task reward 也沒有大量互動 budget）；RL$^2$ 的 hidden state 適應方式天生就比較貼合 zero-shot / few-shot context conditioning。我們的「弱語言標註」可以視為**直接灌進 RNN 輸入的 context tokens**——相當於把 task identity 半 grounded 地暴露給 fast learner。
2. **Exploration/exploitation 在我們 setting 變形**：因為是 offline 資料 + 弱語言，agent 在訓練期沒有真正的 online exploration 自由度，但「在 hidden state 裡進行 belief update / language grounding」這件事仍然存在。RL$^2$ 提醒我們：**只要把 history（包括語言）整段餵給 sequence model + 用整段 episode 的 reward / return-to-go 做訓練訊號，模型就會 implicit 學出「task 解析 + 快速決策」**。
3. **POMDP 觀點**：RL$^2$ 把 task identity 當 hidden state，這個視角直接對應到我們「文字 → task spec / task embedding」的設定——文字標註可看成對 task hidden variable 的 noisy observation，可以接到 belief MDP 框架。

### 9.2 可借用的零件

- **GRU + 跨 episode hidden state 保留**：對 robot play 軌跡很自然，因為一條 demo 通常跨多個子任務。我們可以把 robot demo segment 視為 trial 內的多個 episode、把弱語言當作該 trial 的 task descriptor。
- **輸入四元組 $(o, a, r, d)$**：可擴增為 $(o, a, r, d, \ell)$，其中 $\ell$ 是 sparse language token（或 embedding）。
- **Baseline RNN 與 weight sharing**：尤其在視覺任務 share backbone 效果更穩——這對 robot pixel 輸入很實用。
- **「用整個 trial 的 cumulative return 當訓練訊號」**：可在 offline meta-RL 場景轉成「整段 demo trajectory 的 return-to-go 條件式訓練」（與 [[DecisionTransformer|Decision Transformer]] / Trajectory Transformer 類方法 align）。

### 9.3 可當 baseline / ablation

- **Strong baseline**：把我們的 method 與「純 RL$^2$（RNN policy, no language conditioning）」比，可量化「弱語言標註對 zero-shot generalisation 的 marginal 增益」。
- **Ablation A**：把語言通道砍掉，只留 RL$^2$ 風格的 history conditioning → 看 zero-shot 對 unseen task 還剩多少。
- **Ablation B**：保留語言，但把 cross-episode hidden state 重置 → 看是否回退到 single-episode policy。
- **Ablation C**：把 RNN 換成 transformer 序列模型，但保留 RL$^2$ 的訓練目標（trial-level return），檢驗「fast learner」是否確實是 sequence model 表達能力的函數。
- **與 MAML/[[PEARL|PEARL]] 對照**：RL$^2$ 是 memory-based meta-RL 的代表，PEARL 是 context-based + posterior sampling，MAML 是 gradient-based。三條都應該在我們的 robot play benchmark 上跑，定位我們的方法。

### 9.4 與 offline meta-RL / 文字 task embedding 的接合點

- RL$^2$ 是 **on-policy + online meta-training** 的設計，對我們直接套用有兩個明顯阻礙：(i) 無法 sample 新 trajectory；(ii) 沒有 multi-episode 的 fresh interaction budget。
- 但可以 **重新詮釋 RL$^2$ 的 objective 為 sequence-modelling**：把 offline play data 的 trajectory 視為一個 trial 的觀察序列，hidden state 學「在 prefix 後該怎麼 act」。這就是把 RL$^2$ 退化成 behaviour cloning 加 history conditioning。
- 文字 task embedding 可以注入兩個位置：(a) 作為 RNN 的初始 hidden state（強條件）、(b) 作為每個 timestep 的額外輸入 token（弱條件）。RL$^2$ 原文的 placeholder 設計告訴我們，這種「在前綴位置注入 task 描述」並不影響整體 pipeline。

---

## 10. 一句話總結

**RL$^2$ 把「快速 RL 演算法」直接編進一顆跨 episode 保留 hidden state 的 RNN，用一般 RL（TRPO）以整個 trial 的累積 reward 為目標來訓練 weights，使得 weights 是 prior、activations 是針對未知 MDP 的 fast policy state，從而以記憶式（而非 gradient 式）adaptation 統一了 meta-learning 與 exploration/exploitation。**
