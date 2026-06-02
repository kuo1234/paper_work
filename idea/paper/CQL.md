---
type: paper-note
aliases:
  - "CQL"
  - "Conservative Q-Learning"
year: 2020
stage: "2-offline-rl基礎"
tags:
  - offline-rl
  - conservative
  - value-based
  - baseline
  - 2-offline-rl基礎
summary: "保守Q（壓OOD action的Q值得到lower bound），offline policy backbone候選，α當保守度旋鈕。"
---
> **論文標題**：Conservative Q-Learning for Offline Reinforcement Learning
> **作者**：Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine（UC Berkeley / Google Research, Brain Team）
> **出處 / 年份**：NeurIPS 2020 (arXiv:2006.04779v3, 2020 年 8 月)
> **主題**：Offline / Batch Reinforcement Learning；解決 Q-value overestimation；以正則化 Q-function 取得 lower-bound 的價值估計
> **整理目標**：將 CQL 的核心問題定位、正則化公式、lower-bound 理論、gap-expanding 性質、實作演算法與實驗結論完整精讀，並對照本研究主線「robot play + 弱語言標註 + offline meta-RL + 文字→task spec/embedding 之 zero-shot 泛化」評估其作為 offline RL backbone / baseline 的可用性。

---

## 0. 閱讀總覽（白話）

在 offline RL 設定中，agent 只能用一份固定的資料集學策略，沒辦法再去環境裡試錯。如果直接套標準的 Q-learning / actor-critic，會發生一個致命的問題：Bellman backup 裡的 max（或者 actor 取樣的 action）會把 Q-function 推向「資料集裡看不到的 OOD action」，而這些 OOD action 上的 Q 值因為從未被真實 reward 修正，往往會被神經網路「自我增強」地高估，越學越離譜，最後 policy 也就學壞。

CQL 的想法非常直接：既然 Q 被高估，那就「主動把它壓低」。具體做法是在原本的 Bellman error 之外，再加一項「對某個動作分布 μ（通常選成讓當前 Q 最大的 soft policy）下的 Q 取平均並懲罰」的正則項；再加上一項「對資料分布下的 Q 取平均並獎勵」來防止把 in-distribution 的 Q 也壓掉。這樣一推一拉之後，理論上可以證明：學到的 Q 在 policy 期望下，會是真實 value 的 lower bound——也就是「保守估計」。

理論上他們進一步證明 CQL 具有 gap-expanding 性質：在學到的 Q 上，in-distribution action 和 OOD action 的差距，比真實 Q 上的差距更大。也就是說 CQL 不只是把 Q 整體壓低，而是「特別把 OOD 那邊壓得更低」，這就自然地讓 policy 不會被 OOD 高估誘導出去，而不需要像 BCQ / BEAR / BRAC 那樣顯式估一個 behavior policy 再 KL/MMD 約束。實作只需在 SAC / QR-DQN 上加大約 20 行程式碼。實驗在 [[D4RL|D4RL]] 的 gym、AntMaze、Adroit、Kitchen 與 Atari 上都明顯打贏先前 offline RL 方法，尤其是資料混雜（mixed / medium-expert / random-expert）的情況下優勢最大。

---

## 1. Abstract（重點轉述）

- 問題：offline RL 想用既有大型靜態資料學 policy，但 distributional shift 會讓標準 off-policy 方法的 Q 值嚴重高估，特別在資料分布複雜或多模態時。
- 方法：提出 Conservative Q-Learning（CQL），透過在 Bellman error 上加一個 Q-value regularizer，學一個「保守」的 Q，使得 policy 在這個 Q 下的期望值是其真實 value 的 lower bound。
- 理論：CQL 學到的 Q 對當前 policy 的 value 構成 lower bound，且能整合進帶有策略改進保證（safe policy improvement）的學習流程。
- 實作：只要在 deep Q-learning / actor-critic 上加上 CQL 正則項（約 20 行程式碼）。
- 實驗：在離散與連續控制、複雜多模態資料分布上都明顯優於既有 offline RL 方法，常常 final return 高出 2–5 倍。

---

## 2. 問題背景：為什麼 Q overestimation 在 offline RL 特別嚴重

標準 off-policy Q-learning 的更新核心是

\[
\hat{Q}^{k+1} \leftarrow \arg\min_Q \mathbb{E}_{s,a,s'\sim \mathcal{D}}\Big[(r(s,a) + \gamma \mathbb{E}_{a'\sim \hat{\pi}^k}[\hat{Q}^k(s',a')] - Q(s,a))^2\Big]
\]

而 policy improvement 取 \(\hat\pi^{k+1}(a|s) = \arg\max_\pi \mathbb{E}_{a\sim\pi}[\hat{Q}^{k+1}(s,a)]\)。

在 offline 設定下，這個流程有兩個交互的痛點：

1. **Bootstrapping from OOD actions**：target 裡的 \(a' \sim \hat\pi^k\) 不一定在資料 \(\mathcal{D}\) 中出現過。由於神經網路會把不同 (s,a) 的 Q 值耦合在一起，這些 OOD action 上的 Q 值是「外推」出來的，沒有真實 reward 校正。
2. **Optimizer 的選擇性偏誤**：policy 又會去挑 Q 最大的 action，因此會系統性地優先挑那些「碰巧被高估」的 OOD action。online 時這種錯誤可以靠真去環境試試看修正；offline 時 agent 無法再與環境互動，錯誤只會在 Bellman backup 裡被反覆放大，形成正向回饋的 over-estimation。
3. **資料多模態加重問題**：當資料來自多個行為策略（mixed / medium-expert / human demo）時，behavior policy \(\pi_\beta(a|s)\) 本身就難估，prior 工作 (BCQ, BEAR, BRAC, SPIBB) 多半需要先擬合 \(\hat\pi_\beta\) 再對 policy 做 KL / MMD / Wasserstein 約束，這個額外的估計步驟在多模態下非常脆弱。
4. **狀態分布沒有 shift，動作分布才有**：Bellman backup 不會在 OOD state 上查詢 Q，所以 state distribution shift 不是訓練期的問題；但 action distribution shift（OOD action 的 Q 值錯誤）是核心痛點。

CQL 的問題定義因此是：**不要先估 \(\pi_\beta\)，直接在 Q-function 學習階段對 OOD action 的 Q 值加上一個「主動壓低」的正則，使得學到的 Q 在我們關心的 policy 期望下是 lower bound，從而避開 over-estimation 引發的策略漂移。**

---

## 3. CQL 核心方法與公式

### 3.1 基本想法：對某個動作分布 μ 壓低 Q

最簡單的版本是在 Bellman error 上加一項「在某個動作分布 \(\mu(a|s)\)（state-marginal 鎖定在資料分布）下的 Q 平均」做最小化：

\[
\hat{Q}^{k+1} \leftarrow \arg\min_{Q}\ \alpha\,\mathbb{E}_{s\sim \mathcal{D},\ a\sim \mu(a|s)}[Q(s,a)] + \tfrac{1}{2}\,\mathbb{E}_{s,a,s'\sim \mathcal{D}}\Big[(Q(s,a) - \hat{\mathcal{B}}^{\pi}\hat{Q}^k(s,a))^2\Big]
\tag{Eq. 1}
\]

- 直覺：第一項拉低「我們認為可能被高估的那些 (s,a)」上的 Q；第二項仍然要求 Q 滿足 Bellman 一致性。
- 在 tabular、無 sampling error 下解 \(\partial / \partial Q = 0\) 得到固定點關係

\[
\hat{Q}^{k+1}(s,a) = \hat{\mathcal{B}}^\pi \hat{Q}^k(s,a) - \alpha \frac{\mu(a|s)}{\hat\pi_\beta(a|s)},
\]

也就是「每一輪都把 Q 往下推一個量，量正比於 μ 相對於 behavior policy 的密度比」。對 OOD action 而言 \(\hat\pi_\beta(a|s)\) 很小，密度比很大，所以 OOD action 的 Q 會被壓得更兇——這正是我們要的方向。

### 3.2 改良：加一個對資料分布下 Q 的最大化（tighter bound）

Eq. 1 會把所有 (s,a) 都壓低（point-wise lower bound）。但如果我們其實只在乎 policy 的 value \(V^\pi(s) = \mathbb{E}_{a\sim \pi}[Q^\pi(s,a)]\)，那 point-wise 壓得太狠就太保守了。CQL 因此再加一項對「資料分布」下 Q 的最大化：

\[
\hat{Q}^{k+1} \leftarrow \arg\min_{Q}\ \alpha\Big(\mathbb{E}_{s\sim \mathcal{D},\ a\sim\mu(a|s)}[Q(s,a)] - \mathbb{E}_{s\sim \mathcal{D},\ a\sim \hat\pi_\beta(a|s)}[Q(s,a)]\Big) + \tfrac{1}{2}\,\mathbb{E}_{s,a,s'\sim \mathcal{D}}\Big[(Q(s,a) - \hat{\mathcal{B}}^\pi \hat{Q}^k(s,a))^2\Big]
\tag{Eq. 2}
\]

- 第一項：在 μ 下壓低 Q；
- 第二項（新增）：在 behavior 分布下抬高 Q，避免把 in-distribution 的 Q 也一起壓掉，藉此把 lower bound 收緊；
- 第三項：原本的 Bellman error。

直覺：CQL 在 (s,a) 空間做了一個「資料分布抬高、外圍分布壓低」的對比正則，把 Q 的形狀塑造成「資料附近高、OOD 處低」。當取 \(\mu = \pi\) 時，定理 3.2 保證 \(\mathbb{E}_{a\sim\pi}[\hat{Q}^\pi(s,a)] \leq V^\pi(s)\)。

### 3.3 CQL 家族與 μ 的選擇

把 μ 也當成優化變數，並對 μ 加一個正則 \(\mathcal{R}(\mu)\)：

\[
\min_{Q}\max_{\mu}\ \alpha\Big(\mathbb{E}_{s\sim\mathcal{D},a\sim\mu}[Q(s,a)] - \mathbb{E}_{s\sim\mathcal{D},a\sim\hat\pi_\beta}[Q(s,a)]\Big) + \tfrac{1}{2}\mathbb{E}_{s,a,s'\sim\mathcal{D}}[(Q - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k)^2] + \mathcal{R}(\mu)
\tag{CQL(R), Eq. 3}
\]

選擇 \(\mathcal{R}(\mu) = -D_{KL}(\mu \,\Vert\, \rho)\) 時，可解析地得到 \(\mu^*(a|s) \propto \rho(a|s)\exp(Q(s,a))\)。

**CQL(H)（取 \(\rho = \text{Unif}\)，等同於對 μ 加 entropy 正則）**：把 \(\mu^*\) 代回後第一項變成 log-sum-exp（state-wise soft-max），實務上最常用的版本：

\[
\min_{Q}\ \alpha\,\mathbb{E}_{s\sim\mathcal{D}}\Big[\log\sum_a \exp(Q(s,a)) - \mathbb{E}_{a\sim\hat\pi_\beta}[Q(s,a)]\Big] + \tfrac{1}{2}\,\mathbb{E}_{s,a,s'\sim \mathcal{D}}\Big[(Q - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k)^2\Big]
\tag{CQL(H), Eq. 4}
\]

- 第一項在連續動作空間用 importance-sampled 取樣近似（從 uniform、目前 policy、過去 policy 等混合取樣）。
- 第二項仍然鼓勵在資料 (s,a) 上的 Q 不要被壓掉。

**CQL(ρ)（取 \(\rho = \hat\pi^{k-1}\)，前一代 policy）**：解析得到 \(\mu^* \propto \hat\pi^{k-1}\exp(Q)\)，等於對前一代 policy 取 exponential-weighted 平均的 Q。對高維 action（例如 24-DoF Adroit hand）方差更低、更穩定。

### 3.4 為什麼這能得到「保守」的 Q：直覺

把 Eq. 2 的解析固定點直接寫出來（tabular、無 sampling error、\(\mu = \pi\)）：

\[
\hat{Q}^{k+1}(s,a) = \mathcal{B}^\pi \hat{Q}^k(s,a) - \alpha\Big(\frac{\mu(a|s)}{\hat\pi_\beta(a|s)} - 1\Big)
\]

- 對 OOD action，\(\mu/\hat\pi_\beta \gg 1\)，修正項是負的：Q 被壓低。
- 對 in-distribution action，\(\mu/\hat\pi_\beta \approx 1\)，修正項接近 0：Q 不變。
- 對「比資料還更稀有的 action」修正甚至可能略為正：但因為這些 action 很少被取樣，影響不大。

把這個算子展開到 fixed point，誤差項形式為 \((I-\gamma P^\pi)^{-1}\Big(\frac{\mu}{\hat\pi_\beta} - 1\Big)\)，這是個非負 matrix 作用在「OOD 處為正、in-distribution 為 0」的向量上，所以整個 Q 都會被往下修正。再考慮取期望時，\(\mathbb{E}_{\pi}[\mu/\hat\pi_\beta - 1] = \chi^2(\pi \Vert \hat\pi_\beta)\ge 0\)，所以 \(\mathbb{E}_{a\sim\pi}[\hat{Q}^\pi(s,a)] \le V^\pi(s)\)。這就是 lower-bound 結論的數學直覺。

---

## 4. 演算法流程

論文 Algorithm 1（與 SAC / QR-DQN 相比只多了紅字部分）：

1. 初始化 Q-function \(Q_\theta\)；actor-critic 版額外初始化 policy \(\pi_\phi\)。
2. 重複 \(N\) 步：
   - **(critic update)** 用 CQL(H)（或一般 CQL(R)）目標更新 \(\theta\)：
     \[\theta_t \leftarrow \theta_{t-1} - \eta_Q \nabla_\theta \mathcal{L}_{\text{CQL(R)}}(\theta)\]
     對 Q-learning 用 \(\mathcal{B}^*\)、對 actor-critic 用 \(\mathcal{B}^{\pi_\phi}\)。
   - **(actor update, actor-critic only)** 以 SAC 風格 entropy 正則更新 \(\phi\)：
     \[\phi_t \leftarrow \phi_{t-1} + \eta_\pi \mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\phi}[Q_\theta(s,a) - \log\pi_\phi(a|s)]\]

實作細節：
- SAC 上加 ~20 行；QR-DQN 上類似改 backup target。
- 連續控制：\(\alpha\) 用 Lagrangian dual gradient 自動調整；離散控制：固定 α。
- policy 學習率取 \(3 \times 10^{-5}\)（比 Q 的 \(3 \times 10^{-4}\) 小），呼應定理 3.3「policy 變化要慢」的條件。
- **不需要擬合 behavior policy**：相較 BCQ/BEAR/BRAC 是一大實作簡化。

---

## 5. 理論性質（白話解釋）

### 定理 3.1（Eq. 1 是 point-wise lower bound）
假設 \(\mathrm{supp}\,\mu \subset \mathrm{supp}\,\hat\pi_\beta\)，以高機率：

\[
\hat{Q}^\pi(s,a) \le Q^\pi(s,a) - \alpha (I-\gamma P^\pi)^{-1}\!\left[\frac{\mu}{\hat\pi_\beta}\right](s,a) + (I-\gamma P^\pi)^{-1}\!\left[\frac{C_{r,T,\delta}R_{\max}}{(1-\gamma)\sqrt{|\mathcal{D}|}}\right](s,a)
\]

意義：α 夠大時，CQL Eq. 1 在「每個 (s,a)」上都是 \(Q^\pi\) 的 lower bound。樣本越多（\(|\mathcal{D}|\) 越大），所需 α 越小；無 sampling error 時任意 α>0 都成立。

### 定理 3.2（Eq. 2 是 tighter 的「policy 期望」lower bound）
取 \(\mu = \pi\) 時：

\[
\hat{V}^\pi(s) \le V^\pi(s) - \alpha (I-\gamma P^\pi)^{-1}\,\mathbb{E}_\pi\!\Big[\tfrac{\pi}{\hat\pi_\beta} - 1\Big](s) + (\text{sampling-error 項})
\]

意義：Eq. 2 不再保證 point-wise lower bound（in-distribution 的 Q 不會被壓），但保證「policy 期望下的 V」是 lower bound——這正是 policy improvement 階段真正需要的量。因為沒有對 in-distribution 過度壓低，所以這個 bound 比 Eq. 1 緊（實驗 Table 4 也驗證：CQL(H) 的差距比 CQL(Eq.1) 小）。

### 定理 3.3（CQL 學到 lower-bounded Q-values）
在 policy 變化緩慢（\(D_{TV}(\hat\pi^{k+1}, \pi_{\hat{Q}^k}) \le \varepsilon\)）的假設下，CQL 學到的 \(\hat V^{k+1}(s) \le V^{k+1}(s)\)。直覺：定理 3.2 的保守量必須蓋過 policy 變化帶來的潛在高估量；只要 policy 學得夠慢，就能維持 lower bound——這也是論文設定 actor learning rate << critic learning rate 的理論依據。

### 定理 3.4（CQL 是 gap-expanding）
這是 CQL 最關鍵的「防 OOD」性質：對任意 k，

\[
\mathbb{E}_{a\sim\pi_\beta}[\hat{Q}^k(s,a)] - \mathbb{E}_{a\sim \mu_k}[\hat{Q}^k(s,a)] > \mathbb{E}_{a\sim\pi_\beta}[Q^k(s,a)] - \mathbb{E}_{a\sim\mu_k}[Q^k(s,a)]
\]

也就是說，在學到的 Q 上，in-distribution 與 OOD action 之間的差距「比真實 Q 還要大」。這意味著當你用 \(\pi(a|s)\propto\exp(\hat Q(s,a))\) 推導 policy 時，policy 會更傾向 in-distribution action，**隱式地完成了 policy constraint 的功能而不需要顯式 KL/MMD**。Appendix B 在 hopper-expert/medium 上用 \(\hat\Delta^k\) 指標實證：CQL 的 \(\hat\Delta^k < 0\)（OOD Q < in-dist Q），BEAR 則往往 \(>0\) 且隨訓練上升，最終出現 unlearning。

### 定理 3.5 / 3.6（Safe Policy Improvement）
- 3.5：CQL 的最優解等同於在 empirical MDP \(\hat M\) 上做 RL，同時對 policy 與 \(\hat\pi_\beta\) 之間的 \(D_{CQL}\)（\(\chi^2\)-like）距離做懲罰。
- 3.6：以高機率 \(J(\pi^*, M) \ge J(\hat\pi_\beta, M) - \zeta\)，\(\zeta\) 的兩項分別為 sampling error 與 empirical 改進量；\(|\mathcal{D}(s)|\) 越大 \(\zeta\) 越小，越能用更小的 α 取得改進。

### 函數逼近的延伸
Theorem D.1（linear）、D.2（NTK）將 lower-bound 結論推廣到 linear 與 neural tangent kernel 框架。深度神經網路的嚴謹分析仍是 open problem。

---

## 6. 實驗與結論

### D4RL — MuJoCo Gym（Table 1）
- 單一 policy 來源（random / expert / medium）：CQL(H) 與最佳先前方法持平或略勝。
- **多模態混合（mixed / medium-expert / random-expert）**：CQL 大幅領先（常常 2–3x）。例如 walker2d-medium-expert：CQL 98.7 vs BRAC-v 0.9；hopper-random-expert：CQL 110.5 vs BEAR 10.1。

### D4RL — AntMaze（Table 2）
要把 sub-optimal trajectory 片段「拼接」成達標 policy。簡單的 umaze 各方法都還能跑，但在 medium / large 上**只有 CQL 能拿到非零 return**（antmaze-medium-play：CQL 61.2 vs 其他 0），對「需要 stitching」的 offline 資料是質的差異。

### D4RL — Adroit human demo（Table 2）
24-DoF 機械手的真人示範，資料量很少。先前 offline RL 方法都打不過 BC，**只有 CQL 變體能贏 BC**。其中 CQL(ρ)（用前一代 policy 當 prior）比 CQL(H) 更穩，因為 log-sum-exp 在高維 action 下變異很大。

### D4RL — Franka Kitchen（Table 2）
長序操作多物件，sparse 0-1 reward + 真人 teleoperation 資料。CQL 是唯一在三個子任務上都 >40% 成功率、且都打贏 BC 的方法。

### Atari（Figure 1, Table 3）
- 20% DQN replay：CQL ≈ QR-DQN/REM 或更穩。
- **1% / 10% replay（少資料）**：CQL 大幅領先，1% Q*bert 上 CQL 14012 vs REM 343（≈36x）；Breakout ≈6x。少資料下 CQL 的保守正則更顯優勢。

### Lower-bound 與 gap-expanding 的實證（Table 4 / Appendix B）
- CQL(H) 與 CQL(Eq.1) 估出的 V 與真實 return 之差**為負**：實證 lower bound。
- Ensemble 方法的 V 估計 grossly overestimated（\(10^6 \sim 10^{12}\) 量級爆炸）。
- BEAR 仍會輕度高估。
- Appendix B \(\hat\Delta^k\) 圖：CQL 的 OOD-vs-in-dist Q 差距為負（in-dist 高），BEAR 為正且持續增加，最終 hopper-expert 上出現 unlearning。

### CQL 適合的資料分布
從理論與實驗綜合來看，CQL 特別擅長：
- **多模態、混合行為**（mixed / medium-expert / human demo）：因為它不需要先擬合 \(\pi_\beta\)，避開了 prior 工作擬合多模態行為策略的脆弱性。
- **資料覆蓋窄 + 需要 stitching**（AntMaze medium/large、Kitchen）：gap-expanding 防止 policy 漂到 OOD 而 unlearn。
- **少量資料**（Atari 1%）：保守估計在 sampling error 大時優勢更明顯，α 也只需設大一些即可。
- 反過來，當資料分布是**純 random** 或**純 expert** 時，CQL 與既有方法差距較小；純 expert 時 BC 已經非常強，CQL 的優勢主要體現在訓練穩定不退化。

### 主要結論
- CQL 是一個簡單（~20 行）、不需要估 behavior policy、有 lower-bound / safe improvement / gap-expanding 理論支撐的 offline RL 框架。
- 在 D4RL 多項複雜資料分布任務與 Atari 少資料設定上明顯超越 BCQ/BEAR/BRAC/SAC/BC/REM/QR-DQN 等。
- Open problems：deep network 嚴格 lower-bound 分析、offline RL 的 early stopping / validation 機制。

---

## 7. 與本研究主線的關聯

研究主線：**robot play data + 極少弱語言標註 + offline meta-RL + 文字→task spec/embedding 做 zero-shot 泛化**，並強調 meta-learning × zero-shot 並行方向。CQL 在這條主線中扮演的角色：

1. **作為 per-task offline RL backbone**：
   - Play data 來自無監督探索 / 多任務示範，天然是「多模態 + 混合行為」分布——這正是 CQL 最擅長的領域（D4RL mixed / medium-expert / Adroit human demo）。
   - 不需要擬合 \(\pi_\beta\) 是極大優點，因為 play data 的 behavior policy 高度多模態，傳統 BCQ/BEAR 在這種資料上特別脆弱。

2. **作為 offline meta-RL 的 inner 演算法**：
   - meta-training 階段，每個 task / context 都需要從固定資料學一個 task-conditioned Q / policy。CQL 的 gap-expanding 性質讓 policy 不會在 task 切換時被 OOD 高估誤導，這對「短 context、要快速適應」的 meta-RL 特別重要。
   - 在 [[PEARL|PEARL]] / [[VariBAD|VariBAD]] 的 offline 版本（如 FOCAL、BOReL、[[CORRO|CORRO]] 等）中，CQL 已是預設的 backbone；本研究若把 task embedding 從 trajectory-derived 改成 language-derived，仍可直接複用 CQL 的 critic loss，只需把 Q-function 改成 \(Q_\theta(s, a; z_\text{text})\)。

3. **作為 zero-shot text-conditioned policy 的 baseline**：
   - 一個天然 baseline 流程：(a) 用 text→task embedding \(z\) 標註 play data；(b) 把 \(z\) 當條件丟進 CQL 的 \(Q_\theta(s,a;z)\) 與 actor \(\pi_\phi(a|s;z)\)；(c) 推論時對未見任務的文字描述產 \(z\)，直接執行 \(\pi_\phi(\cdot|s;z)\)。這就是 conditioned-CQL，可作為「無 meta 結構、無 latent inference」的最強對照組，用來凸顯 meta-learning（PEARL/VariBAD-style posterior inference）真正帶來的增益。

4. **與弱語言標註結合的考量**：
   - 弱標註下 \(z\) 是噪音較大的條件。CQL 的保守性對「條件估計也帶不確定性」的場景特別重要——它防止 actor 在 \(z\) 噪音方向上被高估的 Q 拉走。
   - 可考慮在 CQL 正則中把 μ 從「\(\exp(Q)\)」改成「\(\exp(Q(s,a;z_\text{noisy}))\)」，自然把語言條件下的 OOD action 也壓低。

5. **可作為消融基線**：
   - Pure CQL（無 task embedding）→ multi-task CQL（one-hot task id）→ language-conditioned CQL → meta-RL + CQL → meta-RL + CQL + text-z，這條 ladder 可以清楚分離出「offline 保守性」、「task conditioning」、「language grounding」、「meta-learning posterior」各自貢獻多少。

6. **侷限提示**：
   - CQL 的 α 在不同任務上敏感；meta 設定下不同 task 對 α 的最佳值可能不同，需要 task-conditional α（或 Lagrangian 自動調）。
   - CQL 並不主動鼓勵跨 task generalization，所以 zero-shot 能力來自 task embedding \(z\) 的學習，而不是 CQL 本身——這就是為什麼仍需要 meta-RL / language alignment 的 outer 結構。

---

## 8. 一句話總結

CQL 在 Bellman error 之外加上「壓低 μ 下、抬高 behavior 分布下」的 Q 對比正則，學到對 policy value 為 lower bound 且具 gap-expanding 性質的保守 Q，因而不需估 behavior policy 就能在 offline、多模態、少資料設定下穩定打贏既有方法，是 offline RL（包含 offline meta-RL 與 text-conditioned 變體）最自然的 backbone 之一。
