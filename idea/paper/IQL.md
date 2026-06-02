---
type: paper-note
aliases:
  - "IQL"
  - "Implicit Q-Learning"
year: 2021
stage: "2-offline-rl基礎"
tags:
  - offline-rl
  - expectile
  - in-sample
  - baseline
  - 推薦backbone
  - 2-offline-rl基礎
summary: "expectile regression + AWR，in-sample不評估OOD action；對弱標註/次優play data友善的強baseline。"
---
# IQL 論文逐段詳細整理

> 論文：**Offline Reinforcement Learning with Implicit Q-Learning**
> 作者：Ilya Kostrikov, Ashvin Nair, Sergey Levine (UC Berkeley)
> 出處／年份：arXiv:2110.06169，2021；後續發表於 ICLR 2022
> 主題：Offline RL、Expectile Regression、In-sample Learning、Advantage-Weighted Policy Extraction、D4RL Benchmark
> 整理目標：逐段說明這篇論文如何用 expectile regression「不必查詢 OOD action」就完成 multi-step dynamic programming，以及為何 IQL 後續成為 offline / offline-to-online RL 的標準強 baseline。

---

## 0. 閱讀總覽（白話）

這篇論文是 offline RL 的一個重要里程碑。它要解決的問題是：

> Offline RL 想學一個比 dataset 行為策略 \(\pi_\beta\) 還要好的 policy，但「想知道某個新 action 好不好」就要用 Q-function 去估它的價值；而 dataset 沒看過的 action 很容易被高估，造成 policy 走偏。傳統作法是「policy 不准離 \(\pi_\beta\) 太遠」或「對 OOD action 的 Q 加 penalty（例如 [[CQL|CQL]]）」，但這些都需要在訓練時對 OOD action 評估，而且帶來 trade-off。

IQL 提出一個非常乾淨的點子：

1. **完全只用 dataset 裡看到的 (s, a, s′, a′) 對 Q 做 TD update（SARSA-style），訓練期間從來不查任何 dataset 之外的 a 的 Q 值。**
2. **但是要怎麼還是學到「比行為策略還好」的 Q？** 不直接對 action 取 max，而是把 \(Q(s, a)\) 視為以 \(a \sim \pi_\beta(\cdot|s)\) 為隨機性的隨機變數，估計它在這個分佈下的 **upper expectile**。當 expectile parameter \(\tau \to 1\) 時，這個 upper expectile 會收斂到「dataset 支持範圍內」最大的 Q 值，因此實質上就是在做 in-sample 的 max。
3. **transition 的隨機性怎麼辦？** 為了避免「碰巧 transition 到好的 s′」被當成這個 action 真的好，引入額外的 value function \(V_\psi(s)\)：先用 expectile regression 從 \(Q_{\hat\theta}(s,a)\) 學 \(V\)（只對 action 取 expectile），再用 MSE 從 \(r + \gamma V_\psi(s')\) 反推 \(Q\)（對 transition 取期望）。這個 V/Q 解耦才是讓「expectile 對 action、MSE 對 transition」乾淨拆開的關鍵。
4. **policy extraction**：value 學完後，用 **advantage-weighted regression (AWR)** 抽出 policy，也就是一個加權的 behavioral cloning，權重是 \(\exp(\beta (Q - V))\)。同樣不查 OOD action。

效果：在 [[D4RL|D4RL]] 上 state-of-the-art，特別是需要「stitching 拼接次優軌跡」的 antmaze 大幅領先 single-step 與多數 multi-step baseline；在 GTX1080 上 1M update 跑不到 20 分鐘；對 online fine-tuning 也很友善。

**為什麼 IQL 之後成為強 baseline？** 因為它 (i) 實作極簡：只是把 standard SARSA TD 的 MSE loss 換成 asymmetric L2，(ii) 不需要 explicit policy 影響 value，(iii) 算得快，(iv) 在 D4RL 全分數上很硬，(v) 對 online fine-tuning 友善（AWAC-like 的 weighted BC policy extraction）。

---

## Abstract 逐段

### Abstract 段落 1：offline RL 的核心矛盾

Offline RL 必須同時做兩件衝突的事：要比 behavior policy 進步，又不能因為 distributional shift 而崩掉。多數現有方法在 policy improvement 時，會把 latest policy 提出的 unseen action 丟到 Q 裡查價值，所以必須要嘛把 policy 限制在 in-distribution，要嘛 regularize 這些 unseen action 的 Q 值（例如 CQL）。

### Abstract 段落 2：IQL 的核心點子

IQL 主張：**完全不查 dataset 之外 action 的價值**。它的洞見是把 state value 看作以 action 為隨機性的 random variable，固定 dynamics 取期望（避免「樂觀 transition」），然後估它的 **state-conditional upper expectile**。這等於借用 function approximator 的 generalization，間接得到「該 state 下最佳 in-support action 的 Q」，而完全不需要把那個 action 餵給 Q。

### Abstract 段落 3：演算法形式與結果

整體是一個 implicit Q-learning：alternate 之間 fit upper expectile V-function 和往 Q backup，**完全沒有 explicit policy 參與 value 學習**。最後用 advantage-weighted behavioral cloning 抽出 policy，同樣不查 OOD action。實作只需要多 fit 一個用 asymmetric L2 loss 的 critic。D4RL 上 state-of-the-art，且 online fine-tuning 效果也好。

---

## 1. 問題背景與 Related Work

### 1.1 為什麼 offline RL 難

Offline RL 的目標是「只用過去蒐集到的資料學 policy」，這在 robotics、operations research 等真實場景很關鍵，因為 online exploration 又貴又危險。但要比 behavior policy 進步，就必須估計沒在資料裡出現的 action 的價值，而這正是 distributional shift 出現的地方。

過去主流方法兩大類：

* **Policy constraint**：把 policy 限制在 \(\pi_\beta\) 附近（BCQ, BEAR, AWAC, TD3+BC 等）。
* **Value regularization**：對 OOD action 的 Q 值加 penalty（CQL, Fisher-BRC 等）。

兩種都有「constraint 太強會限制 improvement，constraint 太弱會崩」的 trade-off。

### 1.2 Multi-step DP vs Single-step

* **Multi-step dynamic programming**（如 CQL、TD3+BC、AWAC、IQL）：做真正的 Bellman backup 多次迭代，理論上 high-coverage data 下能拿到最優 policy。
* **Single-step**（如 Onestep RL, Brandfonbrener et al. 2021；[[DecisionTransformer|Decision Transformer]]）：只 fit \(Q^{\pi_\beta}\) 或乾脆完全不用 value，policy extraction 一次到位。實作簡單，但在需要 **stitching 拼接次優軌跡** 的環境（antmaze）會崩。

IQL 想兼具兩者：multi-step 的能力 + single-step 的簡單與不查 OOD action。

### 1.3 與 distributional RL 的關係

過去 quantile regression 在 RL 主要用來估 Q 的分佈（隨 transition 的 stochasticity 變化）。IQL 用的 expectile regression 也是估隨機變數的統計量，但是目的不同：

* Distributional RL：估「同一個 (s,a) 下，未來 outcome 的分佈」。
* IQL：估「同一個 s 下，不同 action 造成的 value 分佈」並取 upper expectile，average 掉 transition 的 stochasticity。

---

## 2. Preliminaries

標準 MDP：\((\mathcal{S}, \mathcal{A}, p_0, p(s'|s,a), r(s,a), \gamma)\)。

標準 off-policy TD loss：

\[
L_{TD}(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\Big[\big(r(s,a) + \gamma \max_{a'} Q_{\hat\theta}(s', a') - Q_\theta(s,a)\big)^2\Big]
\]

問題出在 \(\max_{a'} Q_{\hat\theta}(s', a')\)：那個 \(a'\) 是當下 policy 給的，未必在 dataset 裡，於是 \(Q_{\hat\theta}\) 對它的估計常常過度樂觀，引發 overestimation。Offline RL 不能像 online 一樣靠互動修正，所以 overestimation 直接傳到下個迭代，最後 policy 失效。

---

## 3. 核心方法與公式

### 3.1 出發點：SARSA-style TD 不查 OOD

如果 target 用 dataset 裡的 \(a'\)：

\[
L(\theta) = \mathbb{E}_{(s,a,s',a') \sim \mathcal{D}}\Big[\big(r(s,a) + \gamma Q_{\hat\theta}(s', a') - Q_\theta(s,a)\big)^2\Big]
\]

這個 loss 從不查 OOD action，理論上 \(Q_\theta\) 會收斂到 \(Q^{\pi_\beta}\)。問題是這只能讓你知道行為策略有多好，無法比 \(\pi_\beta\) 更好（無法做 multi-step improvement）。

### 3.2 目標：受 support 限制的 max

IQL 真正想學的 value，是「對 dataset 支持下的 action」取 max：

\[
L(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\Big[\big(r(s,a) + \gamma \max_{a' \in \mathcal{A},\ \pi_\beta(a'|s')>0} Q_{\hat\theta}(s', a') - Q_\theta(s,a)\big)^2\Big]
\]

直接做這個 max 要列舉 in-support action，做不到。IQL 的關鍵 trick 是用 **expectile regression 隱式逼近這個 in-support max**。

### 3.3 Expectile regression

對隨機變數 \(X\)，其 \(\tau\)-th expectile 是 asymmetric squared loss 的最小化解：

\[
m_\tau = \arg\min_{m_\tau} \mathbb{E}_{x \sim X}\big[L_2^\tau(x - m_\tau)\big],
\quad L_2^\tau(u) = |\tau - \mathbb{1}(u<0)|\, u^2
\]

直觀理解：

* \(\tau = 0.5\)：對稱權重，就是 standard MSE，估的是 **mean**。
* \(\tau > 0.5\)：對「\(x > m\)」的殘差（正向誤差）加大權重，因此 \(m_\tau\) 會被往大值的方向推。
* \(\tau \to 1\)：\(m_\tau\) 趨近 \(X\) **support 的上界**（Lemma 1）。

也可以做 conditional expectile：

\[
\arg\min_{m_\tau(x)}\ \mathbb{E}_{(x,y) \sim \mathcal{D}}\big[L_2^\tau(y - m_\tau(x))\big]
\]

這就跟 MSE-based 神經網路訓練幾乎一樣，只需要對 loss 加 asymmetric weighting，隨機梯度下降即可。

### 3.4 為何 expectile 可以「不查 OOD」

關鍵直覺：把 \(Q(s, a)\) 視為隨機變數，其隨機性來自 \(a \sim \pi_\beta(\cdot|s)\)（dataset 經驗分佈）。

* 我們只能 sample dataset 裡的 \((s, a)\)，所以這個分佈天然只覆蓋 in-support 的 action。
* 對這個分佈估 upper expectile \(m_\tau(s)\)，在 \(\tau \to 1\) 時會收斂到 \(\max_{a: \pi_\beta(a|s)>0} Q(s, a)\)。
* 整個過程都只用 dataset 出現的 \(a\)，**不需要把任何 unseen action 丟給 Q-network 求值**。
* 「隱式」之處：max 是透過 loss 形狀（asymmetric L2）和 function approximator 的 generalization 合作而得到的，不是顯式枚舉或取 argmax。

這就是為什麼 IQL 號稱 "never evaluate actions outside of the dataset"。

### 3.5 為何要分開 V 與 Q（避開「lucky transition」）

如果直接對 TD target \(r(s,a) + \gamma Q_{\hat\theta}(s', a')\) 取 expectile，random variable 的隨機性同時來自 action 和 transition。這樣 upper expectile 不只反映「最好的 action」，也會反映「碰巧 transition 到好的 s′」（lucky sample），這在 stochastic dynamics 下會造成 over-optimism。

解法：拆成兩個 loss。

**V 的學習（expectile，只對 action 取統計量）：**

\[
L_V(\psi) = \mathbb{E}_{(s, a) \sim \mathcal{D}}\Big[ L_2^\tau\big(Q_{\hat\theta}(s, a) - V_\psi(s)\big) \Big]
\quad (5)
\]

對固定 \(s\)，隨機性只來自 dataset 中該 state 對應的 action \(a \sim \pi_\beta(\cdot|s)\)。所以 \(V_\psi(s)\) 學的是 \(Q\) 在 in-support action 上的 upper expectile：當 \(\tau\) 大，\(V_\psi(s)\) 逼近 in-support 的 \(\max_a Q(s,a)\)。

**Q 的學習（MSE，對 transition 取平均）：**

\[
L_Q(\theta) = \mathbb{E}_{(s, a, s') \sim \mathcal{D}}\Big[ \big(r(s,a) + \gamma V_\psi(s') - Q_\theta(s, a)\big)^2 \Big]
\quad (6)
\]

這裡 target 是 \(r(s,a) + \gamma V_\psi(s')\)，**根本沒有 \(a'\)**，所以從頭到尾沒有任何 unseen action 出現。MSE 對 transition 取期望，剛好把「lucky 下一步」平均掉。

符號整理：

* \(\psi\)：value network \(V_\psi(s)\)。
* \(\theta\)：Q-network \(Q_\theta(s, a)\)；\(\hat\theta\) 是 target Q（Polyak averaging）。
* \(\tau \in (0, 1)\)：expectile 參數，越大越像 max；論文 antmaze 用 0.9、locomotion 用 0.7。

### 3.6 Policy extraction：Advantage-Weighted Regression

Value 學完後，policy 還沒被「定義」過。IQL 用 AWR 做 policy extraction，依然只用 dataset action：

\[
L_\pi(\phi) = \mathbb{E}_{(s, a) \sim \mathcal{D}}\Big[ \exp\big(\beta \cdot (Q_{\hat\theta}(s, a) - V_\psi(s))\big)\, \log \pi_\phi(a|s) \Big]
\quad (7)
\]

符號：

* \(\beta \in [0, \infty)\)：inverse temperature。
* \(Q_{\hat\theta}(s, a) - V_\psi(s)\)：advantage，反映該 in-support action 相對於 expectile-V 有多好。

直覺：

* \(\beta \to 0\)：退化成 standard behavioral cloning（所有 dataset action 等權重模仿）。
* \(\beta\) 大：給高 advantage 的 dataset action 更高權重，policy 趨近於對 in-support action 取 argmax。

關鍵性質：

* policy 只看 dataset 裡的 \(a\)，所以從頭到尾沒任何 OOD query。
* policy 是事後抽出的，**不影響 value training**。意思是 value 和 actor 可以分階段或同時訓練，這對 online fine-tuning 特別友善（直接拿 offline 練好的 V/Q/π，再開始 online 收集資料即可）。
* AWR 等價於某種 KL-constrained policy improvement，蘊含 implicit 的 distribution constraint。

### 3.7 理論分析（4.4）

論文證明（在無限容量、無 sampling error 假設下）：

* **Lemma 1**：bounded support 的 random variable，\(\lim_{\tau \to 1} m_\tau = x^*\)（supremum）。
* **Lemma 2**：\(\tau_1 < \tau_2 \Rightarrow V_{\tau_1}(s) \le V_{\tau_2}(s)\)。直覺是更大的 \(\tau\) 對應「更積極挑好 action」的 policy improvement，整套類似 Sutton 的 policy improvement argument。
* **Corollary 2.1**：\(V_\tau(s) \le \max_{a: \pi_\beta(a|s)>0} Q^*(s,a)\)，其中 \(Q^*\) 是 support-constrained 的最優。
* **Theorem 3**：\(\lim_{\tau \to 1} V_\tau(s) = \max_{a: \pi_\beta(a|s)>0} Q^*(s,a)\)。

合起來說明：IQL 的整套 update 在 \(\tau \to 1\) 時就是 support-constrained 的 Q-learning。

論文也順帶指出：**IQL 是 SARSA（\(\tau = 0.5\)）與 Q-learning（\(\tau \to 1\)）之間的整個光譜**，由 \(\tau\) 做插值。實作上 \(\tau\) 太大會難 optimize，所以是 hyperparameter。

---

## 4. 演算法流程（Algorithm 1）

初始化：\(\psi, \theta, \hat\theta, \phi\)。

**TD learning 階段**：對每個 gradient step

1. \(\psi \leftarrow \psi - \lambda_V \nabla_\psi L_V(\psi)\)（expectile loss，公式 (5)）
2. \(\theta \leftarrow \theta - \lambda_Q \nabla_\theta L_Q(\theta)\)（MSE loss with \(V_\psi(s')\)，公式 (6)）
3. \(\hat\theta \leftarrow (1-\alpha)\hat\theta + \alpha\theta\)（soft target update）

**Policy extraction 階段**：對每個 gradient step

* \(\phi \leftarrow \phi - \lambda_\pi \nabla_\phi L_\pi(\phi)\)（AWR，公式 (7)）

實作細節：

* clipped double Q（取兩個 Q 的 min）用於 V update 和 policy update，降低 overestimation。
* policy 用 state-independent std 的 Gaussian。
* exponentiated advantage clip 到 \((-\infty, 100]\)（沿用 Brandfonbrener et al.）。
* policy 不影響 value，所以兩階段可以 concurrent，也可以先 value 後 policy；concurrent 對 online fine-tuning 較方便。

---

## 5. 實驗與結論

### 5.1 Toy umaze：說明 multi-step DP 的必要

一個 u-maze MDP，dataset = 1 條 optimal trajectory + 99 條 random trajectory，transition 還帶 0.25 noise。

* One-step policy evaluation：只 fit \(Q^{\pi_\beta}\)，因為大部分資料是 random，遠離 reward 的 state value 快速衰減到 0，policy 變很差。
* IQL（\(\tau = 0.95\)）：upper expectile 加 multi-step backup，正確把 reward 信號傳播回起點，value function 幾乎等於 optimal value function。

說明 **single-step 方法在需要 stitching 的環境會崩，IQL 能做到 multi-step DP**。

### 5.2 D4RL Benchmark（Table 1）

對比：BC、10%BC、Decision Transformer、AWAC、Onestep RL、TD3+BC、CQL。

* **Gym locomotion**（halfcheetah / hopper / walker2d 的 medium / medium-replay / medium-expert）：IQL 與 CQL、TD3+BC、Onestep RL 都差不多接近上限，沒有壓倒性差距（這類資料含大量近最優軌跡，single-step 也夠用）。
* **Ant Maze**（umaze / medium / large × play / diverse）：IQL 全面壓 single-step（Onestep RL、DT、AWAC 在 medium/large 接近 0），也比 CQL 顯著好（例如 large-play 39.6 vs CQL 15.8，large-diverse 47.5 vs CQL 14.9）。
* **Kitchen / Adroit**：IQL 也比 CQL、BC 略好或相當（Table 3）。
* 總和：locomotion+antmaze+kitchen+adroit 加起來 1348.3，state-of-the-art。

### 5.3 Runtime

JAX 實作下 1M update：

* IQL：≈ 20 分鐘（單 GTX1080）。
* CQL：≈ 80 分鐘（原版 4 小時以上）。

IQL 比 CQL 快約 4 倍，跟 single-step 方法相當。原因是只多訓練一個 V-network，loss 只是 asymmetric L2，沒有 inner optimization 或對 OOD action 取 sample。

### 5.4 \(\tau\) 的影響（Fig. 3）

在 antmaze（需要 stitching）上，把 \(\tau\) 從 0.5 提到 0.9 表現顯著上升，驗證理論：\(\tau\) 越大越像 Q-learning，更能拼接次優軌跡。但太大 optimization 不穩，需要平衡。

### 5.5 Online fine-tuning（Table 2）

流程：先 offline 訓 1M steps，再 online 1M steps（每步 1 gradient update）。對比 AWAC、CQL。

* Antmaze 系列：IQL 從 370.1 → 473.7，明顯壓 AWAC（107.7 → 108.3）與 CQL（151.5 → 231.1）。
* Adroit 系列（pen / door / relocate-binary）：IQL 從 38.1 → 124.0，總分最高。

兩個原因：(i) offline initialization 強，(ii) AWR-style 的 weighted BC 對 online improvement 友善（Nair et al. AWAC 已指出）。policy 不影響 value 也讓兩階段可以無縫銜接。

### 5.6 結論

IQL = 第一個同時做到「完全不查 OOD action 的價值」與「multi-step dynamic programming」的 offline RL 方法。重點貢獻：

1. **演算法簡單**：MSE loss → asymmetric L2，多訓練一個 V-network。
2. **計算快**：20 分鐘 / 1M updates。
3. **D4RL SOTA**，特別在 stitching 任務大幅領先。
4. **適合 online fine-tuning**。

附錄 D 補充：IQL 的目標跟 BCQ 的 batch-constrained Q-learning 概念相通（max over support），但 BCQ 透過 fit 一個 generative model 來 sample candidate action，仍然會跑出 OOD；IQL 用 expectile 直接強制 in-support 約束，不需 density model，也不需要 explicit policy 干預 value training。

---

## A. 與本研究主線的關聯

本研究主線：**robot play data + 極少弱語言標註 + offline meta-RL + 文字 → task spec / embedding 做 zero-shot 泛化**，並把 meta-learning × zero-shot 並行視為核心。IQL 的角色：

### A.1 為什麼 IQL 常被當強 baseline

1. **完全 in-sample**：robot play data 本身就是混雜、含大量次優與探索行為的 dataset，distributional shift 非常嚴重。IQL 不查 OOD action 的設計，讓它在不確定 dataset 品質的場景特別 robust，因此幾乎所有 offline / offline meta-RL 後續論文都把 IQL 列為 baseline。
2. **跟 task conditioning 相容**：把 \(s\) 換成 \((s, z)\)（其中 \(z\) 是 task embedding / language embedding），IQL 的所有公式 (5)(6)(7) 形式不變。對「想做 task-conditioned offline policy」非常自然。
3. **跟 multi-task / meta-RL data structure 相容**：play data 通常標 reward 困難（弱語言標註的副作用），IQL 不依賴對 reward 的強假設，AWR-style policy extraction 也不要求 reward dense；只要能算出 advantage 就能 weight。
4. **適合 offline → online fine-tuning**：robot 場景常見「先 offline pretrain，再放到實機 fine-tune」。IQL 因為 value 與 policy 解耦，pretrain 完直接拿來 online 跑，policy update rule 不用換。這在 robot play + zero-shot generalization 場景特別實用。
5. **算力友善**：對需要跑大量 ablation（不同 task embedding、不同 language encoder、不同 \(\tau\)/\(\beta\)）的 meta-RL 實驗來說，IQL 一次 1M updates 20 分鐘，整個 evaluation matrix 變得可行。
6. **D4RL 上 SOTA**：在「offline meta-RL」與「language-conditioned offline RL」題目，社群預期 baseline 至少要比 BC 與 CQL 好，IQL 通常就是「最容易超越 CQL 且實作最快」的選擇。

### A.2 可借用之處（對本研究）

1. **Expectile-V 拿來條件化 task embedding**：把 \(V_\psi(s, z)\) 與 \(Q_\theta(s, a, z)\) 條件化在 language-derived task spec \(z\) 上。expectile regression 對「z 沒見過、但是 in-support action 是熟悉的」狀況有 implicit 的 generalization 行為，相當於先借 function approximator 在 z 維度的內插能力。
2. **AWR policy extraction 對 weak language label 友善**：當語言標註稀疏，部分 trajectory 可能沒 label。AWR 是 BC + advantage weighting，可以用 \(\beta = 0\) 退化成純 BC 作為缺 label 時的 fallback，自然形成「有標籤就 advantage-weighted、沒標籤就 BC」的混合 objective。
3. **\(\tau\) 當作 conservativeness knob**：在 play data + weak language 場景，把 \(\tau\) 當 hyperparameter 拿來在「保守模仿（小 \(\tau\)）」與「積極 stitching（大 \(\tau\)）」之間 sweep，可以做為 zero-shot generalization vs in-distribution safety 的取捨工具。
4. **跟 [[PEARL|PEARL]] / [[VariBAD|VariBAD]]-style context inference 結合**：IQL 不影響 latent 推論模組（不像 AWAC 需要 actor-critic 同步），可以把 PEARL 的 \(q_\phi(z|c)\) 直接接在 V/Q 的 input 上，meta-train 階段把 \(z\) 取樣後跑 IQL TD update，這在 offline meta-RL 是自然且穩定的組合。
5. **Stitching 能力對 robot play 特別重要**：play data 在語言任務下，通常需要把不同 sub-task 的片段拼起來（先靠近物體、再抓、再移動）。Antmaze stitching 實驗顯示 IQL 在這種子軌跡拼接任務上比 one-step 強很多，這正是 robot play + zero-shot task spec 場景最需要的能力。
6. **作為 zero-shot generalization 的 lower-bound baseline**：在 language-conditioned offline RL setting，IQL（不含語言）→ IQL + language embedding → IQL + meta-learning context 是合理的 ablation 階梯。

### A.3 局限

* IQL 本身 **不是 meta-RL**：它對 task distribution、context、language 都沒結構性假設。要做 zero-shot 對 unseen task spec 泛化，要靠 conditioning 變數 \(z\) 的 generalization，而 IQL 不提供如何 train 出好的 \(z\)。
* expectile 對 support 假設敏感：若 dataset 對某個 (s, z) 切片 action 覆蓋過稀，upper expectile 仍只能在那少數 action 裡選最好，無法真正 zero-shot 推出沒見過的 action 模式。這時要靠 language embedding 在 \(z\) 維度的 generalization 補。
* 對 reward shaping 仍敏感：antmaze 在 D4RL 是減 1 reward 才好用，這在 play data + 弱語言 reward 推論的情境下要小心。

---

## B. 一句話總結

IQL 把 offline RL 的 \(\max_{a'} Q\) 用「對 in-support action 的 upper expectile（asymmetric L2 regression）」隱式取代，再以 advantage-weighted BC 抽 policy，全程不查任何 OOD action 卻仍做完整 multi-step dynamic programming，是 offline / offline-to-online RL 最簡單、最快、也最強的 baseline 之一。
