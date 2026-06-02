---
type: paper-note
aliases:
  - "Offline RL 綜述"
  - "Offline RL Survey"
  - "Offline RL Tutorial"
year: 2020
stage: "2-offline-rl基礎"
tags:
  - offline-rl
  - survey
  - distribution-shift
  - 2-offline-rl基礎
summary: "offline RL的問題框架與核心難點（distribution shift、extrapolation error、OOD action）。"
---
> **論文**：Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems
> **作者**：Sergey Levine, Aviral Kumar, George Tucker, Justin Fu (UC Berkeley / Google Research, Brain Team)
> **出處 / 年份**：arXiv:2005.01643v3，2020 年 11 月版本
> **主題**：以 tutorial + survey 形式系統介紹 offline RL（又稱 batch RL）的問題框架、為什麼把標準 online/off-policy RL 直接搬過來會崩、distribution shift / extrapolation error / OOD action 的數學根源，以及目前的方法家族（importance sampling、policy constraint、conservative value、uncertainty、model-based）與開放問題。
> **整理目標**：抽取本研究「robot play data + 弱語言標註 + offline meta-RL + 文字 → task spec/embedding 做 zero-shot 泛化」會直接踩到的 offline RL 核心難點，特別釐清 distribution shift 與 OOD action 對 Q-function 的傷害，並建立後續評估 meta-RL × offline × zero-shot 結合方法時的判斷骨架。

---

## 0. 閱讀總覽（白話版）

Offline RL 的目標跟一般 RL 一樣：找到一個能最大化期望累積 reward 的 policy。差別只有一個——agent 不能再跟環境互動，所有資料都是過去某個 (可能未知的) behavior policy π_β 收集好的固定資料集 D。

直覺上感覺很像 supervised learning：拿一坨資料 → 訓練模型 → 部署。但只要稍微想深一點，就會發現問題遠比 supervised learning 嚴重：

1. **我們想要的不是「複製資料」而是「比資料更好」**。要更好，policy 必定會選跟 π_β 不同的 action，這就是 counterfactual query（「如果走了沒人走過的路會怎樣？」）。
2. **Q-function / model 一旦被問到沒看過的 (s, a)，預測就會錯**，而且 RL 的訓練目標（max 操作）會「主動找出」那些被高估的洞，然後拼命把 policy 推往那邊，這叫 *exploitation of extrapolation error*。
3. **Online RL 之所以能活，是因為它會去試錯後修正錯估**；offline RL 沒有這個 self-correcting 機制，錯估會被 Bellman backup 一路滾雪球。

整篇 survey 的核心 message：offline RL 的全部技術細節幾乎都圍繞「如何防止 policy / Q-function / model 在 OOD 區域亂飆」這件事打轉。解法分成三大主軸：限制 policy 不要離 π_β 太遠、Q-function 自己加保守 penalty、用 uncertainty 估出不確定區域。Model-based 則把同樣思路套在 dynamics model 上。

---

## 1. 問題定義與 formalism

### 1.1 MDP

標準 MDP `M = (S, A, T, d_0, r, γ)`：state、action、轉移分布 `T(s'|s,a)`、初始分布 `d_0`、reward `r(s,a)`、折扣 `γ∈(0,1]`。Policy 為 `π(a|s)`，目標：

$$
J(\pi) \;=\; \mathbb{E}_{\tau \sim p_\pi(\tau)}\!\Big[\,\sum_{t=0}^{H} \gamma^t r(s_t, a_t)\Big].
$$

定義 state 訪問分布 `d^π(s)`（時間平均）與 `d^π_t(s_t)`。

### 1.2 三種 RL 設定（圖 1 的關鍵）

- **Online RL**：每次更新都由最新 policy 收新資料。
- **Off-policy RL**（含 replay buffer 的 Q-learning、SAC 等）：buffer 累積各代 π_0,...,π_k 的資料，但仍持續加新 sample。
- **Offline RL**：D 由某個未知 π_β 一次收齊，訓練期間**完全不互動**，只在訓練完才部署。

形式上 offline RL 假設 `(s, a) ∈ D` 滿足 `s ∼ d^{π_β}(s), a ∼ π_β(a|s)`。

### 1.3 為什麼不能把 online 演算法直接拿來用

把 SAC / DQN 把 buffer 換成 D、停止 data collection（S=0），就「形式上」是 offline 演算法了。Survey 開頭就說：這幾乎都會炸（Section 4.2 的 HalfCheetah 圖 2）。原因見下節。

---

## 2. 核心難點（read.md 必涵蓋重點）

### 2.1 Distribution shift（分布偏移）—— 全篇的中心

Counterfactual inference 與標準 ML 的 i.i.d. 假設衝突。Offline RL 同時遭遇兩種 shift：

- **State distribution shift（測試時）**：學到的 π 走出去後造訪 `d^π(s)`，但 Q / model 只在 `d^{π_β}(s)` 上訓練。
- **Action distribution shift（訓練時，更致命）**：Bellman backup 目標值依賴 `a' ∼ π(a'|s')`，但 Q 從未在這些 action 上被監督。

### 2.2 Behavioral cloning 的 H² vs H 對照（直觀的 shift 痛點）

引 Ross et al. (2011) 結果：在 offline 設定下，即使給最優 action label，學到的 policy 誤差上界為

$$
\ell(\pi) \le C + H^2 \varepsilon,
$$

而 online (DAgger) 只有 `C + H ε`。多出一個 H，因為一旦 π 漂出 `d^{π_β}` 就再也回不來，generalization bound 失效並累積。

這個結果 even apply 於只是 BC，offline RL 因為還要做 counterfactual evaluation，狀況只會更糟。

### 2.3 Extrapolation error / OOD action：為什麼 Q 會被「主動」高估

這是 Section 4.2 最重要的觀念。考慮 Bellman 目標

$$
y(s,a) \;=\; r(s,a) + \gamma\, \mathbb{E}_{a' \sim \pi(\cdot|s')}\!\big[Q(s', a')\big].
$$

關鍵兩件事同時發生：

1. `a' ∼ π(a'|s')` 可能落在 `π_β` 從未涵蓋的區域 → `Q(s', a')` 的值由 neural net 「外推」而來，幾乎可以是任意值。
2. Policy improvement 步驟在做 `π ← argmax_π E_{a∼π}[Q(s,a)]`，這個 max **主動挑出** Q 被高估最嚴重的那些 OOD action。

於是有一個正回饋：估錯 → 被選 → 進入 backup target → 把鄰居 Q 也拉高 → 下個 iteration 估更錯。Online 時，policy 真的去試這個動作會收到實際 reward 修正它；offline 沒有這個機制，誤差在 Bellman backup 中**反覆累積放大**，產生圖 2 那條「先升後崩」的 unlearning 曲線（增加 dataset size 也救不了 → 不是一般的 statistical overfitting）。

Fujimoto et al. (2018) 把這稱作 *extrapolation error*；Kumar et al. (2019) 用 distributional shift 框架描述同件事。

數學上 error propagation：策略表現誤差的 horizon dependence 為 `O(1/(1-γ)^2) ≈ H^2`（Farahmand 2010, Kidambi 2020 證明這個 H² 在 worst case 不可避免）。意味著每步只要一點點分布偏移，long horizon 就會被放大成災難。

### 2.4 為什麼 offline RL 不是直接把 online RL 拿過來用（總結）

1. **沒有 self-correction**：online 時錯估的 Q 會被實際軌跡修正；offline 時錯估只會被 Bellman backup 強化。
2. **max 與 OOD 的惡性結合**：policy improvement 步驟會「找」出 Q 函數最樂觀的洞。
3. **Function approximator 在 OOD 沒保證**：neural net 在訓練分布外可以任意外推。
4. **沒有 exploration 可以彌補**：dataset 沒覆蓋的高 reward 區永遠不會被發現（survey 把這標為「不在本文範圍」的限制）。
5. **H² 的 error compounding**：相較 imitation/BC online 的 H，offline RL 的 worst case 比這更糟。

---

## 3. 主要方法家族（章節 3 / 4 / 5 整理）

### 3.1 Importance sampling 系（Section 3）

目標：直接估 `J(π)` 或 `∇J(π)`，把樣本從 `π_β` reweight 到 `π`。

- **Per-trajectory IS**（公式 5）：`w_H^i = ∏_t π(a_t|s_t)/π_β(a_t|s_t)`，variance 隨 H 指數爆炸。
- **Per-decision IS、weighted IS、doubly robust**：用 control variate（Q estimator）降 variance；DR 在 model 或 π_β 任一正確即無偏。
- **Marginalized IS / DICE 家族**（DualDICE, GenDICE, AlgaeDICE）：估 state(-action) marginal ratio `ρ^π(s) = d^π(s)/d^{π_β}(s)`，避免逐步乘積。用 forward Bellman（公式 7）或 backward Bellman + convex duality（公式 14）求解。
- **限制**：對 deep + 高維 + 長 horizon 仍變異過大；要求 π 不能離 π_β 太遠。Survey 結論：純 IS 在 fully offline、高維深度設定下不是主流，但在 OPE / contextual bandit / 廣告領域仍重要。

### 3.2 Policy constraint 系（Section 4.3）

把「Q 不要被 OOD action 問到」這件事用 actor update 約束實現：

$$
\pi_{k+1} = \arg\max_\pi \mathbb{E}_{s\sim D}\,\mathbb{E}_{a\sim\pi(\cdot|s)}[\hat Q^\pi_{k+1}(s,a)] \quad \text{s.t. } D(\pi, \pi_\beta) \le \epsilon.
$$

或把約束塞進 reward / target Q（policy penalty）。

子分支：

- **Explicit f-divergence**（KL, χ², TV...）：KL 形式 = entropy-regularized RL（control as inference, Levine 2018）。缺點：需要顯式 fit `π_β`，多模態行為下會誤差很大。
- **Implicit f-divergence**：AWR、AWAC、ABM。用 `π̄ ∝ π_β · exp(Q/α)` 後加權回歸，等價於 KL 約束但不需顯式建模 π_β。
- **IPM**：MMD（BEAR）、Wasserstein。MMD 在 finite sample 下近似 support constraint，比 KL 更合適（圖 3 的 lineworld 例子：support constraint 才能找出最優解，distribution constraint 會被多數行為策略拖偏）。
- **Support vs density 的關鍵洞見**（Kumar 2019）：限制 support（不出 π_β 的覆蓋範圍）就夠，限制 density（要長得像 π_β）會阻擋 policy 在高機率 action 上集中變成 deterministic，造成過度保守。

### 3.3 Uncertainty-based 系（Section 4.4）

學一個 `P_D(Q^π)`（bootstrap ensemble、Bayesian net、Gaussian head），用保守估計做 policy improvement：

$$
\pi_{k+1} = \arg\max_\pi \mathbb{E}_{s,a\sim\pi}\!\big[\,\mathbb{E}[Q] - \alpha \mathrm{Unc}(P_D(Q))\,\big].
$$

Unc 可以是 ensemble variance（BEAR）、worst-case combination、或 mean - σ。

實務缺點：要在 OOD action 上得到「校準」的 uncertainty 很難；offline RL 對 uncertainty 的要求遠高於 exploration（exploration 只要 cover 好行為就行；offline 要 Q 本身可信）。

### 3.4 Conservative value / CQL 系（Section 4.5）

不對 policy 約束、不估 uncertainty，直接讓 Q 本身保守：

$$
\tilde E(B,\phi) = \alpha\, C(B,\phi) + E_{\text{Bellman}}(B,\phi).
$$

`C_{CQL_0} = E_{s\sim B, a\sim\mu}[Q_\phi(s,a)]`，adversarially 選 μ 把高 Q 壓低。closed-form：`log Σ_a exp(Q(s,a))`。CQL 證明在合適 α 下，學到的 Q 是 true Q 的 pointwise 下界。改良版 CQL_1 再減去 `E_{(s,a)∼B}[Q(s,a)]`，在 batch 內 action 上把 Q 拉回，保留期望下界但減少 underestimation。

優點：對 actor-critic / Q-learning 都通用、不需估 π_β、不需 ensemble。缺點：可能過度悲觀，dataset 小時抑制 undersampled action。

### 3.5 Model-based 系（Section 5）

學 `T_ψ(s'|s,a)` 後 planning 或 rollout。

- **天然優勢**：supervised learning fit model 比 RL 穩；rollouts 提供類似 data augmentation 的效果。
- **distribution shift 仍在**：policy 會 exploit model 在 OOD (s, a) 的錯誤預測（model exploitation）。
- **理論界**（Janner et al. 2019）：

$$
J(\pi) \ge J_\psi(\pi) - \frac{2\gamma r_{\max}(\epsilon_m + 2\epsilon_\pi)}{(1-\gamma)^2} - \frac{4 r_{\max}\epsilon_\pi}{1-\gamma}.
$$

第一項 ε_m 是 model error，第二項 ε_π 是 policy 與 π_β 的 TVD；兩者都被 horizon `1/(1-γ)^2` 放大。

- **代表方法**：MOPO（reward 減 `λ·u(s,a)`，u 是 model error oracle）、MoREL（model error 大時導向 absorbing low-reward state）、DIM（用 normalizing flow 做 distributional constraint）。
- **挑戰**：高維（image）long-horizon 預測仍困難；error oracle 的 consistency 在 sampling error 下無保證。

---

## 4. 評估與 benchmark（Section 6）

- **資料分布是關鍵**：
  - 用 optimal policy 收的資料 → 過於樂觀。
  - 用 partial-run replay buffer（含探索）→ 較真實，但容易低估 offline RL 的難度。
  - 高 entropy / wide coverage 的 π_β 反而比較好學（OOD action 少）。
  - 多模態 π_β 對 explicit behavior-policy estimation 是噩夢。
- **compositional generalization**：好的 offline RL 應該能「縫合」軌跡片段（圖 4，maze2D，從 1→2 和 2→3 的 sub-trajectories 縫出 1→3）。[[D4RL|D4RL]] benchmark 專門測這件事。
- **OPE 仍是 open problem**：在 healthcare/廣告 中重要，但與 policy 學習一樣面臨 distribution shift。

### 應用領域

- Robotics：QT-Opt grasping、visual foresight、RoboNet。
- Healthcare：MIMIC-III sepsis / 呼吸器、epilepsy 刺激。
- 自駕：BDD100K、RobotCar、DIM。
- 推薦 / 廣告：常以 contextual bandit 處理，A/B testing、doubly robust。
- 對話 / NLP：以 logged dialogue 訓練 goal-oriented chatbot（Jaques 2019）。

---

## 5. 開放問題（Section 4.6 / 5.3 / 7）

1. **constraint vs improvement 的張力**：constraint 太緊 → 學不過 π_β；太鬆 → OOD 爆炸。如何 *adaptive* 調整保守程度仍未解。
2. **state distribution 的副作用**：即使 Q 從不在 OOD state 被 query，function approx coupling 仍會讓低密度 state 的解被高密度 state 帶歪（DisCor 系列觀察）。
3. **uncertainty 校準**：deep net 的 epistemic uncertainty 在 OOD 上的可靠度仍不足以支撐 fully offline RL。
4. **H² lower bound**：Kidambi 2020 證明 worst case 不可繞過，需要靠 dataset 結構或先驗知識。
5. **model-based vs model-free 的理論等價性**：在 linear FA 下 fitted VI 與 model-based update 相等；nonlinear 下尚未知。
6. **OPE 仍未真正可信**：對 safety critical 部署（醫療、自駕）這是直接 blocker。
7. **counterfactual inference 工具的引入**：causal inference、density estimation、distributionally robust optimization、invariance learning 都被 survey 點名為未來方向。

---

## 6. 結論（survey 的 take-away）

Offline RL 的本質是 **counterfactual inference 在 sequential decision making 上的延伸**。三大解法（policy constraint / conservative Q / model-based with conservatism）都可以視為「不同的方式去防止 max 操作 + function approximator 在 OOD 區域產生不可控外推」。Survey 強調：未來進步可能不在算法本身，而在 **資料的規模、多樣性與覆蓋度**——這對機器學習其他子領域已是公開的秘密。

---

## 與本研究主線的關聯

本研究主線：以大量 **robot play data** + **極少且弱的 language 標註**，做 **offline meta-RL**，並讓「文字 description → task spec / embedding」在測試時實現 **zero-shot 泛化**（特別關注 meta-learning × zero-shot 並行軸）。Offline RL 的所有難點在此設定下會被放大或重新混合，整理如下：

### A. 直接踩到的難點

1. **Play data 本身是高度多模態、且 reward 標註稀疏**
   - 來自 unscripted human/random/scripted play 的混合 → π_β 是高度 multi-modal 的混合分布。
   - Survey 明確指出（4.6）：multimodal π_β 對 explicit behavior cloning + KL constraint 是噩夢。
   - **implication**：應該優先採 implicit constraint（AWAC / AWR / ABM）或 support-based constraint（BEAR-MMD），或直接走 [[CQL|CQL]] / [[IQL|IQL]] 這類**完全不顯式建模 π_β** 的家族。
2. **Reward 由弱語言標註推導 → reward 本身也是 distribution-shifted**
   - 文字標註只覆蓋少數 trajectory；對 unlabeled play data 必須由 task embedding 推 reward。這等同把 distribution shift 從 (s, a) 擴張到 (s, a, z_task)，z 也會 OOD。
   - **implication**：任何 OOD 偵測 / uncertainty 機制需要在 **(state, action, task-embedding)** 聯合空間上做，不只 (s, a)。
3. **Meta-RL 的 task latent 自帶 OOD 風險**
   - 測試時要 zero-shot 接受新文字 task，意味 z_test ∉ supp(z_train)。
   - 對 Q(s, a, z) 而言，z 的 OOD 與 a 的 OOD 在 Bellman backup 中**同時**被 max 操作放大。
   - **implication**：訓練時須有「task-embedding distribution shift」對應的 constraint（語意相近 z 之間的 smoothness / consistency loss），等價於把 policy constraint 從 action 空間延伸到 task 空間。
4. **H² 的 horizon error 在 long-horizon manipulation 上必爆**
   - 機器人操作 horizon 動輒 100–500 steps，理論上 worst-case error ~ H² 會吃掉很多 advantage。
   - **implication**：偏好 short-horizon model-based rollout（MBPO/MOPO 風格）或 hierarchical decomposition（用 sub-skill embedding 縮短有效 H）。

### B. 可借用的方法清單（與本研究結合的優先順序）

| 方法 | 借用點 | 注意事項 |
| --- | --- | --- |
| **CQL / Conservative Q** | 不需顯式 π_β、直接給 Q 下界，跟 task-conditioned Q(s,a,z) 結合方便 | 對 z 也要 push down OOD task-embedding 的 Q，避免文字輸入端的外推爆炸 |
| **AWAC / AWR / IQL（implicit constraint）** | play data 多模態下避免 KL 約束的失敗模式；advantage-weighted 更新等價於 supervised regression，與「文字-動作」對齊很自然 | weight = exp(Q/α) 中 Q 仍需可信 → 與 CQL 結合更穩 |
| **BEAR / MMD support constraint** | support constraint 比 density constraint 適合 play data | MMD kernel 對連續高維 action / image embedding 設計要小心 |
| **MOPO / MoREL（model-based + uncertainty penalty）** | dataset 大且 dynamics 可學時最 sample-efficient；reward shaping 自然 | uncertainty oracle 對 (s, a, z) 都要 calibrated；目前 ensemble 在 z 上 calibration 未驗證 |
| **DICE 家族 (state-marginal ratio)** | 對 task-conditioned d^π(s|z) 做 marginalized correction，可避免逐步乘積 IS | 高維仍困難，但用於 OPE / model selection 有價值 |
| **doubly robust OPE** | meta-RL 部署前要在 unseen task 上做 model selection / safety check | 需要 task-conditioned baseline value estimator |

### C. 與「文字 → task embedding → zero-shot」的具體連結

- **核心對應關係**：本研究的 task embedding `z` 在功能上像 meta-RL 的 task latent，但因為 zero-shot，**永遠不會在訓練 z 分布中見到測試 z**。這是「task-space 的 OOD」。
- 把 Survey 的洞見映射過來：
  - Policy constraint → 對應「z 空間的 smoothness」：相近文字 → 相近 policy 行為，避免文字端 OOD 觸發 Q-function 在 unseen z 的外推爆炸。
  - Conservative Q → 對應「對 unseen z 的 Q 給保守估計」：可以在 task encoder 的 OOD 區直接降低 Q-target。
  - Model-based with uncertainty → 對應「用 dynamics model 在 z-conditioned 軌跡上 hallucinate 並對 model error 加 penalty」：與 imagination-based meta-RL 自然結合。
- **與 meta-learning × zero-shot 並行方向的契合度**：
  - meta-learning 軸需要 fast adaptation；offline 限制下「adaptation」只能是 in-context inference，因此 [[PEARL|PEARL]] / [[VariBAD|VariBAD]] 風格的 belief / latent inference 比 [[MAML|MAML]] 風格更合適（MAML offline 會被 H² error 痛擊）。
  - zero-shot 軸需要 text encoder 的 generalization；Survey 點名 **causal inference / invariance / distributional robustness** 是處理 distribution shift 的工具集合，正是「文字端 OOD」的天然解法。
  - 兩軸的交集：訓練一個 **task-conditioned conservative actor-critic**，actor 走 implicit constraint（AWAC 風），critic 走 CQL-style penalty 同時對 (a, z) 的 OOD 區壓制，搭配文字 encoder 的 invariance / contrastive consistency loss 來控 z 的 distributional shift。

### D. 該避開的失敗模式

- 直接套 SAC / TD3 + replay buffer 跑 offline：圖 2 的 unlearning 幾乎必現。
- 對 multimodal play data 用 BC 風 fit π_β 再做 KL constraint：誤差會在文字-條件分支間互相污染。
- 只在 (s, a) 上做 uncertainty estimation 而忽略 z 的 OOD：測試時的災難會誤判為演算法不收斂。
- 假設能 online 微調補救：本研究的核心承諾就是零互動，這條退路不存在。

---

## 一句話總結

Offline RL 的全部技術糾結都可以歸到一件事：**Bellman backup 的 max 操作會主動挑出 function approximator 在 OOD action 上被高估的洞並把 policy 推進去，而 offline 沒有 online 試錯的修正機制**；本研究的「task embedding zero-shot」會把這個 OOD 從 (s, a) 擴張到 (s, a, z)，因此 conservative value（CQL）+ implicit constraint（AWAC/IQL）+ task-space invariance/uncertainty 的組合是直接踩在巨人肩膀上的可行起點。
