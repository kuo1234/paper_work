---
type: paper-note
aliases:
  - "Decision Transformer"
  - "DT"
year: 2021
stage: "2-offline-rl基礎"
tags:
  - offline-rl
  - sequence-modeling
  - transformer
  - return-to-go
  - baseline
  - 2-offline-rl基礎
summary: "把RL寫成sequence modeling，return-to-go當prompt；是T2DA-T骨架、語言當條件token的最輕量橋。"
---
> **論文標題**：Decision Transformer: Reinforcement Learning via Sequence Modeling
> **作者**：Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch（UC Berkeley / FAIR / Google Brain）
> **出處 / 年份**：NeurIPS 2021（arXiv:2106.01345）
> **主題**：把離線強化學習（offline RL）改寫成「條件式序列建模」（conditional sequence modeling），以 GPT 結構自迴歸地預測動作，並以「回報目標（return-to-go）」作為條件指定希望達到的表現水準。
> **整理目標**：搞清楚為什麼 RL 可以寫成 autoregressive modeling、return-to-go conditioning 的角色，以及它跟 value-based offline RL（如 CQL）有何根本差別；同時對接到本研究主線（robot play + 弱語言條件 + offline meta-RL + 文字 → task spec / embedding）的可遷移性。

---

## 0. 閱讀總覽（白話版）

Decision Transformer（DT）做了一個極為大膽且簡潔的轉換：**完全不學 value function、不做 Bellman backup、不做 policy gradient**，而是把一條 trajectory 寫成「想要的剩餘回報 → 觀察狀態 → 動作」這樣三種 token 的序列，然後丟給 GPT 訓練「預測下一個動作」。

整個 RL 的訓練被換成跟訓練 GPT 一樣的監督式語言模型——cross-entropy / MSE loss、teacher forcing、自迴歸 decode。

關鍵概念是 **return-to-go**（從當下到 episode 結束的累積獎勵）作為一種「prompt」：在測試時，使用者「想要多好」就把目標 return-to-go 給高一點（例如填 expert 的回報值），模型就會生成能達到該目標的動作；想 mediocre 就給低一點。**「希望多好」是一個可條件的輸入變數**，不是 reward maximization 的副產品。

實驗顯示：在 Atari、[[D4RL|D4RL]]（HalfCheetah/Hopper/Walker/Reacher）、Key-to-Door 等 offline RL benchmark 上，DT 跟當時 SOTA（[[CQL|CQL]]、BEAR、BRAC、AWR）打平或更好，在長期 credit assignment、sparse reward、delayed return 等場景特別有優勢。

對本研究而言（robot play + 弱語言 + offline meta-RL + zero-shot），DT 是把「條件式 prompt + transformer + 離線軌跡」這條路打通的奠基性 paper：後續 [[RT1|RT-1]]、[[RT2|RT-2]]、Gato、[[Octo|Octo]] 等 robot transformer 幾乎都吃這條路線。

---

## Abstract

主張把 RL 抽象成一個 sequence modeling 問題，藉此繼承 Transformer / GPT / BERT 的可擴展性與簡潔性。
DT 是一個 causally masked Transformer，輸入是 desired return、過去 states、過去 actions，輸出對應的動作。不擬合 value function、不算 policy gradient。在 Atari、OpenAI Gym、Key-to-Door 上對齊或超越 model-free offline RL SOTA（主要對手是 CQL）。

> 關鍵字句：把 RL 視為 conditional sequence modeling。

---

## 1. Introduction

**動機**：
- Transformer 在語言、視覺上已展示大規模建模高維分佈、零樣本泛化的能力，是否能取代「傳統 RL 框架」中的 TD-learning / Bellman backup？
- 過去用 Transformer 是「把它當 RL 模型內的一個元件」（例如 actor-critic 內的 backbone）；DT 想做的是 **把 trajectory 的 joint distribution 直接建模**，取代整套 RL 算法。

**核心 paradigm shift**：
- 不再透過 TD-learning 訓練 policy，而是用 sequence modeling 目標訓練 transformer，從而：
  1. **避開 bootstrapping**，解掉 RL 經典的「deadly triad」（function approximation + bootstrapping + off-policy 三者並存導致不穩定）。
  2. **不需要 discount factor**，避免 short-sighted 行為。
  3. **直接享受成熟的 transformer 訓練工具鏈與規模化經驗**。

**為什麼 transformer 對 credit assignment 有利**：
- Self-attention 可以「直接連結」遠距離的 state-return 關係，不像 Bellman backup 需要慢慢把 reward 往前傳，且不容易受到 distractor signal 干擾。

**為什麼用 offline RL 當主要評估場景**：
- Offline RL 傳統上苦於 distribution shift / value overestimation，但對 sequence modeling 卻是天然舒適區（給一堆軌跡，做 supervised learning 即可）。
- 透過選擇 desired return token 作為 prompt，可以「指定要哪一種專家程度的 policy」。

**Illustrative example（圖 2，附錄 A.3 詳述）**：
- 任務：固定有向圖找最短路徑（reward 0 為到 goal、-1 為其他）。
- 訓練資料：random walk 軌跡（**沒有 expert demonstration**）。
- 訓練：用 GPT 預測下一個 return-to-go / state / action token。
- 測試：在每一步加上「偏好較大 return-to-go」的 prior，模型就能拼湊 random walk 的子片段，**生成超越任何單一訓練軌跡的最短路徑**。
- 寓意：sequence modeling + hindsight return information 可以在**沒有 DP** 的情況下做到 policy improvement（"stitching"）。

---

## 2. Preliminaries

### 2.1 Offline RL
- 標準 MDP $(\mathcal{S},\mathcal{A},P,R)$。
- 軌跡：
$$\tau = (s_0,a_0,r_0,s_1,a_1,r_1,\dots,s_T,a_T,r_T)$$
- 在時間步 $t$ 的 return（從現在到結束的 reward 總和）：
$$R_t = \sum_{t'=t}^{T} r_{t'}$$
- Offline RL：只有固定資料集，無法環境互動，因此 exploration / value overestimation 是難點。

### 2.2 Transformers
- 標準 self-attention：query/key/value 線性映射，softmax 加權 value：
$$z_i = \sum_{j=1}^{n} \mathrm{softmax}\big(\{\langle q_i, k_{j'} \rangle\}_{j'=1}^{n}\big)_j \cdot v_j$$
- DT 使用 **causal mask（GPT 風格）**：每個 token 只能看自己以前的 token（$j \in [1,i]$）。
- 作者特別點出：self-attention 的「query/key 內積最大化」可被視為**隱式地建立 state-return association**，這是後面解釋 credit assignment 的關鍵 intuition。

---

## 3. Method（核心：trajectory 表示 + return-to-go conditioning + autoregressive 動作預測）

### 3.1 Trajectory 的表示

**關鍵設計**：把 reward 改成 **return-to-go**（剩餘累積獎勵）：
$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

軌跡寫成三類 token 的交錯序列：
$$\tau = \big(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T\big)$$

**為什麼不直接用 reward？** 因為我們希望「在第 $t$ 步生成動作時，能根據未來想要的回報 conditional 生成」，而不是依賴過去獎勵——這正是 return-to-go 作為 prompt 的精神：**它指定目標，不是描述過去**。

**測試時的更新規則（autoregressive rollout）**：
1. 初始給定目標 return $\hat{R}_1$（例如想要 expert level）與起始狀態 $s_1$。
2. 模型生成 $a_1$，環境回 reward $r_1$，新狀態 $s_2$。
3. **遞減** return-to-go：$\hat{R}_2 = \hat{R}_1 - r_1$。
4. 重複直到 episode 結束。

這個遞減規則就是「目標被消耗」的物理意義：每跨一步，剩下要拿的分數就少一點。

### 3.2 為什麼 RL 可以寫成 autoregressive modeling

從 Bayesian 角度看，標準 RL 想學 $\pi(a|s)$ 最大化 $\mathbb{E}[\sum r_t]$；DT 把它改寫成：

$$p_\theta(a_t \mid s_{\le t},\, a_{<t},\, \hat{R}_{\le t})$$

——亦即，在「過去全部上下文 + 想要的剩餘回報」條件下，預測下一個動作的條件分佈。

**核心洞察**：在 offline data 中，每條軌跡都自然定義了一組 $(\hat{R}_t, s_t, a_t)$；對這個 joint distribution 做 maximum likelihood 估計，就等同於學一個「條件 policy 的家族」 $\{\pi_R\}_R$，其中 $R$ 是欲達成的回報。

這跟 GPT 學語言完全一樣：給定前綴 tokens，預測下一個 token。差別只在 token 類型是 multimodal（return、state、action）。

訓練 loss（連續動作）：
$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}} \Big[ \frac{1}{T}\sum_t \| a_t - \hat{a}_\theta(\hat{R}_{\le t}, s_{\le t}, a_{<t}) \|^2 \Big]$$
離散動作則用 cross-entropy。

### 3.3 架構細節
- 取最後 $K$ 個時間步，每步三個 token（$\hat{R}_t, s_t, a_t$），共 $3K$ 個 token。
- 每種 modality 自己的線性 embedding + LayerNorm；視覺輸入則接 conv encoder（DQN encoder）。
- **每個 timestep 自己一個 learned positional embedding**（注意：不是 per-token 的標準 sinusoidal pos enc，而是 per-timestep，再加到該步的三個 token 上）。
- 通過 GPT backbone（causal mask），最後接 linear head 預測動作。
- 訓練時只對 action token 計算 loss；預測 state / return-to-go 沒有顯著收益（但架構支援，5.4 / 5.5 節有探索）。

### 3.4 Pseudocode（重寫成繁中註解版）

```python
# R, s, a, t: returns-to-go, states, actions, timesteps
# transformer: causal-masked GPT
# embed_s/a/R: 各 modality 的線性 embedding
# embed_t: per-timestep 的 learned positional embedding
# pred_a: 線性 action head

def DecisionTransformer(R, s, a, t):
    pos = embed_t(t)                          # 每個 timestep 一個 pos embedding
    s_e = embed_s(s) + pos
    a_e = embed_a(a) + pos
    R_e = embed_R(R) + pos
    # 交錯成 (R_1, s_1, a_1, ..., R_K, s_K, a_K)
    x = interleave(R_e, s_e, a_e)
    h = transformer(x)
    h_a = select_action_positions(h)
    return pred_a(h_a)

# 訓練
for (R, s, a, t) in dataloader:
    a_hat = DecisionTransformer(R, s, a, t)
    loss = mean((a_hat - a) ** 2)             # 連續動作
    loss.backward(); optimizer.step()

# 推論 / rollout
target_return = R_expert                       # 想要的 episode 總回報
R, s, a, t = [target_return], [env.reset()], [], [1]
done = False
while not done:
    action = DecisionTransformer(R, s, a, t)[-1]
    new_s, r, done, _ = env.step(action)
    R.append(R[-1] - r)                       # **return-to-go 遞減 reward**
    s.append(new_s); a.append(action); t.append(len(R))
    R, s, a, t = R[-K:], s[-K:], a[-K:], t[-K:]
```

---

## 4. 演算法 / 架構流程（圖 1 + Algorithm 1 串連）

整體流程簡述如下：

1. **資料**：收集任意 policy 的 offline trajectories（不需要 expert）。
2. **預處理**：對每條軌跡，將每步 reward 累加為 return-to-go；建立 $(\hat{R}_t, s_t, a_t)$ 三元 token。
3. **批次**：從資料集隨機抽長度為 $K$ 的片段。
4. **embedding**：三種 modality 各自線性映射；加上 timestep 的 learned positional embedding。
5. **GPT forward**：通過 causal-masked transformer。
6. **prediction**：在每個 $s_t$ 對應位置的 hidden state 上 head 出 $\hat{a}_t$。
7. **loss**：MSE（連續）或 cross-entropy（離散）平均。
8. **推論**：
   - 給目標 return $\hat{R}_1$；
   - 每步 transformer 預測 $a_t$；
   - 環境執行 → 取得 $r_t, s_{t+1}$；
   - $\hat{R}_{t+1} \leftarrow \hat{R}_t - r_t$；
   - 維持 context window 為最近 $K$ 步。

---

## 5. 實驗與結論

### 5.1 Atari（Table 1）
- 1% DQN-replay 資料，4 個遊戲：Breakout、Qbert、Pong、Seaquest。
- DT 在 3/4 遊戲跟 CQL 競爭、在多數情況超越 REM、QR-DQN、BC。
- Qbert 是 DT 落後 CQL 的例外（15.4 vs 104.2 normalized）。
- Breakout 超越甚至遠勝 CQL（267.5 vs 211.1）。

### 5.2 OpenAI Gym / D4RL（Table 2）
- HalfCheetah / Hopper / Walker / Reacher × {Medium, Medium-Replay, Medium-Expert}。
- 平均（不含 Reacher）DT 74.7、CQL 63.9、BEAR 48.2、BRAC-v 36.9、AWR 34.3、BC 46.4。
- DT 在 Medium-Expert 場景幾乎全勝（HalfCheetah 86.8、Walker 108.1 都贏 CQL）。
- 結論：DT 是強有力的 offline RL baseline，且不需要 value pessimism、不需要 behavior regularization。

### 5.3 Discussion 子議題（DT 為什麼有效）

**Q1：DT 是不是只是在做 subset 上的 BC？**（5.1）
- 對比 Percentile BC（取前 X% return 軌跡做 BC）：
  - D4RL 等資料充足時，DT 跟最強的 %BC 持平，但 DT **不需要選 X**。
  - Atari 等資料稀疏時，DT 顯著贏 %BC——因為 DT 能利用「全部」資料（包含與條件 return 不相符的軌跡）來提升泛化。
- 結論：DT 不只是 subset BC，它能把「不同水準的軌跡」當成一個分佈來學，再用 return-to-go 做條件 query。

**Q2：DT 能多準地模擬 return 分佈？**（5.2，Figure 4）
- 把 target return 從低到高掃描，發現 sampled return 跟 target return **高度正相關**（很多任務幾乎完美對齊 oracle 線）。
- 在 Seaquest 等任務，**target return 超過資料集中最高 return** 時，DT 仍能推外插出更好的行為（extrapolation）。

**Q3：context length 重要嗎？**（5.3，Table 5）
- 對 Atari 而言，$K=1$（只看當前 state）的表現遠遜於 $K=30\text{-}50$。
- 假設：當在學一個 policy 的分佈時，context 幫助 transformer **辨識當前是哪一個 sub-policy 生成的軌跡**，提升學習效率與穩定性。

**Q4：long-term credit assignment？**（5.4，Table 6，Key-to-Door）
- 三相環境：拿鑰匙 → 走過空房 → 開門才得 reward；只有 random walk 訓練資料。
- DT 在 10K random trajectories 達 94.6%，CQL 13.3%，BC 1.6%，%BC（只看成功軌跡）95.1%。
- 用 hindsight return 的方法（DT、%BC）能學會，TD learning 在長 horizon 上難以傳遞 Q-value。

**Q5：transformer 能不能當 critic？**（5.5，Figure 5）
- 把 DT 改成同時預測 return token（包括初始 $p(\hat{R}_1)$）。
- 觀察到：模型會根據事件**連續更新 return 機率**（拿鑰匙後機率上升，沒拿則保持低）。
- Attention weights 集中在 pivotal events（拿鑰匙、到門），印證 self-attention 形成了 state-reward association。

**Q6：sparse / delayed reward？**（5.6，Table 7）
- 把 D4RL 改成「整條 trajectory 的 reward 只在最後一步給」。
- TD learning（CQL）大幅崩潰（Hopper Medium-Expert 從 111.0 掉到 9.0）。
- DT 幾乎不受影響（從 107.6 變 107.3），%BC、BC 同樣抗 sparse（因為它們對 reward 結構不敏感）。

**Q7：為什麼不需要 value pessimism / behavior regularization？**（5.7）
- TD-based 方法學近似 value，再對它做 argmax，**會放大 value 的誤差**（off-policy + bootstrapping 的副作用），所以需要 conservatism（CQL 的悲觀懲罰）或 action-space constraint（BEAR/BRAC）來壓住。
- DT 不對學到的 value function 做 explicit optimization，**直接做監督式 likelihood**，因此不需要這些 hack。

**Q8：online RL？**（5.8）
- DT 可以當 behavior generation 的「記憶引擎」，搭配 Go-Explore 等 exploration 算法，潛力很大；但本文未做 online 實驗。

### 與 CQL / TD learning 的根本差別總結

| 面向 | TD-based offline RL（CQL/BEAR/BRAC） | Decision Transformer |
|---|---|---|
| 學什麼 | Q-function（隱式 policy）| trajectory 上的條件分佈 $p(a \mid s, \hat{R})$ |
| 訓練目標 | Bellman backup + 悲觀懲罰 / KL 正則 | Supervised MLE / MSE |
| Credit assignment | Bellman 緩慢回傳，怕 distractor | Self-attention 直接連 long-range |
| Sparse / delayed reward | 嚴重退化 | 幾乎不受影響 |
| Discount factor | 必要 | 不需要 |
| Value overestimation | 主要痛點，要 pessimism 控制 | 不存在（沒在優化 learned function）|
| Stitching（拼接 sub-optimal 軌跡）| DP 提供 | Sequence modeling 隱式達成（如附錄 graph 範例）|
| Test-time 表現指定 | 隱式（policy 已固定）| 顯式：給 target return |

---

## 6. 相關工作

- **Offline RL**：BCQ、BEAR、BRAC、CQL、MOReL、MOPO 等用 action constraint / value pessimism / model pessimism 處理 distribution shift；DT 不需這些。
- **RL 中的監督式學習**：Q-learning（仍含 bootstrapping）、BC（無 reward 條件）；UDRL（"upside-down RL"，Srivastava et al. 2019、Kumar et al. 2019 / Reward-Conditioned Policies）跟 DT 動機接近，**但前者強調「監督學習」、DT 強調「序列建模」**——這個轉換讓 DT 可以利用 long context、scale 化、未來甚至可在 reward-free 下訓練（類似 language pretrain）。
- **Concurrent work**：Janner et al. 的 Trajectory Transformer 同期出現，亦做序列建模，但加上 state / return 的預測與 discretization，偏向 model-based；DT 走純 model-free。
- **Credit assignment**：RUDDER、Synthetic Returns 等顯式學分解 reward；DT 不顯式學 reward function，而是讓 transformer 自然出現 state-reward association。
- **Conditional language generation**：CTRL、PPLM、GeDi 等做 controllable text，DT 與其概念呼應；差別是 reward 是時間變化量，且 prompt（return-to-go）隨環境互動而更新。
- **Transformers in RL**：Stabilizing Transformers for RL (Parisotto et al.)、Relational RL、Episodic memory transformer 等過去是「拿 transformer 當 RL 模型內的 backbone」；DT 是「拿 transformer 取代 RL 演算法本身」，這是定位上的根本差異。

---

## 7. 結論

DT 試圖把 language / sequence modeling 與 RL 的概念統合。
在標準 offline RL benchmark 上，僅以 GPT 加上極少改動就能匹敵或超過為 offline RL 量身訂做的算法。

作者強調未來方向：
- 大規模 self-supervised pretrain（拿大量無 reward 軌跡先預訓練）。
- 更精細的 embedding（例如對 return 分佈建模以處理 stochastic 環境而非確定性 return）。
- 用 transformer 取代 model-based RL 的 dynamics model。
- MDP 設定下 transformer 會犯哪些錯、安全性、reward 設計的潛在 misuse。

---

## 與本研究主線的關聯

研究主線：**robot play + 極少弱語言 + offline meta-RL + 文字 → task spec / embedding 做 zero-shot**。

DT 對這條主線的啟發點與接口非常多：

1. **把 RL 改寫為「條件式序列建模」是後續所有 robot transformer 的範式起點**
   - RT-1 / RT-2 / Gato / Octo / [[OpenVLA|OpenVLA]] 都繼承了 DT 的核心邏輯：把 trajectory tokenize、用 causal transformer 自迴歸預測 action、用某種 "prompt"（在 robot 設定裡通常是語言指令或 goal image）做條件。
   - DT 的 return-to-go 在 robot 領域被替換成 **language instruction embedding**（RT-1：文字 → FiLM 條件；RT-2：直接用 PaLM-E 級的 VLM）；但「以 prompt 條件、用 transformer 自迴歸 decode action」的骨架是同一個。

2. **return-to-go ↔ task embedding / language prompt 的對應**
   - 在主線中，我想用「文字 → task embedding」作為 zero-shot 任務的條件；DT 教我們：**只要那個條件 token 跟 trajectory 在訓練資料中是 jointly observable 的**，transformer 就能自然學會「給定條件 → 生成適合的行為分佈」。
   - 對 offline meta-RL 而言，這意味著：弱語言標註（甚至是 noisy 自動描述）可以扮演 return-to-go 的角色——每段 play data 配上一句「正在抓杯子」就是該段軌跡的條件 prompt。Test-time 給新指令，就如同給新 return 目標。

3. **與 value-based offline meta-RL 對比（[[PEARL|PEARL]] / [[VariBAD|VariBAD]] 路線）**
   - PEARL / VariBAD 走「學 task latent + 條件 policy」+「value-based RL」的路線，需要 reward 訊號驅動學習，且仍受 distribution shift / value overestimation 困擾。
   - DT 走「條件 sequence modeling」路線，**對 reward 結構（sparse/delayed）和 dataset 品質都更寬容**，這對 robot play data（reward 經常稀疏、甚至需要事後標註）是極大優勢。
   - 因此在「offline meta-RL + 弱語言」的設計上，DT 風格更自然：language embedding 直接當 prompt token，不必走 value function。

4. **Credit assignment / sparse reward 的優勢**
   - 真實 robot play 中，「拿到玩具」「打開蓋子」的事件常常稀疏。
   - DT 在 Key-to-Door、delayed D4RL 等實驗證明：sequence model 對這類訊號比 TD learning 友善太多。對未來「弱監督 + 稀疏 reward」設定是強烈支持。

5. **作為實驗 baseline 的角色**
   - 任何「語言條件 / 任務條件 + offline RL」的新方法，幾乎都要跟「Language-Conditioned DT」比較——直接把 instruction embedding 拼在 return token 旁邊當作 prompt，就是極強的 baseline。
   - 同時也是檢驗「我的方法是否真的學到 meta-knowledge」的對照：如果一個簡單 conditional DT 已經接近性能上限，就需要重新思考方法的價值定位。

6. **stitching 與 zero-shot 任務組合的潛力**
   - 附錄 graph 範例顯示，DT 可以**從 sub-optimal 軌跡片段組合出全新的最優路徑**。
   - 對 robot play → task spec 的應用而言，這暗示：給定大量未標註 play data，只要在 test time 用合適的 prompt（語言/task embedding），就可能組合出未曾出現過的整套行為——這正是 zero-shot task spec 的本質訴求。

7. **限制與需要補的後續論文**
   - DT 本身沒處理 stochastic environment（return 是確定性的）。對 robot 而言環境隨機性大，需配合 Stochastic DT、Q-Transformer 或 RvS 等後續工作。
   - DT 不顯式學世界模型，無法做 explicit planning；若主線想做 model-based meta-RL 推理，需配合 Trajectory Transformer / Dreamer 系列。
   - DT 沒做 online fine-tuning；主線若需 sim-to-real / online adaptation，需擴充 ODT (Online Decision Transformer) 或 RvS-online 等。

---

## 一句話總結

Decision Transformer 證明「**把離線 RL 改寫成條件式 (return-to-go) 序列建模，再用 GPT 自迴歸預測動作**」就能在不用 value function、不用 Bellman backup、不用悲觀懲罰的情況下，跟 SOTA offline RL 並駕齊驅、甚至更擅長 long-horizon credit assignment 與 sparse reward——這條路線正是後續所有 language-conditioned robot transformer 的奠基範式。
