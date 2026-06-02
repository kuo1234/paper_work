---
type: paper-note
aliases:
  - "MACAW"
year: 2021
stage: "3-offline-meta-rl"
tags:
  - offline-meta-rl
  - advantage-weighting
  - bilevel
  - baseline
  - 3-offline-meta-rl
summary: "offline meta-RL起點之一，AWR + enriched policy update + bilevel；offline meta-RL baseline。"
---
> **論文**：Offline Meta-Reinforcement Learning with Advantage Weighting (MACAW)
> **作者**：Eric Mitchell, Rafael Rafailov, Xue Bin Peng, Sergey Levine, Chelsea Finn
> **發表**：ICML 2021 (PMLR 139)
> **連結**：https://sites.google.com/view/macaw-metarl
> **關鍵字**：Offline Meta-RL、[[MAML|MAML]]、Advantage-Weighted Regression (AWR)、Universality、Weight Transform Layer

---

## 0. 閱讀總覽（白話）

這篇是 **offline meta-RL** 這個問題設定的奠基論文之一。一句話：把「離線資料 + meta-learning」結合，做到「meta-train 時完全不跟環境互動、meta-test 時也只用極少資料（< 5 條 trajectory）就要適應到新任務」。

作者觀察到一個關鍵的方法論張力：
- 想要 **consistent**（test 時用越多資料就越好），就應該走 MAML 這種「test 時真的跑 RL fine-tune」的路線。
- 但 MAML 原版用 policy gradient，**完全不能離線**。
- 反過來，offline RL 圈成熟的是 value-based 方法（如 AWR），但 value-based + MAML 又會有 bootstrap 不穩、需要很多 step 才能傳遞 reward 等問題。

MACAW 的解法：在 inner / outer loop 都用「不靠 bootstrap、純監督回歸」的 AWR 目標。再做兩個關鍵改造：
1. **Enriched policy update**：inner loop 除了 AWR 的 weighted regression，多加一條 advantage regression head，讓 gradient 同時帶有「regression target」和「regression weight」資訊，理論上把 inner update 變成 universal。
2. **Weight transform layer**：因為 MLP 一步梯度只能對 weight matrix 做 rank-1 更新，太弱。改用 latent code → weight matrix 的線性 hypernetwork，提升 gradient 的表達能力。

實驗在 Half-Cheetah / Walker / Ant 的 offline meta-RL 變體上，把 Offline [[PEARL|PEARL]]、Multi-task AWR + fine-tune、Meta-BC 全部打贏，並且在 task 數量稀疏時也很穩定。

對主線研究的重要性：**MACAW 幾乎是「offline meta-RL」這條線最乾淨的基準起點**——它把問題正式化（task 是 (MDP, behavior policy)、test 只有 < 5 trajectories）、給了一個 consistent + offline 友善的演算法骨架，後續所有 offline meta-RL（含 robot play、語言條件化 task）幾乎都要跟它比。

---

## Abstract

提出 offline meta-RL 問題：meta-train 時只能使用多個任務的固定離線 buffer、meta-test 時新任務也只給很少資料（不到 5 條 trajectory），全程不准跟環境互動。這跟「pre-train + fine-tune」的監督式典範類比。

提出的演算法 **MACAW = Meta Actor-Critic with Advantage Weighting**：
- Optimization-based meta-learning（MAML 系）
- Inner / outer loop 都用簡單的 supervised regression 目標（基於 AWR）
- 可完全離線 meta-train
- 在標準 meta-RL benchmark 的 offline 版本上明顯贏過先前方法

---

## 問題背景

### Offline RL vs. Offline Meta-RL vs. Online Meta-RL

| 設定 | meta-train 期間是否能與環境互動 | test 時資料量 | 任務分佈 |
|---|---|---|---|
| **Online RL** | 是（單一任務） | 不適用 | 單一任務 |
| **Offline RL** | 否（單一任務固定 buffer） | 不適用 | 單一任務 |
| **Online Meta-RL**（MAML、PEARL） | 是（多任務，可在線收集） | 通常數十條 rollout | 多任務 |
| **Offline Meta-RL**（本文） | **否**（每個 train task 只給固定 buffer） | **< 5 trajectory** | 多任務 |

關鍵差別：
- **Offline RL 不處理跨任務轉移**，也不處理 test 任務只有極少資料的情況。
- **Offline Meta-RL 同時繼承了**：offline RL 的分佈偏移 / extrapolation error 問題 + meta-RL 的「test 時要快速適應新任務」問題。

### 為什麼 test task 只能給極少資料？

論文設定 test 用不到 5 條 trajectory，原因是：
1. 真實應用（醫療、機器人）中，新任務的資料往往昂貴或危險。
2. 如果 test 可以給很多資料，那就退化成「offline RL 解這個新任務」而不是 meta-learning。
3. 它逼迫演算法在 meta-train 時學會「task prior」，而不是只靠 test 時硬學。

### Consistency desideratum（一致性需求）

作者強調 offline meta-RL 演算法應該是 **consistent**：給夠多 test 任務的資料，adaptation 應該能找到該任務的好 policy，不管 meta-train 任務分佈長什麼樣。
- Contextual meta-RL（如 PEARL）通常 **不 consistent**：test 用的 inference network 在 OOD task 上就會壞。
- MAML 因為 test 時真的在跑 gradient descent，**天然 consistent**。
- 這是作者選擇 optimization-based meta-learning 而非 contextual 的核心理由。

### 為什麼 inner adaptation 在離線設定特別難？

這是 read.md 指定要涵蓋的重點：
1. **沒有探索可言**：inner loop 只能看 Dtest 這個固定 batch，如果 behavior policy 收集到的資料不能消歧 task，就沒辦法靠「再多走幾步」補救。
2. **Bootstrapping 不穩**：value-based 方法 inner loop 想用 TD update，會碰到 offline RL 經典的 Q-function extrapolation error；但 MAML 又要求 inner loop 的 gradient 穩定可微。
3. **Truncated optimization 的需求**：inner loop 為了記憶體 / 計算只能跑幾步 gradient，但 TD-based 方法需要很多步才會把 reward 傳出去——矛盾。
4. **AWR 的 universality 限制**：直接拿 AWR 當 inner loop 損失，gradient 形式是「(reward 加權) × log π 的梯度」，無法同時恢復「regression weight」與「regression target」，理論上不夠 universal（見後文 Theorem 1）。

---

## 核心方法與公式

### 符號

- 任務 \( T_i = (M_i, \mu_i) \)：MDP \( M_i \) 加上一個未知的 behavior policy \( \mu_i \)；任務分佈 \( p(T) = p(M, \mu) \)。
- 每個 train task 的離線 buffer：\( D_i = \{(s_{i,j}, a_{i,j}, s'_{i,j}, r_{i,j})\} \)，由 \( \mu_i \) sample 出來。
- Meta-parameters：policy 初始 \( \theta \)、value function 初始 \( \phi \)。
- 任務內適應後參數：\( \theta_i', \phi_i' \)。
- 學習率：inner \( \alpha_1, \eta_1 \)，outer \( \alpha_2, \eta_2 \)。
- Temperature：\( T > 0 \)。

### 回顧 AWR (offline RL backbone)

AWR 的 policy loss 是「以 advantage 加權的最大概似」：

\[
L_{\text{AWR}}(\vartheta, \phi, B) = \mathbb{E}_{s,a \sim B}\!\left[ -\log \pi_\vartheta(a \mid s) \cdot \exp\!\left( \frac{1}{T} \big( R_B(s,a) - V_\phi(s) \big) \right) \right]
\]

其中：
- \( R_B(s,a) \)：buffer 中從 \((s,a)\) 開始的 Monte Carlo return（不依賴 bootstrap）。
- \( V_\phi(s) \)：學到的 value function（評估 behavior policy）。
- \( R_B(s,a) - V_\phi(s) \)：advantage estimate。

直覺：把 advantage 高的 (s,a) 給更大權重去做 imitation。

Value function 直接對 Monte Carlo return 做 MSE 回歸（沒有 bootstrap）：

\[
L_V(\phi, D) = \mathbb{E}_{s,a \sim D}\!\left[ \big( V_\phi(s) - R_D(s,a) \big)^2 \right]
\]

選 AWR 作為 backbone 的理由：**不需要 bootstrap，inner loop 一兩步梯度就能得到有意義的 update**——這正是 MAML 需要的。

### MACAW 的 Enriched Policy Update（核心創新 1）

**問題**：直接用 \( L_{\text{AWR}} \) 當 inner loss 不是 universal——梯度只攜帶「以 advantage 為權的 log π 梯度」，無法分離出「regression target = action」與「regression weight = advantage」。Finn & Levine (2018) 已證明 MAML 的 inner gradient 必須保留「推斷 task 所需」的全部資訊。

**解法**：policy 網路加一個 advantage prediction head \( A_\theta(s, a) \)，並在 inner loop 加上一條 advantage regression loss：

\[
L_{\text{ADV}}(\theta, \phi_i', D) = \mathbb{E}_{s,a \sim D}\!\left[ \Big( A_\theta(s,a) - \big( R_D(s,a) - V_{\phi_i'}(s) \big) \Big)^2 \right]
\]

Inner loop 的 policy loss 變成：

\[
L_\pi = L_{\text{AWR}} + \lambda \, L_{\text{ADV}}
\]

Inner adaptation step：

\[
\phi_i' \leftarrow \phi - \eta_1 \nabla_\phi L_V(\phi, D_i^{\text{tr}})
\]
\[
\theta_i' \leftarrow \theta - \alpha_1 \nabla_\theta L_\pi(\theta, \phi_i', D_i^{\text{tr}})
\]

**重點**：
- Advantage head \( A_\theta \) **只在 inner loop 用**；meta-test 時直接丟掉，policy 拿出來就是 RL policy。
- 加 ADV head 的意義是「強迫 inner gradient 同時帶 action 和 advantage 兩種資訊」，因此理論上 MAML inner step 可變 universal（論文 Theorem 2）。
- 實驗（Figure 5 左）顯示 adaptation data 越爛、enriched update 越重要——因為這時 task 訊號弱，inner gradient 需要更多資訊才能 disambiguate。

### Outer Loop（meta-objective）

Value function outer：

\[
\min_\phi \; \mathbb{E}_{T_i} \!\left[ L_V\!\big( \phi - \eta_1 \nabla_\phi L_V(\phi, D_i^{\text{tr}}),\; D_i^{\text{ts}} \big) \right]
\]

Policy outer（注意 outer loop 用回原本的 AWR loss，不再加 ADV——因為 outer loop 的梯度路徑足夠複雜，universality 顧慮只發生在「淺淺幾步 inner」這側）：

\[
\min_\theta \; \mathbb{E}_{T_i} \!\left[ L_{\text{AWR}}\!\big( \theta - \alpha_1 \nabla_\theta L_\pi(\theta, \phi_i', D_i^{\text{tr}}),\; \phi_i',\; D_i^{\text{ts}} \big) \right]
\]

實作小技巧：每個任務的 inner support 與 outer query 用 **disjoint batches**（\( D_i^{\text{tr}} \) vs \( D_i^{\text{ts}} \)），避免 memorize adaptation data。

### Weight Transform Layer（核心創新 2）

**動機**：MLP 一個 fully-connected layer 的 weight matrix \( W \)，一步梯度只能對它做 rank-1 更新（\( \Delta W = -\eta \, g \, x^\top \) 是 rank-1）。Finn & Levine 證明這會讓 MAML 需要不切實際地深才能 universal。

**解法**：把每層 \( W \) 改成由一個 latent code \( z \) 透過線性映射生成：
- 維護 latent code \( z \)，前向時用 \( W = \text{Linear}_W(z),\; b = \text{Linear}_b(z) \)，再算 \( y = Wx + b \)。
- Inner loop 的 gradient descent 是對 **latent code \( z \)** 做的，再映射回 weight matrix。
- 因為 \( z \) 可能高維，所映出的 weight update 可以到 rank 高達 dim(\( z \))。

這跟 Hypernetwork (Ha et al., 2016) 和 LEO (Rusu et al., 2019) 同源，但 MACAW 用最簡單的線性映射，並且 policy / value 的每一層都套用、每層獨立 latent code。實驗（Figure 5 中）：移掉 weight transform，學習速度和穩定性大幅下降。

---

## 演算法流程

### Algorithm 1：MACAW Meta-Training

```
輸入：任務 {T_i}、離線 buffer {D_i}
初始化 meta-params θ, φ
for 訓練 step do
    for 每個任務 T_i do
        從 D_i 取出 disjoint 兩批：D_i^tr, D_i^ts
        # Inner loop（在 support set 上適應）
        φ_i' ← φ - η_1 ∇_φ L_V(φ, D_i^tr)
        θ_i' ← θ - α_1 ∇_θ L_π(θ, φ_i', D_i^tr)         # L_π = L_AWR + λ L_ADV
    end for
    # Outer loop（在 query set 上更新 meta-params）
    φ ← φ - η_2 Σ_i ∇_φ L_V(φ_i', D_i^ts)
    θ ← θ - α_2 Σ_i ∇_θ L_AWR(θ_i', φ_i', D_i^ts)
end for
```

### Algorithm 2：MACAW Meta-Testing

```
輸入：test task T_j、offline 資料 D（極少量）、meta-policy π_θ、meta-value V_φ
θ_0 ← θ, φ_0 ← φ
for n 步 do
    φ_{t+1} ← φ_t - η_1 ∇ L_V(φ_t, D)
    θ_{t+1} ← θ_t - α_1 ∇ L_π(θ_t, φ_{t+1}, D)
end for
```

**Consistency 體現**：test 階段跑的就是一個良好定義的 AWR fine-tune 子程序，因此資料越多越好，符合 consistency。

---

## 實驗與結論

### Benchmark

Offline 變體：Half-Cheetah-Direction、Half-Cheetah-Velocity、Walker-Params（動力學變化）、Ant-Direction。每個 task 的離線 buffer 取自從零訓 RL agent 的 replay buffer。

### 主要結果（Figure 3）

對手：
1. **Offline PEARL**：state-of-the-art off-policy meta-RL 的離線變體。
2. **Multi-task AWR + fine-tune**：合所有 task 做 multi-task AWR，再 fine-tune（20 步 Adam）。
3. **Meta-Behavior Cloning**：純監督式 meta-BC。

結論：
- **MACAW 是唯一在所有環境上都打贏 meta-BC 的方法**。
- Offline PEARL 幾乎全面失敗，作者歸因於 Q-function extrapolation error + offline bootstrap 不穩。
- MT + fine-tune 在 cheetah 還行，但 walker / ant 上崩。

### 線上 fine-tune（Table 1）

允許 test 時再額外蒐集 10k / 20k 步：MACAW 在 2/3 環境上能進一步進步、PEARL 在 2/3 環境上反而退步。MACAW 因為「meta-train 時就是在學如何 fine-tune」，所以線上資料能直接接上。

### Ablation：Enriched policy update（Figure 5 左）

用 first / middle / last 100k 的 replay buffer 模擬 poor / medium / expert data。
- Expert adaptation data：有沒有 enriched update 都差不多。
- Random / poor adaptation data：**enriched update 大幅勝出**——確認其「資料弱時保住 task 訊號」的角色。

### Ablation：Weight transform layer（Figure 5 中）

- 移掉 WT、等寬 FC：學得很慢、不穩。
- 移掉 WT、等參數量 FC：依然輸給 MACAW。
- WT 對學習速度和穩定性都關鍵。

### Ablation：訓練任務稀疏性（Figure 5 右）

只給 3 個 train task 也行：**MACAW 在 task 稀疏時退化最緩**，Offline PEARL 在 task 多或少時都崩、只在「剛剛好」的中間區段才行。作者歸因於 PEARL 的 task inference net + value net 在離線設定容易不穩定。

### 結論與未來工作

- 正式定義 offline meta-RL 問題。
- MACAW 同時做到：sample-efficient、可完全離線 meta-train、test 時 consistent。
- 首個成功結合 gradient-based meta-learning 與 off-policy value-based RL 的演算法。
- 限制：未學 exploration policy（test 線上資料來自 random policy）；用 MC return 而非 bootstrap，asymptotic 表現可能受限；未來可結合 TD-λ 或 explore-aware objectives。

---

## 與本研究主線的關聯

主線設定：**robot play 蒐集到的大量無標籤離線資料 + 極少量弱語言標註 + 目標是 zero-shot 對新文字任務做出動作**，並關注 meta-learning × zero-shot 的並行路線。MACAW 對這條主線的價值：

### 1. MACAW 是 offline meta-RL 的「乾淨骨架」

主線「robot play + offline meta-RL + 文字 task spec」的最自然實作之一是：
- 把每段 play 視為一個 (隱含) task 的 behavior data。
- Meta-train 階段用 MACAW-like 演算法學一個能快速適應的 (π_θ, V_φ)。
- Test 階段把文字 task spec 轉成 task embedding，當作 \( \theta_i' \) 的條件 / 提示，做 zero-shot 或 few-shot adaptation。

→ MACAW 是這條 pipeline 中 **「offline meta-RL 主幹」最少自由度的合理選擇**：bootstrap-free、consistent、跟監督式 pre-train + fine-tune 思維對齊。

### 2. 直接作為 baseline

論文做 offline meta-RL 時，baseline 幾乎必含：
- **Offline PEARL**（contextual + offline）
- **MACAW**（optimization-based + offline）
- Multi-task offline RL + per-task fine-tune

我的主線 paper 若做 offline meta-RL + 文字 conditioning，MACAW 是 contextual 路線（PEARL / [[VariBAD|VariBAD]]）的對照組——而且 MACAW 因為 consistent，能驗證「文字 conditioning 是否額外帶來收益」這件事。

### 3. Universality + weight transform 的啟示

主線目標是「文字 → task embedding → 動作」的 zero-shot。如果走 optimization-based 路線（test 時對少量 text-action 對做 gradient adaptation），那 **MACAW 對 inner-loop 表達力的處理（enriched loss + weight transform）幾乎可以直接搬過去**：
- 把 enriched ADV head 換成「text alignment head」之類的 auxiliary loss，讓 inner gradient 帶有「任務語意」資訊。
- Weight transform 對 robot policy 的 rank-1 限制問題同樣存在，可作標配。

### 4. 對「為什麼 test task 只有極少資料」的設定共鳴

主線的「弱語言 + zero-shot」本質就是「test task 只有 1 句話 / 0 條 trajectory」的極端情況。MACAW 已論證：當 test data 弱（甚至是 random policy 收的），enriched policy update 越關鍵。這直接告訴我們：**在「test 只有文字、沒有動作示範」這種極端弱訊號情境，inner-loop 必須額外塞 auxiliary 訊號（如語言對齊 loss）才能 disambiguate task**。

### 5. Consistency 對 zero-shot 評估的意義

主線要做 zero-shot，但實務上一定有「test 給 k 條示範」的 few-shot 對照組。MACAW 的 consistency 保證讓我們可以用同一個 meta-trained policy 連續評估 k = 0, 1, 2, … 的 trade-off 曲線；contextual 方法在 OOD task 上的 inference network 會失準，不適合做這條曲線。

### 6. 與 PEARL / VariBAD 的對照位置

- PEARL：contextual、off-policy（離線時崩）。
- VariBAD：contextual、Bayesian、原版 on-policy（Dorfman & Tamar 把它擴到 offline）。
- **MACAW：optimization-based、offline、consistent**——形成主線文獻三角的第三個頂點，主線論文寫 related work 時這三者是必比的核心。

### 一個具體可做的延伸

把 MACAW 的 enriched policy update 中的 ADV head 換成 / 並列「language-conditioned advantage head」：inner loop 同時對 (action, advantage, language goal) 三項做監督，meta-test 時只給語言，policy 仍然能 zero-shot 出動作。這幾乎是用最小改動把 MACAW 的 universality 論證套到 language-conditioned offline meta-RL 上的方案。

---

## 一句話總結

MACAW 用 bootstrap-free 的 advantage-weighted regression 同時當 inner 和 outer loop 的目標，再透過 advantage prediction head（提升 universality）與 weight transform layer（突破 rank-1 限制），首次讓 gradient-based meta-RL 能在完全離線、test 只有極少資料的設定下穩定運作，奠定了 offline meta-RL 領域的 optimization-based 骨幹。
