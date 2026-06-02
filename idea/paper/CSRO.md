---
type: paper-note
aliases:
  - "CSRO"
year: 2023
stage: "3-offline-meta-rl"
tags:
  - offline-meta-rl
  - context-shift
  - mutual-information
  - 核心警訊
  - 3-offline-meta-rl
summary: "max-min MI處理context shift；關鍵洞見：語言可當policy-invariant task anchor，潛在理論貢獻。"
---
# CSRO：Context Shift Reduction for Offline Meta-RL 逐段精讀

> Gao, Zhang, Guo 等，NeurIPS 2023。
> 主題：在 offline meta-RL（OMRL）中正面處理「context shift」問題——meta-train 時 context 來自 behavior policy、meta-test 時 context 來自 exploration policy，兩者分布不一致導致 task representation 推論錯誤。
> 解法：(1) max-min mutual information representation learning：用 CLUB 把 task representation $z$ 與 behavior policy 之間的互信息壓低、同時用 metric learning 把 $z$ 與 task 標籤的互信息拉高；(2) non-prior context collection strategy：測試初期先隨機探索若干步，再以 posterior 條件下的 meta-policy 探索，避免從 prior 採樣的 $z_0$ 把 context 拖離 behavior policy 的分布。
> Backbone 為 BRAC + [[PEARL|PEARL]] 風格 context-based meta-RL。

---

## 0. 閱讀總覽（白話）

這篇處理的是 offline meta-RL 一個非常具體、但常常被掃進地毯下的問題：

- **訓練時**：每個 task 都有一個事先收集好的 dataset，它是某個「在這個 task 上訓練到不錯」的 behavior policy 跑出來的。所以 dataset 裡只有「往目標走的軌跡」這類具有 task 特徵的軌跡。
- **測試時**：要面對新 task，agent 必須先「亂試一下」收集 context 來推 task representation $z$。這時 agent 用的是 meta-policy 自己 + 從 prior $p(z)$ sample 出來的 $z_0$，**根本不是訓練資料的那個 behavior policy**。
- **後果**：context encoder 在訓練時偷偷把 behavior policy 的「往哪走」之類的特徵當成 task 線索；測試時 context 來自完全不同的 policy，task representation 就推錯了，meta-policy 也跟著錯。

CSRO 的核心思路非常直接：「**task representation 應該只跟 task 有關，不應該跟 policy 有關**」。所以：

1. 訓練時用 CLUB（一種互信息上界估計）把 $z$ 與 $(s,a)$ 之間的互信息壓低——$(s,a)$ 是 behavior policy 的代理。同時保留 FOCAL 風格的 metric loss 讓同 task 的 $z$ 拉近、不同 task 的 $z$ 推遠。這就是 max-min。
2. 測試時不要從 prior 抽 $z_0$ 來條件 meta-policy。前 $t_r$ 步 agent 隨機亂走（與 prior 無關），再從累積的 context 算 posterior $q_\phi(z|c)$，後續才用 $\pi_\theta(\cdot|s,z)$ 探索。這就是 non-prior context collection。

實驗在 Point-Robot、Half-Cheetah-Vel、Ant-Goal、Humanoid-Dir、Hopper/Walker-Rand-Params 六個環境上贏過 FOCAL、[[CORRO|CORRO]]、OffPEARL、BOReL；在 t-SNE 上 CSRO 的 task embedding 也比 baseline 更乾淨地依 task 分群。弱點是稀疏獎勵環境（Sparse-Point-Robot）——因為非先驗的隨機探索很難碰到 reward，這時 IDAQ 那種「在訓練資料中找最接近的 context」的策略反而更好。

---

## Abstract

OMRL 利用預先收集的離線資料提升 agent 對未見任務的泛化能力。但因 meta-train context（來自 behavior policy）與 meta-test context（來自 exploration policy）的分布差異，產生 **context shift** 問題，會讓 task inference 出錯、進一步損害 meta-policy 的泛化。既有 OMRL 方法不是忽略這個問題，就是必須借助額外資訊（如所有 task 的 reward function、額外的線上資料）來緩解。本文提出 CSRO，僅靠離線資料就解決 context shift。核心是「在訓練與測試兩階段都把 policy 在 context 中的影響壓到最低」：訓練階段用 max-min mutual information representation learning 削弱 behavior policy 對 task representation 的影響；測試階段用 non-prior context collection strategy 削弱 exploration policy 的影響。實驗顯示 CSRO 顯著減少 context shift 並超越過去方法。

---

## 1. 問題背景：context shift 的正式定義

### 1.1 設定（preliminaries）

每個 task 是一個 MDP $M_i = (S, A, P_i, \rho_0, R_i, \gamma) \sim p(M)$，不同 task 共享 $S, A, \rho_0, \gamma$，但 reward function 或 transition dynamics 不同。Task 分布可寫為 $p(R, P)$。

每個 task 的 offline dataset 由「task-dependent behavior policy」$\pi_{\beta_i}(a|s)$ 收集：

$$D_i = \{(s_{i,j}, a_{i,j}, r_{i,j}, s'_{i,j})\}_{j=1}^{N_{\text{size}}}.$$

Context-based OMRL 從 $D_i$ sample 一個 mini-batch context $c = \{(s_j, a_j, r_j, s'_j)\}_{j=1}^{n}$，再用 context encoder $q_\phi$ 得到 task representation $z = q_\phi(z|c)$。Policy、value function、Q-function 都 condition 在 $z$ 上。

OMRL 的目標：

$$J(\pi_\theta) = \mathbb{E}_{M \sim p(M)} [J_M(\pi_\theta)].$$

### 1.2 Context shift 的正式定義

關鍵 observation：每筆 offline data 的分布同時被 task 和 behavior policy 共同決定：

$$p(D_i) = \rho(s_0) \prod_{j=0}^{N_{\text{size}}} \pi_{\beta_i}(a_j | s_j) \, R(s_j, a_j) \, P(s_{j+1} | s_j, a_j).$$

於是：

- **Meta-train 階段**：context $c \sim p(D_i)$，分布同時由 task 與 $\pi_{\beta_i}$ 決定。
- **Meta-test 階段**：context $c$ 是線上由 exploration policy $\pi_e$ 收集，分布同時由 task 與 $\pi_e$ 決定。

由於 $\pi_e \ne \pi_{\beta_i}$，context 在兩階段的分布不一致，這就是 **context shift**：

> Context shift 不是普通的 state/action distribution shift；它是因為「context 本身的生成 policy」在訓練與測試不同，而 context 又被用來推 task representation——所以這個 shift 直接污染 task 推論本身，不僅是 policy 估計的偏差。

### 1.3 為什麼不是普通的 distribution shift

普通 offline RL 的 distribution shift 指的是「學到的 policy 在新 state 上選的 action 落在 behavior policy 沒覆蓋的地方」，後果是 Q 值高估、policy 在 OOD 上崩。CSRO 處理的 shift 不一樣：

1. **影響對象不同**：普通 shift 影響的是 $Q$、$\pi$ 的估計準確度；context shift 影響的是 **task representation 本身的語意**。
2. **混淆來源**：因為 task encoder 從 context 中學特徵，若 behavior policy 的特徵（例如往目標方向移動）跟 task（目標位置）高度相關，encoder 會錯把 policy 特徵當作 task 特徵編進 $z$。
3. **測試端也有 shift**：常見 meta-RL exploration 從 prior $p(z)$ sample $z_0$，再用 $\pi_\theta(\cdot|s, z_0)$ 探索，這個 $z_0$ 帶有特定 policy 特徵，會把 context 推離 behavior policy 的分布——這是 exploration 端引入的 shift。
4. **解決方式不同**：BRAC/[[CQL|CQL]] 這類做「policy 的 distribution 約束」並不能解 context shift；要從「task representation 與 policy 之間的相依」下手。

論文 Table 1 用 FOCAL、OffPEARL 直接證實：把 test 時的 context 換成從 offline data 抽（context A，沒有 shift）vs. 從 meta-policy 線上收集（context B，有 shift），return 差距很大（如 Point-Robot：FOCAL 從 −4.4 掉到 −14.9；OffPEARL 從 −5.1 掉到 −17.8）。

### 1.4 與其他 OMRL 方法的差別

- **Offline test 派**（FOCAL, MBML 等）：測試時直接從 test task 的 offline data 抽 context——回避而非解決 shift；而且在實務上根本不可能有 unseen task 的 offline data。
- **Online test 派 + 額外資訊**：BOReL 假設知道所有 task 的 reward function；[[OfflineMetaRL_OnlineSelfSupervision|SMAC]]（[26]）測試前要先用大量線上資料做 self-supervision。
- **CSRO**：online test，**僅用 offline datasets**，不需要額外資訊。

---

## 2. 核心方法與公式

CSRO 的關鍵 insight：「**在 meta-train 與 meta-test 兩端都降低 policy 在 context 中的影響**」。對應兩個元件：

### 2.1 Max-min mutual information representation learning（meta-train）

從 $D_i$ sample context $c_i = \{(s_{i,j}, a_{i,j}, r_{i,j}, s'_{i,j})\}_{j=1}^{N_c}$。Context encoder 對每筆 transition 抽 transition embedding $z_{i,j} = q_\phi(z | (s_{i,j}, a_{i,j}, r_{i,j}, s'_{i,j}))$，再取平均作為 task representation：$z_i = \mathbb{E}_j [z_{i,j}]$。

**最大化** $I(z; \text{task})$：採用 FOCAL [20] 的 metric learning loss，

$$L_{\text{maxMI}}(\phi) = \mathbf{1}\{y_i = y_j\} \, \|z_i - z_j\|_2^2 \; + \; \mathbf{1}\{y_i \ne y_j\} \cdot \frac{\beta}{\|z_i - z_j\|_2^n + \epsilon},$$

其中 $y_i$ 是 task label：同 task pair 拉近，不同 task pair 推遠。直觀上這是 task identifiability 的 surrogate。

**最小化** $I(z; \pi_\beta)$：這部分是 CSRO 的關鍵。

**Proposition 1（CSRO Section 4.2 / Appendix B）**：對 task $M_i$ 與其 behavior policy $\pi_{\beta_i}$，當 exploration policy $\pi_e \ne \pi_{\beta_i}$，
$$J_{M_i}(\pi_\theta, \pi_e) = J_{M_i}(\pi_\theta, \pi_{\beta_i}) \quad \Longleftrightarrow \quad I(z; \pi_e) = 0.$$

含意：只有 $z$ 與 policy 互信息為零，meta-test 的 return 才不因 policy 改變而退化。所以需要把 $I(z; \pi_\beta)$ 壓低。

**實作問題**：每個 task 只有「一個」behavior policy 在收集資料，無法直接估計 $\pi_\beta$ 的分布表徵。論文用 $(s, a)$ pair 當作 behavior policy 的代理——因為 transition 的條件分布拆解為
$$p(a, r, s' | s) = \pi_\beta(a|s) \, p(s' | s, a) \, r(s, a),$$
其中 $\pi_\beta$ 的資訊只進到 $(s, a)$ 這個邊緣。

接著用 **CLUB**（Cheng et al. 2020，互信息「上界」估計）來最小化 $I(z; (s,a))$。CLUB 上界：

$$I_{\text{CLUB}}(z, (s,a)) = \mathbb{E}_i \big[ \log p(z_i | (s_i, a_i)) - \mathbb{E}_j [\log p(z_j | (s_i, a_i))] \big].$$

但 $p(z | (s,a))$ 不可得，故引入 variational 估計 $q_\psi(z | (s,a))$，先最小化負對數似然來逼近：

$$L_{\text{VD}}(\psi) = -\mathbb{E}_{M \sim p(M)} \, \mathbb{E}_i [\log q_\psi(z_i | (s_i, a_i))].$$

固定 $\psi$ 後，對 $\phi$ 最小化：

$$L_{\text{minMI}}(\phi) = \mathbb{E}_{M \sim p(M)} \, \mathbb{E}_i \big[ \log q_\psi(z_i | (s_i, a_i)) - \mathbb{E}_j [\log q_\psi(z_j | (s_i, a_i))] \big].$$

這形成 $\phi$ 與 $\psi$ 的對抗訓練：
- $\psi$ 想學會從 $(s,a)$ 預測 $z$，估出較緊的上界；
- $\phi$ 想讓 $z$ 對 $(s,a)$ 不可預測（任何 $(s,a)$ 對任何 $z$ 的對數機率差不多），把上界壓低，等同最小化 $I(z; \pi_\beta)$。

**總 encoder loss**：

$$L_{\text{encoder}}(\phi) = L_{\text{maxMI}}(\phi) + \lambda \cdot L_{\text{minMI}}(\phi),$$

$\lambda$ 是兩個目標的權重，論文 Appendix Table 5 中視環境給 10–50。

**Offline RL 端**：用 BRAC [32] 約束 learned policy $\pi_\theta$ 與 behavior policy $\pi_\beta$ 的 KL 散度，避免動作的 distribution shift 造成 Q 高估：

$$L_{\text{critic}}(\omega) = \mathbb{E}\big[(Q_\omega(s,a,z) - r - \gamma Q^{\text{target}}_\omega(s', a', z))^2\big],$$

$$L_{\text{actor}}(\theta) = -\mathbb{E}\big[Q_\omega(s, a'', z) - \alpha \cdot D\big(\pi_\theta(\cdot|s, z), \pi_\beta(\cdot|s, z)\big)\big].$$

### 2.2 Non-prior context collection strategy（meta-test）

問題：傳統 meta-RL 的 exploration 先從 prior $p(z)$ sample 一個 $z_0$，再用 $\pi_\theta(\cdot | s, z_0)$ 探索新 task，問題是 $z_0$ 帶有特定 policy 特徵，會把 context 推離訓練分布。即使 encoder 已被 max-min MI 處理，也無法把 policy 影響完全消掉。

訓練 exploration policy（如 MetaCURE [38]）也不可行——offline 無法線上互動，純從 offline data 訓 explorer 會被限制在資料附近、與 offline 的 conservatism 衝突（論文 Appendix F.5 實驗 CSRO+MetaCURE 反而退步）。

**CSRO 提案（Algorithm 2）**：

```
for t = 0, ..., T-1:
    if t < t_r:
        a_t ~ Uniform(A)        # 純隨機 action，不依賴任何 z
    else:
        z = q_phi(z | c)        # 用目前累積 context 算 posterior
        a_t ~ pi_theta(a | s_t, z)
    c <- c U {(s_t, a_t, r_t, s'_t)}
最後 z = q_phi(z|c)，再用 pi_theta(a|s, z) 跑 evaluation。
```

直觀：前 $t_r$ 步用 task-agnostic 的隨機 action，避免 prior $z_0$ 把 context 拖偏；隨機探索之後 agent 對 task 有粗略認識，再進入 posterior-conditioned 探索，逐步精化。

---

## 3. 演算法流程整理

**Meta-training（Algorithm 1）**：
1. 從 $D_i$ sample context $c$ 與 history transitions $h$。
2. 計算 transition embedding $z = q_\phi(z|(s,a,r,s'))$、CLUB 端 $z = q_\psi(z|(s,a))$、task representation $z = q_\phi(z|c)$。
3. 更新 $\psi$ 最小化 $L_{\text{VD}}(\psi)$（學 variational 估計）。
4. 更新 $\phi$ 最小化 $L_{\text{encoder}}(\phi) = L_{\text{maxMI}} + \lambda L_{\text{minMI}}$。
5. 用 history $h$ 計算 $L_{\text{critic}}(\omega), L_{\text{actor}}(\theta)$，以 BRAC 風格更新 actor-critic。

**Meta-testing（Algorithm 2）**：見 2.2 節虛擬碼。

**模型構造**：
- $q_\phi$：context encoder（從 $(s,a,r,s')$ 取 transition embedding，再平均成 $z$）。
- $q_\psi$：CLUB variational 估計 $z$ 給定 $(s,a)$ 的條件分布。
- $\pi_\theta(a|s,z)$, $Q_\omega(s,a,z)$：BRAC actor-critic，都 condition 在 $z$ 上。

---

## 4. 實驗與結論

### 4.1 環境

六個 benchmark：
- **Reward function 變化**：Point-Robot（2D 導航到半圓上目標）、Half-Cheetah-Vel（目標速度 ∈ [1,3]）、Ant-Goal（半徑 2 的圓上目標）、Humanoid-Dir（目標方向 ∈ [0, 2π]）。
- **Dynamics 變化**：Hopper-Rand-Params、Walker-Rand-Params（質量、慣量、阻尼、摩擦從 $[1.5^{-3}, 1.5^{3}]$ 採樣）。

每個環境 30 train tasks + 10 test tasks。對每 train task 用 SAC 訓練 + 不同訓練 step 存 checkpoint 作為 behavior policy，每個 policy roll 50 trajectories。

### 4.2 對照方法

- **OffPEARL**（[27] 改 offline）
- **FOCAL**（[20] metric distance）
- **CORRO**（[37] CVAE+InfoNCE）
- **BOReL**（[4]，使用不含 oracle reward 的變體以公平比較）

全部用同樣的 offline backbone（BRAC）和同樣的 offline data。

### 4.3 主要結果（Figure 3）

在六個環境中 CSRO 全部贏過 baseline，Point-Robot 與 Ant-Goal 領先幅度最大——這兩個的 behavior policy 方向特徵跟 task 高度耦合，context shift 最嚴重，CSRO 修正幅度最明顯。

### 4.4 Ablation（Figure 4 / Table 2）

四種組合：
- 不用 minMI、不用 Np：最差。
- 只加 minMI：明顯提升。
- 只加 Np：在大部分環境也有提升。
- 都加（完整 CSRO）：最好。

Table 2 顯示其他 baseline 雖然套用 Np 也有些改善，但仍輸給 CSRO，因為它們的 encoder 沒把 behavior policy 影響壓低。

### 4.5 視覺化（Figure 5、Figure 9）

t-SNE 上 CSRO 在 Half-Cheetah-Vel 與 Point-Robot 的 task embedding 比 FOCAL、CORRO 更乾淨地依 task 分群（速度 1→3 對應彩虹色帶）。

### 4.6 額外實驗（Appendix F）

- **Offline test**（F.1）：context 從 test task 的 offline data 抽，故沒有 shift——CSRO 仍與 baseline 持平或更好。
- **Online vs. offline gap**（F.2）：CSRO + Np 在大多數環境逼近 offline test，但 Point-Robot 與 Ant-Goal 仍有 gap（這兩個 shift 最嚴重）。
- **訓 explorer 不行**（F.5）：CSRO+MetaCURE 反而退步（offline 無法為 explorer 提供環境互動）。
- **稀疏獎勵弱點（F.6, Table 8）**：在 Sparse-Point-Robot 上 CSRO 與 FOCAL/OffPEARL 一樣差（≈0.8），IDAQ 則達 6.5。因為非先驗隨機探索很難碰到稀疏 reward，沒有 reward signal 就無法做 task 推論。IDAQ 的優勢來自「保留訓練 task 的高 reward $z$」並從中挑差異最大者，但若 test goal 半徑內沒有訓練 goal（Setting A/B/C），IDAQ 也崩。

### 4.7 結論

CSRO 形式化了 OMRL 中的 context shift，並從「壓低 policy 在 context 中的影響」這一條主軸給出 train+test 兩端的解：max-min MI representation learning + non-prior context collection。整體在 reward / dynamics 變化環境上都優於既有 OMRL 方法，且不需要額外資訊（reward function、線上資料）。

---

## 與本研究主線的關聯

本研究主線：**robot play 收集大量無標籤行為資料 + 極少的弱語言描述 + offline meta-RL + 用文字當 task spec / task embedding 做 zero-shot**，並關注 meta-learning × zero-shot 並行方向。CSRO 是這條線最該反覆讀的論文之一，因為它把 OMRL 中「task representation 與 policy 糾纏」的問題講得最清楚。對應到本研究的啟示：

### 5.1 Context shift 在 robot play 場景幾乎注定更嚴重

CSRO 假設「每個 task 都由一個 RL-trained behavior policy 收集資料」，那種 policy 通常很有 task-bias（往目標走）。Robot play 場景看似不同——資料是「玩耍」式的、不一定指向特定 task——但其實 shift 來源更複雜：

- Play 資料是 **task-agnostic 但 skill-biased**：每段 play 軌跡通常聚焦在某種互動（抓握、推、堆疊），這個 skill bias 本身就是 policy 特徵。
- Test 時 agent 要做新 task，exploration 又是另一種 policy。若把 play 軌跡當 context，context encoder 可能會把「play 期間 agent 偏好的互動模式」當成 task 的一部分。

因此 CSRO 的「壓低 $I(z; \pi_\beta)$」這個目標在 robot play 場景同樣有效，但 $(s, a)$ pair 作為 policy 代理在高維機器人觀察下可能不夠——也許需要更結構化的 policy 代理（例如 skill primitives 或 trajectory 片段的 embedding）。

### 5.2 弱語言 supervision 能不能幫忙降低 context shift？答案是「能、但角度不同」

CSRO 的核心是把 task representation 變得只與 task 有關、不與 policy 有關。語言 supervision 在這個結構裡能扮演幾個角色：

**(a) 提供 task identity 的直接 signal（取代 FOCAL 風格的 metric loss）。**
CSRO 的 $L_{\text{maxMI}}$ 依賴 task label $y_i$ 來建構同 task / 不同 task 的 metric loss。Robot play 的設定裡，task label 通常不存在；但弱語言描述（即使只是稀疏標註幾段軌跡）可以直接當作 task identity 的 supervision——例如要求 text encoder 出來的 $\phi_{\text{lang}}(\ell_i)$ 與 context encoder 出來的 $z_i$ 對齊（cosine similarity / contrastive）。這比 task label 更語意豐富，也支援 zero-shot：新 task 來時只要給文字描述就有 $z$。

**(b) 提供 policy-invariant 的 anchor。**
Context shift 之所以發生，是因為 $z$ 沒有「外部錨點」告訴它什麼算 task。語言描述本質上是「task spec」而非「behavior spec」——人類描述 task 時通常說目標（"reach the red block"）而非 behavior（"move slowly in y-direction first"）。所以把 $z$ 推向與語言 embedding 對齊，等於同時在做：
- max $I(z; \text{task})$：透過語言 embedding 表達 task。
- min $I(z; \text{policy})$：因為語言錨點與 policy 無關，靠近語言就遠離 policy。

某種意義上，**語言可以同時當 max-min MI 的兩個項的 surrogate**——不再需要顯式的 CLUB 對抗訓練。

**(c) Non-prior context collection 的語言版本：language-conditioned exploration。**
CSRO 的 Np 策略是「前 $t_r$ 步隨機」，這在稀疏 reward 下會崩（Sparse-Point-Robot）。若有弱語言 supervision，可以用語言 embedding 作為 $z_0$ 來條件 exploration，而不是從 prior 抽——這時 $z_0$ 不再帶 policy bias，而是帶 task bias，理論上比隨機探索高效，又能避免 prior 的 policy 污染。這正是「meta-learning × zero-shot」的點：訓練時把 $z$ 對齊文字，測試時用文字直接生成 $z$，跳過 exploration 收集 context 的步驟，或至少當作極強的 prior。

### 5.3 風險：語言 supervision 也可能引入新的 shift

要注意：訓練時若語言描述是「人類事後標註玩耍片段」，這些描述可能會偏向描述 policy（"the arm pushes from left"）而非 task。這樣對齊出來的 $z$ 反而會把 policy 特徵編進去——與 CSRO 想避免的情況一樣。可行的緩解：
- 標註指南強制描述 outcome / goal 而非 behavior。
- 多人標註取共識，policy-specific 的形容詞會被平均掉。
- 對齊 loss 加上 CSRO 風格的 $L_{\text{minMI}}$ 作為輔助，確保即使語言含 policy 訊息也被抑制。

### 5.4 跟 Np 在稀疏環境弱點的對接

CSRO 在 Sparse-Point-Robot 失敗的原因是「沒有 reward 就沒辦法推 task」。在本研究的 robot play 設定下，這個問題會更嚴重——稀疏成功訊號是常態。若有語言 supervision，可以用 language → $z$ 的 mapping 跳過「靠 reward 推 task」這條依賴，這正是 zero-shot task spec 的優勢。換句話說，**語言 supervision 能補上 CSRO 在稀疏環境的最大短板**。

### 5.5 與 PEARL/VariBAD 對照下的位置

- PEARL/[[VariBAD|VariBAD]] 強調 task representation 的 Bayesian / posterior 結構，但沒在意 context 來源的 policy 影響。
- FOCAL 用 metric loss 補上 task identifiability，但同樣假設 context 的 policy 是訓練端的 behavior policy。
- CSRO 是第一個把「policy 在 context 中的影響」從 representation 中明確扣除的工作。
- 本研究若引入語言 supervision，可以視為「用語言 anchor 取代 / 增強 CSRO 的 max-min MI」，並在 meta-test 階段提供 zero-shot 的 task spec 路徑。

---

## 一句話總結

CSRO 把 OMRL 的 context shift 形式化為「task representation 與 behavior/exploration policy 之間的互信息」並用 CLUB 對抗訓練壓低、再用 non-prior 隨機探索切斷 prior $z_0$ 對 context 的污染——這套「讓 $z$ 與 policy 解耦」的框架直接指向：在本研究的 robot play + 弱語言 + offline meta-RL 設定中，語言描述可以同時充當 task identity 的 anchor 與 policy-invariant 的監督訊號，既取代 max-min MI 的兩個項、又能在稀疏獎勵下提供 zero-shot task spec，補上 CSRO 在 Sparse 環境的弱點。
