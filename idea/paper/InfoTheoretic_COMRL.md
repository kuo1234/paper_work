---
type: paper-note
aliases:
  - "Info-Theoretic COMRL"
  - "UNICORN"
  - "Information Theoretic Framework"
year: 2024
stage: "3-offline-meta-rl"
tags:
  - offline-meta-rl
  - information-theory
  - unified-framework
  - 可借用
  - 3-offline-meta-rl
summary: "證明FOCAL/CORRO/CSRO都在優化I(Z;M)不同界；可設計弱語言半監督objective（I(Z;L)≤I(Z;M)）。"
---
# Towards an Information Theoretic Framework of Context-Based Offline Meta-Reinforcement Learning (UNICORN)

> Li, Zhang, Zhang, Zhu, Yu, Zhao, Heng — NeurIPS 2024
> 用資訊理論視角統一 FOCAL / [[CORRO|CORRO]] / [[CSRO|CSRO]] 等 context-based offline meta-RL (COMRL) 方法，證明它們其實都在優化同一個 $I(Z;M)$ 的不同上下界，並提出 UNICORN-SUP（監督式）與 UNICORN-SS（自監督式）兩個新實作。

---

## 0. 閱讀總覽（白話版）

這篇論文要解決一個「混亂的局面」：COMRL 領域近三年（2021–2024）冒出 FOCAL、CORRO、CSRO 等方法，每家都有自己一套 loss，看起來各做各的；FOCAL 用度量學習（push/pull embeddings）、CORRO 用 InfoNCE 對比、CSRO 又把兩者加起來再多塞一個 CLUB 上界。作者的主張是：**這些方法本質上都在做「task representation learning」，而 task representation learning 的最佳目標就是最大化 $I(Z; M)$——context latent $Z$ 與 task variable $M$ 的互資訊**。差別只是大家用了不同的「上界 / 下界 / 線性組合」當作近似。

論文的三個關鍵貢獻：

1. **統一定理 (Theorem 2.3)**：證明 FOCAL 等價於最大化「上界」$I(Z;X)$、CORRO 等價於最大化「下界」$I(Z;X_t|X_b)$、CSRO 則是上下界的線性插值。這個視角讓「方法演進的故事」變成了「對 $I(Z;M)$ 的逼近愈來愈準」的工程史。
2. **解釋 context shift 為什麼會發生**：FOCAL 把 spurious 的 behavior-policy 訊息也吸進 $Z$ 裡（lesser causality $I(Z;X_b)$），所以一旦測試時的行為策略換掉，$Z$ 就失靈。CORRO/CSRO 透過 condition on $(s,a)$ 或上界扣掉 $X_b$，所以比較穩。
3. **新算法**：因為現在有了統一目標，就可以自由選擇怎麼逼近——作者提出 UNICORN-SUP（用 task label 做交叉熵分類）與 UNICORN-SS（FOCAL 對比 + 重建 $X_t$ 的 generative）。實驗顯示在 IID、OOD 行為策略、不同資料品質、不同 backbone（MLP / [[DecisionTransformer|Decision Transformer]]）下都是 SOTA。

白話總結：**「context encoder 該保留什麼？」答案是「對 task identity 有用的、跟 behavior policy 無關的資訊」**——也就是 $I(Z;X_t|X_b)$ 那一部分強因果訊息，盡量壓低 $I(Z;X_b)$ 弱因果訊息。

---

## 1. Abstract

論文把 COMRL 視為 offline RL × meta-RL 的交集，主張既存的方法（FOCAL、CORRO、CSRO 等）雖然 loss 形式各異，**實質上都在優化同一個互資訊目標 $I(Z;M)$ 的不同近似上下界**。基於此理論統一，作者提出 supervised 與 self-supervised 兩個實作（合稱 UNICORN），在 MuJoCo / Meta-World 等 RL benchmark 上、跨資料品質與架構（MLP、DT）展現強大的 in-distribution 與 OOD 泛化能力，並將之定位為未來「決策基礎模型」的 offline pre-training 範式。

---

## 2. 問題背景：context-based offline meta-RL 的統一視角

### 2.1 為什麼需要 OMRL

- Classical meta-RL（[[MAML|MAML]]、[[PEARL|PEARL]] 等）需要大量線上探索，在醫療、自駕、機器人等安全敏感場景無法接受。
- Offline RL 允許僅從 logged data 學習，OMRL = offline + meta 把「快速適應」與「不能線上探索」這兩個需求合在一起。
- 與大模型「multi-task pre-training + 微調」的精神類似，作者明白定位 OMRL 為 RL foundation model 的前身。

### 2.2 COMRL 的核心：學一個好的 task representation

每個 task $M^i$ 是一個 MDP，offline dataset $X^i = \{(s,a,r,s')\}$ 由 behavior policy $\pi_\beta^i$ 收集；給定一段 context $c^i_{1:n}$，context encoder $q_\phi(z\mid c^i_{1:n})$ 產生 task latent $z^i$。若 $z$ 對 task identity 是 sufficient statistic，下游 policy $\pi_\theta(a\mid s, z)$、value $V_\pi(s, z)$ 退化為一般 RL，可用 Bellman update 訓練。

### 2.3 既存方法的 loss 一覽

- **FOCAL**：度量學習，把同 task 的 $z$ 拉近、不同 task 的推遠：

  $$
  \mathcal{L}_{\text{FOCAL}} = \min_\phi \mathbb{E}_{i,j}\Bigl[\mathbf{1}\{i=j\}\|z^i-z^j\|_2^2 + \mathbf{1}\{i\neq j\}\frac{\beta}{\|z^i-z^j\|_2^n + \epsilon}\Bigr]
  $$
- **CORRO**：InfoNCE 對比下界 $I(Z;M)$，正負樣本條件在相同 $(s,a)$ 上：

  $$
  \mathcal{L}_{\text{CORRO}} = \min_\phi \mathbb{E}_{x,z}\Bigl[-\log \tfrac{h(x,z)}{\sum_{M^*\in\mathcal{M}} h(x^*, z)}\Bigr], \quad h(x,z)=\tfrac{p(z\mid x)}{p(z)}
  $$
- **CSRO**：FOCAL + 一個 CLUB 互資訊上界把 $Z$ 與 $(s,a)$ 解耦：

  $$
  \mathcal{L}_{\text{CSRO}} = \min_\phi \{\mathcal{L}_{\text{FOCAL}} + \lambda\, \mathcal{L}_{\text{CLUB}}\}
  $$

  其中 $\mathcal{L}_{\text{CLUB}} = \mathbb{E}_i[\log q_\phi(z_i\mid s_i,a_i)] - \mathbb{E}_i\mathbb{E}_j[\log q_\phi(z_j\mid s_i,a_i)]$。

### 2.4 Context shift 問題

FOCAL 對 context shift 脆弱：當測試時 context 由 OOD behavior policy 收集，$Z$ 中本來吸收了 behavior policy 的 spurious 訊號（捷徑學習 shortcut learning），就無法泛化。這也是後續 CORRO/CSRO 改進的動機。

---

## 3. 核心理論與公式

### 3.1 因果分解（Definition 2.2, Causal Decomposition）

把 context $X$ 拆成兩部分：

- $X_b = (s, a)$：**behavior-related**，主要由 $\pi_\beta$ 決定，與 $M$ 是「弱因果」（除非 task 之間 $\rho_0$ 或 $T$ 不同，否則 $s,a$ 不直接反映 task identity）。
- $X_t = (s', r)$：**task-related**，給定 $(s,a)$ 後完全由 $T(s'\mid s,a)$ 與 $R(s,a)$ 決定，因此與 $M$ 強因果。

由 $M\to X \to Z$ 的 Markov chain 得 $I(Z; M\mid X)=0$。

對 $I(Z;X)$ 套用 chain rule：

$$
I(Z;X) = I(Z;X_t\mid X_b) + I(Z;X_b)
$$

作者命名：
- $I(Z;X_t\mid X_b)$ = **primary causality**（保留 task-related 強因果訊號）
- $I(Z;X_b)$ = **lesser causality**（與 behavior policy 相關的弱因果/spurious 訊號）

### 3.2 統一定理 (Theorem 2.3 — 本文核心)

$$
\underbrace{I(Z;X_t\mid X_b)}_{\text{primary}} \;\le\; I(Z;M) \;\le\; \underbrace{I(Z;X_t\mid X_b) + I(Z;X_b)}_{\text{primary + lesser}} = I(Z;X)
$$

並且：
1. $\mathcal{L}_{\text{FOCAL}} \equiv -I(Z;X)$（操作上界）
2. $\mathcal{L}_{\text{CORRO}} \equiv -I(Z;X_t\mid X_b)$（操作下界）
3. $\mathcal{L}_{\text{CSRO}} \ge -\bigl((1-\lambda) I(Z;X) + \lambda I(Z;X_t\mid X_b)\bigr)$（兩個界的凸組合）

**直觀解讀**：
- 想要最大化 $I(Z;M)$ 時，**最大化下界（CORRO）有保證**，因為提高下界一定不會比 $I(Z;M)$ 高估太多。
- **最大化上界（FOCAL）沒有保證**：上界提升可能來自把 $I(Z;X_b)$ 拉高，等於把 spurious 訊號吸進去。
- CSRO 的線性插值 $\lambda\in[0,1]$ 等於在「保留多少 $I(Z;X_b)$」與「保留多少 $I(Z;X_t\mid X_b)$」之間取 trade-off：當下游 task 之間 $\rho_0, T$ 確實不同（$X_b$ 也帶 task 訊號），上界稍微保留是合理的；反之就該往下界靠。

引理 B.1 (Lemma)：$I(Z; M) \ge I(Z; M\mid X_b)$，由 $I(Z;M;X_b)\ge 0$ 加上 $I(Z;M\mid X_t, X_b)=0$ 推得，是上述推導的關鍵中介。

### 3.3 上下界證明的關鍵步驟（Appendix B 摘要）

下界（CORRO）的關鍵推導鏈：

$$
\begin{aligned}
I(Z;M) &\ge I(Z;M\mid X_b) \\
&= I(M; Z, X_t\mid X_b) - I(M; X_t\mid Z, X_b) \\
&= I(M; X_t\mid X_b) - H(X_t\mid Z, X_b) + H(X_t\mid M, Z, X_b) \\
&\ge I(M; X_t\mid X_b) - H(X_t) + I(X_t; Z, X_b)\\
&\equiv I(X_t; Z, X_b) \equiv I(Z; X_t\mid X_b)
\end{aligned}
$$

最後一行用了 offline 設定下 $I(X_t; X_b)$ 為常數的事實。重要訊息：**在 offline RL，$I(Z; X_t\mid X_b) \equiv I(X_t; Z, X_b)$（差一個常數）**，因此 contrastive 的下界 (CORRO) 與 generative reconstruction 的目標其實等價——這正是後面 UNICORN-SS 把 FOCAL contrastive 與 $X_t$ reconstruction 混合的依據。

### 3.4 監督式 UNICORN-SUP 目標

由 $I(Z;M) = H(M) - H(M\mid Z) \equiv -H(M\mid Z)$，再用 task label 已知這件事，把 $p(M\mid z)$ 視為對 $z^i$ 指派正確 $M^i$ 的分類器。導出：

$$
\mathcal{L}_{\text{UNICORN-SUP}} = -\mathbb{E}_{x,z\sim q_\phi(z\mid x)}\Bigl[\sum_{j=1}^{n_M} \mathbf{1}(M^j = M) \log p_\theta(M^j\mid z)\Bigr]
$$

這就是 $n_M$-way classification 的負交叉熵，**直接優化 $I(Z;M)$ 本身**（不是上下界），但代價是需要 task label。

**Theorem 2.4 — Concentration bound**：

$$
\Bigl|\hat{I}(Z;M) - \bar{I}(Z;M)\Bigr| \le \sqrt{\frac{\mathrm{Var}(H(Z\mid M))}{n_M \delta}}
$$

以機率至少 $1-\delta$ 成立。這告訴我們：**訓練 task 數量 $n_M$ 太少時，SUP 的估計誤差會大**——也是論文後面解釋為何 UNICORN-SUP 在小規模實驗中遜於 UNICORN-SS 的原因。實作中 $n_M = 20$。

### 3.5 自監督式 UNICORN-SS 目標

當 task label 缺失時，用上下界凸組合來逼近：

$$
I(Z;M) \approx \alpha I(Z;X) + (1-\alpha) I(Z; X_t\mid X_b), \quad \alpha\in[0,1]
$$

實作上：
- $I(Z;X)$ → 用 FOCAL contrastive loss $\mathcal{L}_{\text{FOCAL}}$ 估計。
- $I(Z; X_t\mid X_b) \equiv I(X_t; Z, X_b)$（offline 下等價，由 §3.3）→ 用 reconstruction：

  $$
  I(X_t; Z, X_b) \ge \mathbb{E}\bigl[\log p_\theta(x_t\mid z, x_b)\bigr]
  $$

  即用 decoder $p_\theta(x_t\mid z, x_b)$ 重建 $X_t = (s', r)$，得 $\mathcal{L}_{\text{recon}} := -I(X_t; Z, X_b)$。

最終 loss：

$$
\mathcal{L}_{\text{UNICORN-SS}} = \mathcal{L}_{\text{recon}} + \tfrac{\alpha}{1-\alpha}\, \mathcal{L}_{\text{FOCAL}}
$$

實務中 $\alpha/(1-\alpha) \in \{0.15, 1.5, 0.3\}$ 視環境而定（Table 7）。

### 3.6 統一表 (Table 1)

| 方法 | Representation 目標 | 實作 |
|---|---|---|
| UNICORN-SUP | $I(Z;M)$ | Predictive (分類) |
| UNICORN-SS | $\alpha I(Z;X) + (1-\alpha) I(X_t; Z, X_b)$ | Contrastive + Generative |
| FOCAL | $I(Z;X)$ | Contrastive |
| CORRO | $I(Z; X_t\mid X_b)$ | Contrastive |
| CSRO | $(1-\lambda) I(Z;X) + \lambda I(Z; X_t\mid X_b)$ | Contrastive |
| GENTLE | $I(X_t; Z, X_b)$ | Generative |
| BOReL | $I(X_t; Z, X_b) - D_{KL}(q_\phi(Z\mid X)\|p_\theta(Z))$ | Generative |
| VariBAD | 同 BOReL（online） | Generative |
| PEARL | $-D_{KL}(q_\phi(Z\mid X)\|p_\theta(Z))$（online） | N/A |
| ContraBAR | $I(Z; X_t\mid A)$ | Contrastive |

**關鍵領悟**：「所有 COMRL 方法本質上都是 representation learning，只是在優化 $I(Z;M)$ 的不同近似」——這就是 read.md 想看的答案。

[[VariBAD|VariBAD]]/PEARL 的 KL 項可解為 **information bottleneck** 的變分近似，限制 $I(Z;X)$；在 offline 設定下，作者實驗（Table 6）顯示 KL constraint 反而傷害性能，因此 UNICORN-SS 不採用。

---

## 4. 演算法與實作

### 4.1 Meta-training (Algorithm 1)

對每個 iter：
1. 取 task batch $\{M^j\}$ 與對應 replay buffer。
2. **Encoder 更新**：sample context $c^j$ → 得 $z^j \sim q_\phi(z\mid c^j)$ → 估 $\mathcal{L}_{\text{UNICORN}}$ → 更新 $\phi, \theta$（decoder 或 classifier）。
3. **Actor-Critic 更新**：detach $z$ → 用 BRAC (behavior-regularized actor-critic) 防止 bootstrapping error → 更新 $\omega, \psi$。

明確採用 **representation 與 policy decoupled 的 two-phase 訓練**，這是 FOCAL 系列的傳統。

### 4.2 Meta-testing (Algorithm 2)

對每個 test task：採樣 context $c^i$ → 得 $z^i \sim q_\phi(z\mid c^i)$ → 用 $\pi_\omega(a\mid s, z^i)$ rollout。少樣本（few-shot）下只用 1 條 trajectory 推 $z$。

### 4.3 架構選擇

- 預設 MLP encoder / decoder。
- §4.1 試驗 Decision Transformer 變體（UNICORN-SS-DT / UNICORN-SUP-DT / FOCAL-DT）：把 $z$ 當作序列第一個 token 餵 DT，比 Prompt-DT（非可訓 prompt）顯著好。

### 4.4 Hyper-parameters（Table 7 摘要）

- task representation dim 5–40（看環境）
- $\alpha/(1-\alpha)$：0.15、0.3、1.5
- 訓練 20 task、 200K steps、 task batch 16、 RL batch 256、 lr 3e-4
- context = 1 trajectory（200 或 500 steps）

---

## 5. 實驗與結論

### 5.1 三個核心 RQ

1. **IID 泛化**：在 6 個 benchmark（HalfCheetah-Dir/Vel、Ant-Dir、Hopper/Walker-Param、Reach），UNICORN-SS 全勝或並列最佳，UNICORN-SUP 緊隨；CSRO 因為也是上下界線性組合，是最強的 baseline。
2. **OOD behavior policy 泛化** (Table 2)：把測試 context 改成由其他任務的 behavior policy checkpoints 收集；UNICORN 系列下降幅度最小，FOCAL 大幅退步。例如 HalfCheetah-Dir：FOCAL OOD 從 1186 → 861，CSRO 從 1180 → 458（大崩），UNICORN-SS 從 1307 → 1296（幾乎不掉）。
3. **資料品質敏感度** (Table 3)：在 Ant-Dir 用 random / medium / expert 三種資料；UNICORN 在 narrow distribution (medium, expert) 大勝 CSRO；CORRO 在所有資料品質下都崩，因為 negative sample generator 失效。

### 5.2 Model-agnostic (§4.1, Table 4)

把 UNICORN 思想移植到 DT backbone：UNICORN-SS-DT 與 UNICORN-SUP-DT 顯著優於 Prompt-DT 與 FOCAL-DT。作者主張這代表 UNICORN 是 representation-level 的方法、跟 backbone 無關，可作為 RL foundation model 的 pre-training paradigm。

### 5.3 Model-based 延伸 (§4.2, Figure 5)

利用 decoder $p_\theta(x_t\mid z, x_b)$ 當 world model，對 $z$ 加高斯噪音 → 生 imaginary $(s', r)$ → 訓練 RL agent。在 Ant-Dir task-level OOD（goal direction 訓練/測試完全不重疊）只有 UNICORN-SS + model-based 達到正報酬，其他方法全部負分。

### 5.4 額外驗證 (Appendix C)

- **C.1 UNICORN-SS-0**（$\alpha=0$，只用 reconstruction）≈ GENTLE，與 BOReL 相當；確認 reconstruction 路線可行。
- **C.2 t-SNE 視覺化**：UNICORN-SS embedding 對 OOD context 比 UNICORN-SS-0 更可分；但 FOCAL 雖然 cluster 更乾淨，下游性能更差——作者解釋 FOCAL 過度分離反而破壞了相似 task 間的 shared structure。
- **C.4 Ablation on $\alpha/(1-\alpha)$**：性能對此超參相對穩健，過高（趨近純 FOCAL）會下滑，驗證理論。

### 5.5 限制

1. 假設了 representation learning 與 policy optimization 解耦，因此無法量化「representation 好多少 → 下游 RL 提升多少」。
2. 實驗規模有限（至多 40 task、450K transitions），這也讓 UNICORN-SUP 不及 UNICORN-SS（$n_M$ 太小，Theorem 2.4 的估計誤差項偏大）。
3. 推導重度依賴 $M, X$ 的靜態（offline）假設，不能直接搬到 online。

### 5.6 結論

論文用「一個目標 + 上下界家族」把 COMRL 的 representation learning 收斂成統一語言，並指出未來方向：(1) 找更緊的 $I(Z;M)$ 上下界、(2) 探索更多 $I(Z;X)$ 與 $I(X_t; Z, X_b)$ 的實作組合、(3) DT backbone 的 scaling、(4) 擴展到 online setting。

---

## 與本研究主線的關聯

使用者主線：**robot play 收集無 reward 的軌跡 + 極少弱語言標註 + offline meta-RL + 文字 → task spec/embedding 做 zero-shot 任務**，並關心 meta-learning 與 zero-shot 的交集。

這篇論文對研究主線的價值極大、值得深讀，原因如下：

### 1. 替「弱語言 supervision」提供精確的數學定位

主線想做的是：用稀缺的語言標註 $L$（task description / instruction）來引導 task encoder。將語言視為一種「partial / weak label of $M$」，那就有：

- **完整 supervision**（UNICORN-SUP 的情境）：直接最大化 $I(Z; M)$，相當於 $-H(M\mid Z)$，用 cross-entropy 分類器實現。
- **弱 supervision**：把 $M$ 替換成可觀測代理 $L$（語言描述），目標變為 $I(Z; L)$。若 $L$ 與 $M$ 之間有 $L \to M$ 或 $M \to L$ 的因果鏈，則 $I(Z; L) \le I(Z; M)$（DPI），所以最大化 $I(Z; L)$ 是 $I(Z;M)$ 的**有原則的下界代理**。
- 弱語言可以視為「對 $M$ 的稀疏觀測」——也就是說，弱語言 supervision 可以被自然納入 UNICORN 框架的「分類器/迴歸器」項，與 self-supervised 的 FOCAL + reconstruction loss 並肩存在。

具體可能的新弱語言 supervision objective（直接由 Theorem 2.3 推導）：

$$
\mathcal{L}_{\text{weak-lang}} = -\mathbb{E}_{(x,l)\sim \text{labeled}}[\log p_\theta(l \mid z)] \;\;+\;\; \tfrac{\alpha}{1-\alpha}\mathcal{L}_{\text{FOCAL}} \;\;+\;\; \mathcal{L}_{\text{recon}}
$$

其中第一項只在「有語言標註的子集」上計算，後兩項在全資料上計算（self-supervised 半監督混合）。

### 2. zero-shot 與 meta-learning 的橋

主線希望「給一句新指令 → 直接生成 task embedding $z_L$ → policy $\pi(a\mid s, z_L)$ 零樣本執行」。在 UNICORN 框架下這正好是：

- 訓練時：學一個 language encoder $g_\psi(L)$，使其落在與 trajectory encoder $q_\phi(z\mid c)$ 同一空間。Loss 可以是 $\|g_\psi(L) - q_\phi(z\mid c)\|^2$ 或 InfoNCE 對齊，這在資訊理論意義上是在最大化 $I(g_\psi(L); q_\phi(z\mid c))$，等價於拉緊 $L$ 與 $Z$ 的互資訊。
- 測試時：丟掉 trajectory encoder，只用 $z_L = g_\psi(L)$ 餵給 policy → zero-shot。

關鍵洞察：**作者已經證明 $Z$ 的「最佳目標」是 $I(Z; M)$**。語言是 $M$ 的弱代理，所以「對齊 $z_L$ 到 trajectory $z_c$」 = 「讓 $z_L$ 繼承 $q_\phi$ 已經保留的 task-causal 訊號」。

### 3. 對抗 context shift 對機器人非常關鍵

機器人 play data 必然有極強的 behavior-policy diversity（不同 motion primitives、不同 demonstrator）。Theorem 2.3 直接告訴主線：**只用 FOCAL/InfoNCE 對軌跡分群是危險的**，因為 $z$ 會吸收 $(s,a)$ 的 spurious 訊號。應該優先優化 $I(Z; X_t\mid X_b)$（reconstruction 或 conditional contrastive），讓 $z$ 鎖在 reward / dynamics 訊號上。對 play data（reward 通常缺乏），**$X_t$ 應該重新定義為「dynamics next state $s'$」單獨重建**，避開 reward 缺失問題。

### 4. 對「meta-learning × zero-shot」的策略建議

- 主線可以採用 **UNICORN-SS 為骨幹**，加入 **語言對齊頭** 作為第三個 loss，形成「contrastive + reconstruction + language alignment」三項。
- 語言對齊頭可視為 UNICORN-SUP 的「弱版本」：原 UNICORN-SUP 是 hard label $M^i$（cross-entropy），弱語言版本是 soft / sparse 文字訊號（contrastive 對齊或 captioning loss）。
- 框架的好處：你可以分開 ablate 每一項的貢獻，並用 Theorem 2.3 的下界保證來說服 reviewer 「這個語言 loss 為什麼合理」。

### 5. 潛在新貢獻方向

基於這個框架，主線研究可以提出：

- **Theorem (擬議)**：當語言 $L$ 是 $M$ 的 noisy observation 時，$I(Z;L)$ 與 $I(Z;M)$ 的 gap 可由 $I(M; L\mid Z)$ 控制；在弱 supervision 下，可加入一個「context augmentation reconstruction」項作為輔助下界。
- **算法**：把 Algorithm 1 改寫成「半監督版」——大部分 task 沒有語言，少部分有；訓練 encoder 時，有語言時用語言對齊損失，無語言時退化成純 UNICORN-SS。
- **實驗**：在 Meta-World 加上 task description 標註，比較 (a) UNICORN-SS、(b) UNICORN-SS + 語言對齊、(c) 純 language→task spec 的 zero-shot 性能差異。

---

## 一句話總結

UNICORN 證明 COMRL 各路方法都在用不同的上下界逼近同一個 $I(Z;M)$，並由 $I(Z;X)=I(Z;X_t\mid X_b)+I(Z;X_b)$ 的因果分解清楚告訴你：好的 task representation 應該抓 task-causal 訊號、丟 behavior-policy 訊號——這也精準地為主線「弱語言 + offline meta-RL + zero-shot」提供了設計新 supervision objective 的數學骨架。
