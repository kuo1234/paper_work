---
type: paper-note
aliases:
  - "CORRO"
year: 2022
stage: "3-offline-meta-rl"
tags:
  - offline-meta-rl
  - contrastive
  - robust-representation
  - 可借用
  - 3-offline-meta-rl
summary: "transition-level contrastive去除behavior policy對task latent的汙染；與語言supervision互補。"
---
# CORRO：用對比學習打造對 behavior policy 穩健的 offline meta-RL 任務表徵

> Yuan & Lu, "Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning", ICML 2022。
> 主軸：在「完全離線 + 多任務」設定下，offline 資料的分布同時被 task 與 behavior policy 決定，導致 context encoder 學到的 task representation 容易把行為策略的特徵也吃進來；CORRO 用 transition 級別的 contrastive learning，加上 generative modeling 與 reward randomization 兩種 negative pair 生成方式，把 behavior policy 的影響從 task embedding 中剝離。

---

## 0. 閱讀總覽（白話）

想像一張地圖上的 2D 導航環境：每個 task 的差異只是「目標點在哪」。每個 task 都用「直接走向該目標」的策略蒐集了一批離線資料。如果我把這些資料丟給 context encoder，它要怎麼分辨「這條軌跡屬於哪個任務」？最快的捷徑就是看「這隻 agent 走的方向」——也就是行為策略的特徵——而不是去看 reward 或 transition 的本質。一旦測試時換成隨機的探索策略，這種「依賴行為策略長相」的編碼器就會崩潰。

CORRO 的觀察是：要學「真正屬於 task 的表徵」，就要強迫編碼器只能從「reward 與下一個 state 對同一個 (s, a) 的反應差異」分辨任務。具體做法是：

1. 把編碼器拆兩層：先對每個一步 transition `(s, a, r, s')` 編碼，再聚合成 task embedding。
2. 對單一 transition 做 contrastive learning：正樣本是同 task 的另一個 transition；負樣本是「**鎖定同一個 (s, a)、但 reward 與 s' 來自另一個 task** 」的反事實 transition。
3. 因為完全離線，這種反事實 transition 不存在，論文提出兩種生成方式：用 CVAE 學 union 分布生成、或直接對 reward 加噪聲（reward randomization）。
4. 編碼器只能靠 reward / transition 的本質差異區分 task，自然就把 behavior policy 的訊息排掉。

最終效果：在 IID 測試上跟既有 baseline 差不多，但在 OOD（換成沒看過的探索策略蒐 context）時大勝 FOCAL、Offline [[PEARL|PEARL]]，甚至贏過給了 ground-truth task label 的 supervised baseline（在 Ant-Dir、Walker/Hopper-Param）。

---

## Abstract（重點濃縮）

- 問題：OMRL 的 offline 資料分布由 behavior policy 與 task 共同決定，現有 context-based 方法無法分離兩者，task representation 對 behavior policy 變化不穩健。
- 方法：bi-level encoder + mutual information maximization 的目標 → 推導出 InfoNCE 形式的 contrastive 目標；針對 negative pair 提出 CVAE 生成與 reward randomization 兩種策略來逼近真實負樣本分布。
- 結果：在 Point-Robot、Ant-Dir、Half-Cheetah-Vel、Walker-Param、Hopper-Param 上，特別是 OOD behavior policy 設定中大幅領先 Offline PEARL 與 FOCAL。

---

## 1. 問題背景：為什麼 task representation 會被 behavior policy 汙染

OMRL 的設定：給定 $N$ 個訓練 task $\{M_i\}_{i=1}^N$，每個 task $M_i=(\mathcal{S},\mathcal{A},T_i,\rho,R_i,\gamma)$ 共享 $\mathcal{S},\mathcal{A}$，僅 $R, T$ 不同。每個 task 由 **某個** behavior policy $\pi_{\beta_i}$ 蒐集離線資料 $X_i=\{(s,a,r,s')\}$。訓練時只能看 $\{X_i\}$，不能與環境互動；測試時拿到未見過的 task，由「任意」探索策略收集 context，再做 task adaptation。

**核心難題**：trajectory 的分布
$$P(\tau\mid M_i,\pi_{\beta_i}) = \rho(s_0)\prod_t \pi_{\beta_i}(a_t\mid s_t)\,T_i(s_{t+1}\mid s_t,a_t)$$
同時被 $M_i$ 與 $\pi_{\beta_i}$ 決定。若訓練資料中 $\pi_{\beta_i}$ 與 $M_i$ 高度相關（例如每個 task 的 behavior policy 都是該 task 的近最優解），encoder 直接記住「行為策略的指紋」就能分辨 task，根本不必看 reward 與 transition。一旦測試 context 換成隨機策略或別的 task 的策略，encoder 立刻失效。

論文舉例：在 Point-Robot 中，每個 task 是不同的目標位置，behavior 都是「直奔目標」，因此 (s, a) 分布本身已透露目標方向；FOCAL 之類用整條 trajectory 做 metric learning 的方法只需看 state-action 分布就能區分 task，完全忽略 reward——這正是 behavior policy 汙染的具體形態。

對比現有處理方式：
- Dorfman et al. 2020：要求同一策略跑不同 task 蒐資料、且需要 reward function 可呼叫 → 不適用完全離線。
- Li et al. 2020：學各 task 的 reward model 來 relabel trajectory → 無法處理 transition dynamics 不同的 task，且小資料下 reward model 不準。
- FOCAL（Li 2021b）：對 trajectory 做 metric learning，距離拉近 / 拉遠 → 仍然吃 behavior policy 特徵。

CORRO 的定位：完全離線、同時支援 reward 與 transition dynamics 變化、用 contrastive learning 主動消除 behavior policy 的影響。

---

## 2. 核心方法與公式

### 2.1 Bi-Level Task Encoder

- **Transition encoder** $E_{\theta_1}$：輸入單一 transition $x=(s,a,r,s')$，輸出潛在向量 $z=E_{\theta_1}(x)$。
- **Aggregator** $E_{\theta_2}$：輸入 context $c=\{(s_i,a_i,r_i,s'_i)\}_{i=1}^k$，先得到 $z_i=E_{\theta_1}(x_i)$，再以 self-attention 風格加權合成：
$$z = \sum_{j=1}^{k} \mathrm{softmax}\bigl(\{\mathrm{MLP}(z_i)\}_{i=1}^k\bigr)_j \cdot z_j$$
- 為什麼選 transition 級別？trajectory 包含時間序列，會挾帶 behavior policy 的步法資訊；單步 transition 在固定 $(s,a)$ 下，$(r,s')$ 只剩 $R, T$ 的內在差異，較容易剝離 policy。

對比學習只訓練 $E_{\theta_1}$；$E_{\theta_2}$ 由下游 RL loss 一同訓練（Qψ、πφ 都 condition on $z$）。

### 2.2 Mutual Information Maximization 的學習目標

把 $E_{\theta_1}$ 視為機率編碼器 $z\sim P(z\mid x)$。task $M\sim P(M)$，且 $x$ 的分布由 $M$ 與 behavior policy 共同決定。目標：最大化 task representation 與 task 本身的互資訊
$$\max_{\theta_1}\; I(z; M) = \mathbb{E}_{z,M}\!\left[\log\frac{p(M\mid z)}{p(M)}\right]\!.$$
直覺：$z$ 應最大限度地降低對 $M$ 的不確定性，同時不保留任何 task-無關（含 behavior policy）的訊息。

### 2.3 InfoNCE 下界（核心定理）

設 task 集合 $\mathcal{M}$，$|\mathcal{M}|=N$。對任意 $M^*\in\mathcal{M}$，定義反事實 transition
$$x^* = (s,a,r^*,s'^*),\quad r^*=R^*(s,a),\;s'^*\sim T^*(\cdot\mid s,a)$$
即「相同 $(s,a)$ 下，用 $M^*$ 的 $R^*, T^*$ 重新生成 reward 與 next state」。再令
$$h(x,z) = \frac{P(z\mid x)}{P(z)}.$$
論文（附錄 A 證明）得到
$$\boxed{\;I(z; M) - \log N \;\ge\; \mathbb{E}_{M,x,z}\!\left[\log \frac{h(x,z)}{\sum_{M^*\in\mathcal{M}} h(x^*,z)}\right]\;}$$

**符號說明**：
- $x$：正樣本 transition，由真實 task $M$ 生成。
- $\{x^*\}_{M^*\in\mathcal{M}}$：負樣本集合，**鎖定相同 $(s,a)$**，由所有候選 task 重新生成 $(r,s')$（其中 $M^*=M$ 那一項就是正項，便於統一寫式子）。
- 分母把所有 $h(x^*,z)$ 加總，等同 InfoNCE 的 $N$-way 分類。

實作上以 cosine similarity 的 exponential 取代 $h$：定義 score $S(z^*, z)$，得到可計算的訓練目標
$$\boxed{\;\max_{\theta_1}\;\sum_{M_i\in\mathcal{M}}\;\sum_{x,x'\in X_i}\;\log\frac{\exp\bigl(S(z, z')\bigr)}{\sum_{M^*\in\mathcal{M}} \exp\bigl(S(z, z^*)\bigr)}\;}$$
- $x, x'$：從 task $M_i$ 的 dataset $X_i$ 採樣的兩個 transition，$z=E_{\theta_1}(x), z'=E_{\theta_1}(x')$ 為正對。
- 對每個 $M^*\ne M_i$，從「鎖定 $x$ 的 $(s,a)$、但用 $M^*$ 生成 $(r^*, s'^*)$」得到 $x^*$，並取 $z^*=E_{\theta_1}(x^*)$ 為負樣本表示；$M^*=M_i$ 時令 $z^*=z'$。

**為什麼這就能去除 behavior policy 影響？** 關鍵在「鎖定 $(s,a)$」：所有正/負樣本共用相同的 state-action，差別只在 reward 與 transition；encoder 要正確把正樣本拉近、把負樣本推遠，**唯一可用的訊號就是 $R, T$ 本身的差異**。behavior policy 的痕跡（state-action 分布的偏好）在這個構造下完全被抵消。

### 2.4 Negative Pair Generation：兩個關鍵原則與兩個方法

要生成 $x^*$ 必須能在「任意 $(s,a)$ 下」呼叫 $R^*, T^*$，但完全離線設定不允許。論文提出兩條原則：
- **Fidelity（保真度）**：生成的 $(r^*, s'^*)$ 分布應近似真實的負樣本分布
$$p(r, s'\mid s, a) \propto \mathbb{E}_{M\sim P(M)}\bigl[T(s'\mid s,a)\,\mathbf{1}\{R(s,a)=r\}\bigr]\!.$$
- **Diversity（多樣性）**：負樣本越多樣，InfoNCE 越難最佳化，學到的表徵越有意義。

#### 方法 1：Generative Modeling（CVAE）

把所有 task 的離線資料合併 $\bigcup_i X_i$ 訓練一個 conditional VAE，學 $p_\xi(r, s' \mid s, a, z)$ 與 $q_\omega(z\mid s,a,r,s')$，潛在向量 $z$ 抓住「在這個 $(s,a)$ 下不同 task 會給出怎樣不同的 $(r, s')$」這個不確定性。CVAE loss：
$$\mathcal{L}_{\text{CVAE}} = -\mathbb{E}_{(s,a,r,s')\in\{X_i\}}\!\left[\mathbb{E}_{q_\omega}\bigl[\log p_\xi(r,s'\mid s,a,z)\bigr] - \mathrm{KL}\bigl(q_\omega(z\mid s,a,r,s')\,\|\,p(z)\bigr)\right]\!.$$
取樣時從先驗 $p(z)$ 抽 $z$，再用 $p_\xi$ 解碼出反事實的 $(r^*, s'^*)$。要求：當不同 task 的 $(s,a)$ 分布有相當重疊時，這個合併分布才近似 Eq. (9)。

**失敗模式**：若不同 task 的 $(s,a)$ 分布幾乎不重疊（如 Point-Robot 每個 task 都往不同方向走），CVAE 退化成「對給定 $(s,a)$ 給出特定 task 的決定性預測」，多樣性塌掉、對比學習失效。

#### 方法 2：Reward Randomization

當 task 只在 reward function 變化時，直接對真實 reward 加噪聲生成負樣本：$r^* = r + \nu,\;\nu\sim p(\nu)$，例如 $\mathcal{N}(0, 0.5)$。雖然這未必逼近真實分布，但提供無限大且多樣的負樣本空間，讓 contrastive learning 更穩健。實驗顯示這在 Point-Robot 與 Ant-Dir 上比 CVAE 還好。

論文也比了 **Relabeling**（每個 task 各自學 reward / transition 模型再 relabel）與 **None**（直接拿其他 task 的真實 transition 當負樣本、不鎖定 $(s,a)$）兩個對照組——見實驗節。

---

## 3. 演算法流程

訓練分三段（Algorithm 1）：

**A. 預訓 CVAE（若採用 generative modeling）**：在 $\bigcup_i X_i$ 上最佳化 $\mathcal{L}_{\text{CVAE}}$。

**B. 訓練 transition encoder $E_{\theta_1}$**：
1. 抽 task $M$ 與兩個 transition $x, x'\in X_M$，計算 $z=E_{\theta_1}(x), z'=E_{\theta_1}(x')$。
2. 對每個 $M^*\in\mathcal{M}$ 生成 $x^*$：
   - 若 generative modeling：由 CVAE 在 $(s, a)$ 條件下取樣 $(r^*, s'^*)$。
   - 若 reward randomization：$r^* = r + \nu$，$s'^* = s'$。
3. 算 $z^*=E_{\theta_1}(x^*)$，套入 contrastive 損失，反向更新 $\theta_1$。

**C. 訓練 aggregator / policy / Q-function**：
1. 抽 task $X$ 與 context $c$，得 $z = E_{\theta_2}(E_{\theta_1}(c))$。
2. 把 $z$ concat 進 state，套 offline RL 演算法（論文用 SAC）更新 $\theta_2, \psi, \phi$。

**測試（Algorithm 2）**：取一條任意策略蒐集的 context $c$ → $z=E_{\theta_2}(E_{\theta_1}(c))$ → 用 $\pi_\phi(a\mid s, z)$ 部署。

---

## 4. 實驗與結論

### 4.1 環境

| 環境 | 變化 | 任務分布 |
|---|---|---|
| Point-Robot | reward（goal 位置） | $g\sim U[-1,1]^2$ |
| Ant-Dir | reward（前進方向） | $\theta\sim U[0,2\pi]$ |
| Half-Cheetah-Vel | reward（目標速度） | $v_g\sim U[0,3]$ |
| Walker-Param | transition（32 個物理參數） | $1.5^\mu, \mu\sim U[-3,3]$ |
| Hopper-Param | transition（41 個物理參數） | 同上 |

各 20 個訓練 / 20 個測試 task，每 task 用 SAC 各自跑出 replay buffer 當作離線資料。

### 4.2 主要結果

**IID 測試（context 來自 single-task 訓練的 replay）**：
- 在 reward 變化的環境（Point-Robot、Ant-Dir）CORRO 與 supervised baseline 領先；Ant-Dir 上 CORRO 甚至贏過 supervised。
- 在 transition 變化的環境（Walker-Param、Hopper-Param）CORRO 領先，學習更穩定，FOCAL 偶爾發散。
- Half-Cheetah-Vel IID 上各方法差不多，但 OOD 拉開差距。

**OOD 測試（context 改用「不同 task / 不同訓練時期 checkpoint」的 behavior policy 蒐集）**（Table 1）：
- Half-Cheetah-Vel：CORRO −89.7 vs FOCAL −204.1 vs Offline PEARL −242.7。
- Ant-Dir：CORRO 154.7 vs FOCAL 53.5。
- 各環境 CORRO 都對 OOD 退化最小。supervised baseline 雖在某些環境 OOD 表現也好，但須用 ground-truth task label，meta-RL 通常拿不到。

**隨機探索策略蒐 context**（Table 3）：CORRO 在 4/5 環境贏，顯示對 context 蒐集策略寬容度高。

### 4.3 Latent space 可視化（Figure 4）

t-SNE 投影 Half-Cheetah-Vel 的 task representation：CORRO 把 task 排成一條 1D manifold，顏色（目標速度）沿著 manifold 平滑漸變；FOCAL、Offline PEARL、supervised 都看不到這種結構。

### 4.4 Negative pair 生成消融（Table 2）

| 方法 | 對比 loss | Half-Cheetah-Vel OOD | Point-Robot OOD |
|---|---|---|---|
| Generative (CVAE) | 0.07 | **−89.7** | −9.42 |
| Randomize | 0.83 | −84.5 | **−6.39** |
| Relabeling | 0.04 | −245.3 | −9.27 |
| None（不鎖 $(s,a)$，跨 task 真 transition） | 1.20 | −97.6 | −6.52 |

觀察：
- 「None」雖然不鎖 $(s,a)$，但其實作仍把編碼器架在 transition tuple 上，已比 FOCAL（用 trajectory）好很多——作者額外把 FOCAL 改用 InfoNCE 結果還是差，說明 **「採用 transition 而非 trajectory」是 OOD 泛化的主要功臣**。
- Relabeling 多樣性最大（loss 最低）但因單 task reward / transition 模型不準導致下游崩潰。
- CVAE 在 Point-Robot 失敗，因 (s,a) 分布幾乎不重疊；Randomize 在 reward-only 設定下穩定又便宜。
- 因此論文在 Walker-Param、Hopper-Param 也用 generative（transition 也要建模），在 Hopper-Param 甚至直接用 None。

### 4.5 結論

完全離線、含 reward + transition 變化的 OMRL 設定下，bi-level encoder + transition-level InfoNCE + 鎖定 $(s,a)$ 的反事實 negative pair，是把 behavior policy 從 task representation 中剝離的有效設計；OOD 蒐 context 時穩健性顯著優於 FOCAL、Offline PEARL，部分環境甚至贏過 ground-truth supervised 上限。作者也明說目前未處理「自我探索策略」與「資料極稀時 moderate 上線互動」兩個方向。

---

## 與本研究主線的關聯

我的主線：**robot play 蒐到的多任務離線資料 + 極少弱語言標註 + offline meta-RL，希望從文字得到 task spec / task embedding 做 zero-shot adaptation，並關注 meta-learning × zero-shot 並行**。CORRO 是這條線上特別關鍵的一篇，理由如下。

**1. Robot play 的資料本身就是「task ↔ behavior policy 高度糾纏」的最壞情況。** Play 資料通常是同一個（或少數幾個）人在做事，behavior 與場景、技能、目的高度耦合：你拿一段「把杯子放進櫃子」的 play，agent 走的軌跡、speed profile、夾爪用法都帶著很重的「這個人這次的習慣」，而不是「這個 task 該怎麼做」的本質。如果直接拿這種資料訓 context-based offline meta-RL（或拿 latent 當 task embedding 對齊到文字），最大的風險就是 latent 學到的是「這次蒐集行為的 signature」，而不是 task 本質。CORRO 描述的「2D goal-reaching 中 encoder 直接看走的方向、不看 reward」的失敗模式，就是我的場景的縮影。

**2. CORRO 的核心構造（鎖 $(s,a)$ 的反事實 transition）為 play 資料指出一條可行的去汙染路徑。** 在 play 設定下，把樣本從 trajectory 降到 transition、再用「同 $(s,a)$ 不同 $(r, s')$」當對比訊號，是在沒有 task label 的前提下迫使表徵只看 reward / transition 內在差異的可行手法。不過 play 通常 reward 不明確、tasks 邊界模糊（一段 play 可能涵蓋多個子目標），所以要把 CORRO 直接搬過來，需要：(a) 先用某種方式為 play 劃分 / 對齊 task 群（例如分段、語言切片）；(b) 因為 (s, a) 分布跨 task 重疊可能也不大，CVAE 可能像在 Point-Robot 一樣 collapse，這時 reward randomization 行不通（play 沒明確 reward），更需要新的負樣本生成策略——例如以「鎖 $(s,a)$ 但替換語言條件」的方式構造反事實，把語言當作生成器的 condition 而不是只當 retrieval target。

**3. Contrastive 與 language supervision 是高度互補的（這是本篇給我最有價值的 takeaway）。**
- CORRO 的 InfoNCE 是 *無監督* 對比，能在「沒有 task 標籤」時把 behavior policy 從 latent 中剝離，但它無法保證 latent 的語意結構與「人類想的 task 概念」對齊——它能讓 representation 對 task 不同/相同敏感，但 latent 的方向、組合性、可解釋性都不受控（只在 Half-Cheetah-Vel 偶然出現 1D 漸變）。
- 弱語言 supervision 剛好補上這一塊：少量「這段是把紅杯放進右抽屜」的 caption 可以當 anchor 把 latent 維度釘到「物件」「目的位置」「動作型態」這些可組合語意上，**但語言本身對 behavior policy 汙染無能為力**——如果 encoder 把「這個人手抖的習慣」也編進 latent，文字也不會告訴你不要編。
- 一個自然的結合方案：用 CORRO 的鎖 $(s,a)$ 反事實 InfoNCE 作為「去 policy 汙染」的 backbone loss，**同時**把弱語言當另一條對比軸（語言 embedding 與 task latent 在相同 / 不同 task 上拉近 / 推遠），或直接用語言 embedding 當 InfoNCE 的 anchor，把「同語言不同 trajectory」當正樣本、「不同語言但同 $(s,a)$ 反事實」當負樣本。這樣 latent 同時具備：對 behavior 不敏感（從 CORRO 來）、對語言語意對齊（從弱 caption 來）→ 才有機會做真正的 zero-shot：給一句新的 task 描述，文字編碼器吐出的 embedding 直接可餵進 policy。
- 換句話說：**CORRO 解決的是「latent 不該含什麼」（behavior policy）；語言 supervision 解決的是「latent 該含什麼且怎麼結構化」（task 語意）。兩者疊加才是 robot play + 弱語言 zero-shot meta-RL 想要的 task representation。**

**4. 對 meta-learning × zero-shot 並行的啟發。** CORRO 屬於 context-based meta-RL：先看 context 才能 adapt。要走 zero-shot，自然的做法是把「從 context 推 latent」換成「從語言推 latent」，但前提是這兩條路徑的 latent 空間相同。CORRO 提供了一個「context 路徑」可以乾淨地避開 behavior policy 干擾的訓練方案；再加上語言對齊損失（contrastive between context-latent and language-latent），就能讓 zero-shot 的語言路徑與 meta-learning 的 context 路徑共享一個語意空間。

**5. 風險與限制（要注意）。**
- CORRO 假設不同 task 共享 $\mathcal{S}, \mathcal{A}$，且 task 是良好定義的 MDP；robot play 並非如此。
- CORRO 沒處理「探索策略」與「線上微調」，但 robot play 的本性是「先有探索 / 再有任務劃分」。
- CVAE / Randomize 兩條負樣本路在 reward 隱式、transition 共享但風格不同的 play 設定下都不直接適用，需要新的負樣本構造（例如以「語言條件下的反事實 transition」生成）。

---

## 一句話總結

CORRO 把 task representation 學習從「整條 trajectory 的判別」搬到「鎖定相同 $(s,a)$、僅替換 reward 與 next state 的反事實 transition 的 InfoNCE」，配合 CVAE 或 reward randomization 來生成負樣本，從而在完全離線的 meta-RL 中把 behavior policy 的痕跡逐出 task embedding，OOD context 下顯著穩健——這正好補上「弱語言 supervision 解決 latent 該含什麼、卻無法解決 latent 不該含什麼」的另一半。
