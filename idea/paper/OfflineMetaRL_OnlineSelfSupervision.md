---
type: paper-note
aliases:
  - "Offline Meta-RL with Online Self-Supervision"
  - "SMAC"
  - "Online Self-Supervision"
year: 2022
stage: "3-offline-meta-rl"
tags:
  - offline-meta-rl
  - self-supervision
  - context-shift
  - 警訊
  - 3-offline-meta-rl
summary: "揭示meta-test收新context造成z-space shift；嚴格zero-shot（只給語言）正好繞過此問題。"
---
# Offline Meta-Reinforcement Learning with Online Self-Supervision (SMAC)

> Pong, Nair, Smith, Huang, Levine. ICML 2022 (PMLR 162)。
> 提出 SMAC：以離線多任務 reward-labelled 資料做 meta-train，再用無 reward 標記的線上互動做自我監督式 fine-tune，補上「離線 meta-RL 在 meta-test 階段出現的 z-space 分佈偏移」。

---

## 0. 閱讀總覽（白話）

這篇要解決的問題是：當我們用一批「離線」（已預先標好 reward 的多任務）資料來訓練一個 meta-RL agent 時，看起來很省事——只要標一次 reward、訓練很多次、不必跟真實環境互動就能 meta-train 出 adaptation 機制。但是當這個 agent 上到 meta-test 時，它必須先用學到的 exploration policy 自己跑出一段 trajectory 當作 context，然後再根據這段 context 去調整自己的行為。問題是：

- 離線資料是「行為策略 $\pi_\beta$」蒐集的（例如人示範、[[PEARL|PEARL]] 早期的探索策略），
- 但 meta-test 時的 context 是「meta-learn 出來的 exploration policy $\pi_\theta$」蒐集的，
- 兩者的軌跡分佈不同 $\Rightarrow$ encoder 輸出的 latent context $z = A_\phi(h)$ 分佈也不同 $\Rightarrow$ 條件在這個「沒見過分佈」的 $z$ 上的 policy 會壞掉。

作者把這個現象命名為 **z-space distribution shift**，並提供實證：在 Ant Direction 上，post-adaptation policy 條件在 $z\sim q(z\mid h_{\text{offline}})$ 時能往各方向跑，但條件在 $z\sim q(z\mid h_{\text{online}})$ 時就只會往某一個方向走，性能掉很多。

SMAC 的解法是「**不必再標 reward**，但允許 agent 跟環境互動」：
1. 先做 offline meta-train，順便學一個 reward decoder $r_{\phi_d}(s,a,z)$；
2. 線上 rollout 用 $z\sim p(z)$（從 prior 採樣）跑出 trajectory $\tau$；
3. 用「離線資料」估出該任務的 $z\sim q(z\mid h_{\text{offline}})$，再用 reward decoder 給 $\tau$ 補上合成 reward；
4. 把這些自我標註的資料丟回 buffer 繼續做 actor/critic 更新（encoder 與 decoder 不再更新）。

這個方法的核心張力是：純 offline meta-RL 假設不能再跟環境互動，但作者放寬「不互動」這條，改為「可以互動但不能再要 reward 監督」，這在現實機器人或網頁瀏覽情境下更切實際——reward 才是真正昂貴的監督。

最後實驗顯示：offline 階段 SMAC 跟 [[MACAW|MACAW]] / BOReL 平手，但開啟 self-supervised 線上階段後，SMAC 在六個 MuJoCo / Sawyer 環境上都顯著超越，逼近全程使用真實 reward 的 online oracle PEARL。

---

## Abstract（重點翻寫）

Meta-RL 雖然能在少量 trial 內適應新任務，但 meta-training 本身耗資源；若能改成從離線資料 meta-train，就能重複利用同一份「標一次 reward 的多任務資料」。然而離線 meta-RL 有額外挑戰：meta-train 時的 context 由先前 behavior policy 產生，但 meta-test 時的 context 由 meta-learn 出的 exploration policy 產生，分佈不同會導致 adaptation 在新分佈上失效。本文提出 hybrid 演算法：先 offline meta-train，再蒐集**無 reward**的線上資料來縫合此分佈偏移；自動生成的合成 reward 讓無監督資料也能參與 meta-train。實驗在 locomotion 與 manipulation 上顯示效能可逼近全程有 reward 的 online meta-RL。

---

## 1. 問題背景：為什麼 meta-test 收集 context 會破壞離線假設

### 1.1 Meta-RL 的 context-based 範式

設任務分佈 $p_\mathcal{T}(\cdot)$，每個任務 $\mathcal{T}=(\mathcal{S},\mathcal{A},r,\gamma,p_0,p_d)$ 是個 MDP。一個 meta-episode：
1. 抽 $\mathcal{T}\sim p_\mathcal{T}$；
2. 用 policy $\pi_\theta$ 跑 $T$ 條 trajectory；
3. 在 trajectory 之間，用 adaptation procedure $z = A_\phi(h)$ 把目前 episode 累積的 history $h=\{(s_i,a_i,r_i,s'_i)\}$ 轉成 context $z$；
4. policy 變成 $\pi_\theta(a\mid s, z)$；
5. 以最後一條 trajectory 的回報衡量。

在 PEARL 中，$A_\phi$ 是隨機 encoder $z\sim q_{\phi_e}(z\mid h)$，輸出對角高斯。

### 1.2 Offline meta-RL 的設定

只給定多個任務的 replay buffer $\mathcal{D}=\{\mathcal{D}_i\}_{i=1}^{N_{\text{buff}}}$，每個 buffer 由某 behavior policy $\pi_\beta$ 蒐集，agent 不能再跟環境互動。

### 1.3 真正的痛點：z-space distribution shift

純 offline meta-RL 只在 $h\sim\mathcal{D}_i$（即 $h\sim\pi_\beta$）下訓練 $A_\phi$。但 meta-test 階段，agent 必須用自己學到的 $\pi_\theta$ 去蒐集 context $h$。於是：

$$
p(z\mid h_{\text{offline}}) \neq p(z\mid h_{\text{online}}),\qquad h_{\text{offline}}\sim\pi_\beta,\; h_{\text{online}}\sim\pi_\theta.
$$

範例：人示範蒐集的軌跡可能平滑，但 stochastic exploration policy 跑出來的可能抖動。雖然抖動不影響探索，但 encoder 對「抖動軌跡」產生的 $z$ 分佈是 offline meta-train 時根本沒見過的，policy 條件在這種 $z$ 上會崩。

作者在 Ant Direction 上實證（Fig 2）：
- $\mathrm{KL}(q_{\phi_e}(z\mid h_{\text{offline}})\,\|\,p(z))$ 與 $\mathrm{KL}(q_{\phi_e}(z\mid h_{\text{online}})\,\|\,p(z))$ 分佈差距大；
- 同一個 policy，只是換 $z$ 的來源 ($h_{\text{offline}}$ vs $h_{\text{online}}$)，post-adaptation 回報直接掉一截。

### 1.4 與「純 offline」設定的張力

純 offline 的精神是 "no further interaction"。但作者指出：若強行用保守策略（讓 $\pi_\theta\approx\pi_\beta$）來消滅 shift，就會犧牲 meta-RL 學「好的 exploration 策略」的能力——meta-RL 的賣點正是學一個比 behavior policy 更聰明的探索器。所以**保守化不是答案**。

作者的取捨：放鬆「不互動」這條，但保留「不再標 reward」。換句話說：
- 純 offline RL：禁止互動。
- 純 online meta-RL：互動 + reward 監督。
- SMAC（semi-supervised）：互動但**無 reward**。

這個張力是論文的核心：offline 假設的「不互動」其實過嚴；真正昂貴的是 reward 標註（人類標 reward、現實安全代價）而不是互動本身。SMAC 把 offline 假設換成「reward 一次就標完」這個更貼近現實的版本。

---

## 2. 核心方法與公式

SMAC 兩階段：(A) offline meta-train（PEARL + AWAC + reward decoder）、(B) self-supervised online meta-train。

### 2.1 符號

- $\mathcal{D}=\{\mathcal{D}_i\}_{i=1}^{N_{\text{buff}}}$：每任務一個 replay buffer，含 $(s,a,r,s')$。
- $h\sim\mathcal{D}_i$：一個 mini-batch / history。
- $\tau=(s_1,a_1,s_2,\dots)$：**無 reward** 的 trajectory。
- $q_{\phi_e}(z\mid h)$：stochastic encoder（與 PEARL 同）。
- $\pi_\theta(a\mid s,z)$：context-conditional policy。
- $Q_w(s,a,z)$：context-conditional Q。
- $r_{\phi_d}(s,a,z)$：**reward decoder**（SMAC 新加）。
- meta-parameters $\phi=\{\phi_e,\phi_d\}$。

### 2.2 Phase A：Offline meta-training

#### Critic loss（與 PEARL 同的 Bellman error，但只用 offline data）：

$$
\mathcal{L}_{\text{critic}}(w)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}_i,\; z\sim q_{\phi_e}(z\mid h),\; a'\sim\pi_\theta(\cdot\mid s',z)}\bigl[(Q_w(s,a,z)-(r+\gamma Q_{\bar w}(s',a',z)))^2\bigr].
$$

#### Actor loss（用 AWAC 取代 SAC，避免 offline bootstrapping error）：

$$
\mathcal{L}_{\text{actor}}(\theta)=-\mathbb{E}_{s,a,s'\sim\mathcal{D},\; z\sim q_{\phi_e}(z\mid h)}\Bigl[\log\pi_\theta(a\mid s,z)\cdot\exp\!\Bigl(\frac{Q(s,a,z)-V(s',z)}{\lambda}\Bigr)\Bigr],
$$

其中 $V(s,z)=\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s,z)}[Q(s,a,z)]$ 用單一樣本近似，$\lambda$ 是約束的 Lagrange multiplier。AWAC 的本意是「以 advantage-weighted likelihood 把 policy 隱式拉近 behavior 分佈」。

#### Reward decoder 與 encoder 的聯合 loss：

$$
\mathcal{L}_{\text{reward}}(\phi_d,\phi_e,h,z)=\sum_{(s,a,r)\in h}\|r-r_{\phi_d}(s,a,z)\|_2^2 \;+\; D_{\mathrm{KL}}\!\bigl(q_{\phi_e}(\cdot\mid h)\,\|\,p_z(\cdot)\bigr).
$$

注意：
- reward loss 反向傳到 encoder（這跟 PEARL 把 critic loss 灌進 encoder 不同；附錄 B ablation 顯示兩者性能差不多）。
- 第二項 KL 把 $q_{\phi_e}$ 拉向 prior $p_z$，形成 information bottleneck，使得**從 prior 採樣的 $z$ 仍代表合理的 latent task**——這在 Phase B 用 $z\sim p(z)$ 探索時很關鍵。

### 2.3 Phase B：Self-supervised online meta-training

關鍵公式是 **synthetic reward labeling**：

$$
r_{\text{generated}}=r_{\phi_d}(s,a,z),\quad z\sim q_{\phi_e}(z\mid h_{\text{offline}}),\; h_{\text{offline}}\sim\mathcal{D}_i. \tag{4}
$$

整體流程每個 iteration：
1. 從 prior 抽 $z_t\sim p(z)$，用 $\pi_\theta(a\mid s,z)$ 跑出無 reward trajectory $\tau$。
2. 隨機抽某任務 $\mathcal{D}_i$，從中抽 history $h_{\text{offline}}$，編碼 $z\sim q_{\phi_e}(z\mid h_{\text{offline}})$。
3. 用 (4) 給 $\tau$ 上每個 transition 補 reward，把標註後資料併入 $\mathcal{D}_i$。
4. 重抽 $h, h'\sim\mathcal{D}_i$，編碼 $z=q_{\phi_e}(h)$，用 $\mathcal{L}_{\text{critic}}, \mathcal{L}_{\text{actor}}$ 更新 $\pi_\theta, Q_w$。
5. **不再更新 $\phi_e, \phi_d$**（因為沒有新的真實 reward 監督）。

#### 為什麼這樣不會造成 reward decoder 自身的分佈偏移？

作者的論點：z-shift 只發生在「用 online history 編碼 $z$」這個動作。但 reward decoder 只在 $z\sim q_{\phi_e}(z\mid h_{\text{offline}})$ 下被呼叫——也就是 reward decoder 看到的 $z$ 分佈永遠維持 offline 分佈，因此沒有自我訓練資料污染問題。reward decoder 唯一要 generalize 的是「新的 $(s,a)$」，而非新的 $z$，作者主張 reward generalization 比 policy generalization 容易。

#### Self-supervised 階段的 actor loss 變體

實驗發現混合 PEARL 的 actor loss 略好：

$$
\mathcal{L}_{\text{actor}}^{\text{self-sup}}(\theta)=\mathcal{L}_{\text{actor}}(\theta)+\lambda_{\text{pearl}}\cdot\mathcal{L}_{\text{actor}}^{\text{PEARL}}(\theta),
$$

其中 PEARL actor loss 是 SAC-style entropy-regularised KL：

$$
\mathcal{L}_{\text{actor}}^{\text{PEARL}}(\theta)=\mathbb{E}_{s\sim\mathcal{D}_i,z\sim q_{\phi_e}(z\mid h)}\Bigl[D_{\mathrm{KL}}\!\Bigl(\pi_\theta(a\mid s,z)\;\Big\|\;\frac{\exp Q_w(s,a,z)}{Z(s)}\Bigr)\Bigr].
$$

$\lambda_{\text{pearl}}=0$ 等同 AWAC。

---

## 3. 演算法流程（Algorithm 1 改寫）

**輸入**：$\mathcal{D}=\{\mathcal{D}_i\}$、$\pi_\theta$、$Q_w$、$q_{\phi_e}$、$r_{\phi_d}$。

**Phase A — Offline（$n=1,\dots,N_{\text{offline}}$）**：
1. 抽 buffer $\mathcal{D}_i\sim\mathcal{D}$，抽兩個 history $h, h'\sim\mathcal{D}_i$。
2. $z\sim q_{\phi_e}(\cdot\mid h)$。
3. 用 $z, h'$ 同時最小化 $\mathcal{L}_{\text{actor}}, \mathcal{L}_{\text{critic}}, \mathcal{L}_{\text{reward}}$ 更新所有參數。

**Phase B — Self-supervised online（$n=1,\dots,N_{\text{online}}$）**：
1. $z_t\sim p(z)$，跑 trajectory $\tau$，用 $\pi_\theta(a\mid s,z_t)$。
2. 抽 $\mathcal{D}_i$，抽 $h_{\text{offline}}\sim\mathcal{D}_i$。
3. 用 (4) 給 $\tau$ 補 reward，併入 $\mathcal{D}_i$。
4. 再抽 $h, h'\sim\mathcal{D}_i$，$z = q_{\phi_e}(h)$。
5. 用 $z, h'$ 更新 $\pi_\theta, Q_w$（最小化 $\mathcal{L}_{\text{actor}}, \mathcal{L}_{\text{critic}}$，**不**更新 encoder / decoder）。

---

## 4. 實驗與結論

### 4.1 環境

- **MuJoCo 系列**：Cheetah Velocity、Ant Direction、Humanoid（376-D 狀態）、Walker Param、Hopper Param（後兩者是隨機 physics 參數，需 adapt to dynamics）。
- **Sawyer Manipulation**（Khazatsky et al. 2021 的 PyBullet 環境改的）：開抽屜、按按鈕、抓物。sparse reward $\{-1,0\}$，offline data 由 scripted random policy 蒐集，成功率僅 46%——高度 suboptimal。

### 4.2 Offline 資料規模刻意比 MACAW/BOReL 少 2–3 個數量級

- BOReL 原論文 Cheetah Velocity 用 400M transitions；MACAW 用 100M；SMAC 只用 240k。
- 作者刻意這樣做來壓力測試 offline 階段對「少且差」資料的學習能力。

### 4.3 比較對象

- **Online Oracle**：PEARL + 真實 reward 線上訓練（上界）。
- **MACAW**（Mitchell et al. 2021）、**BOReL**（Dorfman & Tamar 2020）：純 offline meta-RL，沒有自我監督機制可用，故只報 offline 訓練後結果。
- **Meta behavior cloning**：把 actor update 換成單純 BC。
- **SMAC (actor ablation)**：把 AWAC actor loss 換回 PEARL 的，藉以驗證 AWAC 對 offline 階段是必要的。

### 4.4 主要結果（Fig 5）

- 六個環境上 SMAC 的 offline 階段就跟 MACAW/BOReL 打平甚至贏；
- 加上 self-supervised 階段後，SMAC 顯著超越所有 offline meta-RL baseline，並接近 Online Oracle。
- Sawyer Manipulation 上 actor ablation（用 PEARL-style update）明顯不如 AWAC，顯示 offline RL 階段必須用 conservative update。

### 4.5 視覺化 z-shift（Fig 6）

- Offline 訓練後：post-adaptation policy 條件在 $h_{\text{offline}}$ 時往各方向走；條件在 $h_{\text{online}}$ 時只往左上一個方向。
- Self-supervised 後：兩種來源的 history 得到的軌跡幾乎一致。
- 進一步在附錄 Fig 8 看出：exploration 軌跡本身在 self-supervised 前後變化不大，問題真的在 adaptation procedure 對 history 來源的敏感性，而不是 exploration 變了。

### 4.6 附錄重點

- **B1 Encoder loss 敏感性**：只用 Q-loss 訓 encoder（PEARL 原作法）在 Walker/Hopper 上崩；同時用 reward loss + Q-loss 略優於只用 reward loss；簡化起見 SMAC 只用 reward loss。
- **B2 State-space shift on test tasks**：在 Sawyer 上允許 agent 在「測試任務」環境裡自我監督互動（不要 reward），可以進一步追上一個拿真實 reward 在測試任務上線上學的 oracle，顯示 SMAC 也能緩解 state 分佈偏移。
- **B3 Reward decoder 不必很準**：Sawyer reward 量級 1，decoder MSE 約 0.2~0.25 也足以支撐 SMAC。
- **B4 為何 online 比例越來越大、性能不退化**：作者認為 meta-RL 的目標是「對任何任務都能 adapt」，新生成 reward 對應的就是新任務，不存在「錯誤資料」這回事。

### 4.7 結論與限制

SMAC 識別出 offline meta-RL 特有的 z-space distribution shift 並提出 self-supervised online fine-tune 的解法；可與任何 context-based meta-RL（如 PEARL/MACAW/BOReL）結合。限制：仍須允許無監督互動，需要安全自動的環境介接。

---

## 與本研究主線的關聯

使用者主線：robot play + 極少弱語言 + offline meta-RL + 文字→task spec / task embedding 做 zero-shot；關注 meta-learning × zero-shot 並行方向。

### A1. 本論文揭示的 distribution shift 是使用者題目的核心威脅

使用者規劃中的「offline meta-RL on robot play data + few-shot/zero-shot transfer」幾乎注定踩到 SMAC 點名的兩個 shift：
1. **z-space shift**：robot play data 是人類遠端操作（teleop）的平滑軌跡；但部署時 agent 自己跑出來的 exploration 軌跡必定不像人類示範（抖動、stuck、撞物等）。SMAC 在 Sawyer Manipulation 已直接示範這個情境的崩壞。
2. **State-space shift on novel tasks**：使用者目標是 zero-shot 適應**全新任務**（甚至以語言指定），這必然引入訓練 distribution 沒見過的 state 分佈。本論文附錄 B2 證實在 test task 環境裡做無 reward 互動可顯著彌補 state shift。

換句話說，若使用者真的要做純粹 zero-shot 文字→behavior，且不允許任何測試端互動，則 SMAC 指出的 shift 會成為性能上限。

### A2. 語言 supervision 能緩解的部分

SMAC 的 z 是純統計上的 task latent，沒有語意接地。使用者主線中加入「弱語言」做 task spec 對 z-shift 的影響可從兩個角度想：

- **正面**：若 task 由語言唯一指定，那麼 $z = \mathrm{TextEnc}(\ell)$ 來自語言端，與 trajectory 收集策略**解耦**——也就是 z 的來源不再是 $q_\phi(z\mid h)$ 這個 history-conditioned encoder，自然避開 z-shift。極少弱語言 + zero-shot 的 setup 等於把 SMAC 想處理的 history-based adaptation 換成 language-based adaptation。
- **限制**：但 policy 仍然需要在 robot play 的 $\pi_\beta$ 分佈下被 trained，部署時的 $(s,a)$ 分佈仍會 shift，這時 SMAC 提到的 state shift 與 AWAC-style 保守更新依然必要。換句話說，語言能消掉 z-shift，但消不掉 state-action shift。

### A3. 具體可借鏡的設計

- **AWAC actor loss**：使用者只要用 offline robot play 做 meta-train，AWAC（或 [[IQL|IQL]]、[[CQL|CQL]]）這類 conservative update 幾乎是必需品，不然 bootstrap 會炸。
- **Reward decoder**：若使用者的弱語言只覆蓋部分 trajectory，可以把 SMAC 的 reward decoder 改成「以語言為條件」的 $r_{\phi_d}(s,a,\ell)$ 來為大量無語言、無 reward 的 play data 補標籤，達成「語言→reward 自動標註」的版本，銜接 inverse-RL/preference learning。
- **Prior-driven exploration**：Phase B 用 $z\sim p(z)$ 探索的做法，對應到使用者題目就是「用語言空間的 sampling」推動 exploration coverage。

### A4. 重要警訊

對 zero-shot 從文字到 task embedding 的設計者而言，最該記住一句：**離線 meta-RL 訓練時 agent 從來沒見過自己 exploration 出來的 context**。即便有了乾淨的語言 embedding 當 z，policy 仍可能在自己的「shaky deployment trajectory」上失靈。最低成本的對策就是 SMAC 的精神：放棄「不互動」這條太嚴的假設，改要「不需新 reward 標註」——這在弱語言 + 機器人 setup 下非常合理（採集 trajectory 便宜，重新標 reward / 重寫語言指令昂貴）。

---

## 一句話總結

SMAC 點出離線 meta-RL 在 meta-test 收集 context 時必然產生 z-space distribution shift，並用「允許互動但禁用 reward 監督」的 semi-supervised 折衷，配合 reward decoder 自動補標籤，把純 offline meta-RL 的性能推到接近全程有 reward 的 online meta-RL。
