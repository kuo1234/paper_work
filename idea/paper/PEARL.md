---
type: paper-note
aliases:
  - "PEARL"
year: 2019
stage: "1-meta-rl骨架"
tags:
  - meta-rl
  - context-inference
  - posterior-sampling
  - off-policy
  - 核心
  - 1-meta-rl骨架
summary: "off-policy meta-RL，用probabilistic context q(z|c)做task inference，posterior sampling探索；本專案重現主線。"
---
# PEARL 論文逐段詳細整理

> 論文：**Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables**  
> 作者：Kate Rakelly, Aurick Zhou, Deirdre Quillen, Chelsea Finn, Sergey Levine  
> 主題：Meta-Reinforcement Learning、Off-Policy RL、Probabilistic Context、Posterior Sampling、SAC  
> 整理目標：逐段說明本文想解決的問題、方法設計、公式意義、演算法流程、實驗設計與主要結論。

---

## 0. 閱讀總覽：這篇論文到底在做什麼？

PEARL 的完整名稱是 **Probabilistic Embeddings for Actor-Critic Reinforcement Learning**。這篇論文想解決的是：

> 如何讓 meta-RL 在訓練時更省樣本，同時在新任務上可以用少量互動快速適應？

傳統 deep RL 每個任務通常都要單獨訓練，而且需要大量 environment interaction。Meta-RL 則希望 agent 可以從許多相關任務中學會一種「快速適應新任務」的能力。但早期 meta-RL 方法，例如 [[MAML|MAML]]、[[RL2|RL²]]、ProMP、MAESN，多半依賴 on-policy training，因此訓練樣本效率很差。

PEARL 的核心想法是把 meta-RL 拆成兩個部分：

1. **Task inference**：從少量 context transitions 推論目前任務是什麼。
2. **Control**：在知道任務 latent variable 後，選擇好的 action。

PEARL 用一個 probabilistic encoder：

\[
q_\phi(z \mid c)
\]

從 context \(c\) 推論 latent task variable \(z\)。policy 和 critic 都 conditioned on \(z\)：

\[
\pi_\theta(a \mid s,z), \qquad Q_\theta(s,a,z)
\]

這使得 agent 在新任務上不需要像 MAML 一樣做 gradient update，而是只要收集少量 trajectory，更新 context，再重新推論 posterior over \(z\)。同時，actor 和 critic 可以用 off-policy SAC 訓練，因此 meta-training sample efficiency 比 on-policy meta-RL 高很多。

---

## 1. Abstract 逐段解釋

### Abstract 段落 1：問題與方法總述

論文摘要首先指出 deep RL 的基本問題：單一任務通常需要大量經驗才能學會。Meta-RL 理論上可以讓 agent 從少量新任務經驗中快速學習，但實務上有兩個主要障礙。

第一個障礙是 **sample efficiency**。當時許多 meta-RL 方法依賴 on-policy data，這代表每次 policy 更新都需要重新收集資料，舊資料很難重複使用。這在 RL 裡非常昂貴，尤其如果任務是 robotics 或 continuous control，環境互動成本很高。

第二個障礙是 **task uncertainty**。當 agent 面對一個新任務時，一開始並不知道這個任務的目標、reward function 或 dynamics 是什麼。如果方法沒有明確機制處理「我目前還不確定任務是哪一個」，那它在 sparse reward 環境中會很難探索。因為 sparse reward 下，只有碰到少數重要狀態才會得到 reward，agent 必須能夠做有方向、有假設的探索。

PEARL 的解法是：

1. 使用 **off-policy meta-RL** 提升 meta-training sample efficiency。
2. 使用 **probabilistic context variable** 表示任務不確定性。
3. 將 **task inference** 和 **control** 解耦。
4. 在新任務上透過 **posterior sampling** 做 structured exploration。

摘要最後強調實驗結果：PEARL 在數個 meta-RL benchmark 上，相比 prior methods 有 **20–100 倍的 meta-training sample efficiency 提升**，且 final performance 也更好。

---

## 2. Introduction 逐段解釋

### Introduction 段落 1：Deep RL 很強，但每個任務都要大量資料

第一段先建立背景。Deep RL 結合深度神經網路後，已經能解許多 sequential decision-making 問題。但標準 RL 通常是「每個任務訓練一個 policy」，而每個 policy 都可能需要數百萬甚至更多 environment interactions。

這造成一個問題：如果我們希望 agent 學會大量技能，例如各種不同機器人操作、導航、操控任務，逐一訓練會非常昂貴。

作者舉例：擰瓶蓋和轉門把都涉及手部抓握與旋轉手腕。這些任務雖然目標不同，但共享很多結構。因此合理的想法是：agent 應該能從過去任務中學到共通結構，然後在新任務上快速適應。

這就是 meta-learning 的動機。

---

### Introduction 段落 2：Meta-learning 可以學共通結構，但 meta-training 本身很耗資料

第二段說明 meta-RL 的矛盾。Meta-learning 的目標是讓模型可以在新任務上用少量資料快速適應。但要學會這種能力，meta-training 時通常需要在大量訓練任務上收集大量資料。

也就是說，meta-RL 雖然提高了 **test-time adaptation efficiency**，但可能犧牲 **meta-training efficiency**。

傳統 meta-RL 方法多數依賴 on-policy data，例如：

- MAML
- RL²
- ProMP
- MAESN
- SNAIL 類方法

on-policy 的問題是每次更新都需要目前 policy 新產生的資料，舊資料不能像 off-policy 方法那樣大量重用。這使得 meta-training 非常昂貴。

作者接著指出，把 off-policy data 用到 meta-RL 並不簡單。因為 meta-learning 有一個基本假設：

> meta-training 時的 adaptation data distribution 應該和 meta-test 時相似。

在 few-shot supervised learning 中，如果測試時會看到 5 個 examples，訓練時也應該模擬 5-shot setting。類似地，在 meta-RL 中，測試時 agent 面對新任務會用自己當下 policy 收集 on-policy trajectories 來適應；如果訓練時 encoder 卻用大量舊 replay buffer 的 off-policy transitions 來適應，就會發生 distribution mismatch。

這是 off-policy meta-RL 的第一個技術難點。

---

### Introduction 段落 3：本文目標是 efficient off-policy meta-RL

第三段明確提出本文研究目標：設計一個 efficient off-policy meta-RL algorithm。

作者的解法是把 **online inference of probabilistic context variables** 和 **off-policy RL algorithms** 整合。

這裡有兩個關鍵詞：

1. **probabilistic context variables**：用一個機率分布表示目前任務的 latent context，而不是單一 deterministic embedding。
2. **online inference**：隨著 agent 在新任務上收集更多 transition，逐步更新對任務的 belief。

作者強調快速適應需要 reasoning about distributions。因為一開始 agent 不知道任務是什麼，它需要隨機探索可能有 reward 的區域，同時逐漸縮小任務不確定性。

---

### Introduction 段落 4：PEARL 在 meta-test 時如何探索與適應

第四段描述 PEARL 的 meta-test 行為。

在 meta-training 時，PEARL 學一個 probabilistic encoder，將過去 experience 壓縮成 context variable。到了 meta-test，新任務來了，agent 會從 latent context distribution sample 一個 \(z\)，並在一整個 episode 中固定這個 \(z\)。

這樣做的好處是可以產生 **temporally extended exploration**。

一般 action noise 是每一步都隨機抖動，探索不連貫。PEARL 則是先 sample 一個「任務假設」\(z\)，然後整條 trajectory 都根據這個假設行動。這會產生更有結構的探索。

收集完 trajectory 後，PEARL 把資料加入 context，重新更新 posterior。作者把這件事描述成：

> agent samples task hypotheses, attempts those tasks, and evaluates whether the hypotheses were correct.

也就是：抽一個任務假設，照這個假設去做，然後根據結果判斷假設是否合理。

---

### Introduction 段落 5：task inference 和 action/control 解耦，使 off-policy learning 變可行

第五段說明 PEARL 最重要的設計之一：

> Disentangling task inference from action makes the approach amenable to off-policy meta-learning.

意思是，PEARL 不把「如何辨識任務」和「如何控制」全部塞進同一個 recurrent policy。它把它們拆開：

- encoder 負責根據 context 推論 \(z\)。
- actor/critic 負責在給定 \(z\) 的情況下控制。

這樣 actor 和 critic 可以用 off-policy data 訓練，而 encoder 可以使用較接近 meta-test 的 context data 訓練，減少 distribution mismatch。

這是 PEARL 能同時保有 sample efficiency 和 adaptation ability 的關鍵。

---

### Introduction 段落 6：主要貢獻

這段總結本文貢獻：提出 **PEARL**。PEARL 具備三個特色：

1. meta-training sample efficient：因為使用 off-policy RL。
2. fast adaptation：因為可以 online accumulate context。
3. structured exploration：因為 latent context 是 probabilistic，可以做 posterior sampling。

實驗上，PEARL 在六個 continuous control meta-learning environments 上超過 prior algorithms，並在 sample efficiency 上有 20–100 倍提升。

---

## 3. Related Work 逐段解釋

### 3.1 Meta-learning 段落 1：Meta-RL 的背景

作者先把本文放入 meta-learning 框架。Meta-learning 的核心是「learning to learn」：不是只學某一個任務，而是學會如何從少量資料快速學新任務。

在 RL 中，meta-RL 可以分成幾種方向：

1. meta-learn dynamics model：學一個可以快速適應新 dynamics 的 model。
2. meta-learn policy：直接學一個能快速適應新任務的 policy。

本文屬於第二類：policy-based meta-RL，但它額外引入 probabilistic context inference。

---

### 3.1 Meta-learning 段落 2：Context-based meta-RL

這段介紹 recurrent 或 recursive meta-RL 方法，例如 RL² 和 SNAIL。

這些方法通常把過去經驗丟進 RNN 或 attention 模型，讓 policy 根據歷史資料改變行為。作者稱這類方法為 **context-based meta-RL**，因為它們都使用過去 experience 作為 task-specific context。

PEARL 也屬於 context-based meta-RL，但有兩個差異：

1. PEARL 的 context 是 **probabilistic latent variable**，可以表示 task uncertainty。
2. PEARL 不用 recurrence，而是使用 permutation-invariant encoder。

作者認為，如果 MDP 是 fully observed，那一組 transition 的順序不應該影響任務辨識。任務本身由 transition/reward 結構決定，而不是由資料被觀察到的順序決定。因此 PEARL 使用 unordered set encoder，而不是 RNN。

---

### 3.1 Meta-learning 段落 3：RNN off-policy 的困難

作者提到，過去有人嘗試把 recurrent Q-function 和 off-policy Q-learning 結合，但多數是在較簡單的任務或離散環境中。PEARL 自己的實驗也顯示，直接把 recurrent policy 加到 off-policy learning 中並不容易。

這裡暗示 PEARL 的設計動機：

> 與其讓 RNN 同時負責記憶、任務推論、控制，不如把 task inference 明確拆成 encoder，再把 control 交給 actor-critic。

---

### 3.1 Meta-learning 段落 4：Gradient-based meta-RL

這段介紹 MAML、ProMP 等 gradient-based meta-RL。

這些方法的共同點是：在新任務上透過 gradient descent 更新 policy parameters。它們多數依賴 policy gradient，因此通常是 on-policy。

PEARL 與它們不同：PEARL 不在 meta-test 做 gradient update，而是透過 context inference 更新 posterior over \(z\)。

作者強調，除了 sample efficiency，PEARL 在實驗中也有更高 asymptotic performance。也就是說，它不只是學得比較省資料，也可能學到更好的最終策略。

---

### 3.1 Meta-learning 段落 5：Few-shot supervised learning 的啟發

作者提到 few-shot supervised learning 中的 embedding-based 方法，例如 Prototypical Networks。

Prototypical Networks 將 support examples 映射到 embedding space，再用距離做分類。PEARL 也受到這種 embedding function 啟發，但差異是：

- Prototypical Networks 的 embedding 通常是 deterministic。
- PEARL 的 embedding 是 probabilistic。
- Prototypical Networks 用於分類。
- PEARL 的 embedding 用於 conditioning RL policy。

---

### 3.2 Probabilistic meta-learning 段落

這段說明 probabilistic models 在 meta-learning 中已有先例。例如 hierarchical Bayesian models、probabilistic MAML、latent task variables 等。

PEARL 的創新不是「第一次用 latent variable 做 meta-learning」，而是把 probabilistic latent task inference 放進 **off-policy meta-RL**，並讓它支援 posterior sampling exploration。

作者也比較 MAESN。MAESN 也使用 latent variables 做 structured exploration，但它是 on-policy gradient-based meta-learning，而且 test-time adaptation 是透過 gradient descent 調整 latent variables。PEARL 則是透過 amortized inference 直接從 context 推論 posterior。

---

### 3.3 Posterior sampling 段落

Posterior sampling 在 classical RL 中是指：維護一個可能 MDP 的 posterior，每次 sample 一個 MDP，然後根據這個 MDP 的 optimal policy 行動。

這種方法可以產生 deep exploration。因為 agent 不是每一步隨機，而是在整個 episode 中根據同一個 sampled hypothesis 做行動。

PEARL 可以看成 posterior sampling 的 meta-learned 版本。它不直接維護 MDP posterior，而是維護 latent task variable posterior：

\[
q_\phi(z \mid c)
\]

這個 posterior 由 meta-training 學出來，因此比 classical Bayesian RL 更容易用在高維 neural RL setting。

---

### 3.4 Partially observed MDPs 段落

作者指出 meta-RL 可以被視為 POMDP 問題。因為任務本身是 hidden state 的一部分：agent 看得到環境 state，但看不到目前 task identity、goal、reward function 或 dynamics parameters。

因此，agent 需要根據過去 observation、action、reward 形成 belief over task。

PEARL 的 \(q_\phi(z \mid c)\) 就是這個 belief 的 approximation。

與一般 POMDP 方法不同，PEARL 利用 meta-learning 的結構：任務分布在 meta-training 時可以被學習，因此 encoder 可以學會如何從少量 context 推論任務。

---

## 4. Problem Statement 逐段解釋

### Problem Statement 段落 1：本文問題設定

作者正式定義問題：agent 有一個 task distribution \(p(\mathcal{T})\)。每個 task 都是一個 MDP。

一個 task \(\mathcal{T}\) 包含：

\[
\mathcal{T} = \{p(s_0), p(s_{t+1}\mid s_t,a_t), r(s_t,a_t)\}
\]

其中：

- \(p(s_0)\)：初始狀態分布。
- \(p(s_{t+1}\mid s_t,a_t)\)：transition dynamics。
- \(r(s_t,a_t)\)：reward function。

這個設定允許不同 task 之間有不同 reward function，也允許不同 dynamics。例如：

- goal location 不同：reward function 不同。
- robot mass/friction 不同：transition dynamics 不同。

---

### Problem Statement 段落 2：context 的定義

作者定義 context \(c\) 為目前在任務上收集到的 transition history。

單一 transition 是：

\[
c_n^\mathcal{T} = (s_n, a_n, r_n, s'_n)
\]

一組 context 是：

\[
c_{1:N}^\mathcal{T}
\]

這表示 agent 到目前為止在任務 \(\mathcal{T}\) 上收集到的經驗。

meta-training 時，agent 從 training tasks 中學習如何使用 context 來調整行為。meta-test 時，面對從同一 task distribution 抽出的 unseen task，agent 必須根據少量 context 快速適應。

---

## 5. Section 4：Probabilistic Latent Context 總覽

這一節是 PEARL 的方法核心。作者引入 latent probabilistic context variable \(Z\)，用它表示目前任務中對控制有用的資訊。

policy 被寫成：

\[
\pi_\theta(a \mid s,z)
\]

也就是 policy 的行為會因 \(z\) 不同而改變。

Meta-training 要學兩件事：

1. 如何從 context 推論 \(z\)：

\[
q_\phi(z\mid c)
\]

2. 如何在給定 \(z\) 的條件下控制：

\[
\pi_\theta(a\mid s,z), \qquad Q_\theta(s,a,z)
\]

---

## 6. Section 4.1：Modeling and Learning Latent Contexts 逐段解釋

### 4.1 段落 1：latent context 必須包含任務關鍵資訊

作者指出，為了讓 agent 能適應，latent context \(Z\) 必須 encode task-relevant information。也就是說，\(z\) 不能只是任意 embedding，它必須包含會影響策略選擇的資訊。

例如：

- Half-Cheetah-Vel 中，\(z\) 應該包含 target velocity。
- Ant-Goal-2D 中，\(z\) 應該包含 goal location。
- Walker-2D-Params 中，\(z\) 應該包含 dynamics parameter 的相關資訊。

PEARL 使用 amortized variational inference 來學 \(q_\phi(z\mid c)\)，使其近似真正 posterior \(p(z\mid c)\)。

---

### 4.1 段落 2：variational lower bound 與 information bottleneck

作者提出一個概念性的 variational objective：

\[
\mathbb{E}_{\mathcal{T}}\left[\mathbb{E}_{z\sim q_\phi(z\mid c^\mathcal{T})}\left[R(\mathcal{T},z)+\beta D_{KL}(q_\phi(z\mid c^\mathcal{T})\|p(z))\right]\right]
\]

這裡的重點不是公式符號本身，而是它的意義。

- \(R(\mathcal{T},z)\)：表示 \(z\) 在任務 \(\mathcal{T}\) 上是否有用。它可以是 reward、reconstruction likelihood、value function objective 等。
- \(D_{KL}(q_\phi(z\mid c)\|p(z))\)：限制 posterior 不要離 prior 太遠。
- \(\beta\)：控制 bottleneck 強度。

KL term 的作用可以理解為 information bottleneck。它限制 \(z\) 不要攜帶 context 裡所有細節，只保留對任務適應有必要的資訊。

這很重要，因為如果 \(z\) 太自由，encoder 可能會記住 training tasks 的偶然細節，導致 overfitting。

---

### 4.1 段落 3：inference network 架構設計

作者接著討論 \(q_\phi(z\mid c)\) 的架構。理想上，encoder 要足夠 expressive，可以捕捉 task-relevant sufficient statistics，但又不能建模不必要的 dependencies。

作者提出一個重要觀察：

> 如果 MDP 是 fully observed，要辨識任務時，一組 transitions 的順序不重要。

例如你要判斷某個 goal 在哪裡，看到 transitions A、B、C 的順序不應該影響你對 goal 的判斷。真正重要的是這些 transitions 的集合內容。

因此 PEARL 使用 permutation-invariant encoder，而不是 RNN。

---

### 4.1 段落 4：product of independent factors

PEARL 將 posterior 建模成多個 transition factor 的乘積：

\[
q_\phi(z\mid c_{1:N}) \propto \prod_{n=1}^{N}\Psi_\phi(z\mid c_n)
\]

每個 transition \(c_n\) 都提供一個關於 \(z\) 的 evidence factor。把這些 evidence 相乘，就得到整體 posterior。

直觀上：

- 一筆 transition 只能提供一點任務資訊。
- 多筆 transition 結合後，posterior 會更集中。
- context 越多，對任務的 belief 越準確。

這個設計也自然符合 Bayesian updating 的直覺。

---

### 4.1 段落 5：Gaussian factors

為了讓方法可計算，PEARL 使用 Gaussian factors：

\[
\Psi_\phi(z\mid c_n)=\mathcal{N}(f_\phi^\mu(c_n), f_\phi^\sigma(c_n))
\]

每個 transition 經過 neural network \(f_\phi\)，輸出一個 Gaussian 的 mean 和 variance。

多個 Gaussian factors 相乘後，仍然可以得到 Gaussian posterior。這使得 sampling 和 optimization 都比較簡單。

Figure 1 展示這個架構：每個 transition 都經過共享的 encoder network \(\phi\)，輸出一個 factor，最後這些 factors 相乘得到 \(q_\phi(z\mid c)\)。

---

## 7. Section 4.2：Posterior Sampling and Exploration via Latent Contexts 逐段解釋

### 4.2 段落 1：probabilistic context 讓 posterior sampling 成為可能

這段是 PEARL 的探索機制核心。

因為 \(z\) 不是 deterministic vector，而是一個 posterior distribution，所以 PEARL 可以從 \(q_\phi(z\mid c)\) sample 不同的 \(z\)，並讓 policy 根據這些 \(z\) 行動。

在 classical posterior sampling RL 中，agent 維護一個可能 MDP 的 posterior，每次抽一個 MDP，然後在整個 episode 中根據這個 MDP 的 optimal policy 行動。

PEARL 類似，但它抽的是 latent context \(z\)，不是完整 MDP。

---

### 4.2 段落 2：temporally extended exploration

作者強調 posterior sampling 可以產生 temporally extended exploration。

一般 exploration noise 往往是 step-wise 的：每一步 action 加一點噪音。這種探索很容易不連貫。

PEARL 的方式是：

1. 先 sample 一個 \(z\)。
2. 在整個 episode 中固定這個 \(z\)。
3. policy 根據同一個任務假設持續行動。

因此，agent 會產生一整條連貫 trajectory。例如在 navigation 任務中，如果 sample 到的 \(z\) 對應「goal 可能在左上角」，agent 就會整條 trajectory 往左上角探索，而不是每一步亂走。

---

### 4.2 段落 3：PEARL 和 Bootstrapped DQN 類方法的關係

作者提到，在 single-task deep RL 中，也有人用 bootstrap 方式近似 value function posterior，例如 Bootstrapped DQN。這類方法透過維持多個 Q-functions 來產生 deep exploration。

PEARL 不維持多個 value functions，而是直接推論 latent context posterior。這個 \(z\) 可以代表：

- MDP itself，如果用 reconstruction 訓練。
- optimal behavior，如果用 policy objective 訓練。
- value-function-relevant task information，如果用 critic objective 訓練。

PEARL 實作中最終選擇用 critic objective 訓練 encoder，因此 \(z\) 主要學到對 Q-value 預測有用的 task information。

---

### 4.2 段落 4：meta-test 時的行為

在 meta-test，一開始沒有 context：

\[
c=\emptyset
\]

因此 agent 從 prior sample：

\[
z\sim r(z)
\]

然後執行一個 episode。這代表 agent 根據 training tasks 中學到的 task prior 抽一個任務假設。

收集到 trajectory 後，加入 context，更新 posterior：

\[
q_\phi(z\mid c)
\]

之後再從 posterior sample 新的 \(z\)，繼續探索或利用。

隨著 context 增加，posterior 會越來越集中，policy 的行為也會越來越接近正確任務。

---

## 8. Section 5：Off-Policy Meta-Reinforcement Learning 逐段解釋

### Section 5 段落 1：為什麼要 off-policy meta-RL？

作者指出，probabilistic context 可以和 on-policy policy gradient 結合，但本文真正目標是 off-policy meta-RL。

原因是：on-policy meta-RL 的 meta-training sample efficiency 太差。MAML、RL²、ProMP、MAESN 等方法在每次更新時都需要新資料，訓練成本很高。

PEARL 希望使用 replay buffer 重複利用過去資料，像 SAC 那樣高效率訓練 actor 和 critic。

---

### Section 5 段落 2：off-policy meta-RL 的第一個困難：distribution mismatch

作者再次強調 meta-learning 的 train-test matching 問題。

meta-test 時，agent 面對新任務只能用自己 policy 收集到的 on-policy data 來 adaptation。如果 meta-training 時 encoder 用的是 replay buffer 裡很舊、很 off-policy 的資料，那 encoder 學到的 inference procedure 可能不適用於測試時的資料。

所以 off-policy meta-RL 不能簡單地把所有 replay buffer 資料都拿來訓練 context encoder。

---

### Section 5 段落 3：off-policy RL 不直接學探索分布

作者指出另一個困難：off-policy TD learning 通常是最小化 temporal-difference error，並不直接優化「在新任務中該如何探索」。

Meta-RL 的困難不只是學控制，還要學 exploration strategy。Policy gradient 方法因為直接根據 on-policy trajectory 更新，比較自然能學探索行為；但 off-policy value-based 方法不一定能學到好的 task-identifying exploration。

這也是為什麼 naive off-policy meta-RL 可能失敗。

---

### Section 5 段落 4：PEARL 的關鍵洞察

PEARL 的核心洞察是：

> 訓練 encoder 的 context data 不需要和訓練 actor/critic 的 RL batch 相同。

也就是：

- actor/critic 可以使用整個 replay buffer 中的 off-policy transitions。
- encoder 的 context data 可以用比較接近 on-policy 的 recent data。

這樣同時滿足：

1. actor/critic 訓練 sample efficient。
2. encoder 訓練不會和 meta-test context distribution 差太遠。

作者定義一個 context sampler \(S_C\)，專門負責抽 encoder context batch。

---

### Section 5 段落 5：recent replay buffer 的折衷策略

如果 context 完全從整個 replay buffer 抽，distribution mismatch 太嚴重。

如果 context 完全 on-policy，又會失去 off-policy efficiency。

PEARL 採用中間策略：從最近收集的資料中抽 context。這樣 context 比較接近目前 policy 的資料分布，又可以重複使用最近資料。

Figure 2 展示這個流程：

1. replay buffer 存各 training tasks 的資料。
2. \(S_C\) 抽 context batch 給 encoder。
3. encoder 推論 \(q_\phi(z\mid c)\)。
4. sample \(z\)。
5. actor 和 critic 接收 \(z\)。
6. encoder 透過 critic loss 與 KL bottleneck 更新。

---

## 9. Algorithm 1：PEARL Meta-training 詳解

Algorithm 1 是 PEARL 的訓練流程。它可以分成兩個大階段：資料收集與模型更新。

### Step 1：每個 training task 有自己的 replay buffer

對每個 training task \(T_i\)，建立 replay buffer \(B^i\)。

這樣做的原因是 context 必須屬於同一個 task。如果把不同 task 的 transitions 混在一起，encoder 會得到矛盾訊號。

---

### Step 2：對每個 task 初始化 context

對每個 task，先令：

\[
c^i=\emptyset
\]

這代表 agent 一開始還不知道這個 task 是什麼。

---

### Step 3：根據目前 context sample \(z\)

在每次資料收集前：

\[
z\sim q_\phi(z\mid c^i)
\]

如果 context 是空的，就相當於從 prior sample。這一步會產生一個任務假設。

---

### Step 4：用 conditioned policy 收集資料

使用：

\[
\pi_\theta(a\mid s,z)
\]

在 task \(T_i\) 中 rollout，收集 transitions 並放入 \(B^i\)。

---

### Step 5：更新 context

從 task replay buffer 中抽取 transitions 更新 context：

\[
c^i = \{(s_j,a_j,s'_j,r_j)\}_{j=1}^{N}\sim B^i
\]

這讓下次 sample \(z\) 時能使用更多任務資訊。

---

### Step 6：訓練更新時分別抽 context batch 與 RL batch

在 optimization loop 中，對每個 task：

\[
c^i\sim S_C(B^i)
\]

\[
b^i\sim B^i
\]

其中：

- \(c^i\)：給 encoder 推論 \(z\)。
- \(b^i\)：給 actor/critic 計算 RL loss。

這個 decoupling 是 PEARL 最重要的設計之一。

---

### Step 7：計算 losses

對每個 task 計算：

1. actor loss：

\[
L^i_{actor}=L_{actor}(b^i,z)
\]

2. critic loss：

\[
L^i_{critic}=L_{critic}(b^i,z)
\]

3. KL loss：

\[
L^i_{KL}=\beta D_{KL}(q(z\mid c^i)\|r(z))
\]

KL loss 用來限制 posterior 不要過度偏離 prior，避免 \(z\) 過度記憶 context。

---

### Step 8：更新參數

Algorithm 1 中 encoder 參數 \(\phi\) 由 critic loss 和 KL loss 更新：

\[
\phi \leftarrow \phi - \alpha_1 \nabla_\phi \sum_i (L^i_{critic}+L^i_{KL})
\]

actor 參數 \(\theta_\pi\) 由 actor loss 更新：

\[
\theta_\pi \leftarrow \theta_\pi - \alpha_2 \nabla_\theta \sum_i L^i_{actor}
\]

critic 參數 \(\theta_Q\) 由 critic loss 更新：

\[
\theta_Q \leftarrow \theta_Q - \alpha_3 \nabla_\theta \sum_i L^i_{critic}
\]

這裡可以看出，PEARL 的 encoder 不是直接用 return 最大化訓練，而是主要透過 critic loss 學習對 value function 有用的 task representation。

---

## 10. Algorithm 2：PEARL Meta-testing 詳解

Algorithm 2 描述 PEARL 在新任務上的 adaptation。

### Step 1：初始化 context

新任務 \(T\) 來了，令：

\[
c^T=\emptyset
\]

---

### Step 2：sample latent context

對每次 episode：

\[
z\sim q_\phi(z\mid c^T)
\]

第一個 episode 沒有 context，因此從 prior sample。後續 episode 則根據累積 context sample posterior。

---

### Step 3：rollout policy

用：

\[
\pi_\theta(a\mid s,z)
\]

收集一條 trajectory：

\[
D_k^T=\{(s_j,a_j,s'_j,r_j)\}_{j=1}^{N}
\]

---

### Step 4：累積 context

把新 trajectory 加入 context：

\[
c^T \leftarrow c^T \cup D_k^T
\]

PEARL 在 meta-test 不做 gradient update，所有 adaptation 都透過更新 context posterior 完成。

---

## 11. Section 5.1：Implementation 詳解

### 5.1 段落 1：PEARL 基於 SAC

PEARL 建立在 Soft Actor-Critic 上。SAC 是一種 off-policy actor-critic 方法，使用 maximum entropy RL objective。

SAC 不只最大化 expected return，也鼓勵 policy entropy：

\[
\sum_t r(s_t,a_t)+\alpha \mathcal{H}(\pi(\cdot\mid s_t))
\]

這讓 policy 保持 stochastic，通常能提升探索與訓練穩定性。

PEARL 修改 SAC 的方式是把 \(z\) 加入 actor 和 critic：

\[
\pi_\theta(a\mid s,z)
\]

\[
Q_\theta(s,a,z)
\]

---

### 5.1 段落 2：joint optimization 與 reparameterization trick

PEARL 同時優化：

- encoder \(q_\phi(z\mid c)\)
- actor \(\pi_\theta(a\mid s,z)\)
- critic \(Q_\theta(s,a,z)\)

因為 \(z\) 是 sample 出來的 latent variable，所以需要 reparameterization trick 讓梯度能回傳到 encoder。

如果 \(q_\phi(z\mid c)\) 是 Gaussian，可以寫成：

\[
z = \mu_\phi(c) + \sigma_\phi(c)\epsilon, \qquad \epsilon\sim \mathcal{N}(0,I)
\]

這樣 sampling 操作可微分，encoder 可以用 gradient-based optimization 訓練。

---

### 5.1 段落 3：encoder 用 critic loss 訓練

作者實驗後發現，用 Bellman critic update 訓練 encoder，比以下方式更好：

1. 直接最大化 actor return。
2. 重建 state 和 reward。

原因是 critic loss 要求 \(z\) 幫助預測 \(Q(s,a,z)\)，這會迫使 \(z\) 保留對控制決策有用的任務資訊。

例如，在 Half-Cheetah-Vel 中，如果 \(z\) 不包含 target velocity，critic 很難正確估計某個速度下 action 的價值。

---

### 5.1 段落 4：critic loss 公式

PEARL 的 critic loss 是：

\[
L_{critic}=\mathbb{E}_{(s,a,r,s')\sim B,\ z\sim q_\phi(z\mid c)}\left[Q_\theta(s,a,z)-\left(r+\bar{V}(s',\bar{z})\right)\right]^2
\]

其中：

- \(B\)：replay buffer。
- \(z\sim q_\phi(z\mid c)\)：從 context posterior sample latent variable。
- \(\bar{V}\)：target value network。
- \(\bar{z}\)：stop-gradient 的 latent variable。

這個 loss 要求 critic 在給定 \(z\) 時，符合 Bellman target。

---

### 5.1 段落 5：actor loss 公式

actor loss 與 SAC 類似：

\[
L_{actor}=\mathbb{E}_{s\sim B,a\sim\pi_\theta,z\sim q_\phi(z\mid c)}\left[D_{KL}\left(\pi_\theta(a\mid s,\bar{z})\middle\|\frac{\exp(Q_\theta(s,a,\bar{z}))}{Z_\theta(s)}\right)\right]
\]

這表示 policy 要接近由 Q-value 定義的 Boltzmann distribution。Q 值越高的 action，policy 應該給越高機率。

這裡 \(\bar{z}\) 表示 actor loss 不直接更新 encoder。encoder 主要透過 critic loss 和 KL loss 更新。

---

### 5.1 段落 6：context batch 與 RL batch 的具體取樣

作者最後說明實作細節：

- context sampler \(S_C\) 從最近收集的資料中 uniform sample context。
- 最近資料每 1000 個 meta-training optimization steps 更新一次。
- actor/critic 的 RL batch 則從整個 replay buffer uniform sample。

這再次體現 PEARL 的折衷：

- context 不要太 off-policy，以免測試時不適用。
- actor/critic 可以完全 off-policy，以提升 sample efficiency。

---

## 12. Section 6：Experiments 總覽

實驗要回答三個問題：

1. PEARL 是否比 prior meta-RL methods 更 sample efficient？
2. probabilistic context + posterior sampling 是否真的改善 sparse reward exploration？
3. PEARL 的設計選擇是否必要？

因此實驗分成三節：

- 6.1：sample efficiency and performance。
- 6.2：posterior sampling for exploration。
- 6.3：ablations。

---

## 13. Section 6.1：Sample Efficiency and Performance 逐段解釋

### 6.1 段落 1：實驗環境

作者使用六個 MuJoCo continuous control meta-RL benchmarks：

1. Half-Cheetah-Fwd-Back
2. Half-Cheetah-Vel
3. Humanoid-Direc-2D
4. Ant-Fwd-Back
5. Ant-Goal-2D
6. Walker-2D-Params

這些任務可以分成兩類：

### Reward function varies

任務主要差異在 reward function。例如：

- 要往前或往後跑。
- 要達到某個 target velocity。
- 要走到某個 goal location。

### Dynamics varies

Walker-2D-Params 中，系統 dynamics parameters 會變化，例如身體參數不同。這測試 agent 是否能適應不同 dynamics。

---

### 6.1 段落 2：比較方法

作者比較 PEARL 與：

- ProMP
- MAML-TRPO
- RL² with PPO

這些都是當時重要的 on-policy meta-RL baselines。

作者也提到嘗試 recurrent DDPG，但沒有得到合理結果。他們推測原因包括：

1. adaptation data distribution mismatch。
2. recurrent model 需要處理 trajectory-level 訓練，較不穩定。
3. RNN 同時負責 task inference 和 control，負擔太大。

PEARL 則透過 decoupling task inference and control 避免這些問題。

---

### 6.1 段落 3：meta-test 評估方式

評估時，PEARL 在 test task 上做 trajectory-level adaptation。

第一條 trajectory：

\[
z\sim r(z)
\]

也就是從 prior sample。

後續 trajectories：

\[
z\sim q_\phi(z\mid c)
\]

其中 context 是前面收集到的 trajectories。

最終 test-time performance 是在收集兩條 trajectories 作為 context 後，計算後續 trajectory 的 average return。

---

### 6.1 段落 4：Figure 3 結果

Figure 3 顯示 test-task performance 隨 meta-training samples 增加的變化。

主要結果：

1. PEARL 在六個任務中都比 prior methods 更 sample efficient。
2. PEARL 在 meta-training sample efficiency 上提升約 20–100 倍。
3. PEARL 在五個六個 domains 中 final asymptotic performance 提升約 50–100%。
4. 即使與調得更好的 RL²-PPO 比，PEARL 仍然有明顯優勢。

這個圖是 PEARL 最主要的實驗證據。

---

## 14. Section 6.2：Posterior Sampling for Exploration 逐段解釋

### 6.2 段落 1：為什麼要測 sparse reward exploration？

這一節要驗證 PEARL 的 probabilistic context 是否真的幫助探索。

在 sparse reward MDP 中，agent 只有在到達特定區域時才會得到 reward。如果 exploration 沒有結構，很難碰到 reward。

PEARL 的 posterior sampling 應該能幫助 agent 做 coherent hypothesis-driven exploration。

---

### 6.2 段落 2：2D navigation 任務設定

任務是一個 2D point robot，要走到半圓邊緣上的 goal。

- training tasks：100 個 random goals。
- testing tasks：100 個 unseen goals。
- reward：只有進入 goal radius 內才給。
- 測試兩種 radius：0.2 與 0.8。

radius 0.2 更 sparse、更困難；radius 0.8 比較容易。

---

### 6.2 段落 3：meta-training 使用 dense reward 的原因

作者提到，雖然目標是測 sparse reward adaptation，但從零開始在許多 sparse reward tasks 上 meta-training 很困難。因此他們在 meta-training 使用 dense reward，meta-test 使用 sparse reward。

這是一個實驗設計上的折衷。

這代表 PEARL 在訓練時學會不同 goal 的結構，測試時則要在 sparse feedback 下快速找出 unseen goal。

---

### 6.2 段落 4：與 MAESN 比較

MAESN 是一個 prior method，也使用 latent variables 做 structured exploration。但 MAESN 是 on-policy gradient-based meta-learning。

結果顯示：

- PEARL 平均約 5 條 trajectories 後開始有效適應。
- PEARL final performance 比 MAESN 高。
- PEARL meta-training 約用 \(10^6\) timesteps。
- MAESN 約用 \(10^8\) timesteps。

這代表 PEARL 不只 adaptation 效果較好，meta-training 也更省樣本。

---

### 6.2 段落 5：Figure 4 解讀

Figure 4 上方小圖展示 PEARL 的探索軌跡。

一開始從 prior sample \(z\)，agent 會往不同可能 goal 方向探索。收集到 trajectory 後，posterior 更新，agent 逐漸排除錯誤假設，最後往真正 goal 移動。

曲線顯示 PEARL 的 return 隨 test-time adaptation trajectories 增加而快速上升，明顯優於 MAESN。

---

## 15. Section 6.3：Ablations 逐段解釋

Ablation studies 用來回答：PEARL 的設計是否真的必要？作者測了三個部分：

1. encoder architecture。
2. context sampling strategy。
3. probabilistic vs deterministic latent context。

---

## 16. Ablation 1：Inference Network Architecture

### 段落 1：為什麼比較 RNN encoder？

PEARL 使用 permutation-invariant encoder。作者想知道，如果改成傳統 RNN encoder 會怎樣。

RNN 是很多 meta-RL 方法常用的 context encoder，因為它可以處理 trajectory history。但作者認為 fully observed MDP 中，transition order 對任務辨識不一定必要。

---

### 段落 2：RNN ablation 設定

作者保留 PEARL 其他設計，只改 encoder architecture。

他們測兩種 RL batch sampling：

1. **RNN tran**：RL batch 仍使用 unordered transitions。
2. **RNN traj**：RL batch 使用 full trajectories。

---

### 段落 3：Figure 5 結果

Figure 5 顯示 Half-Cheetah-Vel 上的結果。

主要觀察：

- RNN tran 可以接近 PEARL，但 optimization 比較慢。
- RNN traj 表現明顯下降。

這代表：

1. permutation-invariant encoder 不是唯一可行選項，但更有效率。
2. off-policy actor-critic 訓練中，decorrelated transitions 很重要。
3. 用 full trajectories 當 RL batch 會讓訓練變得更難。

---

## 17. Ablation 2：Data Sampling Strategies

### 段落 1：原始 PEARL 的 sampling 策略

PEARL 的 context sampler \(S_C\) 有兩個特點：

1. 從最近收集的 transitions 抽 context。
2. context batch 和 RL batch 分開抽。

這樣可以兼顧 context distribution matching 和 off-policy efficiency。

---

### 段落 2：兩個替代策略

作者測試兩個替代方案：

1. **off-policy**：context 從整個 replay buffer 抽，但不同於 RL batch。
2. **off-policy RL-batch**：直接用同一個 RL batch 當 context。

---

### 段落 3：Figure 6 結果

Figure 6 顯示：

- context 完全從整個 replay buffer 抽會顯著降低 performance。
- 使用同一個 RL batch 當 context 有時比完全 off-policy context 好，可能是因為 context 和 critic target 有 correlation，讓學習比較簡單。
- 但兩者都不如原始 PEARL。

這證明 off-policy meta-RL 的資料取樣策略非常關鍵。不能簡單把所有東西都從 replay buffer 抽。

---

## 18. Ablation 3：Deterministic Context

### 段落 1：為什麼測 deterministic context？

PEARL 的核心假設之一是 probabilistic context 對 exploration 很重要。作者測試如果把 posterior distribution 改成 deterministic point estimate，會發生什麼。

也就是把：

\[
z\sim q_\phi(z\mid c)
\]

改成：

\[
z=f_\phi(c)
\]

---

### 段落 2：為什麼 deterministic context 在 sparse reward 中會失敗？

如果沒有 context，一個 deterministic encoder 只能輸出固定的 \(z\)。這代表每次新任務第一條 trajectory 都做類似行為，缺乏多樣化 task hypotheses。

在 sparse reward 任務中，這會很嚴重。因為如果第一種探索方向沒有碰到 reward，agent 沒有好的不確定性機制去嘗試其他方向。

probabilistic PEARL 則可以從 prior sample 不同 \(z\)，產生不同探索方向。

---

### 段落 3：Figure 7 結果

Figure 7 顯示 deterministic PEARL 在 sparse 2D navigation 中表現很差，而 probabilistic PEARL 可以有效解任務。

這證明 probabilistic latent context 不是附加功能，而是 PEARL 在 sparse reward exploration 中成功的核心。

---

## 19. Section 7：Conclusion 逐段解釋

### Conclusion 段落 1：方法總結

作者總結 PEARL 是一個新的 meta-RL algorithm，它透過 latent context variable inference 進行 adaptation。

policy conditioned on latent context：

\[
\pi_\theta(a\mid s,z)
\]

而 latent context 由 encoder 從 experience 推論：

\[
q_\phi(z\mid c)
\]

---

### Conclusion 段落 2：為什麼 PEARL 適合 off-policy RL？

因為 PEARL 解耦 task inference 和 task solving。

- task inference：encoder 負責。
- task solving/control：actor-critic 負責。

這讓 actor/critic 可以使用 off-policy replay buffer 訓練，提升 meta-training sample efficiency；同時 encoder 的 context distribution 可以控制得比較接近 meta-test。

---

### Conclusion 段落 3：probabilistic context 的價值

作者強調 probabilistic context 讓 posterior sampling 成為可能，進而產生 temporally extended exploration。

這對 sparse reward 和未知任務尤其重要。

---

### Conclusion 段落 4：實驗結論

PEARL 在多個 continuous control meta-RL domains 上超越 prior meta-RL algorithms，並且使用更少經驗。

整體來說，PEARL 同時改善：

1. meta-training sample efficiency。
2. adaptation efficiency。
3. structured exploration。

---

## 20. Appendix A：Experimental Details 解釋

Appendix A 補充主要實驗的完整時間尺度圖與各環境定義。

### Figure 8：Continuous control tasks

Figure 8 顯示四種 MuJoCo agent：

1. Half-Cheetah
2. Humanoid
3. Ant
4. Walker

這些是實驗中的主要 continuous control agents。

---

### Figure 9：完整時間尺度結果

主文 Figure 3 為了凸顯 PEARL 的收斂速度，x-axis 被截斷。Appendix Figure 9 則顯示完整 \(10^8\) timesteps 時間尺度，並使用 log scale。

這張圖更清楚顯示：

- PEARL 在很少 samples 下就達到高 performance。
- MAML、ProMP、RL² 需要非常多 samples 才逐漸追上。
- 在許多任務中，即使 baseline 訓練很久，也未必達到 PEARL 表現。

---

### 各 benchmark 任務定義

Appendix 列出各 meta-learning domains：

1. **Half-Cheetah-Dir**：向前或向後移動，共 2 個 tasks。
2. **Half-Cheetah-Vel**：達到指定 forward velocity，100 train tasks，30 test tasks。
3. **Humanoid-Dir-2D**：在 2D grid 上朝指定方向跑，100 train tasks，30 test tasks。
4. **Ant-Fwd-Back**：向前或向後移動，共 2 個 tasks。
5. **Ant-Goal-2D**：導航到 2D grid 上指定 goal，100 train tasks，30 test tasks。
6. **Walker-2D-Params**：system dynamics parameters 隨機化，40 train tasks，10 test tasks。

這些任務覆蓋 reward variation 和 dynamics variation，因此能測 PEARL 是否真的能從 context 推論不同類型的 task differences。

---

## 21. 三個核心公式整理

### 21.1 Context posterior

\[
q_\phi(z\mid c_{1:N}) \propto \prod_{n=1}^{N}\Psi_\phi(z\mid c_n)
\]

意義：每個 transition 都提供一個任務證據，多個證據相乘得到 posterior。

---

### 21.2 Critic loss

\[
L_{critic}=\mathbb{E}\left[Q_\theta(s,a,z)-\left(r+\bar{V}(s',\bar{z})\right)\right]^2
\]

意義：critic 在給定 task latent \(z\) 下學習 Bellman consistency。encoder 也透過這個 loss 學會產生對控制有用的 \(z\)。

---

### 21.3 KL bottleneck

\[
L_{KL}=\beta D_{KL}(q_\phi(z\mid c)\|r(z))
\]

意義：限制 posterior 不要攜帶過多 context 資訊，避免 overfitting，也讓 \(z\) 保持合理的不確定性。

---

## 22. PEARL 與 MAML、RL² 的差異總結

| 面向 | MAML | RL² | PEARL |
|---|---|---|---|
| adaptation 方式 | gradient update | RNN hidden state | posterior inference over \(z\) |
| meta-test 是否更新參數 | 是 | 否 | 否 |
| task information 存在哪裡 | policy parameters update | recurrent hidden state | latent context variable |
| 是否 probabilistic | 通常否 | 通常否 | 是 |
| 訓練方式 | on-policy meta-policy gradient | on-policy recurrent RL | off-policy SAC + encoder |
| sample efficiency | 較低 | 較低 | 較高 |
| exploration 機制 | gradient adaptation 後改善 | recurrent memory-based exploration | posterior sampling |
| 主要優化內容 | 好微調的初始化 | 內隱學習演算法 | task inference + off-policy control |

---

## 23. 從實作角度理解 PEARL

如果你要自己實作一個簡化版 PEARL，可以把它拆成以下模組：

### 23.1 Environment task sampler

負責 sample 不同 tasks，例如不同 goal location 或 target velocity。

### 23.2 Replay buffer per task

每個 task 一個 replay buffer，避免 context 混淆。

### 23.3 Context encoder

輸入 transition：

\[
(s,a,r,s')
\]

輸出 Gaussian factor 的 mean 和 variance。

### 23.4 Product aggregation

把多個 transition factor 聚合成 posterior：

\[
q_\phi(z\mid c)
\]

### 23.5 Actor

輸入 state 和 \(z\)，輸出 action distribution：

\[
\pi_\theta(a\mid s,z)
\]

### 23.6 Critic

輸入 state、action、\(z\)，輸出 Q-value：

\[
Q_\theta(s,a,z)
\]

### 23.7 Training loop

每輪訓練包含：

1. 對每個 task 收集資料。
2. 從 recent data 抽 context。
3. 從 replay buffer 抽 RL batch。
4. sample \(z\)。
5. 更新 critic。
6. 更新 actor。
7. 更新 encoder。
8. 更新 target network。

---

## 24. 這篇論文最重要的研究觀念

### 24.1 Meta-RL 不只是快速學習，也要考慮 meta-training cost

早期 meta-RL 很重視 adaptation speed，但較少處理 meta-training sample efficiency。PEARL 指出，如果 meta-training 需要 \(10^8\) environment steps，方法實務上仍然很難用。

---

### 24.2 Task inference 和 control 可以拆開

MAML 把 adaptation 寫成 gradient update。RL² 把 adaptation 放進 recurrent hidden state。PEARL 則明確拆成 encoder 和 actor-critic。

這種 modularization 讓 off-policy learning 更容易。

---

### 24.3 Uncertainty 對 exploration 很重要

如果 agent 不知道任務是什麼，它不應該只輸出單一 deterministic embedding。它需要表示多種可能任務，並透過 sampling 做假設測試。

這是 PEARL 在 sparse reward 任務中勝出的原因。

---

### 24.4 Context distribution matching 是 off-policy meta-RL 的核心難題

off-policy replay 很省資料，但 context encoder 如果看到的資料分布和 meta-test 時差太多，就會失敗。

PEARL 的解法是 actor/critic off-policy，encoder context recent-on-policy-like。

---

## 25. 如果你要接續做實驗，應該觀察什麼？

若你要比較 MAML、RL²、PEARL，可以觀察以下指標：

1. **Adaptation curve**：test return 隨 context/adaptation trajectories 增加的變化。
2. **Meta-training sample efficiency**：達到某個 test return 需要多少 environment steps。
3. **K=0 performance**：沒有 adaptation 時的 prior policy 表現。
4. **K=1 improvement**：一條 trajectory 後提升多少。
5. **Sparse reward exploration**：是否能在沒有 dense feedback 時找到任務線索。
6. **Variance over seeds**：meta-RL 訓練通常高 variance，必須多 seed。

---

## 26. 最後總結

PEARL 的貢獻可以濃縮成一句話：

> PEARL 用 probabilistic latent context 做 task inference，並把它和 off-policy actor-critic 結合，使 meta-RL 同時具備快速適應、有效探索與高 sample efficiency。

它相對於 MAML 和 RL² 的核心差異是：

- 不透過 gradient update adaptation。
- 不依賴 RNN hidden state 做全部記憶。
- 使用顯式 probabilistic posterior over task latent variable。
- 使用 off-policy SAC 重複利用資料。
- 透過 posterior sampling 做 structured exploration。

因此，這篇論文是理解 modern off-policy meta-RL、latent task inference、probabilistic exploration 的重要基礎。

