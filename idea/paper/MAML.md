---
type: paper-note
aliases:
  - "MAML"
  - "Model-Agnostic Meta-Learning"
year: 2017
stage: "1-meta-rl骨架"
tags:
  - meta-learning
  - gradient-based
  - few-shot
  - baseline對照
  - 1-meta-rl骨架
summary: "用 bilevel（inner適應/outer學初始化）讓模型對新任務幾步gradient就能適應；gradient-based，當few-shot上界對照。"
---
> - **論文標題**：Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)
> - **作者**：Chelsea Finn, Pieter Abbeel, Sergey Levine（UC Berkeley / OpenAI）
> - **出處 / 年份**：ICML 2017（arXiv:1703.03400）
> - **主題**：meta-learning、few-shot learning、gradient-based meta-learning、model-agnostic、適用於 supervised classification / regression / RL
> - **整理目標**：弄清楚 MAML 為什麼能透過「優化一組好初始參數」達成 fast adaptation；釐清 task distribution、inner / outer loop 的角色；判斷它對「robot play + 弱語言 + offline meta-RL + zero-shot task embedding」這個主線可以提供哪些零件與 baseline 設計概念。

---

## 0. 閱讀總覽（白話版）

MAML 嘗試解這樣一件事：給你一堆「相關但不同」的任務，希望模型在看到一個全新任務的少量資料後，只用幾步 gradient descent 就把該任務做好。它的核心想法非常乾淨——不引入額外的 meta-learner 模組、不要求特定的網路架構，只是去找一組「初始參數 θ」，使得從 θ 出發、在新任務上做一次（或數次）梯度更新後，得到的參數對該任務的損失最小。

換言之，MAML 並不是教模型「怎麼學習」，而是教模型「站在哪裡比較好開始學」。它把「能快速被微調」本身當成優化目標：對 task distribution \(p(\mathcal{T})\) 內的所有任務，模型 fine-tune 後的損失越低越好。這個視角讓 MAML 與「learn-an-update-rule」、「learn-a-metric-space」這類方法分道揚鑣，並且因為通用，能無痛接到 classification、regression、policy gradient RL 上。

對研究主線而言，MAML 是 gradient-based meta-learning 的奠基之作，是任何 offline meta-RL / zero-shot 工作中無法迴避的對照組與思想原點：「task embedding」這種顯式 latent 方案，正是相對於 MAML「隱式把任務資訊寫進梯度更新」這條路的另一極端。

---

## 1. Abstract 逐段精讀

**核心宣告**：提出一種 *model-agnostic* 的 meta-learning 演算法，意思是只要模型是用 gradient descent 訓練的，這個方法就可以套上去。適用範圍涵蓋 classification、regression、reinforcement learning。

**目標設定**：meta-learning 想做的是——在多種任務上訓練模型，使其面對「新任務」時，只用「少量樣本」即可解決。

**方法直覺**：作者直接對「模型的初始參數」動手腳，要求這組初始參數有一個良好性質：對一個新任務，做「少量的 gradient steps」、用「少量資料」進行更新後，模型在該任務上能有好的 generalization。換句話說，它訓練出來的模型是「容易被 fine-tune 的」。

**結果預告**：在 few-shot image classification 兩個 benchmark（Omniglot、MiniImagenet）達到 SOTA；在 few-shot regression 表現良好；在 policy gradient RL 中能加速 fine-tune。

---

## 2. Introduction 重點整理

- 人類能從少量例子學會新事物，希望 agent 也能。挑戰在於：要結合既有 prior experience 與少量新資訊，又不可 overfit 新資料。
- 由於任務形式多樣，meta-learning 機制必須是 *task-agnostic*——也就是要對任務的具體形式不挑食。
- 既有方法可分幾類：
  - 學一個 update function / learning rule（Schmidhuber 1987, Andrychowicz 2016, Ravi & Larochelle 2017）。
  - 用 recurrent / memory-augmented 模型把整個資料集吃進去（Santoro 2016, Duan 2016b）。
  - 學一個 metric space + nonparametric 分類（Matching nets, Siamese nets）。
- MAML 的差別：
  - 不引入額外的可學參數（learner 本身的參數就是 meta 的全部）。
  - 不對模型架構有限制（FCN、CNN、RNN 通通可以）。
  - 不限制 loss 的形式（supervised 可微 loss 或 RL 不可微 reward 都行）。
- 兩個重要的觀點：
  - **Feature learning view**：若 internal representation 對多任務都通用，那麼只 fine-tune 一點就能適應新任務。MAML 顯式優化這種「容易被 fine-tune 的特徵」。
  - **Dynamical system view**：MAML 等價於 *最大化新任務的 loss 對參數的敏感度*——θ 處於這樣一個位置：任何任務的 loss 在該點的梯度方向都能帶來大幅改善。

- 貢獻：一個簡單、與模型 / 任務無關、訓練成本可控的 meta-learning 演算法，並在 regression、classification、RL 三種域驗證。

---

## 3. 方法與核心公式

### 3.1 問題設定（task distribution）

定義一個 **task distribution** \(p(\mathcal{T})\)。每個任務 \(\mathcal{T}_i\) 是一個四元組：

\[
\mathcal{T}_i = \{ \mathcal{L}_{\mathcal{T}_i}(x_1, a_1, \dots, x_H, a_H),\; q_i(x_1),\; q_i(x_{t+1} \mid x_t, a_t),\; H \}
\]

各符號意義：

- \(\mathcal{L}_{\mathcal{T}_i}\)：該任務的 loss（supervised 是預測誤差；RL 是負 reward）。
- \(q_i(x_1)\)：初始 observation 的分布。
- \(q_i(x_{t+1}\mid x_t, a_t)\)：轉移機率（MDP dynamics）。
- \(H\)：episode 長度。supervised 設定下 \(H=1\)。

模型 \(f_\theta: x \to a\) 是被 meta-train 的對象。**K-shot learning** 指的是模型在新任務上只能看到 \(K\) 個樣本。

> 為什麼這個 task distribution 重要：MAML 的 generalization 能力完全建立在「meta-train 的任務」與「meta-test 的任務」來自同一個 \(p(\mathcal{T})\) 上。\(p(\mathcal{T})\) 越廣、越涵蓋 test 任務的變因（amplitude、phase、goal velocity、character class…），越能期待 fast adaptation 成功。

### 3.2 Inner loop：對單一任務做一步（或幾步）gradient

對採樣到的任務 \(\mathcal{T}_i\)，用其上的 \(K\) 個樣本算出 task-specific 的 loss \(\mathcal{L}_{\mathcal{T}_i}(f_\theta)\)，再用一步 SGD 更新：

\[
\theta_i' = \theta - \alpha \, \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)
\]

- \(\alpha\)：inner-loop 學習率，可固定或 meta-learn。
- \(\theta_i'\)：「為任務 \(i\) 量身打造」的 adapted parameters。
- 注意：這裡的 K 是 fine-tune 用的 support set 大小。
- 可擴展成多步：\(\theta_i' = \theta - \alpha \sum_{k=0}^{K-1} \nabla \mathcal{L}(\theta^{(k)})\)，公式形式類似，只是要 chain rule 多走幾層。

### 3.3 Outer loop：meta-objective

MAML 想要 \(\theta_i'\)（fine-tune 後的參數）對新樣本有好的表現，所以在「新樣本 \(D_i'\)」上算 loss，並對 \(\theta\)（而非 \(\theta_i'\)）做梯度更新：

\[
\min_\theta \;\; \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\!\bigl( f_{\theta_i'} \bigr)
= \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\!\bigl( f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)} \bigr)
\]

對應的 outer-loop 更新（meta-update）：

\[
\theta \leftarrow \theta - \beta \, \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}( f_{\theta_i'} )
\]

- \(\beta\)：outer-loop 學習率（meta step size）。
- 這個梯度是「梯度的梯度」：因為 \(\theta_i'\) 是 \(\theta\) 的函數，對它再做一次微分會出現 Hessian-vector products。
- 實作上，現代 autodiff 框架（TensorFlow / PyTorch）可以自動處理這個二階導數。

### 3.4 為什麼「可快速微調」本身就是一個可學習目標？

這正是 MAML 的核心哲學，可從三個角度理解：

1. **損失敏感度（loss sensitivity）視角**：MAML 等價於在 task distribution 上最大化 loss 對參數的敏感度——θ 落在一個對所有任務「梯度方向都很有效」的位置。在這個位置上，一個小步長就能換來大改善。
2. **特徵學習視角**：若參數的某些子集編碼了 task-general features（如 sine wave 的週期性、字符筆畫的形狀統計），那只要 fine-tune 一些 task-specific 的部分（例如最後一層）就能適應新任務。MAML 顯式優化「這種容易 fine-tune 的結構」。
3. **訓練分布視角**：標準訓練只看「模型在現有任務上的 loss」；MAML 看的是「fine-tune 之後的 loss」，所以它優化的是 *post-update behavior*，而不是 *pre-update behavior*。這是把「fast adaptation」這個元能力，直接寫進目標函數的關鍵手段。

### 3.5 Supervised 版本的 loss

回歸（MSE）：

\[
\mathcal{L}_{\mathcal{T}_i}(f_\phi) = \sum_{(x^{(j)}, y^{(j)}) \sim \mathcal{T}_i} \| f_\phi(x^{(j)}) - y^{(j)} \|_2^2
\]

分類（cross-entropy，論文寫成 binary 形式，多類別則對應 softmax + cross-entropy）：

\[
\mathcal{L}_{\mathcal{T}_i}(f_\phi) = \sum_{(x^{(j)}, y^{(j)}) \sim \mathcal{T}_i} y^{(j)} \log f_\phi(x^{(j)}) + (1-y^{(j)}) \log(1 - f_\phi(x^{(j)}))
\]

N-way K-shot 分類設定：每個 class 提供 K 個 input/output 對，共 \(N K\) 個資料點。

### 3.6 RL 版本的 loss

對 RL 任務，loss 是負 expected return：

\[
\mathcal{L}_{\mathcal{T}_i}(f_\phi) = - \mathbb{E}_{x_t, a_t \sim f_\phi, q_{\mathcal{T}_i}} \left[ \sum_{t=1}^H R_i(x_t, a_t) \right]
\]

由於 dynamics 未知，這個期望對 \(\phi\) 不可微，所以用 policy gradient（REINFORCE）估計梯度。又因為 policy gradient 是 on-policy 的，每做一次 inner-loop 更新都需要從 \(f_{\theta_i'}\) 重新採 trajectory。Meta-optimizer 用 TRPO；為避免出現三階導數，論文用 finite differences 來算 Hessian-vector products。

---

## 4. 演算法流程

### 4.1 通用版（Algorithm 1）

1. 隨機初始化 θ。
2. 重複：
   - Sample 一批 task \(\mathcal{T}_i \sim p(\mathcal{T})\)。
   - 對每個 \(\mathcal{T}_i\)：
     - 用 K 個樣本計算 \(\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)\)。
     - inner-loop 更新：\(\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)\)。
   - outer-loop 更新：\(\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})\)。

### 4.2 Supervised 版（Algorithm 2）

差異點：

- inner-loop 用 support set \(D\)（K 個 (x, y) 對）算 loss。
- outer-loop 改用「另一批 query set \(D_i'\)」算 \(\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})\)，再對 θ 做梯度——這就是 MAML 之所以需要 support / query 切分的原因。

### 4.3 RL 版（Algorithm 3）

差異點：

- inner-loop 從環境採 K 條 trajectory \(D = \{(x_1, a_1, \dots, x_H)\}\)，policy gradient 估出梯度後 update。
- outer-loop 用 *fine-tune 後的 policy* \(f_{\theta_i'}\) 再採新一批 trajectory \(D_i'\)，計算 post-update loss，再做 outer-update。
- 實作上會搭配 baseline（state-dependent baseline）、TRPO 等改良。

### 4.4 一階近似（First-Order MAML, FOMAML）

二階 Hessian 計算成本高。FOMAML 直接在 outer-loop 把二階項丟掉——只用 \(\nabla_{\theta'} \mathcal{L}(f_{\theta'})\) 代替 \(\nabla_\theta \mathcal{L}(f_{\theta'})\)。實驗顯示在 MiniImagenet 上效果與完整 MAML 幾乎相同，且訓練速度提升約 33%。作者推測這與 ReLU 網路在局部近乎線性（二階導數近 0）有關。

---

## 5. 實驗設計與結論

### 5.1 Regression（sinusoid）

- **Task distribution**：sine wave 的 amplitude 在 \([0.1, 5.0]\)、phase 在 \([0, \pi]\)；輸入 \(x \in [-5, 5]\)；輸出 1 維。
- **模型**：兩層 MLP，hidden size 40，ReLU。
- **MAML 超參**：inner-loop 一步、α=0.01、K=10；outer 用 Adam。
- **Baseline**：(a) 在所有任務上 pre-train、再 fine-tune；(b) 給出 amplitude/phase 的 oracle。
- **結果**：MAML 從 K=5 個點即可恢復 sine 波形，並能從「半邊輸入區間」的少量樣本，推斷出另一半的形狀，顯示它真的學到了「sine 的週期結構」。Pre-training baseline 因為任務間目標互相矛盾（同一個 x 對應不同 y），fine-tune 容易 overfit。
- **延伸觀察**：雖然 MAML 只訓練「一步更新後」的表現，但 test 時做更多 gradient steps 仍持續改善——表示它找到的 θ 處於「對 fast adaptation 友善的區域」，並非只在第一步好。

### 5.2 Classification（Omniglot、MiniImagenet）

- **Omniglot**：1623 字符 × 20 instance × 50 alphabets。Train 用 1200 字符，剩下測試。資料 augmentation 用 90° 倍數旋轉。
- **MiniImagenet**：64 train / 12 val / 24 test classes。
- **架構**：模仿 Matching Networks 的 4-block CNN（3×3 conv、64 filters、batch norm、ReLU、2×2 max-pool）。Omniglot 用 strided conv 取代 max-pool；MiniImagenet 用 32 filters 降低 overfit。
- **MAML 超參（見附錄 A.1）**：
  - Omniglot 5-way：1 step α=0.4、meta batch=32；test 時 3 steps。
  - Omniglot 20-way：5 steps α=0.1、meta batch=16。
  - MiniImagenet：5 steps α=0.01、test 10 steps、meta batch=4 (1-shot) / 2 (5-shot)。
- **結果**（節錄）：
  - Omniglot 5-way 1-shot：MAML 98.7%、5-shot：99.9%（與 Matching nets、Memory module 同等或更好）。
  - MiniImagenet 5-way 1-shot：MAML 48.7%、5-shot：63.1%。優於 Matching nets、Meta-learner LSTM。
  - FOMAML（first-order approx.）幾乎等同 full MAML。
- **意義**：在 few-shot classification 達到 SOTA，且不需引入專屬於分類的結構（不像 metric learning / memory networks）。

### 5.3 Reinforcement Learning

- **環境**：rllab benchmark + MuJoCo。
- **架構**：兩層 MLP policy，hidden=100，ReLU。
- **演算法**：inner-loop 用 vanilla policy gradient (REINFORCE)，outer-loop 用 TRPO。為避免三階導數，用 finite differences 算 Hessian-vector product。Baseline 用 linear feature baseline。
- **任務集合**：
  - **2D Navigation**：unit square 上隨機 goal；H=100；用 20 條 trajectory 做一步 update。
  - **HalfCheetah / Ant goal velocity**：reward = -|當前速度 - 目標速度|；速度 0–2（cheetah）或 0–3（ant）。
  - **HalfCheetah / Ant goal direction**：reward = ±forward velocity，方向隨機。
  - H=200，每步 20 條 rollout（ant fwd/bwd 用 40）。
- **超參**：α=0.1（首步 inner），test 時第二步起改 α=0.05；meta batch size 2D=20、locomotion=40；最多 500 個 meta-iteration。
- **結果**：MAML 在 1–3 個 gradient steps 內就能適應新方向 / 新速度，明顯優於 pretraining 與 random init。在某些任務上 pretrain 甚至比 random 還差（multi-task interference）。

### 5.4 附錄補充對照

附錄 C 加入兩種對照：

- **Multi-task in parameter space**：對 500 個任務各別訓練後，把參數平均，再 fine-tune。MSE 約 2.7~7.18，明顯比 MAML（0.35–0.67）差。
- **Context vector adaptation (Rei 2015 style)**：把可學習向量 z 接在輸入上、只更新 z。在簡單 pointmass 表現不錯，但在 Omniglot、HalfCheetah 表現明顯較差。這對應 MAML 「全參數都可被 inner-loop 更新」的彈性帶來的優勢。

### 5.5 主要結論

- 「初始參數即 meta-knowledge」這條路在三大領域都可行。
- 「post-update loss」這個 meta-objective 真的會驅使模型學到 task-general representation。
- 一階近似已經能拿到大部分性能，計算成本可控。
- pretraining 並非 fast adaptation 的好對照——它常因任務間衝突，在某些 RL 任務反而比 random 還差。

---

## 6. 與相關工作的關係

- **Learn-an-update-rule 系**（Schmidhuber 1987、Hochreiter 2001、Andrychowicz 2016、Ravi & Larochelle 2017）：學一個 RNN-like 的「優化器」。MAML 不另設 optimizer 參數，仍用標準 gradient descent。
- **Memory-augmented / RNN meta-learner**（Santoro 2016 MANN、Duan 2016b [[RL2|RL²]]、Wang 2016 LearningToRL）：把整段 task 經驗餵進 RNN，靠 hidden state 暗含 task identity。MAML 在 few-shot 分類上勝出，且不對架構做限制。
- **Metric-based**（Siamese, Matching Networks, Prototypical Networks）：為分類量身打造，難搬到 regression / RL。MAML 是 task-agnostic 的。
- **Pretraining + fine-tune**（Donahue 2014 DeCAF）：MAML 把「快速 fine-tune」顯式寫進 objective，而 pretraining 沒有；經驗上 pretrain 在 task-variability 高的 RL 中常常失敗。
- **Sensitivity / initialization**（Saxe 2014, Krähenbühl 2016 等）：MAML 也可視為一種顯式的「為 fine-tune 而初始化」，但它是 data-dependent + task-distribution-aware 的。

---

## 7. 附錄重點：實作 / 超參數速查

### Classification（A.1）

- N-way K-shot：每個 gradient 用 N·K 個樣本。
- Omniglot 5-way：1 inner step、α=0.4、meta batch=32 task；evaluate 用 3 steps。
- Omniglot 20-way：5 inner steps、α=0.1、meta batch=16。
- MiniImagenet：5 inner steps、α=0.01、test 用 10 steps；每 class 15 個 example 拿來算 post-update meta-gradient；meta batch=4 (1-shot) / 2 (5-shot)；訓練 60000 iter on 1 Pascal Titan X。

### RL（A.2）

- inner-loop α=0.1（首步），test 時後續步 α=0.05。
- meta batch：2D navigation 20、locomotion 40。
- 最多 500 個 meta-iteration，挑訓練期間平均 return 最佳的 model 評估。
- Ant goal velocity 加 per-step bonus 防止 agent 提早結束 episode。

### 多任務對照表（附錄 C，sinusoid 5-shot MSE）

| 方法 | 1 step | 5 step | 10 step |
| --- | --- | --- | --- |
| Multi-task, no reg | 4.19 | 3.85 | 3.69 |
| Multi-task, ℓ2 reg | 7.18 | 5.69 | 5.60 |
| Multi-task, reg to mean θ | 2.91 | 2.72 | 2.71 |
| Pretrain on all tasks | 2.41 | 2.23 | 2.19 |
| **MAML** | **0.67** | **0.38** | **0.35** |

可看到「在參數空間做平均」（multi-task）效果普遍比「在輸出空間做平均」（pretrain）還差，而 MAML 比兩者都好一個量級。

---

## (A) 與本研究主線的關聯

我的主線是：**robot play data + 極少量弱語言標註 + offline meta-RL + 把文字轉成 task specification / task embedding 做 zero-shot 泛化**，並同時關注 meta-learning × zero-shot 的交叉。

MAML 對這條主線的啟發、可借用零件、以及它能擔任的角色：

1. **作為 gradient-based meta-RL 的 baseline**
   - 即使主線會走 task embedding / latent inference 那條（[[PEARL|PEARL]]、[[VariBAD|VariBAD]]），MAML 仍是「沒有顯式 task representation」的最強對照組。任何主張「文字 embedding 帶來 zero-shot 優勢」的方法，都應該與「MAML 從 play 中學 task-general init + 在新任務做 few-step fine-tune」直接比較。
   - 可拿來做 ablation：(i) MAML（不看語言）vs. (ii) MAML + language-conditioned policy（語言當輸入）vs. (iii) embedding-based zero-shot（完全不微調）。差距會直接顯示「語言 zero-shot」相對於「梯度 few-shot」的價值。

2. **Inner / outer loop 概念可直接搬到 offline meta-RL**
   - MAML 的 inner-loop = 對單一任務 fine-tune；outer-loop = 對 task distribution 做 meta-optimization。
   - 在 offline meta-RL 設定下，inner-loop 不能再用環境互動，需改成 *offline policy improvement*（如 [[IQL|IQL]]、[[CQL|CQL]]、AWR 的 actor update）來模擬 fast adaptation；outer-loop 對 \(\theta\) 求梯度的概念則保留。
   - 這指向一個具體的設計分支：「offline MAML-style fine-tune」vs.「context encoder-style zero-shot」。

3. **「可快速微調」當作 latent / 語言 embedding 的另一面**
   - MAML 的 θ 編碼了 task distribution 的「結構先驗」（隱式 task knowledge）。我的主線想把這份結構先驗顯式化為文字 / latent task embedding。
   - 這帶出一個值得實驗的問題：當把語言當作 task 描述輸入時，「我們真的需要 inner-loop 微調嗎？」MAML-style 可作為 lower bound（語言完全不用、靠梯度）；純 zero-shot via language embedding 是 upper-bound 目標。中間段（如 MAML + language conditioning）是合理的混合策略。

4. **Task distribution 設計直接對應 play data 抽取**
   - MAML 高度依賴一個良好的 \(p(\mathcal{T})\)。在 robot play 中，「task」可由 segmentation / 弱語言標註自動構建（每個語言短句對應一段 play trajectory，視為一個 task）。
   - 這給「如何把無結構的 play 切成 task distribution」一個具體的格式：每個 task 必須是「同一意圖、可以給 reward / supervision signal」的片段，這也與我們關心的弱語言標註剛好契合。

5. **FOMAML 的工程價值**
   - 在 offline meta-RL + 大 transformer policy 的情境下，二階導數成本爆炸。FOMAML 證明「丟掉二階項」幾乎不傷效能，是後續做 large-scale offline meta-RL 時的重要工程簡化技巧。

6. **可借用的零件**
   - Algorithm 2/3 的 support / query 切分結構：直接適用於我們對 play data 切片成「demo / evaluation」的 schema。
   - 「inner step 用一個學習率、test 時 halve」這個小技巧（A.2）可以拿來穩定 offline adaptation。
   - Sensitivity-maximization 的視角，可用來理解「為什麼語言 embedding 做 task conditioning 時，要把它注入 attention 而非 layer norm bias」之類的設計選擇。

7. **MAML 不直接做 zero-shot，這正是落差**
   - MAML 永遠需要至少一次 inner-loop 更新；而我們追求「看到語言，就能在新任務上直接跑」。所以 MAML 是「few-shot baseline」，不是「zero-shot 同類方法」——但凡寫論文，這個差距必須明白寫清楚，並用實驗量化（0 step = MAML 的初始 θ 是否已可零樣本 work；vs. 我們的方法在 0 step 已可 work）。

---

## (B) 一句話總結

MAML 證明：只要把「fine-tune 後的損失」直接當成 meta-objective，就能用普通的 gradient descent 學到一組「對任務分佈通用、能在幾步梯度內快速適應新任務」的初始參數——這讓「快速學會學習」變成一個可被 SGD 優化的明確目標，而且完全與模型架構解耦。
