---
type: reading-list
aliases:
  - "read"
  - "閱讀清單"
tags:
  - 閱讀清單
  - MOC
summary: "六階段論文閱讀清單（meta-RL 骨架 → offline RL → offline meta-RL → play/弱語言 → zero-shot/generalist → benchmark）。"
---
# 第一階段：先建立 Meta-Learning / Meta-RL 骨架

這一階段的目標是：  
你要先知道「Meta-RL 到底在 meta 什麼」。

## 1) MAML

**Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**。這篇是 meta-learning 的經典起點：核心思想是讓模型在多個任務上訓練後，對新任務只需少量資料與少量 gradient steps 就能快速適應。它不是專門做 RL，但它奠定了後面很多 meta-RL 的思路。

你讀這篇時要抓的重點：

- task distribution 是什麼
- inner loop / outer loop 是什麼
- 為什麼「可快速微調」本身就是一個可學習目標

## 2) RL²

**RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning**。這篇很重要，因為它把「快速學習器」本身塞進 RNN hidden state，讓 agent 在多個 episode 中自己學 adaptation。這會讓你明白：meta-RL 不只是學 policy，而是學「如何利用過去經驗快速推斷新任務」。

你要抓：

- 為什麼 hidden state 可以被解讀成 fast learner
- 為什麼 meta-RL 會自然碰到 exploration / exploitation
- 這條線和 MAML 式 gradient-based adaptation 有什麼差別

## 3) PEARL

**Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables**。這篇對你後面最有用，因為它把 meta-RL 明確寫成「從 context 推斷 latent task variable」。你之後要做 task embedding、task inference、語言作為 task supervision，都會一直回到這篇。

你要抓：

- context 是什麼
- latent task variable 是什麼
- 為什麼「任務推斷」可以跟「控制」拆開

## 4) VariBAD

**A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning**。這篇把不確定性直接納入決策，讓 agent 在未知環境中做近似 Bayes-optimal exploration。它對你未來理解「task ambiguity」「belief inference」很有幫助。

你要抓：

- belief / uncertainty 在 meta-RL 裡扮演什麼角色
- 為什麼 meta-RL 不只是 adaptation，也包含 inference
- 它和 PEARL 的 latent task 表示有什麼異同

---

# 第二階段：補 Offline RL 基礎

這一階段的目標是：  
你要先知道「為什麼離線學習這麼難」，不然後面 offline meta-RL 會看不懂。

## 5) Offline RL 綜述

**Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems**。這篇是入門 offline RL 最適合的總覽。它系統整理了 offline RL 的核心難點：資料是固定的、不能再互動、策略一旦跑到資料分布外就容易出事。

你要抓：

- distribution shift
- extrapolation error / OOD actions
- 為什麼 offline RL 不是把 online RL 直接拿來用

## 6) D4RL

**Datasets for Deep Data-Driven Reinforcement Learning**。這篇是 benchmark 基礎。它讓 offline RL 變成一個可比較、可重現的研究問題。

你要抓：

- 為什麼 offline RL 需要專門的 benchmark
- dataset quality / coverage 為什麼重要
- benchmark 設計如何影響方法比較

## 7) CQL

**Conservative Q-Learning for Offline Reinforcement Learning**。這篇是典型保守派做法：因為資料外動作的 Q-value 很容易被高估，所以要把 Q 函數估得更保守。

你要抓：

- 為什麼 Q overestimation 在 offline RL 特別嚴重
- conservative learning 的核心直覺
- 它適合什麼型的資料分布

## 8) IQL

**Offline Reinforcement Learning with Implicit Q-Learning**。這篇很值得讀，因為它走的不是 CQL 那條線，而是盡量避免直接評估資料外動作。它對後續你接 robot offline data 很有實際參考價值。

你要抓：

- expectile regression 在做什麼
- 為什麼它可以不用 explicit OOD action evaluation
- 為什麼 IQL 在實作上常被拿來當強 baseline

## 9) Decision Transformer

**Decision Transformer: Reinforcement Learning via Sequence Modeling**。這篇把 RL 重寫成 sequence modeling 問題。對你之後要接 transformer、language-conditioned decision making、trajectory tokenization 都很重要。

你要抓：

- 為什麼 RL 可以寫成 autoregressive modeling
- return-to-go conditioning 的角色
- 這條線和 value-based offline RL 的差別

---

# 第三階段：進入 Offline Meta-RL

這一階段是你題目的理論主幹。  
你的研究很可能最終會落在這區。

## 10) MACAW

**Offline Meta-Reinforcement Learning with Advantage Weighting**。這篇幾乎可以視為 offline meta-RL 的起點論文之一。它明確定義了：在固定離線多任務資料上 meta-train，對新任務只用極少量資料做 adaptation。

你要抓：

- offline meta-RL 和 offline RL 差在哪
- 為什麼 test task 只有極少量資料
- inner adaptation 在離線 setting 為什麼特別難

## 11) Offline Meta-RL with Online Self-Supervision

這篇很重要，因為它指出：meta-RL 在 meta-test 時會收集新 context，但這些 context 的分布通常跟離線資料不同，於是會出現新的 distribution shift。

你要抓：

- adaptation phase 收集到的新資料為什麼會破壞離線假設
- online self-supervision 想補的是哪個洞
- 這和純 offline 的設定有什麼張力

## 12) CORRO

**Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning**。這篇對你很關鍵，因為它在處理一個你未來一定會碰到的問題：task representation 容易被 behavior policy 汙染。

你要抓：

- 為什麼同一 task 的資料仍可能因 behavior policy 不同而長得不一樣
- contrastive learning 怎麼幫 task representation 去除 policy 影響
- 它和語言 supervision 可不可以互補

## 13) CSRO

**Context Shift Reduction for Offline Meta-Reinforcement Learning**。這篇是你題目最值得反覆讀的論文之一。它把 offline meta-RL 的關鍵問題寫得很清楚：訓練時 context 來自 behavior policy，測試時 context 來自 exploration policy，兩者分布不同，導致 task inference 失真。

你要抓：

- context shift 的正式定義
- 為什麼它不是普通的 distribution shift
- 你未來若用語言 supervision，能不能幫忙降低這個 shift

## 14) Information-Theoretic Framework of Context-Based Offline Meta-RL

這篇適合在你前面幾篇看完後讀，因為它是統整型視角：用 information-theoretic 的框架重看 context-based offline meta-RL。

你要抓：

- context encoder 到底在最大化或保留什麼資訊
- 哪些方法其實只是不同形式的 representation learning
- 這個框架能不能幫你設計新的弱語言 supervision objective

## 15) T2DA

**Text-to-Decision Agent: Offline Meta-Reinforcement Learning from Natural Language Supervision**。這篇是你現在選題最直接的近鄰論文之一，因為它明確把自然語言拿來做 offline meta-RL 的 supervision。

你要抓：

- 它的 text supervision 是怎麼進模型的
- 它是在替代 target-task demonstrations，還是在輔助 task inference
- 它和你想做的「弱語言標註 + robot play」還差哪一步

---

# 第四階段：Robot Play Data 與弱語言標註

這一階段是你題目最具機器人味的部分。

## 16) Learning Latent Plans from Play

這篇是 play data 線的經典。它說明為什麼 play 比 task demos 更便宜、更多樣，而且資料覆蓋通常更大。

你要抓：

- play data 的三個特性：便宜、豐富、無需明確任務分段
- 為什麼 latent plan 是合理的中介表示
- 為什麼從 play 學出來的行為會比較有 retry / recovery 性質

## 17) Language Conditioned Imitation Learning over Unstructured Data

這篇很重要，因為它已經在做「無結構資料 + 語言 + imitation」的結合，而且明確說能使用 unlabeled / unstructured demonstrations。

你要抓：

- shared latent goal space 是什麼
- unstructured data 怎麼被利用
- 它還不是 meta-RL；那和你要做的方向差在哪

## 18) Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation

這篇非常實用，因為它走的是比較工程上可落地的路：用 offline robot data 加 crowd-sourced language labels，學 language-conditioned reward，再接 offline multi-task RL。

你要抓：

- 為什麼它不是直接學 text-to-action
- language-conditioned reward 的好處和限制
- crowd-sourced annotation 跟 hindsight annotation 有什麼不同

## 19) From Play to Policy

**Conditional Behavior Generation from Uncurated Robot Data**。這篇是「從雜亂 play data 擠出 task-centric behavior」的重要代表。

你要抓：

- uncurated play data 最大的麻煩是什麼
- multi-modal generation 為什麼重要
- 它怎麼把 play data 變成可用 policy，而不是只做 representation

## 20) PlayFusion

**Skill Acquisition via Diffusion from Language-Annotated Play**。這篇很接近你題目，因為它明確是 language-annotated play，而且用 hindsight language 標註與 diffusion model 去抽技能。

你要抓：

- hindsight language annotation 的角色
- diffusion 為什麼比單純 BC 更適合 noisy multi-modal play
- discrete skill bottleneck 在這裡扮演什麼作用

## 21) LUMOS

**Language-Conditioned Imitation Learning with World Models**。這篇是你一定要精讀的。它從 unstructured play data 學、只用很少量 hindsight language annotations、測試時能接受語言指令，並用 world model 緩解 policy-induced distribution shift。

你要抓：

- 為什麼 world model 能幫忙減少 covariate shift
- 少量語言標註為什麼足夠
- 它和 T2DA 的主要差異：一個偏 robot imitation / world model，一個偏 offline meta-RL

---

# 第五階段：Zero-shot Task Specification 與 Generalist Robot Policy

這一階段是幫你把題目接到大模型與 generalist policy 的趨勢。

## 22) Can Foundation Models Perform Zero-Shot Task Specification for Robot Manipulation?

這篇直接對應你題目裡的「zero-shot 任務規格化」。它研究低成本任務指定方式，例如語言、網路圖片、草圖等，能不能被 foundation model 轉成機器人可執行的任務指定。

你要抓：

- task specification 和 task execution 的差別
- 為什麼「給一張目標圖」不一定是最好的人機介面
- zero-shot task spec 能不能做成結構化中介表示

## 23) RT-1

**Robotics Transformer for Real-World Control at Scale**。這篇幫你建立 generalist robot policy 的基本印象：大量真實機器人資料、語言條件控制、強調規模與泛化。

你要抓：

- 為什麼資料規模在 robotics 也開始變重要
- RT-1 是怎麼把 instruction、image、action 一起編碼的
- 它和你要做的 offline meta-RL 差在哪：它偏 generalist policy，不特別強調 task inference 理論

## 24) RT-2

**Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**。這篇的關鍵不是「又一個機器人模型」，而是它展示了 web-scale vision-language knowledge 可以遷移到 robot control。

你要抓：

- 為什麼 VLM / LLM 知識能幫 robot semantics
- action tokenization 的意義
- 它比較像 VLA；你題目若接它，應該把新意放在 task inference，不是重做 RT-2

## 25) Open X-Embodiment

**Robotic Learning Datasets and RT-X Models**。這篇在資料與生態上很重要。它把多個機構、不同 robot embodiment 的資料標準化，並展示高容量模型跨平台的正遷移。

你要抓：

- 為什麼 cross-embodiment data 很重要
- dataset standardization 對研究可重現性的意義
- 你未來若要做大型資料預訓練，這會是重要基礎

## 26) Octo

**An Open-Source Generalist Robot Policy**。這篇適合你思考一件事：你未來是不是不需要從零訓練 policy，而是把 generalist policy 當 backbone，再加上你的 meta-task layer。

你要抓：

- 為什麼 Octo 適合作為研究 backbone
- 它對 observation / action space adaptation 做了什麼
- 若把你的研究放在上層 task inference，這類 backbone 可以節省多少工程量

## 27) OpenVLA

**An Open-Source Vision-Language-Action Model**。這篇對你很有現實價值，因為它是開源、可 fine-tune、而且直接面向 VLA 訓練與部署。OpenVLA 專案頁面說明它是 7B 開源 VLA，預訓練在約 970k Open X 機器人 episodes 上。

你要抓：

- VLA 的輸入輸出設計
- 哪些部分可以被你拿來當語言到 task spec 的 backbone
- 你的論文是否要把它當比較對象，而不是自己從頭訓練一個 7B 模型

---

# 第六階段：Benchmark 與資料格式

這一階段是做實驗前一定要補的。

## 28) CALVIN

**A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks**。這是長時程語言條件操控的重要 benchmark，LUMOS 也有在這類長時程語言控制場景做評估。

你要抓：

- long-horizon language control 為什麼比單步 manipulation 難很多
- success metric 怎麼定義
- 若你做 zero-shot 任務規格化，CALVIN 為什麼會是合理 benchmark

## 29) RLDS

**an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning**。這篇不是演算法，但非常實用。它提供 RL / imitation / offline RL 資料的標準化格式與處理生態。

你要抓：

- episode / step / metadata 應該怎麼組織
- 為什麼資料格式對跨資料集研究重要
- 你若以後接 Open X 或 robot trajectory dataset，RLDS 會非常常見