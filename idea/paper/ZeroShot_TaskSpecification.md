---
type: paper-note
aliases:
  - "Zero-Shot Task Specification"
  - "ZeST"
  - "Can Foundation Models Perform Zero-Shot Task Specification"
year: 2022
stage: "5-zero-shot與generalist"
tags:
  - zero-shot
  - task-specification
  - foundation-model
  - 概念框架
  - 5-zero-shot與generalist
summary: "研究foundation model把語言/圖/草圖轉成task spec；強調task specification≠task execution。"
---
> **論文標題**：Can Foundation Models Perform Zero-Shot Task Specification For Robot Manipulation?
> **作者**：Yuchen Cui, Scott Niekum (UT Austin); Abhinav Gupta, Vikash Kumar, Aravind Rajeswaran (Facebook AI Research)
> **出處**：Proceedings of Machine Learning Research vol 168, 4th Annual Conference on Learning for Dynamics and Control (L4DC), 2022
> **主題**：使用 foundation models（CLIP、ResNet、MoCo）對機器人操作任務做 zero-shot 的「任務規格化（task specification）」——把網路圖、手繪草圖、自然語言當作目標輸入，並以「embedding 相似度」當作目標一致性指標
> **整理目標**：對應使用者主線研究——以極低成本（弱語言、網路圖、草圖）做 zero-shot task spec/embedding，並與 robot play data + offline meta-RL 結合。本文是「task specification（人類如何告訴機器人目標）」最直接的 baseline 之一。

---

## 0. 閱讀總覽（白話）

這篇論文的核心想法是：「機器人最痛的不是怎麼動作，而是『使用者怎麼告訴它要做什麼』」。傳統做法有兩種，但都很糟糕：

1. **state-vector goal**：給一個低維向量 `s_goal`，但要先做 state estimation，使用者根本看不懂。
2. **same-scene goal image**：給一張「機器人自己場景的目標照片」，但這需要有人先把任務做完一遍，才能拍到目標照片——那要這台「自主」機器人幹嘛？

作者主張兩個更輕鬆的人機介面：
- **從網路找一張圖**（off-domain image / 手繪草圖）：例如想叫機器人打開微波爐，就直接 google 一張「微波爐打開的照片」。
- **語言指令**：例如「open the microwave」。

問題是：機器人沒看過這些圖、不認得語言。**怎麼讓 zero-shot 就 work？** 作者的答案：用 foundation models（CLIP、ResNet、MoCo）把觀察與目標都丟進 embedding 空間，比相似度即可。

他們提出 **ZeST 框架**，三個模組：(1) embedding network、(2) feature selector（raw 或 delta）、(3) similarity metric（cosine 或 L2）。並用三個任務做評估：goal selection、high-level action selection、offline RL（把相似度當 reward proxy）。

**關鍵實證**：CLIP + delta features + cosine 在 Franka Kitchen 上比隨機選好 14 倍；把 ZeST 相似度當 reward 拿給 [[DecisionTransformer|Decision Transformer]] 做 offline RL，可達 expert 90.77%，比 BC 的 80.67% 高。

對使用者主線而言，**這篇就是「task spec ↔ task execution 解耦」這個觀點的代表作**：先不管怎麼執行，先把「人類意圖」用 foundation model 翻譯成可比對的訊號。它沒做到把 spec 變成結構化中介表示，但已經把方向打開了。

---

## Abstract

- 任務規格化（task specification）是自主機器人的核心；要讓非專家使用者用得起來，必須有「低成本（low-effort）」的指定方式。
- 既有做法主要靠 goals：state vector（難解讀、要 state estimation）或 same-scene goal image（要有人先做一次任務，本末倒置）。
- 作者探索更廣義的 goal：網路圖、手繪草圖、語言描述。
- 作為初步工作，研究大規模 pre-trained foundation models 能否做 zero-shot goal specification。
- 在數個模擬機器人操作任務與真實世界資料集上得到正面結果。
- Keywords：Goal-conditioned RL、Visual RL、Robot Learning。

---

## 1. Introduction

- **動機**：機器人正進入家庭，使用者有個人化需求，需要「低成本、直覺」的溝通介面。這要求機器人具備 human-centric 的 common sense。
- **時代背景**：vision（AlexNet 等）與 NLP（BERT、GPT-2）在大規模網路資料上學到的 representation 已展現巨大遷移能力，即 **foundation models** (Bommasani et al., 2021)。
- **問題提出**：foundation models 能不能用於機器人的 task specification？

**既有三種程式機器人方式**：reward function、demonstrations、goal specification。其中 goal specification 對 end user 最友善（比 demo 簡單；比 reward function 不易被 exploit / Goodhart）。

**既有 goal 表示的不足**：
- compact state vector goal：解讀困難、要 state estimation，實驗室外幾乎不可行。
- same-scene image goal：要先做一次任務，違背「自主機器人」的初衷。

**作者提的兩種替代**：
1. **語言 goal**（「open the microwave」）：需要 grounding 與語意（要知道微波爐長什麼樣、開與關的差別）。
2. **off-domain image goal**（從網路找一張開著的微波爐）：需要 domain adaptation、能把目標圖與當前場景的物體建立對應。

**Contributions**：
1. 提出 ZeST 框架（Figure 1）。
2. 用 goal selection tasks 評估 zero-shot policy execution 的效果。
3. 用 ZeST score 作為 reward proxy 做 offline RL 並評估。

**核心發現**：goal-selection 比 random 好 14 倍；用 ZeST score 當 reward 的 DT 比 BC 好。

---

## 2. Background and Related Work

### Block MDP 設定

環境定義為 high-dimensional MDP（Du et al., 2019 的 Block MDP）：

$$M = \langle S, X, A, P, R, d_0, \gamma \rangle$$

- $S$：緊湊 state space（不可直接觀察）。
- $X$：高維 observation space（含足以唯一恢復 state 的資訊；例如多視角相機影像）。
- $A$：action space；$P(s'|s,a)$、$R(s)$、$d_0$、$\gamma$ 為標準 RL 元素。
- Trajectory $\tau = \{(X_0,a_0), \dots, (X_H,a_H)\}$；policy $\pi(a_t|X_t)$。
- 目標：$\max_\pi \mathbb{E}\big[\sum_{t=0}^{\infty} \gamma^t R(s_t)\big]$。

### Goal-Conditioned Policies

把 policy 與 reward 都條件於 goal $G \in \mathcal{G}$：$\pi(a_t|X_t, G)$、$R(s_t, G)$。等價於把 MDP 的 state/observation 擴充為 $S \cup \mathcal{G}$、$X \cup \mathcal{G}$。先前 goal 多半放在：
- compact state space（Kaelbling 1993、HER、IRIS、Gupta et al. 2019）。
- 或 same-scene 高維影像（Nair, Singh, Johns 等）。

兩者都有前述限制；本文用 off-domain images 與語言。

### Foundation Models 與其應用

- 表示學習 / 遷移學習在 AlexNet、ResNet 後成主流。
- 自監督學習（MoCo、BERT、GPT-2）與多模態（CLIP）大幅推動下游任務。
- 機器人 / control 的應用仍少；同期工作：**CLIPort**（CLIP + transporter）、Khandelwal et al. (2021)、Parisi et al. (2022) 都顯示 CLIP/self-supervised 比 ImageNet 預訓練更強。

---

## 3. ZeST: Using Foundation Models for Task Specification

### 整體想法

要支援 off-domain image / language 這類 generic goal，agent 需「人類視角的 common sense」。Foundation models 因為訓練在內含人類產生的資料，正好可能補上這個 gap。本文只做 **zero-shot**：沒有任何 human annotation、也沒有「agent 經驗 ↔ goal spec」的監督配對，貼近真實部署情境。

ZeST 的核心是：**測量「agent observation」與「user-specified goal」在 foundation model embedding 空間的相似度**。
- 相似度高 → 認定 goal 已達成。
- replay buffer 中相似度最高的圖可作為 goal-conditioning 對象。
- 相似度也可作為 reward proxy。

### 三個模組

1. **Embedding network $\psi(\cdot)$**：本文用 ImageNet-supervised ResNet50、ImageNet-MoCo、CLIP。CLIP 因多模態還能吃語言。
2. **Feature selector**：
   - **Raw**：直接用 $\psi(X_t)$、$\psi(G_f)$。
   - **Delta**：取「現在 vs 起始」與「終態 vs 初態」的差。例如使用者給「關著的微波爐」與「開著的微波爐」兩張圖，描述「開」這個 high-level action。
3. **Similarity metric $\alpha$**：cosine similarity 或 L2 distance。

### 公式

Raw similarity：
$$\phi_\text{raw}(X_t, G_f) := \alpha\big(\psi(X_t),\ \psi(G_f)\big) \tag{1}$$

Delta similarity（agent 端 $(X_t, X_0)$；goal 端 $(G_f, G_0)$）：
$$\phi_\text{delta}\big((X_t, X_0),(G_f, G_0)\big) := \alpha\big(\psi(X_t)-\psi(X_0),\ \psi(G_f)-\psi(G_0)\big) \tag{2}$$

Figure 2 顯示：當機器人成功執行「開櫥櫃」時，$\phi_\text{delta}$ 隨時間單調上升。

### 框架定位

作者強調 ZeST 是一個**抽象框架**，可以替換任意 foundation model、任意 featurization、任意 metric；窮舉所有組合超出單篇範圍，他們示範若干設計選擇，並期望啟發後續工作。

---

## 4. Experimental Tasks and Evaluation Metrics

### 4.1 Goal Selection Task

給定經驗資料集 $\mathcal{D}=\{\tau_1,\dots,\tau_N\}$、off-domain goal spec $(G_0, G_f)$ 與起始 obs $X_0$，要從 $\mathcal{D}$ 挑出最符合 goal 的 obs $X_f$。$\mu$ 為資料集中滿足 goal 的觀察數。

兩個指標：
1. **Top-N Success Rate**：依相似度排序，取前 $N \le \mu$ 個，計算其中滿足 goal 的比例。
2. **Dataset Total Variation (DTV)**：考慮整個分布，不只 top 端。$\phi_i$ 為 $X_i$ 的相似度，$\mathbb{I}(X_i)$ 為是否達成 goal 的指示函數：

$$\mathrm{DTV} := \frac{1}{\|\mathcal{D}\|} \sum_{X_i \in \mathcal{D}} \left| \frac{1}{\mu}\mathbb{I}(X_i) - \frac{\phi_i}{\sum_j \phi_j} \right| \tag{3}$$

直覺：把「實際是 goal 的離散分布」與「相似度歸一化後的分布」做 L1 比對；越小越好。

### 4.2 High-Level Action Selection Task

要求 agent 從資料集中找一對影像 $(I_0^k, I_i^k)$，其 delta feature $\Delta a_i^k = \psi(I_i^k)-\psi(I_0^k)$ 與 goal delta $\Delta g = \psi(G_f)-\psi(G_0)$ 相似度最高。用途：當沒有目標物的圖時，可借「開櫥櫃門」的圖去指定「開微波爐」這個共享「開」語意的 high-level action。以 Top-N 評估。

### 4.3 Offline Reinforcement Learning

任務由 $(G_0, G_f)$ 指定，給定 sub-optimal trajectories（offline RL 設定）。比較：
- **BC**：直接監督學 $(X_t \to a_t)$，無法改進資料品質。
- **DT（Decision Transformer）+ ZeST reward**：用 ZeST 相似度當 reward，做 reward-conditioned sequence modeling。

若 DT 顯著贏 BC，代表 ZeST score 確實是合理的 reward proxy。

---

## 5. Experimental Results and Discussion

**設計選擇**：embedding ∈ {ResNet50, MoCo, CLIP}；feature ∈ {raw, delta}；metric ∈ {cosine, L2}。

### 5.1 Franka Kitchen 上的 Goal Selection

- **環境**：Franka Kitchen（Gupta et al. 2019），5 種任務：top/bottom 旋鈕、開微波爐、開鉸鏈門、開推拉門。
- **資料**：以 policy gradient（Rajeswaran et al.）訓出的 expert + 加 uniform action noise 形成混合成功 / 失敗的 dataset。
- **Goal 形式**：(1) same-scene image、(2) 網路圖、(3) 手繪草圖、(4) 語言。後三者來自 Visual Task Dataset (Cui, 2021)。

**設計選擇結果（Figure 5）**：
- 影像 goal → **delta features + cosine** 最好。
- 語言 goal → **delta features + L2** 最好（語言端 delta 較難直觀想像，但實驗如此）。
- 之後實驗都用各自最佳設定。

**兩種資料情境**：
- **Narrow Dataset (ND)**：只含單一任務的 trajectories。較簡單，類似既有 goal classifier reward。
- **Diverse Dataset (DD)**：含全部 5 任務 trajectories；更接近真實開放環境。

**結果（Figure 6, 7）**：
- **CLIP** 在所有 goal 形式下 DTV 最低；作者推測因 CLIP 訓練資料量更大（400M vs ImageNet 15M）。
- same-scene > internet image / drawings（domain gap 越小越好）。
- Top-25 success rate 上，ZeST 全面遠勝 random；CLIP 在較難的 DD 設定中**比 random 好 14 倍**。

### 5.2 Real-World Video（SomethingSomething-V2）

- 用 SSV2 中「opening something」的真實影片做評估；視角不穩、有手部遮擋，難度高。
- 也測試 sim2real：用 3D Warehouse 物件渲染合成圖去當 goal，找對應的真實 video frame。
- ND 與 DD 兩種設定都跑。

**結果（Figure 8, 9）**：
- 平均三個 embedding 模型，ZeST 仍勝 random。
- 然而在 real2real 設定下 **CLIP 只略勝 random**，而 ResNet/MoCo 顯著更好——與 Franka Kitchen 結論相反。
- 結論：真實影片仍有挑戰，未來方向是訓練在 internet-scale video 上的 foundation model。

### 5.3 Offline RL（Franka Kitchen + 網路圖 goal）

- 比較：BC vs DT（Chen et al., 2021）+ ZeST reward。
- 兩演算法差異主要在「是否做 reward conditioning」。

| 方法 | Normalized Return |
|---|---|
| ZeST + DT | **90.77%** |
| Demo. 資料平均 | 81.19% |
| BC | 80.67% |

ZeST reward proxy 讓 DT 超過 demo 平均 + 接近 expert。表示**用 zero-shot 推算的相似度當 reward，已足以驅動 sequence model 做 offline RL 改進**。

---

## 6. Conclusion 與限制

- 提出 ZeST：用 foundation models 做 zero-shot task specification。
- Goal/action selection 上 14× 勝 random；offline RL 上勝 BC。
- 顯示 foundation models 有潛力縮小 real / sim domain gap。

**限制**：
- Embedding 容易受**遮擋**影響，常需把機器人手臂從畫面挪開（可用 segmentation + inpainting 修補）。
- 只適合**有顯著視覺特徵變化**的任務；對於以位置 / 空間關係為主的任務無能為力（例如「把杯子放到 x,y 座標」這類），未來可加上 localization 模組。

---

## 相關工作（補）

- Goal-conditioning：Kaelbling 1993；HER；Fu et al. 2018；Gupta et al. 2019；Nair et al. 2018；Singh et al. 2019；Johns 2021。
- Foundation models：BERT、GPT-2、MoCo、CLIP。
- 機器人 + foundation models 同期工作：CLIPort（Shridhar 2021）、Khandelwal et al. 2021、Parisi et al. 2022。本文與這些工作的差異：**ZeST 完全 zero-shot 且專注於 task specification 階段**，不訓任何 policy / 不做 fine-tune。

---

## 與本研究主線的關聯

使用者主線是 **robot play data + 極少弱語言 + offline meta-RL + 文字→task spec / embedding 做 zero-shot**，並關注 meta-learning × zero-shot。這篇正中靶心，下面拆解三個 read.md 指定重點：

### 1. Task Specification vs Task Execution 的差別

- **Task specification（這篇處理的層）**：「人類如何把意圖告訴機器人」。輸入是人類友善的模態（語言、網路圖、草圖），輸出是「機器人能用的目標訊號」（在 ZeST 是 embedding 相似度）。**完全不關心動作怎麼產生**。
- **Task execution**：給定 task spec，怎麼選 action / 規劃 trajectory / 學 policy。
- 在 ZeST，兩者**完全解耦**：spec 階段用 frozen foundation model 算相似度；execution 階段交給下游（goal-selection、DT offline RL 等）。
- **對主線的意涵**：
  - 你的研究若要做 **offline meta-RL + zero-shot task embedding**，這個解耦正是合理的切法：meta-train 階段學「給 task embedding $z$ 就能 execute」的 policy；deploy 階段用 frozen LM / foundation model 把語言 → $z$。
  - ZeST 用「相似度」做 spec 較粗糙——只回答「像不像 goal」；你的主線可以把 spec 升級為「結構化 $z$」，使 execution 端不必依賴 nearest-neighbor 檢索，而是直接條件 $z$ 跑 policy。

### 2. 為什麼「給一張目標圖」不一定是最好的人機介面？

ZeST 文章本身就用很大篇幅論述這點，整理重點：
- **same-scene goal image 的悖論**：要拿到目標圖，必須先有人做完任務 → 失去自主性。
- **off-domain image**：使用者要去找一張外觀像目標的圖；對非典型物件 / 個性化場景，找圖反而更困難。
- **圖只能傳遞「視覺終態」**，傳不了：
  - 過程約束（例如「不要碰到杯子」）。
  - 抽象高層動作（「整理桌面」沒有單一目標圖）。
  - 物件 binding（「把『左邊那個』杯子放到櫃子」需要指代）。
- **遮擋與位置語意**：ZeST 自己承認對位置 / 空間關係任務失效；圖像 embedding 天然偏 appearance。
- **delta features 的引入即是補救**：單張圖不夠，所以要 $(G_0, G_f)$ 兩張圖夾出「動作意圖」。但這只是部分解，仍無法表達順序、條件、否定。
- **語言的優勢**：天然攜帶 abstraction、否定、組合性（「開但不要全開」），且 0 成本。
- **對主線的啟示**：你的研究主張「極少弱語言」其實非常合理——一張圖不夠就讓使用者加一句話；同時，語言本身可被 LM 編成結構化 $z$，比 image embedding 更接近 task 而非外觀。

### 3. Zero-shot task spec 能不能做成結構化中介表示？

ZeST 目前的中介表示是 **scalar 相似度 $\phi$ 或 vector embedding $\psi(G)$**——這算是「非結構化」的中介。它「能用」但有明顯瓶頸：
- $\phi$ 只能告訴你「像不像 goal」，無法表達**子目標序列 / 物件變數 / 約束**。
- $\psi(G)$ 是 visual / multimodal embedding，但維度高、語意稠密、無組合性。

**可以做成結構化中介表示嗎？三個層次：**

**(a) 弱結構化：把 $z$ 當作可學的 task embedding**
- 像 [[PEARL|PEARL]] / [[VariBAD|VariBAD]] 的 $z$，offline meta-RL 中以 $z$ 條件化 policy。
- ZeST 沒做到，但只需把 $\psi(G)$ 投影到一個學過的 $z$ 空間並做 contrastive。
- 這正是「文字 → task embedding」的核心橋樑：訓練時 $z$ 由 trajectory encoder 給出，推論時可由 LM 給出。

**(b) 中度結構化：分解成 (object, attribute, change)**
- 把 ZeST delta feature 拆成「在哪個物件上做了哪種變化」。
- 可借助 LM 或 VLM 把語言解析為這種 schema；execution 端對每個欄位有獨立條件機制。
- 對 zero-shot 友善，因為 schema 可組合（compositional generalization）。

**(c) 強結構化：程式 / 邏輯式（PDDL、LTL、code）**
- 已是 instruction-following / program-as-policy 方向；對家庭環境太 brittle，且需手工 grounding。

**對主線最務實的選擇**：**(a) + (b) 混合**——以 task embedding $z$ 為主，輔以結構化欄位（target object、target attribute change），由弱語言補充。這樣：
- meta-train 用 offline play data 學 $z$。
- deploy 用 LM 把語言 → $z$（zero-shot）。
- 比 ZeST 強的地方：execution 不靠 nearest-neighbor 檢索，而是 policy 直接吃 $z$；對 unseen 目標泛化來自 $z$ 空間的結構，而非 embedding 相似度。

### 4. 給主線研究的具體借鑑

- **Baseline 選擇**：在你的論文裡，ZeST 是必比的 zero-shot baseline（CLIP + delta + cosine + 檢索）。
- **資料設計**：可借鏡 Franka Kitchen + Visual Task Dataset 的多模態 goal（網路圖 / 草圖 / 語言）來測你的 task embedding 在不同弱模態下的 robustness。
- **Reward proxy 思路**：ZeST 把 score 當 reward 餵 DT 已 work；你可以把 task embedding $z$ 直接餵 DT 或 [[IQL|IQL]]，省掉「shaped reward」這層。
- **失效模式**：對位置任務、遮擋任務無效——你的方法若能 handle 這兩類，是重要的差異化點。
- **延伸方向**：作者自己提到「應該訓 video foundation model」，這呼應 robot play data → video pretraining → task embedding 的 pipeline。

---

## 一句話總結

ZeST 用「foundation model embedding 上的相似度」當作 zero-shot task spec 訊號，證明 frozen CLIP/ResNet/MoCo 足以在 goal-selection 比 random 強 14×、並把 offline RL 推到接近 expert——這把「task specification」與「task execution」乾淨分離，但相似度本身是非結構化中介，留下「升級成結構化 task embedding $z$」這個正是你主線該補的洞。
