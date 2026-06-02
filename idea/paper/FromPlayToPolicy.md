---
type: paper-note
aliases:
  - "From Play to Policy"
  - "C-BeT"
  - "Conditional Behavior Transformer"
year: 2022
stage: "4-play與弱語言"
tags:
  - play-data
  - multi-modal
  - behavior-transformer
  - uncurated
  - 4-play與弱語言
summary: "從uncurated play做conditional behavior generation；強調multi-modal生成避免動作塌成平均。"
---
# From Play to Policy: Conditional Behavior Generation from Uncurated Robot Data

> Cui, Wang, Shafiullah, Pinto (NYU, 2022)
> 提出 **C-BeT (Conditional Behavior Transformer)**：把 Behavior Transformer 的多模態動作生成能力，與 future/goal-observation conditioning 結合，從**沒有任務標籤、沒有 reward、沒有 curation** 的 play data 直接萃取出可條件控制的 policy。在三個模擬基準上平均超越前 SOTA 45.7%，並首次展示「純 offline play data → 真機可用的視覺策略」可行性。

---

## 0. 閱讀總覽（白話）

想像你給一群志工一支機器手臂，告訴他們「想玩什麼就玩什麼，只要去動廚房裡這幾個東西即可」。他們會以各種順序、各種風格去開微波爐、轉旋鈕、移鍋子，有成功也有失敗，路徑長度不一、目標不一致。這種「資料」叫做 **play data（玩耍資料）**：未經整理、沒有 reward、沒有任務標籤、行為高度多模態（同一個起點下有很多種合理走法）。

過去的問題是：

1. 若用一般的 behavior cloning（高斯 head + MSE）去學，多種走法會被平均成一個平庸的「中間值動作」，根本不能用。
2. 若用 offline RL 來學 goal-conditioned policy，幾乎都需要 reward 或手動設計的距離函數，play data 不天然提供這些。
3. 若用 [[DecisionTransformer|Decision Transformer]] 之類 outcome-conditioned 的方法，預設「給定 outcome 後 policy 是 unimodal 高斯」——但作者實驗指出在多模態 play data 上這假設常常不成立。

C-BeT 的關鍵點：

- **生成主幹**用 Behavior Transformer (BeT)，它對連續動作做 **k-means 分箱 + 殘差 offset**，所以可以同時表達「離散 mode 機率」與「每個 mode 內的精細數值」，天然支援多模態。
- **條件機制**用「未來觀察 sequence」，把 (current obs sequence, future/goal obs sequence) concatenate 後丟進 transformer——所以 prompt 與輸出共享同一個 observation token space，不需要 reward/語言/額外標籤。
- **訓練資料**用 hindsight 風格自動標：直接從同一條 play trajectory 內取 t' > t 的未來片段當 goal 條件，零人類標註。

最終 C-BeT 是一個「把 uncurated play data 變成可用 conditional policy」的 offline 監督式管線，不再只是先把 play data 蒸成一個 representation/skill prior，還要再接 RL 才能用。

---

## Abstract（精讀）

- 動機：大規模 sequence model 在 NLP/CV 表現亮眼，但在機器人上難複製，因為 **uncurated robot demonstration（play data）**雜訊大、多樣、分佈多模態，從中抽出「以任務為中心」的行為是難的生成建模問題。
- 方法：C-BeT = Behavior Transformer 的多模態生成能力 + future-conditioned goal specification。
- 結果：在模擬基準上比 prior SOTA 平均高 45.7%；首次展示在真機上「完全用 play data、無任務標籤、無 reward」就能學到有用任務行為。

---

## 1. Introduction（精讀）

論點脈絡：

1. **大型生成模型的成功公式**：在大規模未整理資料上以自監督訓練 → 用 prompt 條件化做 zero-shot。
2. 這套公式在「決策／機器人行為」上幾乎沒被複製成功。
3. 既有 play data 做法分兩派的痛點：
   - 學完還要大量 online + reward 微調（Lynch 2019, Pertsch 2020, Singh 2020）。
   - 用 offline goal-conditioned RL（Levine 2020, Ma 2022），但需要 reward 或人工 distance function；真實世界資料未必有。
4. 核心問題：**從 reward-free play data，怎麼學 conditional behavior generation？**
5. 直接套 GPT 風格 transformer 到 behavior generation 有兩個挑戰：
   - 動作是**連續**且**多模態**；
   - **prompt 與輸出不在同一個 token space**（要 condition on 「未來結果」而不是 「文字」）。
6. C-BeT 的對策：BeT 的離散+offset 動作頭 + Play-GCBC 風格的 future observation conditioning，兩者結合。

表 1（與既有方法定位比較）：

| 方法 | Reward-free | Offline | Multi-modal |
|---|:-:|:-:|:-:|
| GCBC | yes | yes | no |
| GCSL | yes | no | no |
| Offline RL | no | yes | no |
| Decision Transformer | no | yes | no |
| **C-BeT** | yes | yes | yes |

唯一同時符合三項的就是 C-BeT。

主要實驗發現：

1. 在 future-conditioned 任務上大幅勝過 prior work；
2. 在真機 4.5 小時 play data 上，能拼出 long-horizon 的多任務 visual policy（圖 1）。

---

## 2. Background and Preliminaries（精讀）

### Play-like data

- 與傳統 Learning from Demonstration 不同：play data **不假設**示教者是 expert、也不假設單一任務、單一模態。
- 與標準 offline-RL 資料集（如 [[D4RL|D4RL]]）不同：play data **沒有 reward**，但也不是純隨機。
- 常見假設：示教者是「有某種潛在意圖的理性 agent」。

### Behavior Transformers (BeT)

- 多模態 behavior cloning 模型，GPT-like transformer 學 $\pi(a_t \mid s_{t-h:t})$。
- 動作頭：對動作做 **k-means 離散化**，每個動作被拆成「所屬 bin（離散 class）」+「相對 bin 中心的 continuous offset」，故能同時表達多個 mode。
- 限制：原始 BeT 只能 **unconditional** rollout，不能在 inference 時指定要哪個 mode。

### Conditional behavior learning（一般定義）

學一個 $\pi : \mathcal{O} \times \mathcal{G} \to \mathcal{A}$，多吃一個 condition $g \sim p(g)$，可以是 state、latent、image。成功度量可用 reward、距離、或「達成 outcome 的折扣造訪機率」：

$$
d^{\pi(\cdot \mid g)} = \mathbb{E}_{\tau \sim \pi}\Bigl[ \sum_{t=0}^{\infty} \gamma^t \,\delta\bigl(\phi(o_t) = g\bigr) \Bigr]
$$

其中 $\phi$ 是 state → outcome 的映射。

### Goal Conditioned BC (GCBC)

資料是 $(o, a, g)$ 三元組，目標：

$$
\pi^* = \arg\max_\pi \prod_{(o,a,g)} \mathbb{P}\bigl[a \sim \pi(\cdot \mid o, g)\bigr]
$$

在「unimodal 高斯」假設下退化為 MSE：

$$
\theta^* = \arg\min_\theta \sum_{(o,a,g)} \|a - \pi(o,g;\theta)\|^2
$$

play data 沒有 goal label，所以用 **hindsight relabeling**：若 $\mathcal{G} \subset \mathcal{O}$，將 $(o_t, a, o_{t'})$（$t' > t$）加入資料集，把未來真實達到的 state 當作 goal。

---

## 3. Approach：Conditional Behavior Transformer（C-BeT，精讀）

### 3.1 Conditional task formulation

給定當前 obs $o_c$、未來/目標 obs $o_g$，模型要學動作的條件分佈：

$$
\pi(a \mid o_c, o_g) \,\triangleq\, \mathbb{P}_{\tau \in \mathcal{T}}\bigl(a \mid o_c = \tau_t,\; o_g = \tau_{t'},\; t' > t\bigr)
$$

為了在 partial observability 下更穩，把單一觀察換成長度 $N$ 的序列 $\bar o_c = o_c^{(1:N)}$、$\bar o_g = o_g^{(1:N)}$。最終任務：

$$
\pi\!\left(a \,\Big|\, o_c^{(1:N)}, o_g^{(1:N)}\right) \;\triangleq\; \mathbb{P}_{\tau \in \mathcal{T}}\!\left(a \,\Big|\, o_c^{(1:N)} = \tau_{t:t+N},\; o_g^{(1:N)} = \tau_{t':t'+N},\; t' > t\right) \tag{1}
$$

符號說明：

- $\mathcal{T}$：play trajectory 集合；
- $o_c^{(1:N)}$：當前的 $N$ 個連續觀察（含 history）；
- $o_g^{(1:N)}$：未來的 $N$ 個連續觀察（goal window，不必只是 single frame）；
- $\bar o_g$ 可獨立選用長度 $N'$，不必等於 $N$。

### 3.2 架構選擇（Architecture）

- 必須是多模態：同一個 $(\bar o_c, \bar o_g)$ 對下，可能有多種動作序列同時合理（圖 2 的例子：去同一個目標有兩條路）。
- 故取 BeT 為生成主幹（離散 bin + offset，天生多模態）。
- 修改輸入：把 **未來條件序列** 與 **當前觀察序列** **concatenate**（而非 stack）丟入 BeT，這樣可獨立決定 current/future window 長度。
- 因為 BeT 是 seq-to-seq，**只**取對應到當前觀察那段的預測動作作為輸出。

### 3.3 Dataset preparation

訓練資料動態相當於：

$$
\{(o_{t:t+N},\; a_{t:t+N},\; o_{t':t'+N'})\},\quad t' > t
$$

第三項當作 $\bar o_g$。實作上等於把 play trajectory 在 batch loader 裡做 hindsight relabel。

### 3.4 Training objective

沿用 BeT loss：

$$
\mathcal{L}_{\text{BeT}} = \mathcal{L}_{\text{focal}}\bigl(\pi(o)_d,\; \lfloor a \rfloor\bigr) \;+\; \lambda \cdot \mathcal{L}_{\text{MT}}\bigl(\langle a\rangle,\; \pi(o)_c\bigr)
$$

其中：

- $\lfloor a \rfloor$：動作 $a$ 對應到的 k-means bin 索引；
- $\langle a \rangle = a - \text{BinCenter}(\lfloor a \rfloor)$：相對 bin 中心的殘差；
- $\pi(o)_d \in \mathbb{R}^k$：bin 上的 multinomial logits；
- $\pi(o)_c \in \mathbb{R}^{k \times |\mathcal{A}|}$：每個 bin 各自一份 offset 預測；
- Focal loss（Lin 2017）：$\mathcal{L}_{\text{focal}}(p_t) = -(1-p_t)^\gamma \log p_t$，緩解 bin class 不平衡；
- Multi-task loss（Girshick 2015）：只對 ground-truth bin 計 offset 的 $\ell_2$，

$$
\text{MT-Loss}\bigl(\langle a\rangle, \{\langle \hat a^{(j)}\rangle\}_{j=1}^{k}\bigr) = \sum_{j=1}^{k} \mathbb{I}[\lfloor a \rfloor = j] \cdot \bigl\| \langle a\rangle - \langle \hat a^{(j)}\rangle \bigr\|_2^2
$$

直觀：classifier 學「動作落在哪個 mode」，regressor 只對「正確的那個 mode」學殘差，避免不同 mode 在連續空間裡被平均掉。

### 3.5 Test-time conditioning

- 主要實驗：condition on **未來觀察**（single goal frame 或 demo sequence 皆可）。
- 也比較了：binary latent / one-hot 標籤條件（人工標 mode）。
- 推論流程：將 $(\bar o_{c-h:c}, \bar o_{g:g+h'})$ concatenate → BeT → 取 bin 分佈 → 取出對應 offset → 加上 bin center 得到動作 → 取對應於當前 observation 那段。

---

## 架構流程（綜整圖 3）

1. **(A) Dataset**：play data $\{(o, a)\}$，含 semi-optimal、多模態、失敗段，無任何標註。
2. **(B) Training**：
   - 取 $(o_{c:c+h}, o_{g:g+h'})$ → concat → Behavior Transformer → 預測動作分佈（bin 機率 + offset）
   - 用 BeT loss（focal + MT）對 ground-truth $a_{c:c+h}$ 監督。
3. **(C) Evaluation**：
   - 條件可為 **單一 target frame** 或 **整條 target demonstration**；
   - 連同最近 $h$ 步觀察 concat 入 BeT，輸出 $\hat \pi(a_c)$，取樣得 $a_c$ 執行。

---

## 4. 實驗（模擬部分）

### 4.1 Baselines

涵蓋四個家族：

- **GCBC / WGCSL**（goal-conditioned supervised）：MLP-based、unimodal MSE。
- **[[LatentPlansFromPlay|Play-LMP]]**：VAE 編 short-horizon motor primitive。
- **RIL (Relay Imitation Learning)**：hierarchical（high-level 出短期 sub-goal、low-level 出動作）。
- **C-IBC**：conditional Implicit BC，能量基礎模型 $E(a \mid o, g)$。
- **GTI**：用 CVAE 編 goal 潛變數 + autoregressive 出動作序列。
- **GoFAR**：offline goal-conditioned RL，用 state-occupancy 推 proxy reward。
- **Unconditional BeT**：當作「隨機」上限參照——多模態但沒看 goal。
- **Unimodal C-BeT**：拿掉 multi-modal head → 接近一個 outcome-conditioned Decision Transformer 變體。

WGCSL 與 GoFAR 在像素環境下需要 proxy reward，作者用 $\exp(-(1/4 \|g-s\|)^2)$ 套上。

### 4.2 環境

- **CARLA self-driving**：(224,224,3) 像素觀察、二維動作 (油門/煞車 + 方向)。資料：200 demos，有岔路，雙模態路徑。
- **Multi-modal block-pushing**：xArm 推紅、綠兩塊到對應色目標，1000 demos，多模態。
- **Franka relay kitchen**：566 條 VR 人類示教，包含七種互動中四種的不同組合。

### 4.3 主要結果（Table 2，分數越高越好）

| 方法 | CARLA | BlockPush | Kitchen |
|---|:-:|:-:|:-:|
| GCBC | 0.04 | 0.06 | 0.74 |
| WGCSL | 0.02 | 0.10 | 1.17 |
| Play-LMP | 0.00 | 0.02 | 0.04 |
| RIL | 0.59 | 0.07 | 0.39 |
| C-IBC | 0.65 | 0.01 | 0.13 |
| GTI | 0.74 | 0.04 | 1.61 |
| GoFAR | 0.72 | 0.04 | 1.24 |
| BeT (uncond.) | 0.31 | 0.34 | 1.77 |
| C-BeT (unimodal) | 0.62 | 0.35 | 2.74 |
| **C-BeT (multimodal)** | **0.98** | **0.90** | **2.80** |

觀察：

- 三項任務都拿下最高。
- 沒有一個 baseline 在三項上都贏過 unconditional BeT，顯示「multi-modal action backbone」本身就是強需求。
- Play-LMP 在 long-horizon 任務上表現差（與其 short-horizon motor-primitive 設計不合）。

### 4.4 Multi-modal 重要性

對照 unimodal C-BeT 與 multimodal C-BeT：

- BlockPush：0.35 → 0.90，差距巨大；
- CARLA：0.62 → 0.98；
- Kitchen：2.74 → 2.80（已逼近上限）。

這直接反駁 Decision Transformer / RvS 那種「conditioned on outcome 後 policy 必然 unimodal」的隱含假設——在 play data 中，**即使給定 outcome，達成方式仍多模態**，unimodal head 會被多 mode 平均壓垮。

### 4.5 與「人工標籤條件」比較（Table 3）

| | No labels | Labels |
|---|:-:|:-:|
| CARLA | 0.98 | 1.00 |
| BlockPush | 0.90 | 0.89 |
| Kitchen | 2.80 | 2.75 |

未來觀察條件（自動 hindsight）≈ 人工 one-hot/binary 標籤，**完全不需要人類標**。

---

## 5. 真機實驗

### 5.1 環境與資料

- Franka Panda + 兒童玩具廚房（烤箱、微波爐、鍋、兩個旋鈕）。
- 動作空間：7 維 joint angle delta（normalized to [-1,1]）+ binary gripper。
- 資料：460 條、共 265 分鐘（4.5 小時）的志工 VR 自由玩耍，**沒有指定任務**，唯一限制是鍋只能放在左爐或水槽以避免不可恢復狀態。共 45,287 frames。
- 表徵：兩台相機影像分別用 ResNet-18 + BYOL self-supervised fine-tune，得 512 維 embedding；本體感覺把 7 維 joint 的 (sin, cos) 重複 74 次填到 1036 維；總 obs 2060 維。

### 5.2 單任務結果（Table 4）

| 方法 | Knobs | Oven | Microwave | Pot | Cumulative |
|---|:-:|:-:|:-:|:-:|:-:|
| GoFAR | 0/10 | 0/5 | 0/5 | 0/5 | 0/25 |
| Uncond. BeT | 5/20 | 6/10 | 1/10 | 0/10 | 12/50 |
| Unimodal C-BeT | 1/20 | 8/10 | 4/10 | 0/10 | 13/50 |
| **Multimodal C-BeT** | **3/20** | **9/10** | **7/10** | **5/10** | **24/50** |

GoFAR 在真機上完全打不到（雖能朝向目標移動，但無法抓取/操作）。

### 5.3 Long-horizon（Table 5，平均每 run 完成任務數）

| 方法 | Oven→Pot | Microwave→Oven | Pot→Microwave | Avg. tasks/run |
|---|:-:|:-:|:-:|:-:|
| Uncond. BeT | (6,0)/10 | (1,6)/10 | (0,1)/10 | 0.47 |
| Unimodal C-BeT | (1,1)/10 | (2,0)/10 | (8,0)/10 | 0.37 |
| **Multimodal C-BeT** | (5,4)/10 | (8,8)/10 | (4,4)/10 | **1.1** |

亮點：**沒有 high-level controller**，C-BeT 自己就能把 play data 中不同片段的子技能拼接出 long-horizon 行為。

### 5.4 泛化

- 對未見過的條件 demo：保留約 67% 單任務成功率（16/50）。
- 加 2 個干擾物：~67% 表現；4 個以上：失敗。

### 5.5 失敗分析

- **旋鈕任務**持續失敗：BYOL 表徵抓不到旋鈕 state，導致 condition signal 無效。表徵限制是真機落地的主要瓶頸之一。
- **Multi-modal 架構的必要性**：對簡單單模態任務（如開烤箱）unimodal 已足夠；但對 long-horizon、相互糾纏的 play data，無多模態就會塌成 sub-optimal 解。

---

## 6. Related Work（精讀）

四條主線：

1. **Outcome-conditioned behavior learning**：從 Kaelbling 1993 / UVFA、HER 到 Decision Transformer。常用 hindsight relabel。但 Paster 2022、Brandfonbrener 2022 指出 reward-conditioned 在 stochastic 環境下不可靠；C-BeT 走 outcome (state) conditioned 並強調 multi-modal。
2. **Learning from play data**：Lynch 2019、Gupta 2019 是直接前身；其中 Gupta 2022 要 human annotation 才能 reset-free，C-BeT 不需要。
3. **Generative modeling of behavior**：IRL 系列、學 action prior 加速 RL（Pertsch 2020、Singh 2020）、Implicit BC / EBM 系列。C-BeT 是 explicit 多模態 generator。
4. **Transformers for behavior learning**：DT (Chen 2021)、Trajectory Transformer (Janner 2021)、BeT (Shafiullah 2022) 是其骨幹；C-BeT 是 BeT 的 conditional 延伸，並把 unimodal C-BeT 對應到 outcome-conditioned 版的 DT。

## 7. Discussion 與限制

- 限制 1：representation 沒抓到關鍵物件（如旋鈕），policy 就失敗 → 把希望寄託在更好的 visual representation。
- 限制 2：簡單 unimodal 任務 multi-modal 沒太大幫助。
- 展望：更大、更多樣的 play data 才能逼近大模型尺度。

---

## 與本研究主線的關聯

使用者的主線：**robot play data + 極少弱語言 + offline meta-RL + 文字 → task spec/embedding 做 zero-shot**，並重視 meta-learning × zero-shot 並行。C-BeT 對主線的啟發如下：

### (1) Uncurated play data 該怎麼處理

這篇等於把「處理 play data 的最小可行 offline 流程」打磨完成：

- **不需要 reward / task label** → 與弱語言設定相容（語言本來就稀疏）。
- **hindsight 把未來觀察當條件**是免費的 supervision；對主線而言，這個 hindsight 機制可直接平移成「把未來觀察 + 弱語言 caption（若有）一起當 task spec」，不需要人工配對每條軌跡。
- play data 通常**長度不一、有失敗段、有 sub-optimal 段**，C-BeT 證明只要動作頭夠強（多模態）+ 條件正確，就**不需要先做高 cost 的 trajectory cleaning**。對主線 offline meta-RL 設計尤其關鍵：meta-RL 過去常苛求乾淨的 task-labelled episode 集合，這篇示範了「條件 + 多模態」就可吃下髒資料。

### (2) Multi-modal generation 為何重要

主線若要做「文字 → zero-shot task embedding → policy」，會踩到三個多模態陷阱，C-BeT 的證據都剛好打到：

- **同一條 task spec 有多種完成方式**：人在 play 階段就是這樣，模型若用 unimodal head 會直接退化成平均動作（在 BlockPush 0.35 vs 0.90 體現最強）。
- **不同 task spec 共享底層 sub-skill**：多模態 head 等於替每個 sub-mode 維持獨立 offset 分支，避免 task 間互相干擾——對 meta-RL 的多任務泛化是必要條件。
- **文字 embedding 本身是 noisy 條件**：若 conditional policy 不能多模態地容忍 prompt 模糊性，文字驅動的 zero-shot 一致會塌掉。C-BeT 的 focal + MT loss 公式可直接作為主線文字條件版的 head 設計。

### (3) 它怎麼把 play data 變成「可用 policy」而非只做 representation

過往 play data 路線（Lynch 2019、Pertsch 2020、Singh 2020）幾乎都是「play → 學一個 skill prior / latent / action decoder」，最後**還要接 online RL + reward** 才能真正完成任務。C-BeT 的關鍵差異：

- 把 **conditioning** 從「人為 latent / one-hot」改成「**未來觀察 sequence**」——這同時是 policy 輸入空間裡就有的東西，所以 inference 時直接拿任何一段 demo 或 goal frame 當 prompt。
- 把 **action head** 從高斯 MSE 換成 **bin 分類 + 殘差 offset**——這讓 BC-style 純監督的 loss 在多模態資料上仍能收斂出可執行 policy，**不再需要 RL 微調**。
- 結論：play data 不再是「半成品」，可以**直接** end-to-end 成為部署用 policy。對主線「offline meta-RL + 文字 spec」而言，這意味著主流程可以是 **offline only**，不需保留昂貴的 online evaluation/finetune 階段；只要把 future-observation 條件換成 / 額外加上「文字 embedding」就能往主線方向延伸。

### (4) 對 meta-learning × zero-shot 並行的影響

- **Zero-shot 軸**：C-BeT 已示範「prompt = goal frame / demo」做 zero-shot 條件控制；要轉成「prompt = 文字」只需把 future observation tokens 換成 text tokens（或共同 cross-attention），但 multi-modal head 必須保留。
- **Meta-learning 軸**：C-BeT 沒有顯式 meta-learn task latent；主線可以把它升級成「文字 / outcome 雙條件 + meta-learn task embedding」，意即把 unimodal C-BeT（≈ outcome-conditioned DT）那一支替換成「文字 embedding-conditioned + 多模態 head」，作為 meta-RL 的 base policy class。
- 風險點：C-BeT 在旋鈕任務失敗，提醒主線若用 BYOL/CLIP 之類凍結表徵，需先驗證關鍵 object/狀態能被表徵區分，否則文字條件無論多準都會被表徵瓶頸吃掉。

---

## 一句話總結

C-BeT 把 Behavior Transformer 的「k-means bin + 殘差 offset」多模態動作頭，與 future-observation hindsight 條件結合，證明**純 offline、reward-free、無標註的 play data 也能直接訓出可用的多任務 conditional policy**，而 multi-modal head 是讓這條路徑不塌掉的關鍵零件。
