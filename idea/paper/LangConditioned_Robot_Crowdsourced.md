---
type: paper-note
aliases:
  - "LOReL"
  - "Language-Conditioned Robot Behavior from Offline Data"
year: 2021
stage: "4-play與弱語言"
tags:
  - offline-data
  - language-reward
  - crowdsourced
  - 對照設計
  - 4-play與弱語言
summary: "offline data + crowd-sourced語言學language-conditioned reward再接planning；非直接text-to-action。"
---
# LOReL：用 offline data + 群眾外包語言標註學語言條件式機器人行為

> Nair, Mitchell, Chen, Ichter, Savarese, Finn.
> "Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation."
> CoRL 2021.

---

## 0. 閱讀總覽（白話）

這篇處理一個非常實際的問題：**我們希望用一句自然語言去指揮機器人做事**（例如「打開右邊抽屜」），但在真實機器人上很難收集到「動作 + 語言對齊」的完美示範資料。

作者的核心想法是把「指揮」這件事拆成兩件事：

1. **語言→獎勵 (reward)**：用「一堆預先收集好、品質可能很爛的離線資料」加上「群眾外包後補的語言標籤」訓練一個分類器。給定起始畫面 $s_0$、目前畫面 $s$、語言指令 $l$，分類器回答「從 $s_0$ 走到 $s$ 是不是完成了 $l$？」。這就是一個 language-conditioned reward $R_\theta(s_0, s, l)$。
2. **獎勵→動作**：再訓練一個 task-agnostic 的視覺前向動力學模型，配上 CEM 做 visual MPC。MPC 規劃時把候選動作序列丟進動力學模型展開，用上面那個 $R_\theta$ 評分，挑最高分的動作序列執行。

關鍵的設計選擇：

- **不是直接學 text→action（語言條件式模仿學習 / 強化學習）**：因為離線資料 sub-optimal，直接抄不好；學「成功判別器」相對輕鬆，且能反向把好行為「規劃」出來，即使資料裡沒有完整成功 demo。
- **群眾外包 (crowd-sourced) 標註**：人類事後 (post-hoc) 看影片寫「機器人在做什麼」，不需要先有任務範本、也不需要人類先示範。對比 hindsight relabeling 是用「達到的目標 (state/image) 反推任務」這種純機器自動的方式，本文是用「人寫的自然語言」反推任務，因此可以支援抽象、合成、多義的指令。
- **用 distilBERT 預訓練語言模型**：可從合成 (procedurally generated) 訓練語言 zero-shot 推廣到真實人類自然語言。

在 simulation 上，這個方法比 language-conditioned BC、language-conditioned Q-learning、goal-image (pixel/LPIPS) 都好至少 25%；在 Franka 真機 5 個任務上平均成功率 66%。

---

## Abstract

研究主題是「從大量 offline 機器人互動資料學各種視覺操作技能」。任務指定有兩種：(i) goal image 接近觀察空間，自監督容易，但 (a) 對人類不方便、(b) 過度指定造成 sparse reward、(c) 對非到達 (non-goal-reaching) 任務又欠指定；(ii) 自然語言彈性、方便，但需把語言 ground 到觀察空間。作者把高度 sub-optimal、自主收集的離線資料 + 群眾外包文字標籤組合，訓練一個分類器：「狀態變化是否完成這條指令？」此分類器即為 language-conditioned reward，再接 offline multi-task RL（實作上用 visual MPC）。實驗中比 goal-image 與 language-conditioned imitation 高 25% 以上，並在 Franka Emika Panda 上做到「打開右邊抽屜」、「移動釘書機」等視覺動作任務。

---

## 1. Introduction

動機：通用機器人需要「人類好下任務、機器人聽得懂」。Goal image 雖然 grounded 卻有上述三個缺點。自然語言彈性大、可表達多種粒度，也能表達「非到達式」任務（例如「在多支麥克筆中隨便抓一支」這種多解任務 goal image 只能編碼其中一種成功狀態）。

挑戰是把語言 ground 到高維觀察。前人多依賴「人類遙操軌跡 + 標註」，但遠端遙操在真機上慢且難 scale。

本文 key insight：**autonomous 收集的離線資料 + 事後群眾外包語言標籤** 是 scalable 的 grounding 路線。資料可以是隨機 policy、scripted、RL replay buffer、人類示範、甚至無動作標籤的影片。

具體做法：用 crowd-sourced 標籤 trajectory，訓練 $R_\theta(s_0, s_T, l) \in [0,1]$ 預測「轉移是否完成指令」；得到 language-conditioned reward 後做 offline RL（實際是 visual MPC）。

主要貢獻與發現：

- 在 simulation 上即使資料來自 random policy，仍比 language-conditioned imitation 高 25%、比 goal-image 高約 30%。
- 借助預訓練語言模型可從合成語言 zero-shot 推到未見自然語言。
- 真機上用既有 sub-optimal 資料 + AMT 標註，完成 5 個由自然語言指定的任務。

---

## 2. 方法：LOReL

### 2.1 問題設定（Preliminaries）

考慮 $K$ 個目標任務 $\{T_k\}_{k=1}^{K} \subset \mathcal{T}$。每個任務的 MDP：

$$
\mathcal{M}_i = (\mathcal{S}, \mathcal{A}, p, R_i, T),
$$

其中 $\mathcal{S}$ 是 RGB 影像，$R_i : \mathcal{S} \times \mathcal{S} \to \{0,1\}$ 是「從 $s_0$ 出發、達到 $s$ 是否完成任務 $T_i$」的二元獎勵。$\mathcal{L}$ 是自然語言空間，$\mathcal{L}_i \subset \mathcal{L}$ 是描述任務 $T_i$ 的語言集合（many-to-many：一個任務多種說法、一種說法可能對應多個任務）。

真實 $R_i$ 不可見。只可取得離線資料

$$
\mathcal{D} = \{\tau_n\}_{n=1}^{N},\qquad
\tau_n = \big( [(s_0, a_0), \ldots, (s_T)], l_n \big),
$$

且至少假設 $\tau_n$ 的起末確實滿足 $l_n$ 對應的某個任務 $T_i$（即 $R_i(s_0, s_T) = 1$），$T_i$ 不必屬於目標任務集合（資料可以包含「什麼都沒做」等無關任務）。

目標：學

$$
R_\theta : \mathcal{S} \times \mathcal{S} \times \mathcal{L} \to [0,1],
$$

並建構 policy $\pi : \mathcal{S} \times \mathcal{S} \times \mathcal{L} \to \mathcal{A}$ 最大化 $\sum_{t=0}^{T} R_i(s_0, s_t)$。注意此公式只能捕捉「以狀態變化反映」的任務，不能處理 path-dependent 任務（如「慢慢關門」）。

### 2.2 學 Reward $R_\theta$

關鍵假設：軌跡內單一步的 action 可能不 optimal，但**起點到終點整段確實完成了 $l_n$**。所以把 reward 實作成二元分類器 $R_\theta(s_0, s, l)$。

優點：

- 處理 many-to-many：一個指令可有多種成功 $(s_0, s_T)$，一組 $(s_0, s_T)$ 可被多句話描述。
- **依賴 $s_0$ → 適合 closed-loop 規劃**：例如「往右」是相對目前位置，迭代執行可以一直往右。
- 比 single-task 分類獎勵（前人 [70,71,55]）更彈性，能表達多任務。

**正例 (Positive)**：對於有標註 $l$ 的軌跡，取
$(s_i, s_j, l)$ 滿足 $i \le \alpha T$ 且 $j \ge (1-\alpha) T$ 為正例。$\alpha$ 大則多用些樣本，但會引入假正例。

**負例 (Negative)** 兩類：

1. **跨指令負例**：從同 $\mathcal{D}$ 取 $(s_0', s_T', l')$，其中 $l' \neq l$；可能含假負例但可容忍噪音。
2. **時序倒轉負例**：把同條軌跡的終→起 $(s_T, s_0, l)$ 當負例，強迫 reward 捕捉「時間進展」而非單純物件出現的視覺特徵。

**訓練目標（binary cross entropy）**：

$$
\mathcal{J}(\theta) = -\,\mathbb{E}_{(s_0, s_T, l)\sim \mathcal{D}} \big[\log R_\theta(s_0, s_T, l)\big]
\;-\;\mathbb{E}_{(s_0', s_T', l')\sim \mathcal{N}} \big[\log\big(1 - R_\theta(s_0', s_T', l')\big)\big].
$$

（原文式 (1)。論文以 $\log + \log$ 形式列，實作為最大化 log-likelihood / 等價的二元交叉熵。）

**資料增強**：影像做 affine + color jitter；語言 embedding 加 uniform noise，避免分類器過擬合（防止稀疏／錯誤獎勵）。

**預訓練語言模型**：用 fixed distilBERT 把指令 encode 成 $\mathbb{R}^{768}$。Section 5.2 顯示這對 zero-shot 推廣到自然語言至關重要。

### 2.3 用 Visual MPC 學 Policy

理論上 $R_\theta$ 可接任何 offline RL。本文選 model-based：用全部 $\mathcal{D}$（不需語言標籤）訓練 task-agnostic 視覺動力學

$$
s_{t+1} \sim p_\theta(s_t, a_t),
$$

採 stochastic variational video prediction / GHVAE 等 off-the-shelf 方法。

執行時：給 $l$ 與 $s_0$，sample $M$ 條長度 $H$ 的動作序列 $\{a^m_{t:t+H-1}\}$，前向展開得 $\hat{s}^m_{t+H}$，計算

$$
\text{score}^m = R_\theta(s_0, \hat{s}^m_{t+H}, l).
$$

以 cross-entropy method (CEM) [75] 反覆 elite-resample，直到收斂；把最佳序列首動作執行於環境。

---

## 3. 架構流程

```
Offline robot data D ────┬──→ Visual dynamics model p_θ(s_{t+1}|s_t,a_t)
                         │       (用整個 D，不需要語言)
                         │
                         └──→ Crowd-sourced language labels {l_n}
                                       │
                                       ▼
                               Reward classifier R_θ(s_0, s_T, l)
                               正例：(s_{<αT}, s_{>(1-α)T}, l)
                               負例 1：(s_0, s_T, l')，l' ≠ l
                               負例 2：(s_T, s_0, l) 反向
                               含 image augmentation + 語言 embedding noise
                               用固定 distilBERT 編碼語言
                                       │
                                       ▼
執行時：(s_0, l) ─► CEM-sample 動作序列 ─► p_θ 展開 ─► R_θ 評分 ─► 取最高 ─► 執行第一個動作 ─► 迴圈
```

---

## 4. 實驗與結論

### 4.1 模擬實驗（Meta-World 桌上場景：抽屜 + 水龍頭 + 兩個馬克杯）

- 資料：50,000 episodes 隨機 policy；用環境 state 程序產生 2311 unique instructions。
- 6 個評估任務：close drawer / open drawer / turn faucet left / turn faucet right / move black mug right / move white mug down。

**Q1 對照前作**（每法 3 seeds × 100 trials）：

- LCBC（language-conditioned behavior cloning，類似 [6,7]）
- LCRL（language-conditioned offline Q-learning，類似 [43] 之 low-level）
- Pixel / LPIPS（goal-image MPC，類似 [67, 77]）
- Oracle（真實動力學 + 真實 reward 的 CEM 上限）
- Random

結果：LOReL 比次佳的 LCBC 高 25%+；比 LPIPS goal-image 高約 30%。
觀察：LCBC 只學到粗略方向，難做 fine-grained 動作（如 turn faucet left）；LCRL 在二元 sparse reward 下幾乎學不到東西；goal-image with pixel cost 會去匹配機器手臂位置而不是真正操作物件，凸顯 goal-image over-specify 的弊病。

**Q2 zero-shot 至未見自然語言**（Table 1）：

| 設定 | LOReL | LOReL (-PM，無預訓練語言模型) |
| --- | --- | --- |
| Original | 56±1% | 40±1% |
| Unseen Verb | 51±3% | 33±2% |
| Unseen Noun | 51±1% | 39±4% |
| Unseen Verb+Noun | 47±2% | 17±3% |
| Unseen Natural Language（9 位真人改寫） | 46±2% | 21±1% |

只改 verb / noun 只掉 5%；同時改 verb+noun 或讓人類自由改寫，最多掉 10%。沒有預訓練 LM 則對未見指令掉 23%。這支持：**預訓練語言模型把「未 grounded」的語言知識帶進來，使少量機器人 grounding 資料可以 zero-shot 推廣**。

### 4.2 真機實驗（Franka Emika Panda + IKEA 桌 + 兩抽屜 + 櫥櫃）

- 觀察：4 相機 × 64×64 RGB；動作：delta end-effector。
- 資料：直接沿用 concurrent work [78] 的 3000 episodes (~150000 frames) replay buffer，sub-optimal。
- 標註：Amazon Mechanical Turk；不給範本，每段請兩位標註者描述行為，共 6000 條，1699 unique；過濾「機器人沒在做事」或標註者看不懂的片段。

**Table 2** 結果（每任務 10 trials）：

| 任務 | LOReL | LOReL (-FN，去除時序反向負例) |
| --- | --- | --- |
| Open the left drawer | 90% | 30% |
| Open the right drawer | 40% | 0% |
| Move the stapler | 50% | 0% |
| Reach the marker | 70% | 70% |
| Reach the cabinet | 80% | 80% |
| 平均 | 66% | 36% |

去除「反向時序負例」掉 30%：印證該負例對「捕捉時間進展、避免單純物件 over-fitting」是必要的。

複雜改寫測試：把指令改成「Open the small black and white drawer on the left fully」、「Push the small gray stapler around on top of the black desk」也能維持 7/10、5/10，呼應 5.2 對複雜語言的穩健性。

### 4.3 結論與限制

- 任務型態限制：只能表達「狀態變化型」任務，不能 path-dependent（例如「慢慢移動」、「畫個圓」）。未來工作可改用完整影片片段訓練 reward。
- Horizon 限制：目前是 short-horizon skill，要做長 horizon 還需更強的 planner、dynamics model、長行為資料。
- 語言 vs goal image 各有優勢，最終應該結合兩種任務指定形式。

---

## 5. 相關工作（重點選讀）

- **Instruction following / grounding language**：早期語意剖析 + motion primitive [33-38]；近期 end-to-end deep IL/RL [39, 26, 6, 7, 43]。本文不需預定 primitive、直接從影像 + 語言學控制。
- **Language-conditioned RL with rewards**：使用環境 reward [23, 43-47] 或語言當 reward bonus 增稠 [48-52]。本文「**完全不假設環境 reward**」，直接從標註離線資料學 reward。
- **線上學 language-conditioned reward**：[5, 24, 41, 51] 線上 RL 學分類式 reward；對真機昂貴。本文全 offline。
- **Offline play / demo + 語言**：[6, 7] 把資料當近 optimal 用 BC 處理；最相關的 Lynch & Sermanet [6] 也是 crowd-sourcing play data。**本文與這些差別**：不假設動作 optimal，能用 random policy、replay buffer、no-action 影片等更廣資料。
- **Goal-conditioned learning**：[1, 2, 62-69] 用 goal image / state；hindsight relabeling [63] 重新指派目標。本文主張：相比 goal-image，語言更彈性、人類更省力，實證上效果也更好。
- **One-shot / meta**：[58, 59, 61] 用 demo 或 meta-learning 表達任務；與本文以語言為任務 spec 的方向互補。

---

## 與本研究主線的關聯

本研究主線：**robot play + 極少弱語言標籤 + offline meta-RL + 用文字產出 task spec / task embedding 達成 zero-shot**，並重視 meta-learning 與 zero-shot 的交集。LOReL 與此方向高度相關，但有幾個關鍵切點需要區分：

### A. Language-conditioned reward vs 直接 text-to-action

- **直接 text→action（如 LCBC、LCRL）**：必須有「動作」與「語言」高品質對齊；當資料 sub-optimal，policy 直接抄人會把 sub-optimal 行為也學起來；RL 端則因 reward 太 sparse + grounding 同時要學而崩潰。
- **LOReL：text→reward + 任務無關 dynamics + 規劃**：
  - 學「成功與否」這個分類器只需「起點/終點」的弱對齊，遠比學動作對齊容易。
  - dynamics 模型在整個資料集上學，純物理規律不需語言。
  - 規劃器把「會走的軌跡空間」與「成功判別」拆開，能在 sub-optimal 資料上拼出更好的行為（**比 demo 還好**：5.1 中 LOReL 勝過 LCBC 的核心原因）。
- **對主線的啟示**：若主線想用「很少弱語言標籤」走 offline meta-RL，把「弱語言→reward / task embedding」當作 meta-task 表徵，比硬學「弱語言→action / policy params」更省標註且更穩；尤其在 play data 噪音大時，reward classifier 是更 robust 的 task spec。

### B. Crowd-sourced 標註 vs Hindsight 標註

- **Hindsight relabeling (HER 系列, [63])**：標籤完全自動，用「我實際達到的 final state / image / goal」當作此 trajectory 的「目標」。優點：免費、無限量；缺點：標籤就只是「達到狀態」，無語意、無抽象，無法表達「移動釘書機」這種以動作描述、且多種終態都算成功的任務。
- **LOReL 的 crowd-sourced annotation**：人類事後看影片寫自然語言；標籤帶有 (i) 抽象、(ii) 多義（many-to-many）、(iii) 對「行為而非狀態」的描述能力（move / push / open）。缺點是成本、噪音（標註不一致、不完整、有「do nothing」要過濾）。
- **對主線（弱語言 + zero-shot）的影響**：
  - 若你想做「文字 → task embedding 然後 zero-shot」，hindsight label 因無語意所以無法 zero-shot；crowd-sourced + 預訓練 LM 是現實可行的弱標籤 zero-shot 通道。
  - LOReL 的 distilBERT 結果（Table 1）正是「弱、合成、有限的訓練語言 → unseen 自然語言 zero-shot」的存在性證明，這正是主線「文字→task spec→zero-shot」想要的能力。
- **與 meta-learning × zero-shot 的整合方向**：
  - 把 $R_\theta(s_0, s, l)$ 視為一個「以語言為 context」的快速任務識別器 → 與 meta-RL 中 context-based encoder（如 [[PEARL|PEARL]] 的 task latent）等價。可以把 distilBERT(l) + (s_0, s) 視為 task latent 的弱監督對齊訊號。
  - meta-test 時：拿到新指令 $l^\star$，無需新樣本即可給出 reward / latent → 真正的 zero-shot task spec。LOReL 在指令層做到了一半（reward 上的 zero-shot），主線可再進一步把這個 reward / latent 接 offline meta-RL policy（如 [[VariBAD|VariBAD]] / OfflinePEARL）走完「弱語言 → meta-RL zero-shot policy」這條路。
- **限制與啟發**：LOReL 不能表達 path-dependent 任務，這對某些 play-based 細微語意（"slowly"、"in a circle"）是真實的限制；主線若想拓寬到動作風格類弱語言，需要把 reward 從 $(s_0, s_T)$ 擴到完整 video clip 或時序模型。

---

## 一句話總結

LOReL 用 sub-optimal 離線資料 + 群眾外包語言標註訓練一個 $(s_0, s, l) \to [0,1]$ 的成功分類器當作 language-conditioned reward，再配上 task-agnostic 視覺動力學模型與 visual MPC，以「學 reward 而非學 action」的策略繞過 imitation 對最優示範的依賴，並借助預訓練語言模型達成從合成指令到未見自然語言的 zero-shot 泛化。
