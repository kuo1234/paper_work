---
type: paper-note
aliases:
  - "Learning Latent Plans from Play"
  - "Play-LMP"
  - "LMP"
year: 2019
stage: "4-play與弱語言"
tags:
  - play-data
  - latent-plan
  - goal-conditioned
  - VAE
  - 可借用
  - 4-play與弱語言
summary: "play data線經典：play三特性（便宜/多樣/無需分段）+ hindsight relabel + latent plan中介表示。"
---
> **論文資訊**
> Lynch, Khansari, Xiao, Kumar, Tompson, Levine, Sermanet（Google Brain）, *Learning Latent Plans from Play*, 3rd Conference on Robot Learning (CoRL 2019), Osaka, Japan.
> 提出 Play-LMP（Play-supervised Latent Motor Plans）與 Play-GCBC（Play-supervised Goal-Conditioned Behavioral Cloning）兩種自監督方法，從人類遠端遙控（teleoperation）的「玩耍資料（play data）」中學到一個 task-agnostic、以目標狀態為條件的策略；其中 Play-LMP 透過 seq2seq CVAE 學一個隱性計畫（latent plan）空間作為中介表示。

---

## 0. 閱讀總覽（白話）

這篇文章想解決的根本問題是：要做一個「會很多技能、能依使用者下的目標任意切換」的通用機器人時，傳統做法（每個任務各收一批專家示範、或各設一個 reward）成本太貴且不可擴展。作者的核心主張是：**與其針對「離散的任務集合」蒐集昂貴的專家資料，不如讓人類自由「玩」（teleoperate playground 一段時間），收一份「沒有切段、沒有任務標籤、沒有 reset」的連續長序列資料，再用自監督的方式把這份 play data 蒸餾成一個通用的、以 (current state, goal state) 為條件的策略。**

Play data 的三個關鍵特性是：
1. **便宜**：不必切段、不必標籤、不必每次 reset 到初始狀態，可以一直連續地錄。
2. **豐富**：因為人受好奇心驅動，會自然地去操作環境中的可互動物件（看到按鈕會按、看到抽屜會拉），且傾向用多種方式達到同一個結果，因此涵蓋範圍很廣——實驗中比 18 個任務的專家示範還多 4.2 倍互動空間覆蓋率，比隨機策略多 14.4 倍。
3. **不需明確任務分段**：play 是連續流，只要從中隨機取一個視窗 $\tau$，視窗的初始狀態就是 $s_c$、最終狀態就是「可達的 $s_g$」、中間動作就是「自監督的動作標籤」——天然就是 goal-conditioned 的訓練樣本。

但 play data 的同一對 $(s_c, s_g)$ 之間可能有非常多種高層行為（multimodality），這會讓單純的 BC 學出來的策略「平均化」掉所有解，效果變差。作者於是引入一個 **隱性計畫 $z$** 作為中介表示：先把當前的整段行為序列編碼到 $z$（plan recognition），同時讓 $(s_c, s_g)$ 也輸出一個 $z$ 的分布（plan proposal），用 KL 把兩者拉近，使策略 $\pi(a|s_t, s_g, z)$ 在 $z$ 給定後就只需做 unimodal 解碼。整個架構就是一個 sequence-to-sequence CVAE。

實驗中很重要的兩個觀察：(i) 一個 Play-LMP 通用策略，在 18 個視覺操作任務上達到 69.4%（pixels）或 85.5%（states），勝過 18 個各自用專家示範訓練的 BC 專家策略；(ii) play-trained 模型對初始位置擾動 robust，並出現 emergent retrying 行為——失敗後會自然地多試幾次直到成功。

---

## Abstract

摘要主張：以人類遙控的 play data 上做自監督控制，是擴大技能學習規模的可行路徑。play 相對於專家示範有兩個吸引人的性質——便宜（不必分段、不必標籤、不必 reset）與天然豐富（同樣時間內覆蓋範圍比專家示範多約 4 倍）。提出 Play-LMP，自監督地將 play 行為組織進一個 latent space 並於測試時取用以達到指定目標。實證結果：訓練於未標籤 play 之單一通用策略，在 18 個視覺操作任務上勝過 18 個各自訓練的專家策略；對擾動更 robust、會自動 retry；而 latent plan 空間在沒有任務標籤下自發以功能分群。

---

## 1. Introduction

作者把整個學習目標形式化為「task-agnostic control」：學一個 goal-conditioned policy $\pi_\theta(a \mid s_c, s_g)$，能從任一當前狀態 $s_c$ 達到任一可達目標 $s_g$（沿用 Kaelbling 1993 的觀點）。在這個觀點下，任務不再是離散標籤，而是由 $(s_c, s_g)$ 在連續空間中索引。

要訓練這種 policy，理想的訓練資料應該在 $(s_c, s_g)$ 構成的互動空間中 **broad and dense**（Fig. 2a）。實務上隨機探索雖便宜但太貧乏；專家示範雖內容豐富但昂貴且只覆蓋窄分布、會造成 distribution shift（引用 Ross et al. DAgger）。

作者主張第三條路 —— **teleoperated play data**：人類遙控機器人時，依自己的好奇心做出「自我導向、但目標導向」的行為流，連續記錄低階觀測與動作。play 不是隨機，而是被人對物件 affordance 的知識所結構化（看到按鈕會去按、看到抽屜會去拉），且人類在同一個目標下會嘗試多種達成方式。實證上，play 在同樣收集時間下覆蓋面顯著超過 18 任務的專家示範與隨機策略（Fig. 2c, d）。

接著作者預告第 3 節提出兩個方法：Play-GCBC（直接做 goal-conditioned BC）與 Play-LMP（加上隱性計畫的 CVAE）。

---

## 2. Related Work

整體位置：

- **監督來源**：RL 需要設計 reward、imitation 需要每任務示範，兩者都需要大量人力。本文用「不為特定任務」、容易收集的 play data。
- **Goal-conditioned policy**：在 RL（Kaelbling, HER, hierarchical actor-critic, intentional unintentional agent 等）與 inverse model（poke by poking、rope manipulation、zero-shot visual imitation 等）皆有大量研究。但既有的 goal-conditioned 工作多侷限於短而簡單的任務（推、繩、短距離導航），本文想做時序長很多的操作。
- **Latent skill space**：與 Hausman et al.（2018）類似都學連續的技能 latent space，但 Hausman 用離散 RL 任務集合與每任務 reward 來定義空間，本文則是無標籤 play 自監督。
- **Few-shot LfD（Finn et al.、Wang et al.）**：需要 meta-training 與預定義任務分布；本文不需要。
- **Paine et al. 用 RL**：需 reward 與 RL 訓練成本；本文不需要。
- 與 Nachum et al.（hierarchical RL representation learning）導出類似架構但動機不同。

---

## 3. 方法：從 Play Data 學 Task-Agnostic Control

### 3.0 Play 的定義與假設

作者把 play 的生成過程刻畫為：操作者見到 $s_c$，依好奇心/內在動機構想一個 $s_g$（如「球在杯中」），然後從一個「對某 $(s_c, s_g)$ 而言所有可行行為的先驗分布」$p(b \mid s_c, s_g)$（即一個由 affordance/動力學知識構成的行為庫）中抽樣一個高層行為 $b$，並執行之、產出 $(O_t, a_t)$ 序列。重點：play 不是隨機動作，而是「在自身導向下的目標條件行為」。

資料集表示：
$$
D = \{(O_1, a_1), \dots, (O_T, a_T)\}, \quad O_t = \{o_{1,t}, \dots, o_{N,t}\}
$$
在實驗中 $O = \{I, p\}$，包含 RGB 影像 $I$ 與 8-DOF 本體感覺 $p$；$a_t$ 是遙控時記下的動作。

感知編碼器（per-channel）：
$$
s_t = \mathrm{concat}\big(E_1(o_{1,t}), \dots, E_N(o_{N,t})\big) \equiv \Phi(O_t).
$$

### 3.1 Play-GCBC（Play-supervised Goal-Conditioned Behavioral Cloning）

關鍵想法：從 play 中隨機抽一段長度為 $\kappa$ 的視窗 $\tau$，把它最初的狀態視為「當前」、最末的狀態視為「目標」、中間動作視為要被 imitate 的標籤——這天然成為一筆 goal-conditioned 的自監督樣本，且「最末狀態必可由開頭狀態在這些動作下達到」是內建保證。

策略：以 RNN $\pi_{GCBC}(a_t \mid s_t, s_g)$ 表示，目標函數為對隨機抽出的 $\kappa$ 步視窗做動作 log-likelihood 最大化：
$$
\mathcal{L}_{GCBC} = -\frac{1}{\kappa} \sum_{t=k}^{k+\kappa} \log \pi_{GCBC}(a_t \mid s_t, s_g),
$$
其中 $s_t = \Phi(O_t)$、$s_g = \Phi(O_{k+\kappa})$。

**問題：多模性（multimodality）**。同一對 $(s_c, s_g)$ 在 play 裡可能對應到很多種高層行為（例如同一個目標可推、可拉、可繞路），若直接做 BC，等於要單一 policy 同時擬合多種互相牴觸的動作軌跡，會被「平均掉」，學壞。

### 3.2 Play-LMP（Play-supervised Latent Motor Plans）

**動機**：用無監督表示學習解決 multimodality。若能學到 $p(b \mid s_c, s_g)$ 的緊緻表示 $z$，並讓 policy 條件在採樣到的 $z$ 上，就能把多模問題轉成 unimodal 解碼問題——policy 不再需要「隱式地表達所有高層行為」，可把全部容量用在執行單一被指定的計畫。

**架構**（seq2seq CVAE，三個元件 end-to-end 訓練）：

1. **Plan recognition encoder（後驗）**：吃整段 $\tau$，輸出 $q_\phi(z \mid \tau)$。實作為雙向 RNN $V_{\mathrm{enc}}$，輸出對角高斯：
$$
\mu_\phi, \sigma_\phi = V_{\mathrm{enc}}(\Phi(\tau)), \qquad z \sim \mathcal{N}\!\big(\mu_\phi, \mathrm{diag}(\sigma_\phi^2)\big).
$$

2. **Plan proposal encoder（學到的條件先驗）**：只吃首末狀態 $s_c, s_g$，輸出 $p_\theta(z \mid s_c, s_g)$。實作為前饋網路 $CG_{\mathrm{enc}}$：
$$
\mu_\psi, \sigma_\psi = CG_{\mathrm{enc}}(s_c, s_g).
$$
這個分布刻畫「所有可能將 $s_c$ 接到 $s_g$ 的高層行為」。

3. **Plan-and-goal-conditioned policy（解碼器）**：RNN $\pi_{LMP}(a_t \mid s_t, s_g, z)$，既是 VAE 的 decoder，也是測試時的 goal-conditioned policy。

**訓練目標**：

- KL 拉近後驗與條件先驗：
$$
\mathcal{L}_{KL} = \mathrm{KL}\!\Big(\mathcal{N}(z \mid \mu_\phi, \mathrm{diag}(\sigma_\phi^2)) \,\Big\|\, \mathcal{N}(z \mid \mu_\psi, \mathrm{diag}(\sigma_\psi^2))\Big).
$$
意義：迫使「只看首末狀態」的 plan proposal 對「真實 play 中被執行過的計畫」放高機率。

- 動作重建：
$$
\mathcal{L}_\pi = -\frac{1}{\kappa} \sum_{t=k}^{k+\kappa} \log \pi_{LMP}(a_t \mid s_t, s_g, z),
$$
其中 $z$ 從後驗以 reparameterization trick 抽出。

- 完整目標（$\beta$-VAE 式）：
$$
\mathcal{L}_{LMP} = \mathcal{L}_\pi + \beta \mathcal{L}_{KL}.
$$
取 $\beta < 1$ 是為了避免 posterior collapse（強解碼器+過度正則化導致 $z$ 被忽略）。

**符號小結**：
- $\tau$：從 play 隨機抽取的長度 $\kappa$ 視窗。
- $s_c, s_g$：視窗的首末狀態（自監督生成的「當前」與「目標」）。
- $z$：latent plan，刻畫「我要怎麼從 $s_c$ 走到 $s_g$」的高層行為意圖。
- $q_\phi(z \mid \tau)$：plan recognition（看整段執行了什麼）。
- $p_\theta(z \mid s_c, s_g)$：plan proposal（只看首末就分布性地預測有哪些可行計畫）。
- $\pi_{LMP}(a_t \mid s_t, s_g, z)$：以當前狀態、目標、計畫為條件的低階策略。

### 3.3 測試時的 Task-Agnostic 控制

測試時 $V_{\mathrm{enc}}$ 被丟棄。流程：

1. 收到當前觀測 $O_c$ 與使用者指定的目標影像 $O_g$，編碼為 $s_c, s_g$。
2. 餵入 plan proposal $CG_{\mathrm{enc}}$，得到 $z$ 分布並抽樣一個 $z$。
3. 用 $\pi_{LMP}$ 在環境中閉迴路執行，30 Hz 取觀測、出動作。
4. **Replanning**：每 $\kappa = 32$ 步（約 1 Hz）重新推斷與抽樣一次 $z$，與訓練時的計畫水平一致。

---

## 架構流程（圖示文字化）

訓練：
```
play stream D
   ├─ 隨機取一段視窗 τ（長度 κ）
   ├─ 取首末狀態 (s_c, s_g) ← Φ(O_t), Φ(O_{t+κ})
   │
   │       ┌──────── V_enc (bidir RNN, plan recognition) ──── q(z|τ)
   │  τ ──>┤                                                       │
   │       │                                                       │ KL 拉近
   │       │  ┌─ CG_enc (FFN, plan proposal) ──── p(z|s_c, s_g) ───┘
   │       │  └  輸入 (s_c, s_g)
   │
   ├─ 從 q(z|τ) reparam 抽 z
   └─ 對每個 t∈τ 將 (s_t, s_g, z) 餵入 π_LMP → 重建 a_t
```

測試：
```
(O_c, O_g) → Φ → (s_c, s_g) → CG_enc → 抽樣 z
循環 (每 1 Hz 重抽 z):
   for k 步: (s_t, s_g, z) → π_LMP → a_t → 環境 → s_{t+1}
```

---

## 4. 實驗與結論

### 4.1 設置

- **環境**：模擬 Playground，8-DOF 機械臂 + gripper，桌面上有滑門、抽屜、矩形方塊、3 個控制燈光的按鈕。
- **任務**：18 個視覺操作任務（拉抽屜、推按鈕、抓取、推開、滑門開關、放入層架、清掃等）。
- **比較基線**：
  - BC：每任務各 100 條專家示範各訓一個 policy（共 1800 條、約 1.5 小時專家資料）。
  - Multitask BC：所有 18 任務示範一起訓、條件於 task id。
  - Play-GCBC、Play-LMP：訓在最多 ~7 小時的未標籤 play data 上。
- **兩種觀測設定**：pixel 與 state（用 ground-truth 物件位姿）。

### 4.2 主要結果（Table/Fig. 6a）

| Method | labels | input | success |
|---|---|---|---|
| BC (18 個專家 policy) | labeled | pixels | 66.5% ± 12.1 |
| Play-GCBC（單一通用 policy） | unlabeled | pixels | 58.7% ± 11.6 |
| **Play-LMP（單一通用 policy）** | unlabeled | pixels | **69.4% ± 10.8** |
| BC | labeled | states | 70.3% |
| Multitask BC | labeled | states | 66.2% |
| Play-GCBC | unlabeled | states | 77.9% |
| **Play-LMP** | unlabeled | states | **85.5%** |

關鍵觀察：

1. **一個沒有任務標籤的 Play-LMP 通用策略勝過 18 個各自有標籤示範的 BC 專家策略**（回答了問題 1、2）。
2. **Play-LMP > Play-GCBC**：把「計畫推斷」與「計畫解碼」解耦帶來系統性提升（回答問題 3），Fig. 7 顯示在許多任務上有達 50 個百分點的絕對提升。

### 4.3 Scalability（Fig. 8）

只用 30 分鐘 play data 訓 Play-LMP 即可達到 71.8% 平均成功率，已勝過用 90 分鐘專家示範訓的 18 個 BC 模型（70.3%）。play 不僅本身更便宜，效率還更高。

### 4.4 Robustness 與 Emergent Retrying（Fig. 6b）

對初始物件位置加入擾動（0–40 cm）的測試：play 訓出的兩個模型（Play-LMP、Play-GCBC）都顯著比 BC 更 robust。

作者把這歸因於 **play 的覆蓋特性**：play 不僅多樣，還因人類好奇心而會把同一個目標嘗試多種路徑，自然包含「失敗後再試一次」「換個角度抓」「碰到邊緣後修正」這類「recovery sub-trajectory」。BC 的專家資料則只給「乾淨成功的一條路徑」，模型一旦離開分布就無法回拉。

質性上他們也觀察到 play-supervised 模型出現 **emergent retrying**：抓取失敗後會自動再嘗試一次，不是 BC 那種失敗就卡住的行為。

### 4.5 Unsupervised Task Discovery（Fig. 4）

把 512 條 play 視窗 + 全部驗證任務示範餵進 plan recognition encoder $V_{\mathrm{enc}}$，做 t-SNE。結果：雖然訓練從未看過任務標籤，latent plan 空間仍然 **按功能自發分群**（drawer 區、button 區、grasp 區、sweep 區、sliding 區等）——亦即 $z$ 確實當成了「行為類別」的中介表示。

### 4.6 Conclusion

主張：用未標籤 play data 學整段連續技能譜系比針對離散任務收專家示範更可擴展、更 robust。Play-LMP 在沒有任務標籤下自監督地發現任務語義、學成完整深度感知與控制堆疊。未來工作：對 unseen 物件/環境的泛化、處理 play 分布不平衡的問題。

---

## 為什麼 latent plan 是合理的中介表示？（重點梳理）

文中（及附錄 A.1.1）論證可這樣理解：

1. **問題本質的多模性**：給定 $(s_c, s_g)$，存在一個由 affordance/動力學決定的高層行為分布 $p(b \mid s_c, s_g)$。若直接做 $\pi(a \mid s_c, s_g)$ 的 MLE，模型必須在輸出層自己處理這個多模性——但 unimodal Gaussian / 對角輸出根本表達不了，多模之間互相平均化會破壞學習。
2. **latent 變數做「行為選擇」**：引入 $z$ 表「我打算用哪種方式走」，則 $\pi(a \mid s_c, s_g, z)$ 條件在已選定的計畫上後可以是 unimodal 的。多模性全部移到 $p(z \mid s_c, s_g)$ 這個離散選擇/連續混合的 prior 上。
3. **CVAE 自然把表示學習和控制學習綁在一起**：plan recognition 學「對整段執行序列做摘要」、plan proposal 學「從首末狀態預測該摘要的分布」、decoder 學「在已知摘要與目標下產生動作」。decoder 拿來當測試時的 policy 是免費的——這就是「plan representation learning 等價於 goal-conditioned control」的關鍵。
4. **與 play 的相容性**：play 中對同一 $(s_c, s_g)$ 自然存在多條成功路徑（人會嘗試多種方法），剛好提供 $p(z \mid s_c, s_g)$ 多模結構的訓練訊號。專家示範通常每任務只有一條 canonical 軌跡，學不出有意義的 $z$ 分布。

---

## 為什麼從 play 學出來的行為會比較有 retry / recovery 性質？

1. **資料中本就含 recovery 子片段**：play 是連續、不 reset 的長序列，當人類嘗試失敗（手滑、抓不到、碰到邊），下一個動作就是「修正、重抓、調整角度」。隨機取的視窗 $\tau$ 會自然包含這些「失敗 → 修正 → 成功」的小片段；專家示範則被嚴格篩選為乾淨成功軌跡，不含這類訊號。
2. **覆蓋廣 + replan 機制**：play data 對狀態空間覆蓋更廣（4.2× 對比專家），當測試中執行偏離預期時，所到狀態仍在訓練分布之內、$\pi_{LMP}$ 有對應的動作分布；加上每 1 Hz 重抽 $z$ 的 replanning，模型可以即時把計畫切換成「先回正、再嘗試」。
3. **multimodality 被 $z$ 正確處理**：失敗後若有多種修正方式，$p(z \mid s_c, s_g)$ 可以提出不同 $z$，下一個 1 秒視窗就嘗試另一種策略；BC 只會死命重複同一條「平均路徑」。
4. **goal 在 episode 內保持固定**：因為 $s_g$ 是使用者給定的固定 frame，只要還沒到 $s_g$，閉迴路系統就會持續被驅動到目標——這在訓練時就成立（視窗終點是 $s_g$，中途失敗就會被修正），測試時也順理成章地產生 retry。

---

## 與本研究主線的關聯

我的主線：robot **play data** + 極少弱語言（hint level）+ offline meta-RL + 用文字將任務做 task spec / task embedding，目標是在新任務上 zero-shot；研究上想兼顧 meta-learning 與 zero-shot 兩條軸的並行。Play-LMP 對這條主線是重要的「先行作」。對應關係：

1. **Play data 作為主資料假設的合法性背書**：本論文以實證明確顯示 play 在 coverage 上對專家示範與隨機探索有壓倒性優勢、且只需 30 分鐘就能勝過 90 分鐘的 18 任務專家示範。對我「以 play 為唯一/主要訓練資料、不依賴 per-task demonstration」的設定提供強力證據；意味著我可以理直氣壯地 **拒絕 per-task labeled demonstration / per-task reward 的成本**，把人力預算集中在弱語言註記上。

2. **Latent plan ↔ task embedding 的對應**：Play-LMP 的 $z$ 在沒有任務標籤下竟然會「按任務功能分群」（Fig. 4），這正是我希望的——一個 unsupervised 學到的 latent 任務空間。我的工作把這個對應再推一步：
   - Play-LMP：$p_\theta(z \mid s_c, s_g)$，**目標狀態（影像）** 驅動 $z$。
   - 我的方向：把 conditioning 改成或增廣為「**自然語言/弱 hint**」，即學 $p_\theta(z \mid s_c, \ell)$ 或 $p_\theta(z \mid \ell)$。Play-LMP 已經證明 latent 空間會自發以任務分群，只要再加一個「文字 → $z$ 分布」的 mapping，就有機會把語言當 zero-shot 的任務 specifier。
   - 換言之，**latent plan $z$ 就是我的 task embedding 的雛形**，我只需把它的條件接口從「goal image」換到「language / spec」。

3. **與 offline meta-RL 的銜接**：Play-LMP 是 supervised reconstruction，不做 reward；但其架構天然適合改造為 offline meta-RL 的 task encoder——把「視窗 $\tau$」視為「task context」，把 plan recognition 視為 task inference $q(z \mid \mathrm{context})$、把 plan proposal 視為 zero-shot 的 prior（由語言或極少 hint 給定）。這就接上 [[PEARL|PEARL]] / [[VariBAD|VariBAD]] 的精神：context-based 的 task inference + 條件 policy。Play-LMP 等於先示範了「在 supervised 條件下這套 task-context inference 是可學的」。

4. **Retry / recovery 對 generalist 機器人特別重要**：我的目標若是「拿一段文字 hint 就在未見過任務上零樣本執行」，那測試時 distribution shift 與部分失敗幾乎必然發生。Play-LMP 顯示 play data 訓出的模型本身就帶 retry 行為——這意味著我若以 play 為基底，可以較少擔心測試時 fragile，這對 zero-shot 場景特別關鍵。

5. **方法層面的具體啟示**：
   - 視窗式自監督取樣可直接搬到我的流程：每個 batch 抽 $\tau$，首末做 self-supervised label。
   - KL 拉近 recognition 與 proposal 的兩個分布的技巧很乾淨，可作為 language→plan 的訓練骨架（recognition 看軌跡產生 $z$，proposal 看語言產生 $z$）。
   - $\beta$-VAE 的權重設定、replanning 頻率（$\kappa = 32$, 1 Hz）等都是可直接借鑑的工程經驗。
   - 缺點：純 image-goal 沒有處理「抽象任務描述」與「語言 hint 中的 ambiguity」；這正是我要補的角色。

---

## 一句話總結

Play-LMP 用一個 seq2seq CVAE，從便宜、豐富、無分段的人類遙控 play data 中自監督地學出一個「按功能分群」的 latent plan 空間 $z$，並以此把多模的 goal-conditioned 控制問題切成「先選計畫 $p(z \mid s_c, s_g)$、再條件解碼 $\pi(a \mid s_t, s_g, z)$」兩步——單一通用策略因此能勝過 18 個專家策略、對擾動 robust、且自然出現 retry 行為，為「以 play 為基底、把語言當作 task specifier 做 zero-shot」的工作奠定了關鍵基礎。
