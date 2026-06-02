---
type: paper-note
aliases:
  - "Language Conditioned Imitation Learning over Unstructured Data"
  - "LangLfP"
  - "MCIL"
year: 2021
stage: "4-play與弱語言"
tags:
  - play-data
  - language-conditioned
  - imitation
  - shared-latent-goal
  - 4-play與弱語言
summary: "shared latent goal space同時吃goal-image與語言，1%標註；但還不是meta-RL（無task posterior）。"
---
> **論文標題**：Language Conditioned Imitation Learning over Unstructured Data
> **作者**：Corey Lynch, Pierre Sermanet（Robotics at Google）
> **出處 / 年份**：Robotics: Science and Systems (RSS) 2021
> **主題**：在大量「無語言標註、無任務切分」的遙操作 play data 上，學一個吃「自然語言指令」或「目標影像」的端到端視覺動作策略（LangLfP / Multicontext Imitation Learning, MCIL）
> **整理目標**：把這篇與我「robot play + 極少弱語言 + offline meta-RL + 文字→task spec/embedding 做 zero-shot」的研究主線對齊：搞清楚 shared latent goal space 是什麼、unstructured data 如何被使用、為什麼它「不是」meta-RL，以及這對我要做的方向意味著什麼

---

## 0. 閱讀總覽（白話版）

這篇論文要做的事，可以一句話講：**讓機器人聽人話做事，但不要靠很多有標註的示範**。

過去的模仿學習要嘛要 one-hot 任務 id，要嘛要目標影像，要嘛要在很受限的環境（2D 格子、簡化動作、合成語言）裡用 RL 跟著指令做事。對真實機器人來說都不太實際。作者想要：

1. 大量收集**沒有任務標籤**的「play data」（隨便玩，但是有意義地玩各種物件）。
2. 只標註其中極少（< 1%）的片段為「事後語言」，亦即把一段 1–2 秒的影片給人看，問「如果要叫 agent 從這個畫面走到那個畫面，你會講什麼？」拿到自由形式的自然語言指令。
3. 訓練一個策略，它能同時接受「目標影像」或「自然語言」，把兩者編碼到同一個 latent goal space $z$，然後決定動作。這就是 **Multicontext Imitation Learning（MCIL）**。
4. 套上預訓練的多語句子向量（MUSE），就能對沒看過的同義詞、甚至 16 種語言保持穩定。

核心訊息：**用大量無結構 play 學感知與控制，用極少語言標註學語言→latent goal 的對齊**，藉由共享 latent goal space 把兩邊縫合起來。它強的地方是用模仿學習做 long-horizon 語言條件操控，弱的地方是：訓練/測試任務同分布，**這不是 meta-learning，也不是 zero-shot 對「全新任務」泛化**。

---

## Abstract（重點摘譯）

- 過往模仿學習要 task id 或 goal image，這在開放世界不實際；過往 instruction following 又只能在簡化環境跑。
- 本文：把自由形式自然語言併入模仿學習，端到端學感知（像素）、語言理解、多任務連續控制。
- 與前人不同：**能利用大量未標註、無結構示範資料**，把語言標註成本壓到 < 1%。
- 測試時單一策略可在 3D 桌面環境根據語言依序做許多操控任務（"開抽屜...拿方塊...按綠按鈕..."）。
- 加上**預訓練語言模型**，能對 OOD 同義詞、多語指令更穩健，不需要新示範。

---

## I. Introduction（精讀重點）

問題設定：人對機器人下一串自由形式自然語言指令 $\{l_0, ..., l_N\}$，每條描述一個短任務；機器人要用高頻連續控制完成。

過去的 instruction following 研究有幾個典型限制：(a) 2D 環境、(b) 簡化致動器（離散 pick-and-place）、(c) 合成且詞彙/語法受限的語言。這篇要打的設定是：**自然語言 × 像素輸入 × 8-DOF 連續控制 × 多階段操控**，作者自稱是這個組合的第一次嘗試。

關鍵設計訴求（也是這篇與「主線研究」最需要對照的點）：
- 怎麼用最少的語言標註把語言映射到動作？
- 怎麼利用大量沒有語言標籤的示範資料？
- 訓練集有限時，怎麼在測試時跟最多種指令？

貢獻：
1. 定義「語言條件視覺操控」設定。
2. 提出 MCIL（多 context 模仿學習），能同時用 goal image / language / task id 等異質 context 訓練。
3. 證明 < 1% 語言標註就能做出長 horizon 的語言條件視覺策略。
4. 提出把任意語言條件策略接到大型預訓練語言模型的簡單做法（TransferLangLfP），讓 agent 對 OOD 同義詞與 16 種語言穩健。

---

## II. Related Work（精讀重點）

- **從通用感測器學機器人技能**：IL vs. RL，本篇選擇 behavior cloning，因為穩定且實作簡單。
- **從大量無結構資料做模仿**：把多個 skill 從未切分的示範裡同時學起來（[16, 26, 14]）。這篇繼承此線，但讓策略也能吃語言。
- **Task-agnostic control**：把任何可達狀態當作 goal 來訓 policy。Goal relabeling（hindsight）讓任何走過的狀態都能成為「達成的目標」。本文承繼此設定，但把 goal image 換成 / 加上 language。
- **Multicontext learning**：把跨任務（如 [5]）與跨目標（如 [40]）泛化推廣到「跨異質 context」。當其中一種 context 多、另一種少時，等價於透過 shared goal space 做 transfer learning。

---

## III. Problem Formulation（符號與設定）

策略 $\pi_\theta(a \mid s, l)$：
- $a \in \mathcal A$：下個動作（這裡是 8-DOF 連續控制）。
- $s \in \mathcal S$：當前 onboard 觀測，$\mathcal S = \{\text{image}, \text{proprioception}\}$（注意不是真實環境狀態）。
- $l \in \mathcal L$：自由形式自然語言，無詞彙、語法限制。

測試時人類給一串 $\{l_0, ..., l_N\}$，策略以 30 Hz 閉迴路控制。

對照 baseline：
- 標準 IL：$\pi_\theta(a \mid s)$，資料 $\mathcal D = \{\tau_i\}$，$\tau = \{(s_0, a_0), ...\}$。
- Goal-conditioned IL：$\pi_\theta(a \mid s, g)$，$g \in \mathcal G$。當 $g = s_g \in \mathcal S$ 時，任何走過的狀態都可被 hindsight relabel 成「達成的目標」。

當把這個 framework 用在 unstructured play data $\mathcal D_{\text{play}}$ 時，是直接最大化：
$$
\mathcal L_{\text{GCIL}} = \mathbb E_{(\tau, s_g) \sim \mathcal D_{\text{play}}}\Bigl[\,\sum_{t=0}^{|\tau|} \log \pi_\theta(a_t \mid s_t, s_g)\,\Bigr].
$$
但當把 $s_g$ 換成 $l$ 時，**沒辦法輕易把任何 $s$ hindsight 成 $l$**，因為 $\mathcal G = \mathcal L$ 不再等於 $\mathcal S$。這就是為什麼需要 MCIL 來打通。

---

## IV. Learning to Follow Human Instructions from Unstructured Data（方法核心）

整體流程（與 Fig. 1 對應）：
1. **收集**：用遙操作收一段沒切分、沒語言標的長 play data。
2. **Relabel**：把 play data 切成大量 1–2 秒視窗 $(s_{i:i+w}, a_{i:i+w})$，把窗末狀態當作 $s_g$，得到 $\mathcal D_{\text{play}}$（goal image demos）。再對其中極少數（$K \ll |\mathcal D_{\text{play}}|$）視窗請標註者「事後」寫一條自然語言指令，得到 $\mathcal D_{(\text{play, lang})}$。
3. **Multicontext imitation**：訓練一個共享策略，可吃 goal image 或 language。
4. **測試時只用語言條件**。

### IV-A. Pairing unstructured demonstrations with natural language

語言落地（grounding）做法：把 play 隨機切窗 → 給人看影片 → 問「要叫 agent 從第一幀走到最後一幀，你會說什麼？」 → 拿到自由形式語言。鼓勵自然、不要固定詞彙語法。

得到 $\mathcal D_{(\text{play, lang})} = \{(\tau, l)_i\}_{i=0}^{D_{(\text{play,lang})}}$。重點：**不需要把每個 play 視窗都配語言**，後續 MCIL 會利用未配的 $\mathcal D_{\text{play}}$。

### IV-B. Multicontext Imitation Learning（MCIL，本篇關鍵概念）

抽象設定：有 $K$ 個 contextual imitation 資料集 $\mathcal D = \{\mathcal D^0, ..., \mathcal D^K\}$，每個 $\mathcal D^k = \{(\tau_i^k, c_i^k)\}$ 用不同 context（如 one-hot id、goal image、language）描述任務。MCIL 同時學：
- 一個共享的 latent goal 條件策略 $\pi_\theta(a_t \mid s_t, z)$；
- 每個 context type 一個 encoder $f_{\theta^k}$，把 $c^k$ 映射到 **shared latent goal space** $z = f_{\theta^k}(c^k) \in \mathbb R^d$。

訓練目標（演算法 1）：
$$
\mathcal L_{\text{MCIL}} = \frac{1}{|\mathcal D|}\sum_{k=0}^{K} \mathbb E_{(\tau^k, c^k) \sim \mathcal D^k}\Bigl[\,\sum_{t=0}^{|\tau^k|} \log \pi_\theta\bigl(a_t \mid s_t,\ f_{\theta^k}(c^k)\bigr)\,\Bigr].
$$
每步從每個資料集各抽一個 batch，把對應 context 編成 $z$，計算 imitation loss 後平均，policy 與所有 encoder 一起端到端優化。

> **這就是「shared latent goal space」的本義**：不是一個學出來的「任務分布先驗」，而是一個「**目標表徵空間**」$\mathbb R^d$。所有種類的任務描述（影像、id、語言）都被各自的 encoder 投到這個共同空間，下游 policy 只認 $z$，不認 context 是哪種模態。這讓「便宜大量的 goal image hindsight」與「昂貴稀少的語言標註」可以彼此補強。

### IV-C. LangLfP：吃 image 或 language 都行

LangLfP = MCIL 套在這個問題上的特例，$\mathcal D = \{\mathcal D_{\text{play}}, \mathcal D_{(\text{play, lang})}\}$，encoder $F = \{g_{\text{enc}}, s_{\text{enc}}\}$ 分別吃 goal image 與文字。每個訓練步：
1. 從 $\mathcal D_{\text{play}}$ 抽 image-goal batch，從 $\mathcal D_{(\text{play, lang})}$ 抽 language-goal batch；
2. 各自編碼到 $z$；
3. 算 MCIL 目標、一起對感知、語言、控制模組做梯度下降。

底層 policy 沿用 LMP（Latent Motor Plans, Lynch et al. 2019）的 seq2seq CVAE：有 posterior $q(z^p \mid \tau)$、prior $p(z^p \mid s_0, z^g)$、以及 plan + goal 條件的 decoder $p(a_t \mid s_t, z^g, z^p)$，原本吃 $s_g$ 的地方一律換成吃 $z^g$。

### TransferLangLfP（接預訓練語言模型）

把 $s_{\text{enc}}$ 的「從頭學」換成把語言句子先丟到 Multilingual Universal Sentence Encoder (MUSE, [51])，得到 512 維句向量，再用 2 層 2048 ReLU MLP 投到 latent goal space。MUSE 權重凍結。這帶來兩個好處：
- 感知/控制收斂更快、最終表現更高。
- 多語、同義詞（OOD-syn / OOD-16-lang）零示範就能跟。

> **重要：這裡的「zero-shot」是「OOD 語言指令對應到訓練分布內任務」，不是 zero-shot「新任務」**。

---

## 架構流程（端到端視角）

訓練時的資料流：
```
play (無語言) ───hindsight goal image──►  (τ, s_g)  ──g_enc──►  z ┐
                                                                 ├──► π_θ(a_t | s_t, z) ──► MCIL loss
play 子集(配語言) ──hindsight 指令──────►  (τ, l)    ──s_enc──►  z ┘
```
測試時：人類打字 → $l$ → $s_{\text{enc}}(l) = z$ → 機器人觀測 $O_t$ → policy 出動作 → 環境前進；人可以隨時換 $l$。

感知模組：3 層 Conv + spatial softmax + FC → 64 維視覺向量，concat 8 維 proprioception → 72 維「感知嵌入」。Language encoder：subword 分詞 → 8 維 embedding lookup → avg pool → 2 層 2048 ReLU MLP → 32 維 latent goal（或從 MUSE 512 維開始）。控制模組：LMP CVAE。動作離散化為 256 bin / 維。

---

## V. Experiments & Conclusion（精讀重點）

環境：[26] 的 3D Playroom；8-DOF 機械臂、200×200 RGB、30 Hz；桌上有滑門、抽屜、可動方塊、3 顆按鈕、垃圾桶。18 種子任務，長 horizon benchmark 為 Chain-2 / Chain-3 / Chain-4（最高 925 個 4-stage 任務串）。

### 比較對象

- **LangBC**：在預定義的 18 task × 100 expert demos 上配 hindsight 指令學的 baseline，沒用 play data。
- **LfP**：用 play data 但只吃 goal image。
- **LangLfP**（本文）：用 play + 語言。
- **Restricted LangLfP**：把 play 限制成與 $\mathcal D_{(\text{demo, lang})}$ 同樣大小，做受控比較。
- **TransferLangLfP**（本文）：在 MUSE 上接 LangLfP。

### 主要結果（Table I）

像素版（pixel input）：
- LangBC：18 task 20.0%、Chain-4 7.1%
- Restricted LangLfP：47.1% / 25.0%
- LfP（goal image）：66.4% / 53.0%
- LangLfP：68.6% / 52.1%
- TransferLangLfP：74.1% / 61.8%

State 版（用模擬器狀態當輸入，上限視角）：LangBC 38.5% / 13.9%；TransferLangLfP 90.5% / 71.8%。

主要訊息：
- LangLfP **追平** goal-image-only LfP，等於把任務指定方式變得更人性化卻幾乎沒掉性能（Q1）。
- 用大量 unstructured play **顯著贏過** 只用結構化 demo 的 LangBC（Q2）；即使 play 被限制成同樣大小，Restricted LangLfP 也贏 LangBC，說明 unstructured 與大規模兩個因素皆有貢獻。
- 預訓練語言模型對機器人操控有正向轉移（Q3），尤其有助於收斂與最終性能。
- TransferLangLfP 對 OOD 同義詞（~15k 句）達 60.2%、對 16 種語言（~240k 句）達 56.0%，LangLfP 只有 37.6% / 27.9%（Q4）。
- 模型規模愈大、unstructured data 的優勢愈明顯；反之 LangBC 在模型放大後反而退化（過擬合 / 多樣性不足）。

### 限制與未來工作（作者自述）

- 純 IL，沒有自主改進機制，建議將來與 RL 結合（類似 [14] 對 LfP+RL）。
- 訓練/測試任務 i.i.d.，同房間、同物件；尚未測試對「新房間 / 新物件」的泛化。
- Compounding error 仍會發生（手臂進入奇怪姿態出不來），實務上可用人類語言介入救援。

### 結論

提出可擴展的「自由形式語言 × 多任務模仿」框架；關鍵是 MCIL 把 unstructured data 與少量語言透過 shared latent goal space 縫合；再用大型預訓練語言模型獲得指令層的穩健性。

---

## 相關工作（與我主線最關的幾線）

1. **[[LatentPlansFromPlay|Learning Latent Plans from Play]] (LfP, Lynch et al. 2019)**：本篇直接繼承的 backbone；play data + hindsight goal image + LMP（CVAE）。
2. **HER（Andrychowicz et al.）/ goal relabeling**：把走過的狀態當 goal，讓任何軌跡都有監督。
3. **Goal-conditioned IL（Ding et al.）**：用 goal image 做 IL，本文把它擴成多 context。
4. **UVF / multitask 泛化（Schaul et al.; Caruana）**：本篇把「跨 task」與「跨 goal」泛化合一成「跨 context」泛化。
5. **BERT / GPT / MUSE**：把預訓練語言向量遷移到機器人控制，這篇是早期成功案例之一。
6. **Instruction following 在 RL 的 2D / 簡化動作**（BabyAI, Hermann et al., Misra et al., Yu et al.）：本篇刻意打 3D + 連續控制 + 自然語言的組合。
7. **同期 IL instruction following（Stepputtis et al. 2020）**：對方要標註 task demo + 預訓練物件偵測，本篇相反——完全端到端、未標註 play。

---

## A. 與本研究主線的關聯

我的主線是「robot play + 極少弱語言 + offline meta-RL + 文字 → task spec / task embedding 做 zero-shot」，並關注 meta-learning × zero-shot 的並行方向。這篇 LangLfP 是 backbone 級的近親，但**還沒有跨進 meta-RL**，差別需要看清楚。

### A.1 對我有用的東西

1. **shared latent goal space $z$ 是「任務表徵共用空間」的最小可行模板**。
   - 它的設計觀念剛好對應我想做的「文字 → task spec / task embedding」：在 LangLfP 裡，$s_{\text{enc}}(l)$ 就是「文字 → task embedding」。差別在於它的下游是 IL policy，而我的下游想是 offline meta-RL 的 task-conditioned policy / context-conditioned value function。
   - **我可以直接借用「多 encoder → 同一 $z$」這套架構**，把 task spec（goal image、語言、demo trajectory snippet、reward sketch）都打到同一個 $z$ 上，讓 meta-policy 對 context 來源不敏感。

2. **unstructured data 的利用方式可以直接搬**。
   - 它用 hindsight relabel 把無標 play 變成 (trajectory, $s_g$) supervision；對應到 offline meta-RL，可以用同樣思路把 robot play 的軌跡切成短窗，每個短窗自動產生一個「pseudo-task」（窗末狀態 / pseudo-reward / 從文本標註而來的弱語言），讓 meta-learner 在「任務數爆量、每任務資料極少」的環境裡學會 context → policy 的對應。
   - 「< 1% 的語言標註」這個比例是個強而具體的工程目標：對我而言意義是「我可以收 100h play，只標 30 分鐘語言」，是可實現的弱語言預算。

3. **預訓練語言句向量帶來的 OOD-synonym 穩健性**幾乎是「文字 → task embedding」這條路的必經之路：直接吃凍結的多語 sentence encoder，輸出穩定到可作 zero-shot 同義/跨語的 task spec。我做 zero-shot 任務泛化時應採同樣設計。

4. **MCIL 的訓練流程**對「混合多種 task descriptor」的 offline meta-RL 設定有直接啟發：每 step 從不同 descriptor 對應的資料集分別抽 batch、各自 encode、共用 policy/Q、loss 平均。

### A.2 它與「我要做的方向」差在哪：為什麼還不是 meta-RL

這篇本質上是 **goal-conditioned imitation learning**，**不是 meta-learning**，也不是 meta-RL：

| 維度 | LangLfP | 我要的 offline meta-RL × 文字 zero-shot |
| --- | --- | --- |
| 學習問題 | 監督式 BC（最大化 $\log \pi(a\|s,z)$） | RL / offline RL，要最大化 return；要能利用次優或非示範資料 |
| 任務分布 | i.i.d. 訓練/測試，同房間同物件 | 明確 train tasks / unseen test tasks；對 unseen 要 zero-shot 或 few-shot |
| 「meta」結構 | 沒有；只有「context-conditioned policy」 | 有 inner/outer 或 belief-based 結構（PEARL / VariBAD）；context 是 task 的後驗變數 |
| 改進機制 | 純模仿，沒有自我改進 | 有 reward signal，可離線策略改進；可結合 task inference |
| 文字角色 | 直接條件化 BC 的 goal | 文字是 **task descriptor / 弱 reward / posterior prior**；可在沒有環境互動下做 task identification |
| Adaptation | 沒有 adaptation 階段；測試直接 forward | 有 adaptation：靠 trajectories 或文字推 task posterior，再選 policy |
| 評估 | 同分布長 horizon | **跨任務 generalization、zero-shot 對新任務描述** |

換句話說，LangLfP 把「任務指定 = $z$」這件事做得很乾淨，但它沒回答：
- **遇到全新任務時怎麼辦？** 它只能依賴 MUSE 把新文字映到舊任務鄰近區域；對真正未見過的 task family 沒有 adaptation 機制。
- **task posterior 怎麼推？** 它沒有「任務不確定性」的概念；不像 [[PEARL|PEARL]]/[[VariBAD|VariBAD]] 會用 context 推 $p(z \mid c_{1:t})$。
- **怎麼用非示範資料？** 它要的是 demo（再加 hindsight），雖然 play 沒任務標但仍是「人覺得有意義的軌跡」；它不能直接吃失敗軌跡 / 隨機探索資料。

### A.3 我可以怎麼把它接到我的主線上

1. **把 LangLfP 的 $z$ 升級為「task latent」**：把 $z$ 視為 meta-RL 裡的 task variable，給它先驗 $p(z)$、後驗 $q(z \mid c)$，其中 $c$ 可以是「文字 + 少量 trajectory」。MCIL 結構自然能擴展成「多 descriptor → 同一 task posterior」。
2. **以 hindsight 弱語言 + reward-free play 餵 offline meta-RL**：拿 play 切窗、極小比例配自然語言，多數窗以 pseudo-reward（goal-reaching 距離、子目標到達訊號）餵 offline RL；用 MCIL-style 共享 encoder 讓「文字 task spec」與「goal-image task spec」對同一個 $z$ 收斂。
3. **Zero-shot 評估設計**：除了 i.i.d. 同分布，要刻意切出 unseen task descriptor（新物件、新詞）測試；可借鏡 LangLfP 在 OOD-syn / OOD-16-lang 的設計範式。
4. **明確切分 backbone 與 meta 部分**：感知 + 控制可直接借 LMP / LangLfP；上面再疊「task inference + adaptation」（PEARL-style context encoder 或 VariBAD-style belief over $z$）才是「我加值」的位置。

簡言之：**LangLfP 給了我 task-encoder + 共享 latent goal + 弱語言 hindsight 的工程藍圖；它沒給的是 meta-RL 的 task posterior / adaptation / 對未見任務的 zero-shot 結構，這正是我要補上的洞**。

---

## B. 一句話總結

LangLfP 用 MCIL 把「大量無標 play 的目標影像 hindsight」和「< 1% 的事後自然語言標註」投到同一個 latent goal space $z$，學出能聽自由文字做長 horizon 操控的端到端 IL 策略；它解決了「弱語言 + unstructured data」的監督問題，但仍是 goal-conditioned IL，缺 meta-learning 的 task posterior 與對未見任務的 zero-shot 結構，因此是我「offline meta-RL × 文字 task embedding × zero-shot」主線的 backbone 級近親，而不是同一件事。
