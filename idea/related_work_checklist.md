---
type: search-checklist
aliases:
  - "related work 查詢清單"
tags:
  - 研究主線
  - related-work
  - zero-shot
  - 待辦
summary: "查證『歧義多模態任務規格 → task belief → zero-shot』有沒有撞題用的搜尋清單與結果記錄模板。"
---

# Related Work 查詢清單

> 為什麼要這份：[[zero_shot_proposal|zero-shot 創新提案]] 的核心是「歧義的多模態任務規格 → task 後驗信念 → risk-aware zero-shot」。投 conference 前必須確認 2024–2025 沒有人已經做過同一件事，否則會被視為抄襲或增量。沙箱無法上網，所以這份清單給你拿到能上網的電腦執行。

---

## 怎麼用

1. **平台優先順序**：OpenReview（看尚未發表的最新動向）→ arXiv（看最新預印本）→ Google Scholar（看引用與年份）→ Semantic Scholar（看相關推薦）。
2. **時間範圍**：限定 **2023–2026**（含尚未正式 publish 的 2025/2026 ICLR、NeurIPS 投稿）。
3. **每組查詢**：看標題與 abstract，符合「歧義 + 規格(語言或圖) + belief/uncertainty + zero-shot」其中至少**兩個**關鍵字組合的論文，記到「## 找到的相關工作」。
4. **回報給我**：把結果整理成「標題 / 年份 / 一句話做什麼 / 你覺得跟我們的提案多近」貼回來，我幫你判斷要怎麼區隔或調整。

---

## 查詢字串(按重要性排序，最關鍵的先查)

### 第一優先：直接撞題檢查(必查)
這些是「如果有人做了一樣的題目,我們就完蛋」的關鍵字。

1. `"task uncertainty" "in-context" reinforcement learning`
2. `"posterior over tasks" language instruction robot manipulation`
3. `ambiguous instruction robot manipulation belief uncertainty`
4. `"goal image" ambiguity reinforcement learning posterior`
5. `multimodal task specification belief robot manipulation`

### 第二優先:相鄰方向(看誰已經做了一半)

6. `Bayesian in-context reinforcement learning task inference`
7. `uncertainty-aware in-context reinforcement learning`
8. `risk-aware language-conditioned policy robot`
9. `compositional zero-shot generalization robot manipulation CALVIN`
10. `task posterior diffusion policy robot`

### 第三優先:技術底座的最新發展(影響你方法的現代性)

11. `algorithm distillation 2024 2025 uncertainty`
12. `decision-pretrained transformer task inference`
13. `in-context reinforcement learning meta-RL 2025`
14. `Prior-Fitted Networks PFN reinforcement learning task`
15. `transformer Bayesian meta-learning robotics`

### 第四優先:近鄰的後續工作(T2DA / LUMOS / ZeST 的衍生)

16. T2DA 的引用 → `cites:T2DA` 或在 Semantic Scholar 看 T2DA 的 **citations**
17. LUMOS 的引用 → 同上
18. ZeST 的引用 → 同上,特別注意有沒有人把它加上 belief / meta-RL
19. `CALVIN 2024 2025 zero-shot` → 看 CALVIN 上最新的方法

---

## 查證的判斷準則(看到論文後怎麼判斷威脅程度)

| 威脅度 | 條件 | 對策 |
| --- | --- | --- |
| **🔴 高(撞題)** | 同時做了:歧義規格 + belief 分布 + zero-shot | 我們的角度要大改,可能要轉到角度 B(組合泛化)或 C(可靠性) |
| **🟡 中(部分重疊)** | 做了三項中的兩項(例如:有 belief 但只用語言、或有歧義但沒 belief) | 在我們的 paper related work 明確區分,強調剩下那一項是我們的 novelty |
| **🟢 低(相關但不撞)** | 只做了一項,或在不同 setting | 列為 related work,不用改方向 |

---

---

## Claude 訓練知識內的相關工作(非即時查證,知識截止 2025-05,務必自行查證/補新)

> 沙箱的 WebSearch / WebFetch 都無法真正連網（多次測試確認），所以以下是 Claude 依訓練知識列出的「最接近的已知工作 + 威脅判斷」，當你自己搜尋時的對照基準。**這不是查證結果**，可能漏掉 2025 下半年後的新論文。

**🟡 部分重疊（做了三項中的一兩項，但無完整組合）：**
- **VariBAD / PEARL**：有 task belief，但無語言/圖片規格、無歧義建模。
- **T2DA / LUMOS**：有「規格→latent→zero-shot」，但單點、確定式、無歧義 belief。← 你的主要區隔對象。
- **PFN（Prior-Fitted Networks）/「transformers as Bayesian learners」（Müller et al.）**：證明 meta-trained transformer 在 context 近似 Bayesian 後驗。與「in-context + belief」相關，但不在 robot task-spec 場景、未處理歧義規格。**→ 自己查它有無 2024-2025 的 RL/robot 衍生。**
- **互動式澄清（interactive disambiguation / clarification questions）**：機器人遇模糊指令時反問使用者。主題相關但解法不同（問人 vs belief+自主探索），可當對照凸顯「不需人介入」。
- **referring expression grounding / 語言指代歧義**：多在感知層物件指代，非 task belief 分布 + zero-shot meta-RL。

**🟢 就 Claude 所知「沒看到」的（= 你可能的 novelty，務必自行確認）：**
- 把「歧義的**圖片** goal 規格」明確建模成 **task 後驗分布**。
- 「依**模態歧義程度**動態融合圖片+語言規格」的具體機制。
- 「現代 in-context RL 上**顯式輸出 task uncertainty** 驅動 risk-aware zero-shot」的 robot task-spec 版本。

> 自己查到真實結果後，比對上面這份名單有沒有遺漏或過時，再填到下方。

---

## 找到的相關工作(由使用者填寫,我之後接續分析)

> 格式範例:
> ### [標題](URL)
> - **年份/出處**:2024 NeurIPS / arXiv:2410.xxxxx
> - **一句話做什麼**:用 transformer 對 task belief 做 in-context 推論,但只用語言、無歧義建模。
> - **威脅度**:🟡 中(belief + zero-shot 撞,缺歧義建模與圖片模態)
> - **是否需要改方向**:不需要,但要在 related work 區分

---

### 第一輪查證結果（2026-06-03，由 Cowork WebSearch 跑出，搜尋結果不穩定，部分查詢未回傳實際命中，需自行補查）

> **方法說明**：用 WebSearch（不限定 arxiv.org，因為限定反而命中為零）跑了第一、二優先的查詢字串，加上 KnowNo / VIMA / MUTEX / DPT / AD 等已知相關工作的延伸查詢。部分查詢（LUMOS、ZeST、posterior task inference + language、risk-aware language-conditioned policy）這次沒拿到實際結果，下面只列已實際命中的論文。需要再針對未命中項手動補查。

---

#### A. In-context RL × task uncertainty 線（技術底座）

##### [Supervised Pretraining Can Learn In-Context Reinforcement Learning (DPT)](https://arxiv.org/abs/2306.14892)
- **年份/出處**：Lee et al., NeurIPS 2023, arXiv:2306.14892
- **一句話做什麼**：Decision-Pretrained Transformer，證明監督式預訓練的 transformer 在新 RL task 上會做出近似**後驗取樣（posterior sampling over tasks）**的最優決策，**隱式**處理 task uncertainty。
- **威脅度**：🟡 中（in-context + 隱式 belief 撞，但無語言/圖片規格、無歧義建模、無顯式 uncertainty 輸出）
- **是否需要改方向**：不需要。**這是你的最有力技術底座之一**，論文要寫「我們把 DPT 的隱式 posterior 改為顯式、並讓它接受歧義的多模態 task spec」。

##### [In-context Reinforcement Learning with Algorithm Distillation (AD)](https://arxiv.org/abs/2210.14215)
- **年份/出處**：Laskin et al., ICLR 2023, arXiv:2210.14215
- **一句話做什麼**：把 RL learning history 蒸餾進 transformer，跨 episode 隱式平衡探索/利用，隱式處理 task uncertainty。
- **威脅度**：🟢 低（無 belief、無 spec、無歧義建模）
- **是否需要改方向**：不需要。列為 in-context RL 主幹工作。

##### [A Survey of In-Context Reinforcement Learning](https://arxiv.org/abs/2502.07978)
- **年份/出處**：Moeini et al., 2025（arXiv:2502.07978）
- **一句話做什麼**：ICRL 綜述，含 Bayesian 詮釋、uncertainty、探索章節。
- **威脅度**：🟢 低（綜述）
- **用途**：寫 related work 時用來定位「現代 ICRL 已有 Bayesian 解讀但未做歧義規格」這個 gap。

##### [VariBAD](https://arxiv.org/abs/1910.08348) / RL² / TrMRL（Transformers are Meta-RL）
- **年份/出處**：VariBAD (Zintgraf et al., ICLR 2020), RL² (Duan et al., 2016), TrMRL (Melo, ICML 2022, arXiv:2206.06614)
- **一句話做什麼**：分別用 VAE belief、RNN、transformer 做 meta-RL 的 task 推論。
- **威脅度**：🟡 中（VariBAD 有 belief 但無 spec／歧義）
- **用途**：你 checklist 中已列，這次只補確認。

---

#### B. 語言指令歧義 × belief 線（最關鍵的撞題檢查）

##### [KnowNo: Robots That Ask For Help](https://arxiv.org/abs/2307.01928)
- **年份/出處**：Ren et al., CoRL 2023 Best Student Paper, arXiv:2307.01928
- **一句話做什麼**：LLM planner 上包 **conformal prediction**，把語言指令的歧義轉成 prediction set，多選項時**反問人**澄清。
- **威脅度**：🟡 中（**有歧義 + 有 uncertainty 量化**，但解法是「問人」而非「belief 自主探索」、且只用語言、非 in-context RL）
- **是否需要改方向**：不需要。**這是你最重要的「對照組」**，你要在 paper 中明確說「不需要人介入」是你的賣點，把 conformal/MCQA 換成 posterior + risk-aware exploration。

##### Interactive Disambiguation / Clarification Questions 系列
- INVIGORATE、CLEAR、Thomason 對話式語言學習等。
- **一句話做什麼**：機器人遇模糊指令時用對話/clarification 解歧義。
- **威脅度**：🟢 低（解法不同：問人 vs belief + 自主行為）
- **用途**：related work 對照組，凸顯「免人介入」novelty。

---

#### C. 多模態 task 規格 線（VIMA / MUTEX）

##### [VIMA: General Robot Manipulation with Multimodal Prompts](https://vimalabs.github.io/)
- **年份/出處**：Jiang et al., ICML 2023
- **一句話做什麼**：交錯的文字+圖片 prompt 規格 → manipulation。**用視覺 token 來主動消除語言歧義**（避免「把那個放到那邊」這種 referent 模糊）。
- **威脅度**：🟡 中（**有多模態 spec、有歧義意識**，但策略是「用圖片替換歧義語言」而非「對歧義建 belief 分布」；無 task uncertainty、無 zero-shot meta-RL）
- **是否需要改方向**：不需要，但要清楚區分：VIMA 是「設計 spec 來避免歧義」，你是「面對歧義 spec、用 belief 主動處理」。

##### [MUTEX: Learning Unified Policies from Multimodal Task Specifications](https://ut-austin-rpl.github.io/MUTEX/)
- **年份/出處**：Shah et al., CoRL 2023
- **一句話做什麼**：用 6 種模態（文字、語音、圖、影片、聲音…）的 spec 訓 unified policy，跨模態 masked modeling + matching。
- **威脅度**：🟡 中（**有多模態 spec、有跨模態對齊**，但是確定式映射、無歧義 belief、無 task uncertainty）
- **是否需要改方向**：不需要。**這是 T2DA/LUMOS 之外的另一個主要區隔對象** ── 「他們把多模態當作可互換的等價 spec；你把模態歧義當作 belief 寬度的來源」。

---

#### D. Goal image / 視覺 goal 線

##### Goal-conditioned RL with imagined / latent goals 系列
- Chane-Sane (ICML 2021) Imagined Subgoals、Eysenbach Contrastive GCRL (NeurIPS 2022)、C-Learning (ICLR 2021)、RIG/PIG (Nair)、Variational Empowerment (ICML 2021)。
- **一句話做什麼**：在 goal image / latent goal 上做表示學習、posterior、VAE。
- **威脅度**：🟢 低～🟡 中（**有 visual goal 的 posterior 思想，但場景是 GCRL 而非 task-spec meta-RL**，未處理「同一張圖對應多個底層 task」的歧義）
- **用途**：related work 補充。寫作時可說「他們做的是 state-level goal posterior，我們做的是 task-level posterior」。

---

### 第一輪後待補查項目（已於下方第二輪補查完成）

1. **LUMOS** 的精確論文連結與 abstract（checklist 已列為主要區隔對象，但要找到正確的 arXiv ID）
2. **ZeST** 的精確論文連結（同上）
3. `posterior task inference language-conditioned policy robot 2024-2025`
4. `risk-aware language-conditioned policy uncertainty`
5. **DPT 的 2024-2025 後續工作**（特別是有人把 DPT 顯式 belief output 化或加上歧義 spec 的）
6. **Prior-Fitted Networks（PFN）在 RL/robotics 的衍生**
7. **CALVIN 上 2024-2025 的最新 zero-shot 方法**（檢查有沒有人剛好做了 belief + ambiguous spec）
8. OpenReview 上 ICLR 2025 / NeurIPS 2025 投稿區的 in-context RL + uncertainty 相關 submission

---

### 第一輪整體威脅評估（暫定，已由第二輪更新）

- **🔴 高（撞題）**：目前沒看到。
- **🟡 中（最重要的對照組）**：
  - **KnowNo**（歧義 + uncertainty，但「問人」）
  - **VIMA / MUTEX**（多模態 spec，但無 belief / 確定式）
  - **DPT**（in-context + 隱式 posterior，但無歧義 spec）
  - **T2DA / LUMOS**（你原 checklist 已列，仍是主要區隔對象；第二輪已補查）
- **🟢 低**：AD、GCRL goal-posterior 系列、interactive disambiguation 系列。

**初步結論**：第一輪沒看到把「歧義多模態 spec + task belief 分布 + zero-shot in-context RL」三項全做的論文。但 LUMOS、ZeST 的最新狀態、以及 OpenReview 的 2025/2026 投稿區當時還沒查到；這些項目已在下方第二輪補查段落更新。

第二輪補查若確認沒有 🔴，就可以接著展開 problem statement + 演算法。

---

### 第二輪查證結果（2026-06-03 同日續查）

> **說明**：原本 WebSearch 對 LUMOS、ZeST、T2DA、OpenReview 的查詢沒有拿到 verifiable 搜尋結果；本段先列當時已命中的 CALVIN 2024 SOTA 與 PFN，原本未命中的項目已在後面的「第二輪原待查項目的 verifiable 補查結果」補上。

#### E. CALVIN 2024 SOTA（你的可能 proof-of-concept 平台）

##### RoboFlamingo (ICLR 2024)
- **一句話做什麼**：把 OpenFlamingo 這個 VLM 微調成 manipulation policy，CALVIN long-horizon 表現強。
- **威脅度**：🟢 低（無 belief、無歧義建模，是 VLA-style 確定式映射）
- **用途**：你 CALVIN proof-of-concept 的對照 baseline。

##### 3D Diffuser Actor (2024)
- **一句話做什麼**：3D scene + diffusion policy，CALVIN ABC→D 強。
- **威脅度**：🟢 低（同上）
- **用途**：diffusion-policy 方向的對照 baseline。

##### GR-1 / GR-2 (2024)
- **一句話做什麼**：GPT-style transformer 先在影片上 pre-train，再微調機器人資料；CALVIN ABC→D 約 94%。
- **威脅度**：🟢 低（無 belief / 歧義建模）
- **用途**：對照 baseline。

##### SuSIE (ICLR 2024)
- **一句話做什麼**：用 image-editing diffusion 從語言**生成 subgoal**，再走 GCRL。
- **威脅度**：🟡 中（**有「語言→（隱含的）視覺 goal 想像」這條路徑**，但只生成一個 subgoal、非 belief 分布；可作為「為什麼需要 belief 而不是 sample 一個」的反例）
- **用途**：related work 需要明確區分。

##### MDT (Multimodal Diffusion Transformer, RSS 2024) / DeeR-VLA (NeurIPS 2024)
- **一句話做什麼**：diffusion-based policy / 動態 early-exit 加速 VLA。
- **威脅度**：🟢 低
- **用途**：補對照 baseline。

**CALVIN SOTA 觀察**：2024 主流是 VLA / diffusion / GCRL，**沒人把 task belief 顯式建模成 CALVIN 上的方法**。對你有利。

---

#### F. Prior-Fitted Networks 線（你 checklist 提到的）

##### [Transformers Can Do Bayesian Inference (PFN)](https://arxiv.org/abs/2112.10510)
- **年份/出處**：Müller, Hollmann, Pineda Arango, Grabocka, Hutter, ICLR 2022, arXiv:2112.10510
- **一句話做什麼**：用 prior 採樣的合成資料訓 transformer，使其在一個 forward pass 內近似 Bayesian posterior predictive。
- **威脅度**：🟢 低（非 RL/robot 場景）
- **用途**：你的理論依據之一——「transformer 能做 amortized Bayesian inference」。

##### PFNs4BO (ICML 2023)
- **一句話做什麼**：把 PFN 用在 Bayesian Optimization，跟 RL 探索有相關性。
- **威脅度**：🟢 低
- **用途**：相關證據，PFN 已被用到「決策 + 不確定性」場景。

**Verifiable 結論**：到目前為止**沒看到** PFN 被用在 robot task-spec、ambiguous spec、或 CALVIN 上。你的 novelty 仍站得住。

---

### 第二輪原待查項目的 verifiable 補查結果（2026-06-03）

> **補查方式**：這輪改用可直接開啟的 arXiv / OpenReview / project page。下面只列能驗證到標題、年份、摘要或投稿頁的項目；搜尋結果仍不等於完整文獻綜述，但已足夠把原本的 ❓ 待查項移出「必須使用者親自查」。

##### [LUMOS: Language-Conditioned Imitation Learning with World Models](https://arxiv.org/abs/2503.10370)
- **年份/出處**：Nematollahi et al., ICRA 2025 / arXiv:2503.10370
- **一句話做什麼**：language-conditioned multi-task imitation learning；在 learned world model 的 latent space 中做長視野 rollout 訓練，並 zero-shot transfer 到真實機器人；實驗含 CALVIN。
- **威脅度**：🟡 中（有 language-conditioned + zero-shot + CALVIN/real robot，但沒有把歧義 spec 顯式建成 task belief 分布，也不是 risk-aware in-context RL）
- **是否需要改方向**：不需要，但 **LUMOS 必寫 related work**。區隔句應放在「LUMOS 是確定式/latent world-model skill learning；我們處理 ambiguous multimodal spec → explicit task posterior → risk-aware action」。

##### [Text-to-Decision Agent: Offline Meta-Reinforcement Learning from Natural Language Supervision (T2DA)](https://arxiv.org/abs/2504.15046)
- **年份/出處**：Zhang et al., arXiv:2504.15046（v1: 2025-04-21；v5: 2025-11-22）
- **一句話做什麼**：用自然語言監督 offline meta-RL；把 multi-task decision data 編成 dynamics-aware embedding，再用 contrastive language-decision pretraining 對齊文字與決策 embedding，讓 policy 對文字指令做 zero-shot text-to-decision。
- **威脅度**：🟡 中偏高（有 language → decision、offline meta-RL、zero-shot，且 abstract 明確討論 task belief；但目前看是文字到單一 decision embedding/policy，沒有 ambiguous multimodal spec 的 posterior 分布與 risk-aware uncertainty 使用）
- **是否需要改方向**：不需要，但 **T2DA 是最接近的主要區隔對象之一**。寫作時要避免只說「language-conditioned zero-shot meta-RL」；novelty 必須鎖定在「歧義、多模態、顯式 task belief distribution、基於 uncertainty 的行為」。

##### [ZeST / Can Foundation Models Perform Zero-Shot Task Specification For Robot Manipulation?](https://arxiv.org/abs/2204.11134)
- **年份/出處**：Cui et al., L4DC 2022, arXiv:2204.11134；project page: https://sites.google.com/view/zestproject
- **一句話做什麼**：用 foundation models（如 CLIP 等）把網路圖片、手繪、語言等低成本規格轉成 robot manipulation 的 zero-shot goal/task specification；也把 ZeST score 當 offline RL reward proxy。
- **威脅度**：🟡 中（有 zero-shot task specification + 視覺/語言規格，但無 task belief、無歧義 posterior、無 in-context RL）
- **是否需要改方向**：不需要。ZeST 是「zero-shot task-specification」的早期直接前作；我們的區隔是從 similarity/reward proxy 推進到 ambiguous spec 的 task-level posterior。

##### OpenReview：ICLR 2025 / ICLR 2026 的 ICRL + uncertainty 近鄰
- **已驗證命中**：
  - [XLand-100B: A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning](https://openreview.net/forum?id=p9OsTj0nMP)（ICLR 2025 poster）：大型 ICRL dataset / benchmark。
  - [Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning](https://openreview.net/forum?id=BfUugGfBE5)（ICLR 2025 poster）：transformer 同時學 dynamics 與 in-context planning。
  - [Transformers Can Learn Temporal Difference Methods for In-Context Reinforcement Learning](https://openreview.net/forum?id=Pj06mxCXPl)（ICLR 2025 poster）：證明 transformer forward pass 可學 TD-style policy evaluation。
  - [Safe In-Context Reinforcement Learning](https://openreview.net/forum?id=F8a6dAw3Yg)（ICLR 2026 submission）：constrained/safe ICRL，讓 adaptation 同時顧 reward 與 cost。
  - [In-Context Reinforcement Learning through Bayesian Fusion of Context and Value Prior](https://openreview.net/forum?id=tFtqUSOUcM)（ICLR 2026 submission）：SPICE，用 value ensemble + Bayesian context fusion + posterior UCB 做 ICRL。
- **威脅度**：🟡 中（SPICE / Safe ICRL 很接近「Bayesian/risk-aware ICRL」技術底座，但目前不是 robot language/multimodal task-spec，也沒有 ambiguous spec posterior）
- **是否需要改方向**：不需要。把 SPICE / Safe ICRL 當「ICRL + uncertainty/risk」最新近鄰，並強調我們把這條線接到 robot ambiguous multimodal task specification。

##### DPT 的 2024-2025 後續工作
- **已驗證命中**：
  - [Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining](https://proceedings.iclr.cc/paper_files/paper/2024/hash/84b5e56a80cbea27de332d456f0d5a4c-Abstract-Conference.html)（ICLR 2024）：理論分析 AD / DPT；證明 supervised-pretrained transformer 可在 context 中近似 LinUCB、Thompson sampling、UCB-VI 等。
  - [HVAC-DPT: A Decision Pretrained Transformer for HVAC Control](https://arxiv.org/abs/2411.19746)（arXiv:2411.19746）：把 DPT/ICRL 用在 multi-zone HVAC control，無需針對新建築重新訓練。
- **威脅度**：🟢 低～🟡 中（DPT 後續主要是理論、benchmark、控制應用；未看到把 DPT 改成 robot ambiguous spec + explicit task belief output 的論文）
- **是否需要改方向**：不需要。DPT 仍可作為技術底座，不是撞題。

##### PFN 在 RL/robotics 的衍生
- **已驗證命中**：
  - [Position: The Future of Bayesian Prediction Is Prior-Fitted](https://proceedings.mlr.press/v267/muller25d.html)（ICML 2025）：PFN 作為 amortized Bayesian prediction 的 position paper。
- **補查結論**：這輪沒有找到可驗證的「PFN + robot task-spec / ambiguous spec / CALVIN」直接撞題。PFN 仍是 Bayesian inference / in-context posterior 的背景技術，不是主要威脅。

##### risk-aware language-conditioned policy / uncertainty 近鄰
- **已驗證命中**：[Risk-Calibrated Human-Robot Interaction via Set-Valued Intent Prediction](https://arxiv.org/abs/2403.15959)（RSS 2024 / arXiv:2403.15959）
- **一句話做什麼**：RCIP 用 set-valued intent prediction 做 risk calibration；當 human intent uncertainty 的風險不可控時，機器人會請人澄清。
- **威脅度**：🟡 中（有 risk + uncertainty + robot intent，但解法是 human clarification / intent prediction，不是 zero-shot in-context RL 的 autonomous task belief exploration）
- **是否需要改方向**：不需要。和 KnowNo 一樣可當「問人/校準風險」對照組。

**第二輪補查結論**：原本 5 個 ❓ 項目現在都已有可驗證結果或可驗證的 negative finding。最接近的新增威脅是 **T2DA、LUMOS、SPICE/Safe ICRL、RCIP**，但仍未看到同時包含「歧義多模態 spec + explicit task belief distribution + zero-shot in-context RL / risk-aware autonomous action」的 🔴 撞題論文。

---

### 第二輪後最終威脅評估

- **🔴 高（撞題）**：**兩輪補查後仍沒看到**。目前你的「歧義多模態 spec + task belief + zero-shot in-context RL」三項組合**仍是空白**。
- **🟡 中（最重要的對照組，related work 必寫）**：
  - **T2DA**（language → decision / offline meta-RL / zero-shot，最接近但缺 ambiguous multimodal posterior）
  - **LUMOS**（language-conditioned world-model IL + CALVIN/real robot zero-shot，但缺 belief / ambiguity）
  - **ZeST**（zero-shot task specification，缺 task belief / ICRL）
  - **SPICE / Safe ICRL**（Bayesian/risk-aware ICRL，缺 robot multimodal spec）
  - **RCIP**（risk-calibrated intent prediction，問人澄清，不是 autonomous belief exploration）
  - **KnowNo**（歧義 + uncertainty 但「問人」）
  - **VIMA / MUTEX**（多模態 spec 但確定式、無 belief）
  - **DPT**（in-context + 隱式 posterior 但無歧義 spec）
  - **SuSIE**（語言→單一 subgoal 但非 belief 分布）
- **🟢 低**：AD、TrMRL、GCRL、interactive disambiguation、PFN、RoboFlamingo / GR-1 / 3D Diffuser Actor / MDT / DeeR-VLA。

**結論（保守版）**：在 2026-06-03 可驗證來源能查到的範圍內，**沒有 🔴 高威脅論文**。原本 ❓ 標記的項目已完成補查；真正需要防守的是 T2DA / LUMOS / ZeST / SPICE / RCIP 這幾條中威脅近鄰。

**建議行動順序**：
1. 先把 T2DA / LUMOS / ZeST / SPICE / RCIP 放進 related work 必寫清單。
2. 接著寫 problem statement + 演算法 + CALVIN proof-of-concept 計畫，重點鎖定「ambiguous multimodal spec → explicit task posterior → risk-aware autonomous action」。
3. 正式投稿前一週再做一次 OpenReview / arXiv keyword sweep，防止 2026 新投稿剛公開但尚未被搜尋引擎完整索引。

---

### Claude 對查證結果的關鍵判斷（2026-06-03）

> 使用者已完成兩輪 verifiable 查證，以下是 Claude 接手後的策略判斷，供寫 method / related work 時直接使用。

**1. 最需要正面防守的兩個近鄰（比一般 🟡 更近）：**

- **SPICE（ICLR 2026 submission, In-Context RL through Bayesian Fusion of Context and Value Prior）** — 機制最接近：`Bayesian fusion` + `posterior UCB` + risk-aware ICRL。**差異防守點**：SPICE 融合「context + value prior」於純 RL/bandit 設定；我們融合「**多模態任務規格（prior）+ 觀察（likelihood）**」於 robot manipulation，且規格本身**歧義**、跨模態歧義度不同。method 必須有一段明確對比 SPICE。
- **T2DA** — abstract 已出現 "task belief" 字樣。**不能**用「language→zero-shot meta-RL with belief」當賣點（會被視為 T2DA 增量）。novelty 必須鎖死在 T2DA 沒有的三點：(a) 歧義建模、(b) 多模態規格、(c) 顯式 uncertainty 驅動 risk-aware 行為。

**2. 最強的故事 framing（把所有 🟡 中威脅轉成對照組）：**

> 面對歧義的任務規格，現有方法二選一：要嘛**問人澄清**（KnowNo、RCIP），要嘛**假裝無歧義直接執行**（T2DA、LUMOS、MUTEX、VIMA）。我們提出第三條路：把歧義建成 **task 後驗信念**，讓 agent **自主**用觀察 inference-time 收斂信念、做 risk-aware 行為——**免人介入**。

這個「三選一、我們是第三條」的定位，比「我們加了 belief」有力，且讓 KnowNo/RCIP（問人線）與 T2DA/LUMOS/MUTEX（確定式線）都成為對照而非競爭。

**3. 對技術底座選擇的影響：** 查證顯示 in-context RL × Bayesian/risk（DPT、SPICE、Safe ICRL、PFN 理論）這條線正熱且有理論支撐（DPT 證明可近似 Thompson sampling、PFN 證明可做 amortized Bayesian inference）——這強化了**用 in-context RL 當主底座**的合理性：你站在「transformer 能做 amortized Bayesian posterior」的理論肩膀上，把它推進到歧義多模態 robot task-spec。

---

## 沒撞題的話,下一步

把這份清單填好後告訴我「沒有 🔴 高威脅、最高只到 🟡 中」,我就接著把選定的技術底座(in-context / VariBAD-style / diffusion)展開成正式的 problem statement + 演算法 + loss + CALVIN proof-of-concept 實驗計畫。

## 如果有 🔴 高威脅論文

把那篇論文的標題、abstract、跟我們提案的具體重疊處貼給我,我幫你判斷:(a) 改方向、(b) 換子問題、(c) 在它之上做明確的增量但要找新賣點。
