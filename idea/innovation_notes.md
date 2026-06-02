---
type: idea-notes
aliases:
  - "innovation_notes"
  - "創新筆記"
tags:
  - 研究主線
  - idea
  - ablation
summary: "可作為題目零件/ablation/baseline 的 idea 萃取（含 VariBAD 與第二批文獻 15–29 節）。"
---
# 非原論文設計整理：可作為未來優化題材

本文件整理目前實驗中偏離 PEARL 原論文的設計。這些設計不應混入「faithful PEARL reproduction」主線，但可以保留為日後論文優化、ablation 或研究 idea。

## 分類標準

| 類型 | 定義 |
| --- | --- |
| Diagnostic | 用來看懂失敗原因，不應當作正式方法貢獻 |
| Stabilizer | 為了讓 toy setting 或離線訓練穩定而加 |
| Proxy | 用簡化機制近似 meta-RL adaptation |
| Potential Contribution | 未來可發展成方法改良，但需要嚴格 ablation |

## 1. Analytic PEARL-Latent Candidate Posterior

類型：Proxy / Potential Contribution

做法：

- 建立 latent candidates，例如 `(goal_x, goal_y, wind_x, wind_y)`。
- 用 reward likelihood 和 transition likelihood 對 candidate 打分。
- 取 posterior-weighted mean 作為 task latent。

效果：

- 在 Run 002 表現很好，K=1 快速改善。
- 很清楚展示「context inference」比逐步 gradient / memory update 更快。

和 PEARL 差異：

- PEARL 用 neural encoder 學 posterior，不枚舉手工 latent candidates。
- 這個方法利用了 toy task 的已知 latent 結構。

未來研究價值：

- 可作為「structured latent inference」或「model-informed PEARL」方向。
- 適合在已知 task family 結構時作為 hybrid Bayesian meta-RL baseline。

風險：

- 不適合宣稱是通用 meta-RL。
- 在高維 MuJoCo task 上 candidate grid 不可行。

## 2. Hidden Goal + Hidden Wind Toy Task

類型：Potential Contribution

做法：

```text
z = (goal_x, goal_y, wind_x, wind_y)
next_state = state + action + wind
reward = -distance(next_state, goal) - action_penalty
```

效果：

- 比單純 2D goal navigation 更能看出 task inference 差異。
- K=0 明顯較差，context 後才有可能補償 wind。

和 PEARL 差異：

- PEARL 論文常用 MuJoCo benchmark 和 sparse 2D navigation。
- 這是我們設計的中間難度 benchmark。

未來研究價值：

- 可作為教學 / debugging benchmark。
- 可用來比較「reward latent」與「dynamics latent」的 inference 難度。

風險：

- 任務仍太低維，不能取代正式 benchmark。

## 3. True Latent Supervision

類型：Stabilizer / Diagnostic

做法：

```text
latent_loss = MSE(mu, true_task_latent)
```

效果：

- 讓 encoder 更容易學會 goal/wind。
- 能診斷 neural encoder 是否有能力恢復 task latent。

和 PEARL 差異：

- PEARL 不使用 true latent label。
- PEARL 的 latent 由 critic loss + KL 學出，只要求對 control 有用。

未來研究價值：

- 可作為 upper-bound / auxiliary-supervision ablation。
- 若某些 domain 有 task descriptor，可研究 semi-supervised PEARL。

風險：

- 會破壞 faithful reproduction。
- latent label 可能逼 encoder 學「可解釋 latent」而非「最有控制價值的 latent」。

## 4. Oracle Behavior Cloning Regularization

類型：Stabilizer

做法：

```text
actor_loss += bc_weight * MSE(actor(s, z), oracle_action)
```

效果：

- 防止 offline SAC actor 被錯誤 Q 值推到 replay distribution 外。
- Run 004 中提高 `bc_weight` 後避免嚴重崩壞。

和 PEARL 差異：

- PEARL-SAC 不使用 oracle action。
- PEARL actor 由 SAC objective 訓練。

未來研究價值：

- 可作為「demonstration-assisted meta-RL」或「offline meta-RL」方向。
- 若有 expert demonstrations，可以研究 PEARL + BC / TD3+BC。

風險：

- actor 可能學成平均 imitation policy，忽略 z。
- 會掩蓋真正 exploration / task inference 能力。

## 5. Scripted Probe / Noisy-Oracle Bootstrap

類型：Stabilizer / Diagnostic

做法：

- 用 deterministic probe trajectory 收集 context。
- replay buffer 初期加入 noisy-oracle transitions。

效果：

- 讓 early training 有足夠 reward / dynamics 訊號。
- 避免 sparse 或 random exploration 讓模型完全學不到。

和 PEARL 差異：

- PEARL 主要由 policy 收集資料。
- meta-test 時 PEARL 使用 prior / posterior sampling exploration，不依賴手工 probe。

未來研究價值：

- 可研究「active probing policy」或「learned exploration prior」。
- 可作為 meta-test exploration strategy 的比較 baseline。

風險：

- 若 probe 太強，會讓 task inference 過簡單。
- 不公平地幫助所有方法取得高品質 context。

## 6. Posterior Rollout Ensemble / Posterior Mean Evaluation

類型：Diagnostic / Stabilizer

做法：

- posterior sample 多次後平均 z。
- 或多次 posterior rollout 後平均 return。
- 或直接使用 posterior mean。

效果：

- Run 009 顯著降低 sample evaluation 的 variance。
- 幫助分辨「sampling noise」和「posterior 本身錯」。

和 PEARL 差異：

- PEARL 使用 posterior sampling，但不一定把 ensemble 當標準 protocol。

未來研究價值：

- 可研究 uncertainty-aware evaluation / risk-sensitive adaptation。
- 對高 variance meta-RL 可能有穩定效果。

風險：

- 增加 meta-test compute。
- 如果只報 ensemble 結果，可能和標準 benchmark 不可比。

## 7. Actor-z Sensitivity Regularizer

類型：Diagnostic / Potential Contribution

做法：

- 人為 shift `z` 的 goal 維度。
- 要求 actor action 隨 z 改變。

效果：

- Run 008 中 actor-z sensitivity 有上升。
- 但 performance 沒改善，表示 actor 使用 z 的方式仍不正確。

和 PEARL 差異：

- PEARL 沒有這個 regularizer。

未來研究價值：

- 可作為 representation collapse 防治方法。
- 可研究 actor 是否忽略 context latent 的 regularization。

風險：

- 可能強迫錯誤 sensitivity。
- 若 z 維度沒有固定語意，這個 regularizer 不成立。

## 8. 手工 Context Summary Features

類型：Stabilizer / Proxy

做法：

- 把 variable-length context 聚合成 13 維 summary features。
- 包含 mean state/action/next_state/reward/residual 等。

效果：

- 訓練快。
- 對 hidden wind task 很容易提供 transition residual 線索。

和 PEARL 差異：

- PEARL encoder 通常處理 transition set，並使用 permutation-invariant aggregation / product of Gaussians。
- 手工 summary 可能把 task inference 做得太人工。

未來研究價值：

- 可作為 low-cost encoder baseline。
- 可比較 learned transition encoder vs engineered context statistics。

風險：

- 不通用。
- 在複雜狀態空間不可行。

## 9. VARIBAD-Style Belief-Conditioned Task Embedding

類型：Potential Contribution / Proxy

做法：

- 參考 VARIBAD，把 task uncertainty 顯式放進 policy input。
- 用 trajectory/context encoder 學 `q(z | τ)`，其中 `z` 表示 hidden task belief，而不是只當一般 representation。
- policy 使用 `π(a | s, z)`，讓行為同時依賴目前 observation 和 inferred task belief。
- 在 robot play data setting 中，可以先用 play trajectory 學 trajectory-to-task embedding，再用少量弱語言標註對齊 text encoder：

```text
play trajectory τ -> q(z | τ)
weak language instruction l -> g(l)
align g(l) ≈ z
policy π(a | observation, z)
```

效果：

- 可把 VARIBAD 的 Bayes-adaptive / belief-based meta-RL 概念接到 offline robot play data。
- 測試時若只給新語言指令 `l`，用 `g(l)` 產生 task embedding，可形成 zero-shot task specification。
- 比單純 language-conditioned BC 更明確：文字不是直接當 command token，而是作為 task belief / task latent 的 interface。

和 PEARL 差異：

- PEARL 主要是從 context inference 出 task latent，並透過 off-policy actor-critic 學 adaptation。
- VARIBAD 更強調把 belief state 視為 Bayes-adaptive MDP 的狀態部分，policy 應根據 task uncertainty 行動。
- 若接到語言，VARIBAD-style latent 可以成為「文字任務規格」和「trajectory belief」之間的橋。

未來研究價值：

- 可形成題目：Weak-Language Offline Meta-RL from Robot Play for Zero-Shot Task Generalization。
- 可研究極少量語言標註下，language-to-belief alignment 是否比直接 language-conditioned imitation 更有效。
- 可比較 `q(z | τ)`、`g(l)`、以及 true task descriptor 是否在 latent space 中對齊。
- 可作為 robot play data、weak language annotation、offline meta-RL、zero-shot task embedding 四者整合的主線。

風險：

- zero-shot 定義必須嚴格：測試時不能額外 gradient update，也不能依賴新 task demonstration。
- 若 play data 沒有足夠 task diversity，`z` 可能只學到 behavior style 而不是 task semantics。
- 語言標註太弱時，`g(l)` 可能無法穩定對齊 `q(z | τ)`。
- 若 policy 忽略 `z`，會退化成普通 imitation / offline RL，需要 actor-z sensitivity 或 latent intervention 診斷。

## 10. VariBAD-Style 全軌跡重建 ELBO（含未來）作為 latent 學習訊號

類型：Potential Contribution

來源：VariBAD（ICLR 2020），詳見 `paper/VariBAD.md` 第 4.2 節。

做法：

- 用 encoder 從**過去**軌跡 `τ:t` 推 posterior `q(m | τ:t)`。
- 但 decoder 重建**整段軌跡（含未來）** `τ:H+`，而不是只重建已看過的 transition：

```text
q(m | τ:t)                         # 只用過去
maximize  E_q[ log p(τ:H+ | m) ]   # 解碼過去 + 未來 transition / reward
        - KL( q(m|τ:t) || prior )
prior := q(m | τ:t-1),  初始 prior = N(0, I)
```

效果：

- 強迫 latent 不只「記住看過的東西」，而是學會**從過去推論未見 state 的 dynamics/reward**。
- 在 PEARL 主線中，目前 latent 只由 critic loss + KL 學出（見 innovation 第 3 節討論）。加上「重建未來」的 VAE 目標，可能讓 task latent 更早、更穩定地收斂。

和 PEARL 差異：

- PEARL 沒有 generative decoder，latent 只要對控制有用即可。
- VariBAD 的重建是 auxiliary loss，明確要求 latent 能解釋 transition/reward 結構。

未來研究價值：

- 可作為 ablation：「PEARL latent vs PEARL + 全軌跡重建（含未來）latent」，比較 K=0/K=1 的收斂速度與穩定度。
- 對 hidden-wind toy task（innovation 第 2 節）特別合適：重建未來 transition residual 可直接逼 latent 表示 wind。

風險：

- 需要訓練時可存取整段軌跡（含未來），離線 robot play data 通常可行。
- decoder 可能讓 latent 偏向「可重建」而非「最有控制價值」，需用 actor-z sensitivity（第 7 節）監控。

## 11. Belief-Conditioned Policy + 不反傳 RL loss 到 Encoder

類型：Stabilizer / Potential Contribution

來源：VariBAD 第 4.3 節（訓練目標 (10) 與實作 trick）。

做法：

- policy 直接 condition 在「posterior 分布的參數（mean, log-var）」上，而不是只吃一個 sample：`π(a | s, μ, σ)`。
- **不把 RL loss 反傳穿過 encoder**：encoder 只用 VAE/重建 loss 學；policy 與 VAE 用不同 optimiser、不同 LR、不同 data buffer。

效果：

- VariBAD 報告：這樣可大幅加速、避免 RL loss 與重建 loss 的梯度互相干擾、不必 trade off 兩個 loss。
- 也是 VariBAD 比 RL² 訓練快的主因（PPO minibatch 更新不必重算 embedding）。

和 PEARL 差異：

- PEARL 也分開訓 encoder（用 critic gradient）與 actor，但 encoder 由 critic loss 驅動；VariBAD 是「encoder 只吃重建/KL、完全不吃 RL loss」。
- PEARL 對 z 做 sampling；這裡也可改成讓 policy 吃整個 posterior 參數（belief-conditioned）。

未來研究價值：

- 可作為穩定 offline meta-RL 的 trick：梯度解耦可能緩解 innovation 第 4 節（BC regularization）想處理的 actor 崩壞問題，但不依賴 oracle action。
- 可做 ablation：「policy 吃 sample z」vs「policy 吃 posterior (μ,σ)」對 variance 的影響（呼應第 6 節 posterior ensemble）。

風險：

- 完全不反傳 RL loss 到 encoder 時，latent 是否「對控制最有用」要靠重建目標保證；若任務的關鍵變因對 reward/transition 重建不顯著，latent 可能漏掉它。

## 12. 把 Task Belief 顯式當成 State 的一部分（BAMDP 視角）

類型：Potential Contribution / Proxy

來源：VariBAD 第 3.2 節 BAMDP 框架（hyper-state、Bayes-optimal）。

做法：

- 把 agent 對任務的不確定性（posterior over task latent）視為 **hyper-state** `s+ = (s, b)` 的一部分，policy 對它最優化。
- 訓練目標明確以「在 horizon H+ 內的 online return」為準（強調 **H+ ≠ H**：要決定要不要花步數探索，取決於還剩多少時間）。

效果：

- 把問題從「學一個好 representation」提升為「學近似 Bayes-optimal 探索」，理論定位更清楚。
- 對 sparse reward toy task（hidden goal+wind）能解釋「為何 K=0 較差、需要先探索」。

和 PEARL 差異：

- PEARL 用 posterior sampling 探索（抽 z 走最優路），非 Bayes-optimal。
- BAMDP 視角讓探索行為由「最大化 H+ 內 return」直接導出，而非啟發式 bonus。

未來研究價值：

- 可作為研究主線命題：在 robot play / offline 設定下，能否近似 Bayes-optimal 探索（而非只是 posterior sampling）。
- 可比較「posterior-sampling 風格（PEARL）」與「belief-conditioned 風格（VariBAD）」在 toy task 的探索效率與重訪率。

風險：

- 真正 Bayes-optimal 需要訓練時有任務分布可抽樣；OOD 任務上 prior/posterior 都會錯（見第 14 節）。
- 在離線資料上「H+ 內最優探索」不一定能由固定資料學到，可能需 on-policy 修正。

## 13. 時間固定的 Task Embedding（利用 BAMDP 任務不變性）

類型：Stabilizer / Diagnostic

來源：VariBAD 第 5.4 節（BAMDP vs POMDP）與附錄 C 的 RL² 穩定性分析。

做法：

- 利用「單一任務內 task latent 應固定」這個性質：posterior 收斂後**凍結**或正則化使 latent 不再隨更多資料漂移。
- 對照 RL²：把會隨時間變的 RNN hidden state 拿去當 task 表示，reset 時容易劇烈位移而失準。

效果：

- VariBAD 報告其 latent 收斂後不隨更多資料改變，跨多 rollout/reset 較穩；RL² 在 CheetahVel reset 後表現會掉。

和 PEARL 差異：

- PEARL latent 本就對固定任務不變（context 聚合），但未必把「凍結 posterior」當標準 protocol。

未來研究價值：

- 可作為診斷：在我們的 toy/MuJoCo 重現中，檢查 latent 是否在 episode reset 後漂移；若漂移就加「收斂後凍結」正則。
- 與第 6 節 posterior ensemble / posterior mean evaluation 互補，一起降低 meta-test variance。

風險：

- 若任務其實會在 episode 內變（非嚴格 BAMDP），凍結 latent 反而有害。

## 14. OOD 任務的 Inference 失效診斷與再訓練

類型：Diagnostic / Potential Contribution

來源：VariBAD 第 7 節未來工作（OOD 泛化的兩個失效模式）。

做法：

- 明確區分兩種 OOD 失效：(a) **inference 錯**（prior/posterior update 對新分布不對）；(b) **policy 無法詮釋改變的 posterior**。
- 用 decoder 的重建誤差當 OOD 偵測訊號：重建很差 → 可能 out-of-distribution → 觸發 encoder/decoder 再訓練或顯式 planning。

效果：

- 把「測試任務分布 ≠ 訓練分布」這個常被忽略的問題，拆成可分別診斷與處理的兩塊。

和 PEARL 差異：

- PEARL 無 generative decoder，較難用重建誤差做 OOD 偵測。

未來研究價值：

- 對 robot play + weak language 主線很關鍵：新語言指令 `g(l)` 可能落在 OOD 區，需要可偵測、可回退的機制。
- 可研究「learned prior」（附錄 B.1 提到的 task-specific prior）以縮小 OOD 落差。

風險：

- decoder 重建誤差作為 OOD 指標需校準；高 reward-noise 任務可能誤報。
- 測試時若允許再訓練，會破壞嚴格 zero-shot 定義（呼應第 9 節風險）。

---

# 第二批文獻萃取（read.md 全套 27 篇，2026-06）

以下是依 read.md 清單精讀全部論文後，萃取出可作為使用者主線（robot play + 極少弱語言 + offline meta-RL + 文字→task spec/embedding zero-shot）的零件、baseline 與 ablation。每篇逐段筆記見 `paper/<名稱>.md`。為避免破壞前面 1–14 節的編號，這批用 15 起編，並按「四塊主線」分群。

## A. Meta-RL 骨架（MAML / RL² / PEARL / VariBAD）

### 15. MAML 式 inner/outer loop 作為「語言對齊」的雙層結構

類型：Potential Contribution / Proxy。來源：`paper/MAML.md`。

做法：把 MAML 的 bilevel（inner 快速適應、outer 學初始化）借用為「outer 學共享 policy + 語言-latent 對齊、inner 用極少語言/context 快速指定任務」。
和主線關係：MAML 是 gradient-based adaptation，與使用者「zero-shot（不做 gradient）」相反——所以它更適合當**對照基準**（few-shot 上界），凸顯 zero-shot 的難度。
風險：inner gradient 在嚴格 zero-shot 不可用，只能當 few-shot baseline。

### 16. RL² 風格 recurrent context 作為「無 gradient adaptation」baseline

類型：Diagnostic / baseline。來源：`paper/RL2.md`。

做法：用 RNN hidden state 當 fast learner，測試時純 forward 即適應，符合 zero-shot「不做 gradient」。
和主線關係：RL²/VariBAD/T2DA 都屬「forward-pass adaptation」，是使用者方法的天然 baseline 家族；RL² 是其中最樸素者（無 latent 結構、無語言）。可作為「拿掉語言、拿掉 belief」的下界 ablation。

（PEARL=context inference baseline、VariBAD=belief/Bayes-optimal 的零件，已於第 9–14 節詳述，不重複。）

## B. Offline RL 基礎（Offline RL 綜述 / D4RL / CQL / IQL / Decision Transformer）

### 17. IQL / CQL 作為 offline policy backbone 與保守度旋鈕

類型：Stabilizer / baseline。來源：`paper/IQL.md`、`paper/CQL.md`、`paper/OfflineRL_Survey.md`。

做法：offline 學 conditional policy 時，用 IQL（expectile + AWR，in-sample、不評估 OOD action）或 CQL（保守 Q）當底層演算法；policy 與 critic 條件化在 task embedding `z`/`ψ(l)` 上。
和主線關係：使用者是 offline 設定，必然踩到 distribution shift / extrapolation error（綜述核心）。IQL 對「弱標註 + 次優 play data」特別友善（AWR 用 advantage 加權，不需 OOD 外推），是最推薦的 backbone baseline；CQL 的 α 可當「保守度旋鈕」做 ablation。
風險：純 offline 下無法靠探索修正錯誤 latent，latent 品質決定上限。

### 18. Decision Transformer / return-to-go conditioning 作為「語言當條件 token」的橋

類型：Potential Contribution / baseline。來源：`paper/DecisionTransformer.md`。

做法：把 RL 寫成 sequence modeling，task 條件（return-to-go、或 `ψ(l)`）當 prompt token autoregressive 產 action。
和主線關係：這正是 T2DA-T 的骨架，也是把「文字 task embedding」接進 policy 的最輕量方式（只多一個 token）。可作為使用者 policy 端的現成架構與 baseline；與 value-based offline RL 形成兩條對照路線。

### 19. D4RL/資料品質維度作為 robustness ablation

類型：Diagnostic。來源：`paper/D4RL.md`。

做法：借 D4RL 的「narrow/undirected/suboptimal/stitching」屬性思維，系統性改變 play data 的覆蓋與品質，量測對齊與 zero-shot 的敏感度。
和主線關係：T2DA 只測了「資料品質、text encoder」兩種 robustness，沒測「語言標註量」；使用者可補上「資料品質 × 語言標註量」的二維 robustness 圖（novelty 空位）。

## C. Offline Meta-RL 主幹（MACAW / Online Self-Sup / CORRO / CSRO / Info-Theoretic COMRL / T2DA）

### 20. CORRO 對比式 robust task representation（去 behavior policy 汙染）

類型：Potential Contribution。來源：`paper/CORRO.md`。

做法：用 transition-level contrastive（鎖 `(s,a)` 造反事實負樣本）讓 task latent 只含任務資訊、不含 behavior policy 特徵。
和主線關係：play data 由雜多 behavior 收集，latent 極易被 policy 風格汙染。**CORRO（決定 latent「不該含什麼」）與語言 supervision（決定 latent「該含什麼、且如何結構化」）互補**——可把語言併進 InfoNCE 當額外正樣本。直接強化 `g(l)≈q(τ)` 的對齊品質。

### 21. CSRO context shift 與語言作為 policy-invariant anchor

類型：Potential Contribution / 核心警訊。來源：`paper/CSRO.md`。

做法：CSRO 用 max-min 互資訊（最小化 latent↔behavior policy、最大化 latent↔task）+ 測試時 non-prior 探索，處理「訓練 context 來自 behavior、測試 context 來自 exploration」的 context shift。
和主線關係：這是使用者題目**最該反覆讀**的問題。關鍵洞見——**語言可當 policy-invariant 的 task anchor**：因為 `l` 只描述任務、與收集 policy 無關，理論上同時達成「max(latent,task)」與「min(latent,policy)」，可能比 CSRO 的對抗式 min-MI 更直接降低 context shift。風險：弱標註若描述的是 behavior 風格而非 task，反而引入 shift（需 actor-z sensitivity 診斷）。

### 22. UNICORN 資訊論框架作為「弱語言 supervision objective」的設計工具

類型：Potential Contribution。來源：`paper/InfoTheoretic_COMRL.md`。

做法：該框架證明 FOCAL/CORRO/CSRO 都在優化 `I(Z;M)` 的不同上下界，並給 supervised / self-supervised 兩種實作。
和主線關係：可把弱語言視為 `I(Z;L) ≤ I(Z;M)` 的下界代理，設計半監督 objective：少量有語言的任務用 supervised 對齊、其餘 play 用 self-supervised——這是替使用者「極少弱標註」量身打造 loss 的理論依據（潛在 theorem/實驗貢獻）。

### 23. MACAW / Online Self-Supervision：offline meta-RL 的張力與洞

類型：baseline / 核心警訊。來源：`paper/MACAW.md`、`paper/OfflineMetaRL_OnlineSelfSupervision.md`。

做法：MACAW = AWR + enriched policy update + bilevel，是 offline meta-RL 起點 baseline；Online Self-Sup 指出 meta-test 收新 context 會造成 `p(z|h_offline)` vs `p(z|h_online)` 的 z-space shift。
和主線關係：使用者若主打**嚴格 zero-shot（測試只給語言、不收 context）**，正好**繞過** Online Self-Sup 揭示的 context shift（因為不在測試時收互動 context）——這可寫成使用者方法相對 context-based OMRL 的一個結構性優勢。MACAW 列為 offline meta-RL baseline。

### 24. T2DA = 最近鄰，CLIP 式對齊直接可用、novelty 空位明確

類型：Potential Contribution / 主要 baseline。來源：`paper/T2DA.md`。

做法：先學 dynamics-aware decision embedding `z=ϕ(τ)` → CLIP 式對比把 `ψ(l)` 對齊到 `z`（LoRA 微調 text encoder）→ policy 吃 `ψ(l)`，zero-shot text-to-decision。
和主線關係：**幾乎驗證了使用者核心機制**。直接可借：對比對齊（公式 3–5）+ LoRA、「先 dynamics embedding 再對齊文字」、「用同任務其他軌跡解碼避免自我重建捷徑」、兩種 backbone（Diffuser/Transformer）。**novelty 空位**：T2DA 用乾淨 template 語言 + SAC 多品質資料；使用者用**極少弱/hindsight 語言 + robot play**——「對齊在弱標註與 play 下是否成立、需多少標註才不崩」正是 T2DA 沒測的維度。列為**首要 baseline**。

## D. Robot Play / 語言 / 生態系（Play-LMP / LangIL / LOReL / C-BeT / PlayFusion / ZeST / RT-1/2 / OpenX / Octo / OpenVLA / CALVIN / RLDS）

### 25. Play 線：latent plan / multi-modal generation / hindsight 語言

類型：Potential Contribution / 資料與表徵基礎。來源：`paper/LatentPlansFromPlay.md`、`paper/LangConditioned_IL_Unstructured.md`、`paper/FromPlayToPolicy.md`、`paper/PlayFusion.md`。

做法與重點：
- **Play-LMP/LangLfP**：play 的三特性（便宜、多樣、無需任務分段）+ hindsight relabel + shared latent goal space，使語言標註可降到 <1%。但**還不是 meta-RL**（i.i.d.、無 task posterior、無 adaptation）——使用者要補的正是「belief/task inference」這層。
- **C-BeT（From Play to Policy）**：uncurated play 最大麻煩是多模態，需 multi-modal generation（k-means + offset）才不會塌成平均動作；把 play 變可用 policy 而非只學 representation。
- **PlayFusion**：hindsight language-annotated play + diffusion + discrete skill bottleneck（codebook）。diffusion 比 BC 更適合 noisy multimodal play；discrete skill 與「可組合 task embedding」呼應。
和主線關係：這四篇提供使用者「如何從 play 學出可用、可組合、抗多模態」的表徵基礎；latent plan / skill code 可當 `q(τ)` 的具體實作選項。

### 26. 語言當 reward vs 當 task embedding（LOReL）

類型：對照設計。來源：`paper/LangConditioned_Robot_Crowdsourced.md`。

做法：LOReL 用 offline robot data + crowd-sourced 語言學 language-conditioned reward，再接 planning，而非直接 text-to-action。
和主線關係：提供使用者另一條路線對照——「語言→reward」vs「語言→task embedding→policy」。crowd-sourced（事後第三方標）vs hindsight（自動由狀態標）的差異，影響弱標註的取得成本與雜訊特性。

### 27. Zero-shot task specification（ZeST）：spec 與 execution 解耦

類型：概念框架。來源：`paper/ZeroShot_TaskSpecification.md`。

做法：研究 foundation model 能否把語言/圖片/草圖等低成本指定，轉成機器人可執行的 task specification；強調 **task specification ≠ task execution**。
和主線關係：支持使用者把「文字→結構化 task embedding `z`」當中介表示（而非給目標圖）。「給一張目標圖不一定是好介面」的論證，正當化「用一句弱語言」的選擇。

### 28. Generalist VLA / 資料生態（RT-1/2、OpenX、Octo、OpenVLA、CALVIN、RLDS）

類型：baseline / backbone / benchmark / 工程基礎。來源：對應各 `paper/*.md`。

重點與主線關係：
- **RT-1 / RT-2 / OpenVLA**：語言直接 condition 的 generalist/VLA policy，**不強調 task inference 理論**。使用者應把它們當**比較對象與 backbone**，而非從頭重做；新意放在「task inference / belief」層。OpenVLA 可 LoRA 微調當「弱語言→z」的視覺-語言前端。
- **Octo**：開源、模組化、diffusion action head、可彈性接 observation/action space——**最適合當研究 backbone**，把使用者的 task-inference 放上層可省大量工程（約 6–7 成）。
- **Open X-Embodiment / RLDS**：cross-embodiment 資料標準化與資料格式（episode/step/metadata）。若使用者接大型 robot trajectory 資料，RLDS 幾乎必經；語言指令/`z` 應放哪一層 metadata 有現成慣例。
- **CALVIN**：**最契合的 benchmark**——本身就是 play + 只標 1% 語言 + 長時程 + 嚴格 zero-shot（新指令 + 新環境）。其 MCIL baseline 在 5 連鏈僅 0.08%，凸顯「imitation → offline meta-RL + 文字 task spec」的改進空間。建議作為主要評估平台。

### 29. LUMOS：play + 弱語言 + world model + zero-shot 的代表作（最近鄰之二）

類型：主要 baseline / 核心對照 / 可借用機制。來源：`paper/LUMOS.md`。

做法：從 play data 學 world model（DreamerV2/RSSM）→ 在凍結 world model 的 latent 空間用 actor-critic on-policy 練習，用 **DITTO latent-matching intrinsic reward**（獎勵 agent latent 軌跡貼近專家，公式 3）壓 covariate shift → <1% hindsight 弱語言做 **CLIP 式「語言↔latent 軌跡」對齊**（沒標註的 window 就略過對比項）→ zero-shot 遷真機。CALVIN 上勝 GCBC/MCIL/HULC（Avg.Len. 2.34）。

和主線關係：**這是文獻地景中「play + 弱語言 + zero-shot robot control」那塊的代表作（與 T2DA「offline meta-RL + 語言」互補）。** LUMOS 已命中使用者三塊中的機器人三塊（play data、<1% 弱語言、zero-shot 語言操控+sim→real），但**沒有 offline meta-RL 的 task belief / unseen-task 泛化**——語言只是直接 condition policy。使用者的 novelty 空位 = 把 LUMOS 的「play+弱語言+world model」接上 T2DA/VariBAD 的「task inference/belief」formalism。

可直接借用：
- CLIP 式「語言↔latent 軌跡」對齊 + MiniLM + 「無標註 window 跳過 L_contrast」——直接是使用者「極少弱語言對齊」的乾淨實作範本。
- **DITTO latent-matching reward**：把模仿從「回歸 action」變成「latent 空間 match 專家」，提供「純 offline 卻能在 world model imagination 裡 on-policy 練習、壓 covariate shift」的機制——正面回應 research_direction_options.md 定義 D「offline 下沒得探索」的難題。
- World model（RSSM `(h_t,z_t)`）當 backbone，可與 VariBAD belief / 第 10 節「含未來重建 ELBO」結合。

警訊：
- LUMOS 已做到「play+弱語言+zero-shot 語言操控」，使用者**不能**把這些當賣點，賣點必須是 LUMOS 沒有的 task belief / unseen-task 泛化（否則被視為增量）。
- No alignment 消融會操作錯顏色 → 弱語言對齊穩定性是成敗關鍵、視覺/latent grounding 要與語言一起對。
- LUMOS 自陳「難事先知道 world model 是否夠好、解析度越高越好但算力暴增」→ world model 品質是隱性瓶頸（呼應第 14 節 OOD/重建誤差診斷）。

## E. 萃取後對主線的三點總判斷

1. **核心機制已被 T2DA 與 LUMOS 雙重驗證可行**：兩篇都用 CLIP 式對比把語言對齊到一個 latent（T2DA→task embedding；LUMOS→world model latent 軌跡），證明「`g(l)≈q(τ)` 對齊」成立。使用者的 novelty 不在「能不能做」，而在**兩個近鄰的交集之外**：LUMOS 有 play+弱語言+world model 但無 task belief；T2DA 有 offline meta-RL+task representation 但用乾淨 template 語言+SAC 資料。空位＝「在 play+極少弱語言下，學一個可做 task inference 的 belief latent，並對 unseen task（而非只 unseen instruction）做 zero-shot」。
2. **最危險的敵人是 context/representation 被 behavior policy 汙染**（CSRO/CORRO 的核心）；語言作為 policy-invariant anchor 可能是比對抗式 min-MI 更乾淨的解法，這是潛在理論貢獻。
3. **工程上可大幅站在巨人肩上**：Octo 當 backbone、CALVIN 當 benchmark、RLDS 當資料格式、IQL/DT 當 policy；**baseline 用 T2DA（meta-RL 線）+ LUMOS/HULC（play+語言線）兩條都要打**。可借 LUMOS 的 DITTO latent-matching reward 解決「offline 下沒得探索」的難題。使用者應聚焦在「弱語言對齊 + task belief + unseen-task 泛化」這層創新，而非重造輪子。

---

## 建議保留方式

這些設計應保留在「research variations」或「diagnostic experiments」中，不應放入 faithful PEARL 主線。

推薦分類：

| 保留 | 用途 |
| --- | --- |
| Analytic PEARL-latent | upper-bound / structured inference baseline |
| Hidden goal + wind task | debugging benchmark |
| True latent supervision | auxiliary supervision ablation |
| BC regularization | offline meta-RL / demonstration ablation |
| Probe bootstrap | exploration strategy ablation |
| Posterior ensemble | uncertainty / variance ablation |
| Actor-z sensitivity | representation collapse diagnostic |
| Context summary features | engineered encoder baseline |
| VARIBAD-style belief embedding | weak-language offline meta-RL / zero-shot task specification |
| 全軌跡重建 ELBO（含未來） | latent 學習訊號 ablation（PEARL latent vs +VAE 重建） |
| Belief-conditioned policy + encoder 梯度解耦 | 訓練穩定性 / 加速 trick ablation |
| Task belief 當 state（BAMDP 視角） | Bayes-optimal vs posterior-sampling 探索主線命題 |
| 時間固定 task embedding | reset/多 rollout 漂移診斷與凍結正則 |
| OOD inference 失效診斷 | OOD 偵測 / 再訓練 / learned prior 研究 |

## 一句話總結

目前這些創新適合作為「理解與診斷 PEARL 為何學不起來」的工具，也可發展成未來研究題材；但如果目標是學習原論文 PEARL，下一階段應把它們關掉或移到 ablation，回到 faithful reproduction。若目標轉向 robot play data + weak language + offline meta-RL，VARIBAD-style belief embedding 可作為較完整的研究主線。

從 VariBAD（第 10–14 節）萃取的 idea，補強了 PEARL 主線缺少的兩塊：一是用「重建過去+未來的 VAE 目標」給 task latent 更明確的學習訊號（而非只靠 critic loss）；二是把問題從「posterior sampling 探索」提升到「Bayes-optimal 探索 + 把 task belief 當 state」的理論定位。這些都可作為 ablation 或研究主線命題，但同樣不應混入 faithful PEARL reproduction。VariBAD 原文逐段筆記見 `paper/VariBAD.md`。
