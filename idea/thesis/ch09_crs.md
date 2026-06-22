# 第 9 章　CRS：跨基礎模型保形組合指稱集合（主方法與主結果）

本章是全篇的台柱。C1–C4 回答「凍結基礎模型的不確定性訊號**有多 informative**」，M4 證明點預測在完整 GREC exact-match 撞牆。CRS 把貢獻從「**量測**可靠度」升級為「**建構有分布無關保證的可靠集合**」：讓信心策略 `π` 輸出一個基數可為 0／1／多的**風險受控指稱集合**，以 Learn-then-Test 聯合校準三個有界風險並給出有限樣本保證，再以跨基礎模型組合把風險控制的代價因式分解為兩個正交瓶頸。

## 9.1 動機：從「測量」到「保證」

CRS 的轉折來自 M4 的診斷數字本身。M4 顯示多目標 exact-match 的 per-sample oracle 天花板僅 0.19–0.24，召回 recall 0.94、瓶頸是 FP 多餘框、單一全域 τ 無法每樣本恰好選對框數。

- **舊解讀**：事後校準做不到（死路）。
- **新解讀**：raw score 把 TP/FP 框在分數軸上交織，任何**點估計**閾值都切不乾淨——但我們不需要點估計，可以輸出一個**有分布無關保證的集合**。問題於是從「能不能精確命中」變成「**能不能在凍結基礎模型上給出有有限樣本保證的框集合，並刻畫保證的可行邊界**」。

這一步把題目從「我們做不到 SOTA exact-match」改寫成「我們給 distribution-free 保證 + 可行域」。審查打不掉「保證」，也打不掉「我們證明了什麼可達、什麼不可達」——這是 measurement 給不了的護城河。

## 9.2 問題形式化：風險受控指稱集合

凍結基礎模型 `g` 對查詢 `(I, e)` 吐出候選框集合 `{(b_i, s_i)}`（OWL-ViT／GroundingDINO 原生有；CLIP-VG 無候選，不適用本章，留作轉移討論）。一個**指稱集合策略** `π_λ` 以閾值／參數 `λ` 從候選中選出集合 `S_λ(I,e) ⊆ {b_i}`，基數可為 0（棄答／無目標）、1 或多。

我們控制的不是準度，而是**風險**。對 target-present 樣本定義**漏檢風險（FNR）**：

```
L_FNR(S, G) = 1 − |{g ∈ G : ∃ b ∈ S, IoU(b,g) ≥ 0.5}| / |G|
```

（G = GT 框集合）。對 no-target 樣本定義**誤選風險** `L_NT(S) = 1[|S| > 0]`。對 target-present 樣本另定義**棄答風險** `L_DEF(S) = 1[|S| = 0]`（真有 target 卻吐空集）。

**目標**：在校準 split 上選 `λ̂`，使得在 unseen test 上同時控制三個有界風險：

- **(R1) 已作答目標漏檢率（answered-target FNR）**：在**未棄答**的 target-present 樣本上 `E[L_FNR] ≤ α`；
- **(R2) 無目標誤選率（no-target false selection）**：no-target 上 `E[L_NT] ≤ β`；
- **(R3) 目標棄答率（target deferral）**：target-present 上 `E[L_DEF] ≤ γ`。

三者皆 distribution-free、finite-sample。α、β、γ 是使用者旋鈕（safety budget）。

**誠實命名（必須寫白）**：R1 是**條件**風險——只在系統選擇回答的 target-present 樣本上計算 FNR，因此**不可**稱為 target recall／coverage guarantee；正式名稱一律 **answered-target FNR**（完整：未棄答 target-present 查詢上的條件 target-set FNR）。R3 的存在正是為了防守「你只是把難的 target 棄答掉，所以 R1 好看」——棄答是被明碼控制的第三風險，不是藏起來的成本。R2 與 R3 語義不同：no-target 吐空集是**正確**決策，target-present 吐空集才是**代價**，兩者分開計、分開控。

**指標精確定義（必寫在 caption）**：

- **answered TP set size**（主表的 "set size"）：平均集合大小只在**已作答 target-present 查詢**（target-present 且最終非空輸出）上計算，不是所有查詢的平均。
- **R3 target deferral**：target-present 最終輸出空集的比例（gate 棄答**或** box 門檻濾光皆計入）。
- **R1 answered-target FNR**：僅在 target-present 且最終非空的查詢上計算。

**校準協定澄清（避免誤解為標準 benchmark held-out test）**：本研究對每個官方 split（val/testA/testB）**內部**以 `ref_id` parity 切成 calibration / evaluation 兩半；閾值網格、LTT 檢定、操作點選擇**只用 calibration 半**，回報的風險與 bootstrap CI 在 held-out parity 半上計算。這是 split 內的風險控制評估，而非標準 train-val-test；跨 split transfer 若報，僅為經驗穩定度，不在 distribution-free LTT 保證範圍內。

這正是 GREC 官方 Pr@(F1=1) 的「機率化、有保證」版本：F1=1 要求零漏檢且零多選，我們不追逐那個 0/1 事件，而是給「answered-target FNR ≤ α」的連續、有保證旋鈕。

## 9.3 單風險 CRS：Conformal Risk Control 與不可化約下限

先只控 R1。`L_FNR` 對「選更多框」單調非增 ⇒ 對閾值 `λ` 單調非減。用 **Conformal Risk Control**（Angelopoulos et al. 2023）在 calib 的 n 個 target-present 樣本上選最大的 λ（最小集合）使

```
(n · R̂_FNR(λ) + B) / (n + 1) ≤ α,    B = 1（loss 上界）
```

**實證**（OWL-ViT gRefCOCO 三個 split，ref_id 奇偶分 calib/test）：nominal α 與 empirical FNR 近乎完美對角線追蹤（val：α 0.2→0.189 / 0.3→0.279 / 0.4→0.374 / 0.5→0.471；testA/B 同）。唯一例外在 α=0.05（testB 連 0.10）——並非 conformal 失效，而是**候選池物理召回上限**：全選候選後殘餘 FNR 仍有 val 0.06 / testA 0.10 / testB 0.14（即 GT-coverage recall ≈ 0.94 的硬上限）。

conformal 框架**精確暴露**了凍結基礎模型的不可化約下限：任何閾值都打不破候選池沒召回到的 GT。這是一個可量測、可報告、distribution-free 的「基礎模型能力上限」——CRS 不只給保證，還診斷出保證在哪裡撞到基礎模型的物理極限。

## 9.4 多風險 CRS：耦合風險與 Learn-then-Test

同時控 R1 + R2 + R3 是本章的技術核心，也是與既有 conformal-OD 文獻（只控單一 box coverage）的關鍵區隔。

**為何單一全域閾值不行**：R1 要「選多」、R2 要「選少」，方向相反；更糟的是兩者透過**重疊的分數分布**耦合——OWL-ViT zero-shot 弱，target 與 no-target 樣本的 top-score 分布高度重疊。

**天真序貫校準失敗（誠實記錄）**：先用 no-target calib 標一個棄答 gate `τ_A`（壓 R2），再在通過 gate 的 target 樣本上標 λ（壓 R1）。實證 R2 守住但 R1 爆到 0.63–0.76——因為 `τ_A` 為壓 no-target 誤選被推高，連帶擋掉大量真 target（它們分數也低），被擋的真 target 吐空集 ⇒ FNR=1。**這個失敗是發現**：兩風險不是獨立可加，序貫校準破壞聯合有效性。

**正解 = Learn-then-Test（LTT, Angelopoulos et al. 2021）**：把 `(τ_A, λ)` 二維網格的每一格當成一個假設，對每個 risk 做 finite-sample 檢定（Hoeffding bound 的 p-value），用 Bonferroni 控制 family-wise error ≤ δ：

```
p_R(config) = exp(−2 n_R (target_R − R̂)²)   若 R̂ < target_R 否則 1
config 有效 ⟺ p_R1 ≤ δ/(3|grid|) 且 p_R2 ≤ δ/(3|grid|) 且 p_R3 ≤ δ/(3|grid|)
```

只保留三 risk 都通過的 config ⇒ **聯合、distribution-free 保證**。Bonferroni 分母是 **3·|grid|**（三風險 × 整個 config 網格），FWER ≤ δ 涵蓋整個搜尋空間。所有通過的 config 構成**可行域（feasible region）**；在可行域內挑「**最小 calibration 集合大小**」的點作為操作點。此 selection 仍 post-hoc 合法：因為 FWER 已涵蓋整個 grid，可行域內任意挑點都不破壞保證（這正是 LTT 的設計目的，直接回答「min-size 選點是否挑數字」的質疑）。

**洩漏防守（必須寫白）**：閾值網格的 quantile **只用 calibration covariates** 建構；evaluation 樣本的 covariates 與 labels 從不參與網格建構、風險檢定或操作點選擇。

**實證**（單一 base OWL-ViT，δ=0.1，30×30 calib-only grid，三風險）：三個 split 聯合保證**全部成立**，**但操作點退化**：純 OWL-ViT 即使取可行域最小集合，仍需約 8–9 框（val 8.23 / testA 9.25 / testB 7.12），GroundingDINO 單 base 則因 gate 弱、需靠高棄答才守住 R2/R3。這個退化正是下一節 cross-base composition 的動機。

在弱零樣本單 base 上，要同時拿 distribution-free 的三風險保證，代價是大集合或高棄答。這**不是方法失敗**，是一個嚴格的數學事實：保證的代價由基礎模型的分數可分性決定。CRS 把這個代價**明碼標出來**（三風險 + 集合大小），而非藏在平均準度後面。這比 M4 的經驗觀察強一個量級——它是有限樣本、有保證、可畫成可行域圖的根本權衡刻畫。

## 9.5 推可行域：nonconformity 設計與更強基礎模型

可行域退化的根因是 nonconformity 太弱（raw score）。兩個正交槓桿把可行域往**有用方向**（小集合、低棄答）推：

**(a) Per-box consistency belief（方法內元件）**：以改寫下的 per-box 出現穩定度重定義 nonconformity：`belief(b) = s(b) · (1 + agreement(b))`，agreement = 該 box 在改寫下被高分框呼應的程度。實證（OWL-ViT gRefCOCO val）：同保證下集合縮小 α 0.1→1.7% / 0.2→3.7% / 0.3→6.2% / 0.4→10.0%，validity 不破。誠實標註：高召回區增益偏弱，當「nonconformity 設計」一節如實報，非台柱。

**(b) 更強凍結基礎模型（GroundingDINO）**：在新 framing 下，基礎模型**不是被比較的對象**，而是「**框架隨基礎模型能力擴張可行域**」的一個軸。同一套 LTT 保證程序，基礎模型的分數可分性越好，可行域裡越早出現有用操作點（低棄答 + 小集合）。

**實證診斷（決定性）**：對 target-present 子問題，達到同一 FNR≤α 保證所需集合大小，GroundingDINO 比 OWL-ViT 小約 **5×**（val，α=0.2：3.3 vs 18.6 框；α=0.3：2.4 vs 12.1），且**不可化約召回下限 0.007 vs 0.062**（val；GroundingDINO 候選池幾乎涵蓋所有 GT；testA/testB floor：GD 0.012/0.017 vs OWL 0.104/0.142）。target 平均約 2 GT 框，GroundingDINO 在 α=0.3 只需 2.4 框 ≈「幾乎恰好選對」。

## 9.6 核心方法：Cross-Base Conformal Composition

**問題的乾淨因式分解**（本身是貢獻）：CRS 的保證代價可分解為兩個**正交**的瓶頸，分別由不同的基礎模型能力決定，且**沒有單一基礎模型兩者都強**：

**表 9.1　兩個瓶頸與兩個基礎模型的互補性**

| | 無目標可分性（棄答 gate） | target 集合可分性（set selection） |
|---|---|---|
| OWL-ViT | **0.82**（好） | 差（α=0.3 需 12 框） |
| GroundingDINO | 0.60（差，對任何 query 都給高信心框） | **強（α=0.3 需 2.4 框），召回下限 0.007** |

OWL-ViT 擅長判斷「**該不該答**」，GroundingDINO 擅長「**答得準**」。單一基礎模型的多風險 LTT 因此必然退化：OWL-ViT 集合爆炸、GroundingDINO 被弱 gate 拖累（高棄答）。

**解法 = 兩個凍結基礎模型各司其職，用 LTT 聯合校準三風險**：

- **棄答 gate** ← OWL-ViT 分數 `top1_score`（無目標可分性 0.82）；
- **set selection** ← GroundingDINO 候選 + 分數（召回下限 0.007、集合小）；
- 兩基礎模型皆 frozen、皆不訓練；join key =`(ref_id, sent_id)`（共享 gRefCOCO 標註，expression／no_target 零 mismatch）；
- LTT 在 `(τ on OWL 分數, λ on GDINO 分數)` 二維 calib-only grid 上聯合校準 `(R1≤α, R2≤β, R3≤γ)`。

**主結果（三個 split 全量，三風險 α=β=0.3 / γ=0.5 聯合保證，calib-only grid，min-size 選點）**：

**表 9.2　Cross-Base Composition 主結果**

| split | set size（answered TP） | R1 | R2 | defer(R3) |
|---|---:|---:|---:|---:|
| val | **3.24** | 0.191 | 0.159 | 0.424 |
| testA | **2.02** | 0.261 | 0.203 | 0.443 |
| testB | **3.50** | 0.168 | 0.236 | 0.413 |

三風險全部守住。其中 testA 的 **2.02 框 ≈ GT 平均基數**，幾乎是「恰好選對數量」；testB 也在三保證下維持 <3.6 框。同一 protocol 下，純 OWL-ViT 需 8.23 / 9.25 / 7.12 框，composition 的集合 CI 與純 OWL-ViT 完全分離。公平比較下的 set-size reduction 約 **2.0×–4.5×**。

這把「沒有單一基礎模型兩者都強」的**限制**，轉成「組合兩個凍結基礎模型互補強項」的**正面方法**：純事後、不訓練、接回 C4 cross-base 主軸、非 detector 比較（是互補組合）。selection rule = 在三風險可行域內取 min calibration set size，post-hoc 合法（FWER Bonferroni 已涵蓋整個 grid）。

## 9.7 機制證實：2×2 gate×box 消融

對調 gate-base 與 box-base 的四種組合（三風險 α=β=0.3 / γ=0.5，calib-only grid，min-size 選點）：

**表 9.3　2×2 gate×box 消融**

| split | gate | box | 集合 | R1 | R2 | defer(R3) | feasible? |
|---|---|---|---:|---:|---:|---:|:--:|
| val | OWL | OWL | 8.23 | 0.239 | 0.183 | 0.366 | ✓ |
| val | GD | GD | — | — | — | — | **EMPTY** |
| val | **OWL** | **GD（COMPOSE）** | **3.24** | 0.191 | 0.159 | 0.424 | ✓ |
| val | GD | OWL（reverse） | 15.27 | 0.211 | 0.251 | 0.429 | ✓(n=2) |
| testA | OWL | OWL | 9.25 | 0.255 | 0.207 | 0.443 | ✓ |
| testA | GD | GD | — | — | — | — | **EMPTY** |
| testA | **OWL** | **GD（COMPOSE）** | **2.02** | 0.261 | 0.203 | 0.443 | ✓ |
| testA | GD | OWL（reverse） | — | — | — | — | **EMPTY** |
| testB | OWL | OWL | 7.12 | 0.234 | 0.236 | 0.413 | ✓ |
| testB | GD | GD | — | — | — | — | **EMPTY** |
| testB | **OWL** | **GD（COMPOSE）** | **3.50** | 0.168 | 0.236 | 0.413 | ✓ |
| testB | GD | OWL（reverse） | — | — | — | — | **EMPTY** |

一旦把目標棄答（R3）也納入受控風險，**pure GD（GD gate + GD box）三個 split 全部可行域變空**——GD gate 守無目標必須靠極高棄答，一旦棄答受 γ=0.5 約束就無解；reverse（GD gate + OWL box）也幾乎全空（val 僅 n=2、集合爆 15）。在三風險下，**只有 OWL gate + GD box 這條對角線同時 feasible 且 compact**。這正面回答「CRS 只是 GroundingDINO 比較強」的質疑：GD 單獨無解，是 **factorization** 才進得了可行域。factorization 主張因此嚴格成立：**OWL gate 控可用棄答／無目標；GD box 控 target 集合緊緻度**。

需限定範圍：2×2 消融在**主 split-specific LTT 協議**下證明 factorization；額外的 stricter validity check（§9.8）只施加在最終 COMPOSE 操作點，未對整個 2×2 重跑。

## 9.8 穩健性與有效性

**統計穩固性（bootstrap CI，config 固定只 resample test）**：三個 split 全量 shared keys = 14229 / 19200 / 16063。

**表 9.4　COMPOSE vs OWL-only 的 bootstrap CI**

| split | OWL-ViT only（answered TP size） | COMPOSE（answered TP size） | COMPOSE R1 | COMPOSE R2 | COMPOSE defer |
|---|---:|---:|---:|---:|---:|
| val | 8.23 [7.85, 8.57] | **3.24 [3.15, 3.36]** | 0.191 [0.179, 0.203] | 0.159 [0.146, 0.171] | 0.424 [0.405, 0.443] |
| testA | 9.25 [9.00, 9.51] | **2.02 [1.98, 2.06]** | 0.260 [0.250, 0.269] | 0.203 [0.187, 0.221] | 0.443 [0.433, 0.454] |
| testB | 7.12 [6.88, 7.35] | **3.50 [3.38, 3.63]** | 0.168 [0.159, 0.177] | 0.236 [0.219, 0.252] | 0.414 [0.401, 0.426] |

三個 split 的 R1/R2 CI 上界都低於 0.3、defer 都低於 γ=0.5；COMPOSE 與純 OWL-ViT 的 size CI 完全分離。結果不是 partial dump 或小樣本僥倖：全量 val/testA/testB 皆在三保證下達到接近 GT 基數的緊緻指稱集合。

需分清：LTT p-value + Bonferroni = 有限樣本風險控制檢定（這是「保證」）；bootstrap CI = 固定 selected config 後對 test 重抽的經驗穩定度（這**不是**保證）。兩者分開報、不混用。

**split-protocol 穩健性（堵 exchangeability）**：COMPOSE 在三種 calib/eval 切分下幾乎不動——parity（主）、5× random ref_id split、image-disjoint（同一影像不跨 calib/eval，最嚴格）：

**表 9.5　三種切分模式的 set size**

| split | parity（主） | random[5] 平均 | image-disjoint |
|---|---|---|---|
| val | 3.25 | 3.24 ± 0.04 | 3.13 |
| testA | 2.02 | 2.05 ± 0.03 | 2.06 |
| testB | 3.51 | 3.53 ± 0.11 | 3.39 |

set size 三模式幾乎重合（random std 僅 0.03–0.11），image-disjoint 亦 feasible 且 compact。誠實標註：random split 的 R2/defer 單次 variance 較大（val R2 偶到 0.26、testB 到 0.29），屬 LTT marginal 保證的正常表現——單次抽樣可略超名目，報 mean±std 並說明即可。

**formal-validity 穩健性**：針對 LTT 保證的三個深層形式問題各做一個對照協議。

> **formal validity 定調（必讀）**：主表（表 9.2/9.4）報的是 **implementation operating-point**
> 版本——R1 用 answered subset 的 Hoeffding，其分母 n1 隨 (τ,λ) 變動。這個版本方便、且與
> ratio-free 版數字幾乎一致，但**嚴格的 distribution-free 保證由 (C) ratio-free 條件風險檢定
> 承擔**（R1 改寫成固定分母=target-present 全數的 bounded 變數，見下）。R2/R3 的分母（no-target /
> target-present）本就固定，Hoeffding p-value 直接給有限樣本保證。因此本論文的 formal claim
> 以 ratio-free R1 為準，operating-point 版僅為等價的實作呈現。

**表 9.6　formal-validity 對照（COMPOSE，α=β=0.3 / γ=0.5）**

| 協議 | val | testA | testB | 結論 |
|---|---|---|---|---|
| operating-point（parity, answered-subset Hoeffding R1） | 3.24 | 2.02 | 3.50 | 實作呈現版 |
| **(C) ratio-free R1（fixed-n 條件風險檢定）= formal 防線** | 3.25 | 2.02 | 3.51 | **幾乎完全重現，formal claim 以此為準** |
| (A) three-way split（grid/calib/eval image-disjoint 三分） | 3.36 | 3.31 | 3.35 | 三 split 全 feasible |
| (B) image-cluster（image=calib unit，最嚴格） | 3.24 (n=28) | EMPTY | EMPTY | 見下方誠實邊界 |

- **(C) ratio-free R1（formal validity 主防線）**：R1 是條件風險，operating-point 版的 Hoeffding 用 answered subset 的 random denominator（n1 隨 config 變）。formal 版改用 `E[L|A]≤α ⟺ E[A(L−α)]≤0` 對 bounded fixed-n 變數 `Z=A(L−α)∈[−α,1−α]` 做 Hoeffding，分母固定為 target-present 全數，因此 p-value 的有限樣本保證不依賴 data-dependent 分母。三個 split 與 operating-point 版幾乎完全一致 → random-denominator 質疑實務上不影響結論，且 distribution-free 保證在 ratio-free 版上嚴格成立。
- **(A) data-dependent grid**：grid 從 calib scores 建、又在同 calib 上做 LTT，理論上可質疑。three-way split（grid-design / calib / eval 三個 image-disjoint 子集）下三 split 仍 feasible，set size 3.31–3.36（略升因 eval 全新 image + 樣本變少），結論不變。
- **combined protocol**：把最嚴格的兩個協議疊加（three-way + ratio-free R1）——val sz=3.36 仍 feasible（n_feas=112），且 three-way 的 5 seeds robustness：val 5/5 feasible，sz=3.29±0.11，非 seed=0 僥倖。

**candidate-pool 截斷（protocol 透明化）**：CRS 的候選池是基礎模型經 `PRED_KEEP=50` + `PRED_MIN_SCORE=0.01` 預處理後的 stored pool。下界檢查顯示：雖然 val 100% 樣本原始 `n_cands>50`，但通過 score≥0.01 門檻的候選平均僅 45.9 個（<50），代表 `KEEP=50` 的 cap 幾乎未實際裁切，主導的是 0.01 門檻；最終 set size 由 LTT 選出的 λ（遠高於 0.01）決定，非 dump cap。故「compact set 靠 cap 作弊」不成立。完整 `KEEP∈{20,50,100}` sweep 需重跑 GPU dump，列為 future robustness。

## 9.9 兩項誠實限制

**1. image-cluster 評估的樣本量限制。** 主協議以 **expression** 為 calibration unit；同影像多 expression 是 record-level exchangeability 假設。改以 **image** 為 unit（最嚴格）時，僅 val 維持 feasible（n=28，勉強），testA/testB EMPTY（見表 9.6）。這**不是 validity 崩塌**，而是樣本量限制：image-as-unit 把有效樣本數從約 9000 records 降到約 3000 images，疊加三風險 × Bonferroni（3·|grid|）的嚴格門檻後，有限校準集不足以同時認證三個保證。這是凍結零樣本基礎模型 + 三風險 + 有限校準的根本張力，誠實標為 limitation；主結果用 expression-level，並以 image-disjoint **split** 穩健性（表 9.5）作為 exchangeability 的經驗緩解。

**2. 長 expression 的 gate 退化。** R2（無目標誤選率）隨 expression **長度顯著惡化**：0–8 token（佔約 70%，gRefCOCO 主體）R2=0.13–0.21 守住，9–16 token 升到 0.40–0.61，17+ token 更高。關鍵：9–16 token 段的 OWL tokenizer 截斷率仍是 0%，R2 卻已惡化 → 這**不純是** OWL `max_length=16` 截斷造成，而是 OWL gate 對長 expression 的無目標判斷本就較弱（長句 top1_score 訊號不可靠）；17+ token 的 100% 截斷讓它雪上加霜。aggregate headline 由短句主導故站得住，但長 expression 的 gate 退化是真 limitation。更強／共享的文字編碼器是 future work。

## 9.10 貢獻定位

**表 9.7　CRS 與既有文獻的區隔**

| 既有工作 | 與 CRS 的區隔 |
|---|---|
| Conformal Object Detection（box coverage、FNR） | 封閉類別、單 box-coverage；無語言條件、無無目標、無 cardinality。CRS 控的是 referring set 的耦合三風險 |
| Conformal for Zero-Shot VLM（CVPR'25） | 只做**分類** label set；非 box set |
| GREC / HieA2G / InstanceVG（multi-target） | **全 trained**（count head / 階層 align）；CRS 是 post-hoc、distribution-free、不訓練 |
| VLM selective prediction（ReCoVERR） | 單答案 answer/abstain；無集合層保證、無 cardinality |
| True-False Verification（2509.09958） | 單答案對錯 verify；CRS 是集合層聯合保證 + 計數 |

**貢獻定位三句話**（禁用「first／第一個」）：
1. 對凍結指稱定位做 **risk-controlled box-set selection**：不追 exact-match，而是輸出 calibrated box set，並用 LTT 對三個有界風險（answered-target FNR、no-target false selection、target deferral）做有限樣本控制；
2. 把風險控制的**代價因式分解**為 gate 與 box 兩個正交瓶頸，並以可行域刻畫；
3. 用 **cross-base conformal composition** 組合兩個異質凍結基礎模型的互補強項（一個管棄答、一個管選框），在三保證下達到接近 GT 基數的精準集合——2×2 消融證明這是 factorization 而非單純 detector 比較（單 base 在三風險下退化或無解）。

## 9.11 小結

CRS 把全篇從「測量凍結基礎模型的不確定性」推進到「在凍結基礎模型上建構有分布無關保證的指稱集合」。它以 LTT 聯合校準三個有界風險、把代價因式分解為 gate 與 box 兩個正交瓶頸、並以跨基礎模型組合在三保證下達到接近真實基數的緊緻集合（val/testA/testB 為 3.24 / 2.02 / 3.50 框，集合大小信賴區間與純 OWL-ViT 完全分離）。2×2 消融、bootstrap CI、三種切分模式與 formal-validity 對照共同支撐結論的穩固性，兩項限制（image-cluster 樣本量、長 expression gate 退化）誠實標註。這是本論文「強框架 + 亮眼正面操作點」的最終解，也是 M4 撞牆之後的回答：不做點估計，改輸出有保證的集合。
