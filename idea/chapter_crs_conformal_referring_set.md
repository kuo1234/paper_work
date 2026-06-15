# Chapter: Conformal Referring Sets (CRS) — 主方法章草稿 v2（紅隊 P0 修補後）

> 寫作原則：論文主體英文，中文〔註〕標 framing/亮點/誠實邊界（最終稿移除）。
> 這是 **2026-06-13 主軸再升級**後的新台柱章，**2026-06-15 套用紅隊 P0/P1 修補**：
> calib-only grid（修 leakage）、三風險 LTT（加 R3 target deferral）、R1 改名 answered-target FNR、
> 移除所有 "first" 與 recall/coverage guarantee、LTT guarantee 與 bootstrap CI 嚴格分離、split robustness。
> 延續 [[chapter_method]] 的「凍結 base + post-hoc policy」框架，
> 但把貢獻從「**量**可靠度（reliability measurement）」升級成「**建構有保證的可靠集合**（reliable set construction with distribution-free guarantees）」。
> M4 那道牆（multi-target exact-match 超出 post-hoc 能力）在這裡從「邊界/失敗」翻成「**根本權衡的嚴格刻畫**」。

---

## 4.1 動機：從「測量」到「保證」

舊四貢獻（C1–C4 + M4）回答的是「frozen base 的不確定性訊號**有多 informative**」。
紅隊與使用者的共識是：這只是 *measurement*，缺一個**方法級、有護城河**的正面貢獻。

CRS 的轉折來自 M4 的診斷數字本身。M4 顯示 multi-target exact-match 的 per-sample oracle 天花板僅 0.19–0.24，
召回 recall 0.94、瓶頸是 FP 多餘框、單一全域 τ 無法每樣本恰好選對框數。
**舊解讀**：post-hoc 做不到（死路）。
**新解讀**：raw score 把 TP/FP 框在分數軸上交織，任何**點估計**閾值都切不乾淨——但我們不需要點估計，
我們可以輸出一個**有分布無關保證的集合**。問題於是從「能不能精確命中」變成
「**能不能在凍結 base 上給出有有限樣本保證的 box 集合，並刻畫保證的可行邊界**」。

> 〔亮點〕這一步把題目從「我們做不到 SOTA exact-match」改寫成「我們給 distribution-free 保證 + 可行域」。
> referee 打不掉「保證」，也打不掉「我們證明了什麼可達、什麼不可達」。這是 measurement 給不了的護城河。

---

## 4.2 問題形式化：Risk-Controlled Referring Set

frozen base `g` 對 query `(I, e)` 吐出候選框集合 `{(b_i, s_i)}`（OWL-ViT / GroundingDINO 原生有；
CLIP-VG 無候選，不適用本章，留作 transfer 討論）。一個 **referring set policy** `π_λ`
以閾值/參數 `λ` 從候選中選出一個集合 `S_λ(I,e) ⊆ {b_i}`，其基數可為 0（abstain / no-target）、1、或多個。

我們要控制的不是準度，而是**風險**。對 target-present 樣本定義 **漏檢風險（FNR）**：

```
L_FNR(S, G) = 1 − |{g ∈ G : ∃ b ∈ S, IoU(b,g) ≥ 0.5}| / |G|
```

（G = GT 框集合）。對 no-target 樣本定義 **誤選風險**：`L_NT(S) = 1[|S| > 0]`。
對 target-present 樣本另定義 **棄答風險（deferral）**：`L_DEF(S) = 1[|S| = 0]`（真有 target 卻吐空集）。

**目標**：在 calibration split 上選 `λ̂`，使得在 unseen test 上同時控制**三個有界風險**：

- **(R1) answered-target FNR**：在「**未棄答**」的 target-present 樣本上 `E[L_FNR] ≤ α`
- **(R2) no-target false selection**：no-target 上 `E[L_NT] ≤ β`
- **(R3) target deferral**：target-present 上 `E[L_DEF] ≤ γ`

三者皆 **distribution-free、finite-sample**。α、β、γ 是使用者旋鈕（safety budget）。

> 〔誠實命名 — 必須寫白〕R1 是**條件**風險：只在系統選擇回答的 target-present 樣本上計算 FNR，
> 因此**不可**稱為 target recall / coverage guarantee（那會被口委直接打穿）。正式名稱一律
> **answered-target FNR**（完整：conditional target-set FNR among non-deferred target-present queries）。
> R3 的存在正是為了防守「你只是把難的 target 棄答掉，所以 R1 好看」——deferral 是被明碼控制的第三風險，
> 不是藏起來的成本。R2（no-target false selection）與 R3（target deferral）語義不同：no-target 吐空集是**正確**決策，
> target-present 吐空集才是**代價**，兩者分開計、分開控。

**指標精確定義（避免誤解，必寫在 caption）**：

- **answered TP set size**（主表的 "set size"）：平均集合大小**只在 answered target-present queries**（target-present 且最終非空輸出）上計算，**不是**所有 query 的平均。caption 一律標：*Set size is averaged over answered target-present queries unless otherwise specified.*
- **R3 target deferral**：target-present 最終輸出空集的比例（gate 棄答 **或** box 門檻濾光皆計入；與形式化定義 `L_DEF=1[|S|=0]` 一致）。
- **R1 answered-target FNR**：僅在 target-present 且最終非空的 query 上計算。

**校準協定澄清（P0：勿寫成標準 benchmark held-out test）**：本研究對每個 official split（val/testA/testB）**內部**以 `ref_id` parity 切成 calibration / evaluation 兩半；threshold grid、LTT 檢定、操作點選擇**只用 calibration 半**，回報的風險與 bootstrap CI 在 held-out parity 半上計算。caption 用：*Split-specific calibration: each official split is partitioned into calibration/evaluation halves by ref_id parity; LTT selection uses only the calibration half.* **不可**寫成「在 testA/testB 上做了完全未用 label 的 held-out benchmark test」——它是 split 內的 risk-control evaluation，不是標準 train-val-test。跨 split transfer 若報，僅為 empirical stability，**不在** distribution-free LTT 保證範圍內。

> 〔framing〕這正是 GREC 官方 Pr@(F1=1) 的「機率化、有保證」版本：F1=1 要求零漏檢且零多選，
> 我們不追逐那個 0/1 事件，而是給「answered-target FNR ≤ α」的連續、有保證旋鈕。CRC 原論文 worked example 即含 bound-FNR，工具天生對齊。

---

## 4.3 單風險 CRS：Conformal Risk Control 與不可化約下限

先只控 R1。`L_FNR` 對「選更多框」單調非增 ⇒ 對閾值 `λ` 單調非減。用 **Conformal Risk Control**
（Angelopoulos et al. 2023）在 calib 的 n 個 target-present 樣本上選最大的 λ（最小集合）使

```
(n · R̂_FNR(λ) + B) / (n + 1) ≤ α,    B = 1（loss 上界）
```

**實證（OWL-ViT gRefCOCO 三 split，ref_id 奇偶分 calib/test）**：nominal α 與 empirical FNR
近乎完美對角線追蹤（val：α0.2→0.189 / 0.3→0.279 / 0.4→0.374 / 0.5→0.471；testA/B 同）。
唯一例外在 α=0.05（testB 連 0.10）——並非 conformal 失效，而是**候選池物理召回上限**：
全選候選後殘餘 FNR 仍有 val 0.06 / testA 0.10 / testB 0.14（即 GT-coverage recall ≈ 0.94 的硬上限）。

> 〔亮點 — 把限制變成貢獻〕conformal 框架**精確暴露**了 frozen base 的不可化約下限：
> 任何閾值都打不破 candidate pool 沒召回到的 GT。這是一個可量測、可報告、distribution-free 的「base 能力上限」。
> 寫作主張：CRS 不只給保證，還**診斷出保證在哪裡撞到 base 的物理極限**。

---

## 4.4 多風險 CRS：耦合風險與 Learn-then-Test

同時控 R1 + R2 + R3 是本章的技術核心，也是與既有 conformal-OD 文獻（只控單一 box coverage）的關鍵區隔。

**為何單一全域閾值不行**：R1 要「選多」、R2 要「選少」，方向相反；更糟的是兩者透過
**重疊的分數分布**耦合——OWL-ViT zero-shot 弱，target 與 no-target 樣本的 top-score 分布高度重疊。

**天真序貫校準失敗（誠實記錄）**：先用 no-target calib 標一個 abstain gate `τ_A`（壓 R2），
再在通過 gate 的 target 樣本上標 λ（壓 R1）。實證 R2 守住但 R1 爆到 0.63–0.76——因為 `τ_A` 為壓
no-target 誤選被推高，連帶擋掉大量真 target（它們分數也低），被擋的真 target 吐空集 ⇒ FNR=1。
**這個失敗是發現**：兩風險不是獨立可加，序貫校準破壞聯合有效性。

**正解 = Learn-then-Test（LTT, Angelopoulos et al. 2021）**：把 `(τ_A, λ)` 二維網格的每一格當成一個假設，
對每個 risk 做 finite-sample 檢定（Hoeffding bound 的 p-value），用 Bonferroni 控制 family-wise error ≤ δ：

```
p_R(config) = exp(−2 n_R (target_R − R̂)^2)   if R̂ < target_R else 1
config 有效 ⟺ p_R1 ≤ δ/(3|grid|)  AND  p_R2 ≤ δ/(3|grid|)  AND  p_R3 ≤ δ/(3|grid|)
```

只保留三 risk 都通過的 config ⇒ **聯合、distribution-free 保證**。Bonferroni 分母是 **3·|grid|**
（三風險 × 整個 config 網格），FWER ≤ δ 涵蓋整個搜尋空間。所有通過的 config 構成
**可行域（feasible region）**；在可行域內挑「**最小 calibration 集合大小**」的點作為操作點。
此 selection 仍 post-hoc 合法：因為 FWER 已涵蓋整個 grid，**可行域內任意挑點都不破壞保證**
（這正是 LTT 的設計目的，直接回答「min-size 選點是否 cherry-pick」的質疑）。

> 〔leakage 防守 — 必須寫白〕threshold grid 的 quantile **只用 calibration covariates** 建構；
> evaluation 樣本的 covariates 與 labels 從不參與 grid 建構、風險檢定或操作點選擇。
> （早期版本曾用 full-split 分數建 grid，雖未用到 eval label，仍會被質疑 transductive；已修正並重跑，主結果位移 ≤0.03。）

**實證（單一 base OWL-ViT，δ=0.1，30×30 calib-only grid，三風險）**：三 split 聯合保證**全部成立**，
**但操作點退化**：純 OWL-ViT 即使取可行域最小集合，仍需 ~8–9 框（val 8.23 / testA 9.25 / testB 7.12），
GroundingDINO 單 base 則因 gate 弱、需靠高棄答才守住 R2/R3。這個退化正是下一節 cross-base composition 的動機。

> 〔誠實邊界 — 必須寫白〕在弱 zero-shot 單 base 上，要同時拿 distribution-free 的三風險保證，
> 代價是大集合或高棄答。這**不是方法失敗**，是一個嚴格的數學事實：
> 保證的代價由 base 的分數可分性決定。CRS 把這個代價**明碼標出來**（三風險 + 集合大小），
> 而非藏在平均準度後面。這比 M4 的經驗觀察強一個量級——它是有限樣本、有保證、可畫成可行域圖的根本權衡刻畫。

---

## 4.5 推可行域：nonconformity 設計與更強 base

可行域退化的根因是 nonconformity 太弱（raw score）。兩個正交槓桿把可行域往**有用方向**（小集合、低棄答）推：

**(a) Per-box consistency belief（方法內元件，P1）**：以 paraphrase 下的 per-box 出現穩定度重定義
nonconformity：`belief(b) = s(b) · (1 + agreement(b))`，agreement = 該 box 在 paraphrase 下被高分框
呼應的程度。實證（OWL-ViT gRefCOCO val）：**同保證下集合縮小** α0.1→1.7% / 0.2→3.7% / 0.3→6.2% / 0.4→10.0%，
validity 不破。〔誠實：高召回區增益偏弱，當「nonconformity 設計」一節如實報，非台柱。〕

**(b) 更強 frozen base（GroundingDINO）**：在新 framing 下，base **不是被比較的對象**，而是
「**框架隨 base 能力擴張可行域**」的一個軸。同一套 LTT 保證程序，base 的分數可分性越好，
可行域裡越早出現有用操作點（低棄答 + 小集合）。〔此處與舊 M4 紅隊「別補第三 base」建議的衝突已釐清：
那建議針對舊 reliability-measurement framing 會把論文拉回 detector comparison；
在 conformal-framework framing 下，多 base 是展示「保證的可達邊界如何隨 base 移動」，角色完全不同。〕

**實證診斷（決定性）**：對 target-present 子問題，達到同一 FNR≤α 保證所需集合大小，
GroundingDINO 比 OWL-ViT 小 **~5×**（val，α=0.2：3.3 vs 18.6 框；α=0.3：2.4 vs 12.1），
且**不可化約召回下限 0.007 vs 0.062**（val；GroundingDINO 候選池幾乎涵蓋所有 GT。testA/testB floor：GD 0.012/0.017 vs OWL 0.104/0.142）。
target 平均 ~2 GT 框，GroundingDINO 在 α=0.3 只需 2.4 框 ≈「幾乎恰好選對」。主圖 `crs_money_setsize.png`。

---

## 4.5b ★核心 invention：Cross-Base Conformal Composition★

**問題的乾淨因式分解**（本身是貢獻）：CRS 的保證代價可分解為兩個**正交**的瓶頸，
分別由不同的 base 能力決定，且**沒有單一 base 兩者都強**：

| | no-target 可分性（abstain gate） | target 集合可分性（set selection） |
|---|---|---|
| OWL-ViT | **0.82**（好） | 差（α=0.3 需 12 框） |
| GroundingDINO | 0.60（差，對任何 query 都給高信心框） | **強（α=0.3 需 2.4 框），召回下限 0.007** |

OWL-ViT 擅長判斷「**該不該答**」，GroundingDINO 擅長「**答得準**」。
單一 base 的多風險 LTT 因此必然退化：OWL-ViT 集合爆炸（~43 框）、GroundingDINO 被弱 gate 拖累
（棄答 79%）。

**解法 = 兩個 frozen base 各司其職，用 LTT 聯合校準三風險**：

- **abstain gate** ← OWL-ViT 分數 `top1_score`（no-target 可分性 0.82）
- **set selection** ← GroundingDINO 候選 + 分數（召回下限 0.007、集合小）
- 兩 base 皆 frozen、皆不訓練；join key = `(ref_id, sent_id)`（共享 gRefCOCO 標註，expression/no_target 零 mismatch）。
- LTT 在 `(τ on OWL 分數, λ on GDINO 分數)` 二維 calib-only grid 上聯合校準 `(R1≤α, R2≤β, R3≤γ)`。

**亮眼操作點（三 split 全量，三風險 α=β=0.3 / γ=0.5 聯合保證，calib-only grid，min-size 選點）**：
Cross-Base Composition 於 val/testA/testB 分別輸出 **3.24 / 2.02 / 3.50 框**，三風險全部守住：
val R1=0.191/R2=0.159/defer=0.424，testA R1=0.261/R2=0.203/defer=0.443，testB R1=0.168/R2=0.236/defer=0.413。
其中 testA 的 **2.02 框 ≈ GT 平均基數**，幾乎是「恰好選對數量」；testB 也在三保證下維持 <3.6 框。
同一 protocol 下，純 OWL-ViT 需 8.23 / 9.25 / 7.12 框，composition 的集合 CI 與純 OWL-ViT 完全分離。
公平比較下的 set-size reduction 約 **2.0×–4.5×**（不使用早期 partial result 的 13× 說法）。

> 〔這是全篇護城河〕把「沒有單一 base 兩者都強」的**限制**，轉成「組合兩個 frozen base 互補強項」的
> **正面方法**。純 post-hoc、不訓練、接回 C4 cross-base 主軸、非 detector 比較（是互補組合）、
> 躲過紅隊三地雷。這是「強框架 + 亮眼正面操作點」的最終解。
>
> 〔誠實技術註〕selection rule = 在三風險可行域內取 **min calibration set size**。
> 此選擇 post-hoc 合法（FWER Bonferroni 已涵蓋整個 grid）。早期 2-risk 版曾用「最小棄答」heuristic
> 誤配最寬鬆 λ（集合爆到 45 框假退化）；升三風險後 R3 直接把 deferral 納入受控集合，問題消失。

**機制證實：2×2 gate×box ablation**（三風險 α=β=0.3 / γ=0.5，calib-only grid，min-size 選點）。
對調 gate-base 與 box-base 的四種組合：

| split | gate | box | 集合 | R1 | R2 | defer(R3) | feasible? |
|---|---|---|---:|---:|---:|---:|:--:|
| val | OWL | OWL | 8.23 | 0.239 | 0.183 | 0.366 | ✓ |
| val | GD | GD | — | — | — | — | **EMPTY** |
| val | **OWL** | **GD（COMPOSE）** | **3.24** | **0.191** | **0.159** | 0.424 | ✓ |
| val | GD | OWL（reverse） | 15.27 | 0.211 | 0.251 | 0.429 | ✓(n=2) |
| testA | OWL | OWL | 9.25 | 0.255 | 0.207 | 0.443 | ✓ |
| testA | GD | GD | — | — | — | — | **EMPTY** |
| testA | **OWL** | **GD（COMPOSE）** | **2.02** | 0.261 | 0.203 | 0.443 | ✓ |
| testA | GD | OWL（reverse） | — | — | — | — | **EMPTY** |
| testB | OWL | OWL | 7.12 | 0.234 | 0.236 | 0.413 | ✓ |
| testB | GD | GD | — | — | — | — | **EMPTY** |
| testB | **OWL** | **GD（COMPOSE）** | **3.50** | **0.168** | 0.236 | 0.413 | ✓ |
| testB | GD | OWL（reverse） | — | — | — | — | **EMPTY** |

> 〔升三風險後 ablation 比舊版更強 — 重點寫白〕一旦把 target deferral（R3）也納入受控風險，
> **pure GD（GD gate + GD box）三 split 全部 feasible region 變空**——GD gate 守 no-target 必須靠極高棄答，
> 一旦棄答受 γ=0.5 約束就無解；reverse（GD gate + OWL box）也幾乎全空（val 僅 n=2、集合爆 15）。
> 在三風險下，**只有 OWL gate + GD box 這條對角線同時 feasible 且 compact**。這正面回答
> 「CRS 只是 GroundingDINO 比較強」的質疑：GD 單獨無解，是 **factorization** 才進得了可行域。
> factorization 主張因此嚴格成立：**OWL gate 控可用 abstention/no-target；GD box 控 target set compactness**。

**統計穩固性（bootstrap CI，config 固定只 resample test）**：三 split 全量 shared keys = 14229 / 19200 / 16063。

| split | OWL-ViT only (answered TP size) | COMPOSE (answered TP size) | COMPOSE R1 | COMPOSE R2 | COMPOSE defer |
|---|---:|---:|---:|---:|---:|
| val | 8.23 [7.85, 8.57] | **3.24 [3.15, 3.36]** | 0.191 [0.179, 0.203] | 0.159 [0.146, 0.171] | 0.424 [0.405, 0.443] |
| testA | 9.25 [9.00, 9.51] | **2.02 [1.98, 2.06]** | 0.260 [0.250, 0.269] | 0.203 [0.187, 0.221] | 0.443 [0.433, 0.454] |
| testB | 7.12 [6.88, 7.35] | **3.50 [3.38, 3.63]** | 0.168 [0.159, 0.177] | 0.236 [0.219, 0.252] | 0.414 [0.401, 0.426] |

三 split 的 R1/R2 CI 上界都低於 0.3、defer 都低於 γ=0.5；COMPOSE 與純 OWL-ViT 的 size CI 完全分離。
因此結果不是 partial dump 或小樣本僥倖：**全量 val/testA/testB 皆在三保證下達到接近 GT 基數的 compact referring set**。

> 〔guarantee vs CI — 必須分清〕LTT p-value + Bonferroni = **有限樣本風險控制檢定**（這是「保證」）；
> bootstrap CI = 固定 selected config 後對 test 重抽的**經驗穩定度**（這**不是**保證）。兩者分開報、不混用。

**split-protocol robustness（堵 exchangeability）**：COMPOSE 在三種 calib/eval 切分下幾乎不動——
parity（主）、5× random ref_id split、image-disjoint（同一影像不跨 calib/eval，最嚴格）：

| split | parity（主） | random[5] 平均 set size | image-disjoint |
|---|---|---|---|
| val | 3.25 | 3.24 ± 0.04 | 3.13 |
| testA | 2.02 | 2.05 ± 0.03 | 2.06 |
| testB | 3.51 | 3.53 ± 0.11 | 3.39 |

set size 三模式幾乎重合（random std 僅 0.03–0.11），image-disjoint 亦 feasible 且 compact。
〔誠實：random split 的 R2/defer 單次 variance 較大（val R2 偶到 0.26、testB 到 0.29），
屬 LTT marginal 保證的正常表現——單次抽樣可略超名目，報 mean±std 並說明即可。〕

---

## 4.5c 〔原 4.5 收尾〕

## 4.6 跨 base 校準的樣本效率（P3，接 C4）

C4 舊結論「校準不可轉移（raw 數值刻度跨 base 近隨機）」在這裡升級成研究問題：
**新 base 要多少 calibration label，LTT 才能達到目標 (α,β) 保證？** 訊號的**結構**可轉移（C4 已證
consistency 兩 base within-AUROC 近相等），故猜測 label-efficiency 高。這把「不可轉移」的限制
轉成「**少量 label 即可重標定保證**」的正面 label-efficiency 命題。

---

## 4.7 貢獻定位（與既有文獻的護城河）

| 既有工作 | 與 CRS 的區隔 |
|---|---|
| Conformal Object Detection（box coverage、FNR；2505.24038 等） | 封閉類別、單 box-coverage；**無語言條件、無 no-target、無 cardinality**。CRS 控的是 referring set 的耦合三風險。 |
| Conformal for Zero-Shot VLM（CVPR'25） | 只做**分類** label set；非 box set。 |
| GREC / HieA2G / InstanceVG（multi-target） | **全 trained**（count head / hierarchical alignment）；CRS 是 post-hoc、distribution-free、不訓練。 |
| VLM selective prediction（ReCoVERR） | 單答案 answer/abstain；無集合層保證、無 cardinality。 |
| True-False Verification（2509.09958） | 單答案對錯 verify；CRS 是集合層聯合保證 + 計數。 |

**貢獻定位三句話（禁用 "first / 第一個"，紅隊明令）**：
(1) 對 frozen referring-grounding 做 **risk-controlled box-set selection**：不追 exact-match，而是輸出
calibrated box set，並用 LTT 對三個有界風險（answered-target FNR、no-target false selection、target deferral）
做有限樣本控制；
(2) 把風險控制的**代價因式分解**為 gate 與 box 兩個正交瓶頸，並以可行域（feasible region）刻畫；
(3) 用 **cross-base conformal composition** 組合兩個異質 frozen grounding base 的互補強項
（一個管棄答、一個管選框），在三保證下達到 ≈GT 基數的精準集合——2×2 ablation 證明這是 factorization
而非單純 detector 比較（單 base 在三風險下退化或無解）。皆躲過紅隊三地雷（非 TTA、非 detector comparison、不撞 verification）。

> 〔wording 安全清單 — 投稿前掃〕禁用：first / 第一個 / target recall(coverage) guarantee /
> unconditional recall / solves full-GREC / finite-sample conformal guarantee for recall / 13× shrink /
> hallucination risk（R2 一律 no-target false selection）。R1 一律 **answered-target FNR**。
> 「保證」只指 LTT 有限樣本檢定；bootstrap CI 一律稱 empirical stability，不稱保證。

---

## 4.8 待辦（實驗面）
- [x] GroundingDINO gref dump 三 split（val/testA/testB rows = 14229/19200/16063，HF 免編譯路徑跑通）。
- [x] 三 split compose + bootstrap CI + 2×2 ablation，確認 Cross-Base Composition 是主結果。
- [x] **紅隊 P0 修補（2026-06-15）**：calib-only grid（修 leakage）+ 三風險 LTT（R3 target deferral，Bonferroni over 3×grid）。
      headline 3.24/2.02/3.50 存活、CI 與 OWL-only 分離；ablation 升級（pure GD/reverse 三 split 全 EMPTY）。
- [x] **P1 robustness**：parity / 5×random / image-disjoint 三模式 set size 幾乎重合（random std 0.03–0.11）。
- [x] **第二輪審查 P0（2026-06-15）**：R3 deferral 改 final-empty 定義（修 gate-pass-but-empty bug，影響 ≤0.001，加 unit test）；
      set size 改名 answered TP set size + split-specific calibration caption；移除 hardcoded paths（env CRS_DUMP_DIR）。
- [x] feasible-region 圖（Pareto 四 panel，val）：`dump/crs_pareto.png`（本地 `paperwork/crs_pareto.png`）。
- [ ] 將本章改寫成英文正式稿，主表放 composition 三 split，副表放 2×2 ablation。
- [ ] 可選：Pareto 圖擴成三 split 一張大圖。
- [ ] 可選深化：P3 label-efficiency 曲線（新 base 少量 calibration label 即可重標定保證）。
- [ ] 投稿前執行 4.7 wording 安全清單全文掃。
