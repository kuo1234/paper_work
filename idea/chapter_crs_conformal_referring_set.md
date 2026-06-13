# Chapter: Conformal Referring Sets (CRS) — 主方法章草稿 v1

> 寫作原則：論文主體英文，中文〔註〕標 framing/亮點/誠實邊界（最終稿移除）。
> 這是 **2026-06-13 主軸再升級**後的新台柱章。延續 [[chapter_method]] 的「凍結 base + post-hoc policy」框架，
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

**目標**：在 calibration split 上選 `λ̂`，使得在 unseen test 上

- **(R1) 召回保證**：target-present 上 `E[L_FNR] ≤ α`
- **(R2) 棄答保證**：no-target 上 `E[L_NT] ≤ β`

兩者皆 **distribution-free、finite-sample**。α、β 是使用者旋鈕（safety budget）。

> 〔framing〕這正是 GREC 官方 Pr@(F1=1) 的「機率化、有保證」版本：F1=1 要求零漏檢且零多選，
> 我們不追逐那個 0/1 事件，而是給「漏檢率 ≤ α」的連續、有保證旋鈕。CRC 原論文 worked example 即含 bound-FNR，工具天生對齊。

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

同時控 R1 + R2 是本章的技術核心，也是與既有 conformal-OD 文獻（只控單一 box coverage）的關鍵區隔。

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
config 有效 ⟺ p_R1 ≤ δ/|grid|  AND  p_R2 ≤ δ/|grid|
```

只保留兩 risk 都通過的 config ⇒ **聯合、distribution-free 保證**。所有通過的 config 構成
**可行域（feasible region）**；在可行域內挑「最小 target 棄答率」的點作為操作點（post-hoc 選擇仍合法，
因為全可行域都滿足保證）。

**實證（OWL-ViT，δ=0.1，25×25 grid）**：三 split 聯合保證**全部成立**（R1、R2 都達標），
**但操作點退化**：代價是棄答 75–80% target 樣本 + 回答時吐 ~45 框（近乎全選）。

> 〔誠實邊界 — 必須寫白〕在弱 zero-shot base 上，要同時拿 distribution-free 的召回+棄答保證，
> LTT 只能靠近退化策略達標。這**不是方法失敗**，是一個嚴格的數學事實：
> 保證的代價由 base 的分數可分性決定。CRS 把這個代價**明碼標出來**（棄答率、集合大小），
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
GroundingDINO 比 OWL-ViT 小 **~5.4×**（α=0.2：3.46 vs 18.62 框；α=0.3：2.25 vs 12.11），
且**不可化約召回下限 0.004 vs 0.062**（GroundingDINO 候選池幾乎涵蓋所有 GT）。
target 平均 ~2 GT 框，GroundingDINO 在 α=0.3 只需 2.25 框 ≈「幾乎恰好選對」。主圖 `crs_money_setsize.png`。

---

## 4.5b ★核心 invention：Cross-Base Conformal Composition★

**問題的乾淨因式分解**（本身是貢獻）：CRS 的保證代價可分解為兩個**正交**的瓶頸，
分別由不同的 base 能力決定，且**沒有單一 base 兩者都強**：

| | no-target 可分性（abstain gate） | target 集合可分性（set selection） |
|---|---|---|
| OWL-ViT | **0.82**（好） | 差（α=0.3 需 12 框） |
| GroundingDINO | 0.60（差，對任何 query 都給高信心框） | **強（α=0.3 需 2.25 框），召回下限 0.004** |

OWL-ViT 擅長判斷「**該不該答**」，GroundingDINO 擅長「**答得準**」。
單一 base 的多風險 LTT 因此必然退化：OWL-ViT 集合爆炸（~43 框）、GroundingDINO 被弱 gate 拖累
（棄答 79%）。

**解法 = 兩個 frozen base 各司其職，用 LTT 聯合校準雙保證**：

- **abstain gate** ← OWL-ViT 分數 `top1_score`（no-target 可分性 0.82）
- **set selection** ← GroundingDINO 候選 + 分數（召回下限 0.004、集合小）
- 兩 base 皆 frozen、皆不訓練；join key = `(ref_id, sent_id)`（共享 gRefCOCO 標註，expression/no_target 零 mismatch）。
- LTT 在 `(τ on OWL 分數, λ on GDINO 分數)` 二維 grid 上聯合校準 `(R1≤α, R2≤β)`。

**亮眼操作點（三 split 全量，α=0.3/β=0.3 聯合保證）**：在 Pareto frontier 上，
Cross-Base Composition 於 val/testA/testB 分別輸出 **3.21 / 2.02 / 3.48 框**，R1/R2 全部守住：
val R1=0.193/R2=0.134，testA R1=0.257/R2=0.182，testB R1=0.169/R2=0.235。
其中 testA 的 **2.02 框 ≈ GT 平均基數**，幾乎是「恰好選對數量」；testB 也在雙保證下維持 <3.6 框。
同一 Pareto protocol 下，純 OWL-ViT 需 8.16 / 9.14 / 7.07 框，composition 的集合 CI 與純 OWL-ViT 完全分離。

> 〔這是全篇護城河〕把「沒有單一 base 兩者都強」的**限制**，轉成「組合兩個 frozen base 互補強項」的
> **正面方法**。純 post-hoc、不訓練、接回 C4 cross-base 主軸、非 detector 比較（是互補組合）、
> 躲過紅隊三地雷。這是「強框架 + 亮眼正面操作點」的最終解。
>
> 〔誠實技術註〕Pareto 選點是關鍵：LTT 可行域常有上百個 valid configs，
> 若只取「最小棄答」會誤配最寬鬆 λ（集合爆到 45 框，假退化）；正確做法是取
> (棄答, 集合) 雙目標 Pareto frontier。這是 selection criterion，不影響保證有效性
> （全 frontier 都滿足保證）。

**機制證實：2×2 gate×box ablation**（α=β=0.3）。對調 gate-base 與 box-base 的四種組合：

| split | gate | box | 集合 | R1 | R2 | 棄答 |
|---|---|---|---:|---:|---:|---:|
| val | OWL | OWL | 8.15 | 0.238 | 0.177 | 0.376 |
| val | GD | GD | 3.22 | 0.207 | 0.215 | 0.795 |
| val | **OWL** | **GD（COMPOSE）** | **3.21** | **0.193** | **0.134** | 0.485 |
| val | GD | OWL（reverse） | 14.82 | 0.214 | 0.236 | 0.466 |
| testA | OWL | OWL | 9.15 | 0.258 | 0.208 | 0.443 |
| testA | **OWL** | **GD（COMPOSE）** | **2.02** | 0.257 | 0.183 | 0.481 |
| testB | OWL | OWL | 7.09 | 0.235 | 0.235 | 0.417 |
| testB | **OWL** | **GD（COMPOSE）** | **3.48** | **0.169** | 0.235 | 0.417 |

val 是最乾淨的 2×2：COMPOSE 同時拿到 GD box 的小集合與 OWL gate 的低 R2，reverse 落到大集合+高 R2。
testA/testB 則暴露另一個重要現象：**GD gate 可以靠 extreme over-abstention 讓表面 risk 變低**
（GD+GD 棄答 0.938/0.873；testB reverse 棄答 0.904），但這不是可用操作點。
在可用棄答區間（約 0.4–0.5）內，OWL gate + GD box 仍是唯一穩定組合：集合壓到 2–3.5 框，且 R1/R2 全部守 α=β=0.3。
因此 factorization 主張更精確地成立：**OWL gate 控可用 abstention/no-target trade-off；GD box 控 target set compactness**。

**統計穩固性（bootstrap CI，config 固定只 resample test）**：三 split 全量 shared keys = 14229 / 19200 / 16063。

| split | OWL-ViT only size | COMPOSE size | COMPOSE R1 | COMPOSE R2 | COMPOSE abst |
|---|---:|---:|---:|---:|---:|
| val | 8.16 [7.78, 8.52] | **3.21 [3.10, 3.32]** | 0.193 [0.180, 0.207] | 0.134 [0.123, 0.146] | 0.485 [0.467, 0.503] |
| testA | 9.14 [8.89, 9.40] | **2.02 [1.97, 2.06]** | 0.257 [0.246, 0.267] | 0.182 [0.167, 0.199] | 0.481 [0.471, 0.492] |
| testB | 7.07 [6.83, 7.30] | **3.48 [3.35, 3.60]** | 0.169 [0.160, 0.178] | 0.235 [0.218, 0.251] | 0.417 [0.405, 0.429] |

三 split 的 R1/R2 CI 上界都低於 0.3；COMPOSE 與純 OWL-ViT 的 size CI 完全分離。
因此結果不是 partial dump 或小樣本僥倖：**全量 val/testA/testB 皆在雙保證下達到接近 GT 基數的 compact referring set**。

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
| Conformal Object Detection（box coverage、FNR；2505.24038 等） | 封閉類別、單 box-coverage；**無語言條件、無 no-target、無 cardinality**。CRS 控的是 referring set 的耦合雙風險。 |
| Conformal for Zero-Shot VLM（CVPR'25） | 只做**分類** label set；非 box set。 |
| GREC / HieA2G / InstanceVG（multi-target） | **全 trained**（count head / hierarchical alignment）；CRS 是 post-hoc、distribution-free、不訓練。 |
| VLM selective prediction（ReCoVERR） | 單答案 answer/abstain；無集合層保證、無 cardinality。 |
| True-False Verification（2509.09958） | 單答案對錯 verify；CRS 是集合層聯合保證 + 計數。 |

**護城河三句話**：(1) 第一個對 frozen referring-grounding 做 risk-controlled box-set selection 的工作；
(2) 第一個同時控召回+棄答耦合雙風險（LTT），並給出可行域；(3) 第一個用 **cross-base conformal composition**
組合兩個異質 frozen grounding base 的互補強項（一個管棄答、一個管選框），在雙保證下達到 ≈GT 基數的精準集合。
皆躲過紅隊三地雷（非 TTA、非 detector comparison、不撞 verification）。

---

## 4.8 待辦（實驗面）
- [x] GroundingDINO gref dump 三 split（val/testA/testB rows = 14229/19200/16063，HF 免編譯路徑跑通）。
- [x] 三 split compose + bootstrap CI + 2×2 ablation，確認 Cross-Base Composition 是主結果。
- [ ] 將本章改寫成英文正式稿，主表放 composition 三 split，副表放 2×2 ablation。
- [ ] 補 cost-Pareto / feasible-region 圖，呈現「保證代價」而非只給單點。
- [ ] 可選深化：P3 label-efficiency 曲線（新 base 少量 calibration label 即可重標定保證）。
