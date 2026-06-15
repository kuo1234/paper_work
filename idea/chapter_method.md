# Chapter: Method (草稿 v2 — CRS 升級後)

> 寫作原則：論文主體用英文（學位論文慣例），中文註解標 framing/亮點機會（最終稿移除）。
> 對應 thesis framing（**2026-06-13 升級**）：從 *Post-hoc Reliability Calibration* 升級為
> *Post-hoc **Risk-Controlled** Referring Sets for Frozen Grounding Models*——
> C1–C4 是「frozen base 的 belief 訊號有多 informative / 可轉移」的 **measurement** 地基；
> M4 證明 point prediction 在 full-GREC exact-match 撞牆；CRS（主台柱）把框架從「測量可靠度」
> 推到「**建構有分布無關保證的 referring set**」。本章定義統一框架，所有實驗（C1–C4 + M4 + CRS）皆為其實例。

---

## 3.1 Problem Formulation

We study **selective referring grounding under a frozen base**. A referring
grounding model `g` maps an image `I` and an expression `e` to one or more
bounding boxes. In the classic REC setting `g` returns a single box; in the
generalized setting (GREC) it must return a set of boxes whose cardinality may
be zero (no-target), one, or many (multi-target).

We treat `g` as a **frozen black box**: its parameters are never updated. Given
`g`, we attach a lightweight **belief policy** `π` that observes cheap
by-products of a single (or a few) forward passes of `g` and decides, per query,
one of the following actions:

- **answer** — emit the base prediction,
- **abstain** — declare no-target / refuse to answer,
- **(optionally) re-rank** — apply a grounding-specific candidate operation,
- **construct a risk-controlled set** — instead of a point prediction, return a
  *calibrated set* of candidate boxes (possibly empty) whose risks are controlled
  with finite-sample guarantees (the CRS chapter, §4).

The thesis quantifies, across bases and datasets, *how reliably* `π` can be
calibrated from `g`'s by-products, *at what cost*, *how well it transfers across
bases*, *where point prediction provably fails* (the oracle gap / boundary, M4),
and *what guarantees a set-valued `π` can provide where point prediction breaks*
(CRS).

> 〔framing〕主角是 π（policy），不是 g。所有貢獻都是「π 能從 g 的副產品學到多少可靠決策」。
> CRS 升級後，π 的輸出從「點」擴成「有保證的集合」——這是把 measurement 推到 construction 的關鍵一步。

---

## 3.2 Frozen Bases

We instantiate the framework on two structurally different frozen bases, chosen
to stress-test whether belief signals are **base-agnostic**:

- **CLIP-VG** (single-box regression). Outputs exactly one box via
  `bbox_embed(reg_token).sigmoid()`; **no candidate list, no scores**. This
  forces us to derive belief from *perturbation* signals rather than score
  statistics — which turns out to favour cross-base transfer (§3.4).
- **OWL-ViT** (open-vocabulary detector). Outputs many candidate boxes with
  scores as a free by-product of one forward pass; enables score-based belief
  and **native abstention** (low max-score ⇒ no-target).

> 〔亮點機會〕兩個 base 的「訊號家族」完全不同（perturbation vs score），卻都 informative
> 且其中一類可跨 base 轉移——這是 C4 的科學主張，是目前最像「發現」的結果。寫作要把它推到最強。

---

## 3.3 Belief Signals

All signals are **post-hoc** (no base update) and fall into two families by cost
and transferability:

**(A) Base-agnostic perturbation signals** (cost K× forward, K paraphrases):
- `cross_prompt_consistency`: mean pairwise IoU of the predicted boxes across K
  paraphrase prompts. Low consistency ⇒ the target identity is unstable under
  benign rewording ⇒ higher error risk. This is *grounding-specific*: it
  measures referential stability, not generic confidence.
- `prompt_box_dispersion`: spread of predicted box centres across paraphrases
  (normalized by √area).

**(B) Base-specific score signals** (cost 1× forward, free by-products; OWL-ViT
only):
- `top1_score`, `margin12` (= top1 − top2), `score_entropy` (entropy of the
  softmaxed candidate scores), `score_mean_topk`.

> 〔framing〕(A) 貴但可轉移、(B) 免費但 base-specific——這個「成本↔可轉移性」的二分法本身是 contribution，
> 直接撐起 cost-Pareto（§5.8）與 C4。不要把訊號當成一堆 feature 平鋪，要講成這個結構。

The calibrator is a logistic regression on standardized signals, fit on a held-
out calibration split; the predicted probability is the confidence/abstain score
that ranks samples for selective prediction. We deliberately keep the calibrator
minimal — the claim is about *signal informativeness*, not calibrator capacity.

---

## 3.4 Selective Prediction & Cross-base Transfer

**Selective prediction (C2).** Given confidence `c(x)` from `π`, answer the
top-`coverage` fraction of samples and abstain on the rest. We report the
**risk–coverage curve** and its area (AURC; lower is better), against random,
single-signal thresholds, and an oracle that ranks by true correctness.

**Cross-base transfer (C4).** The central scientific test: fit `π` on a *source*
base using only **base-normalized** signals, then apply it to a *target* base
**without any refitting**. If the policy transfers, grounding failure has a
*reusable, base-shared structure* — a claim no per-pipeline verification method
(VIRO, True/False) can make. Transfer is measured by the target-base AURC of the
transferred policy vs the target-native policy vs random.

> 〔亮點機會 — 這是全篇最該放大的一節〕
> C4 的乾淨版主張：raw consistency 零參數、無 refit，從 CLIP-VG 套到 OWL-ViT，AURC ≈ native。
> 寫作時把它從「一個 transfer 實驗」升級成「grounding uncertainty 有 base-invariant 結構」這個 thesis-level 主張。
> 這比 C1/C2 的 audit 更像「發現」，是對抗「只是 calibration/threshold」批評的真正護城河。

---

## 3.5 No-target Gate & Generalized REC

**No-target gate (C3).** A frozen base that always emits an argmax box has, by
construction, **zero abstention ability** (N-acc = 0 on no-target queries). On
gRefCOCO we calibrate `π` to output `P(no-target)` from base by-products and
threshold it into an abstain decision. Metrics: no-target AUROC, balanced
accuracy, N-acc — all leak-safe (thresholds fixed on calib).

**Generalized REC / boundary (M4).** Extending `π` to full GREC requires
emitting the *correct set* of boxes, evaluated by the official
**Pr@(F1=1, IoU≥0.5) / N-acc / T-acc** (faithful greedy-IoU matching). This is a
deliberate **stress test**, not a method claim: it locates where post-hoc
calibration stops working (multi-target exact-match is a counting / set-
prediction problem beyond a single confidence threshold). We quantify the
boundary with a per-sample oracle-τ ceiling and bootstrap CIs.

> 〔framing〕M4 在 Method 章就要先聲明是 boundary analysis，否則 Results 讀起來像失敗。
> 「我們刻意把方法推到斷裂點，並量化斷在哪」是成熟度，不是減分。

---

## 3.5b Risk-Controlled Referring Sets (CRS) — 從點預測到有保證的集合

M4 的牆（multi-target exact-match 超出單一信心閾值的 action space）促成框架的最後一步：
讓 `π` 不再輸出**點預測**，而是輸出一個 **risk-controlled referring set** `S(I,e) ⊆ {candidate boxes}`，
基數可為 0（棄答 / no-target）、1、或多。我們以 **Learn-then-Test (LTT)** 在 calibration split 上
聯合校準三個有界風險並給出有限樣本保證：

- **R1 answered-target FNR** ≤ α（在未棄答的 target-present 上的漏檢率；**條件**風險，不稱 recall guarantee）
- **R2 no-target false selection** ≤ β（no-target 樣本吐出任何框的比例）
- **R3 target deferral** ≤ γ（target-present 樣本被吐空集的比例）

進一步地，CRS 把風險控制的代價**因式分解**為 gate（該不該答）與 box（答得準）兩個正交瓶頸，
並用 **cross-base conformal composition**（一個 frozen base 當 gate、另一個當 box selector）
在三保證下達到 ≈GT 基數的 compact set。完整方法、可行域與結果見 **CRS 章（§4）**。

> 〔framing〕這一節在 Method 章只需「宣告框架擴張 + 三風險定義 + 指向 §4」。
> 它把 C1–C4 的 measurement 與 M4 的 boundary 縫進同一條線：訊號可測（C1–C4）→ 點預測撞牆（M4）
> → 換成有保證的集合預測（CRS）。Method 章不放數字（數字在 §4），避免重複。

---

## 3.6 Evaluation Protocol

- **Splits & leakage**: all thresholds / standardization / calibrator fits come
  from the calibration split only; test is evaluated once. Cross-base transfer
  never lets target-test statistics flow back to source training.
- **Cost**: inference cost is reported as **forward passes per query** plus
  measured throughput (GB10), not as an adjective (§5.8 Pareto).
- **Uncertainty**: all headline metrics carry **bootstrap 95% CIs** (≥1000
  resamples of the test set, calibrator fixed).
- **Guarantee vs CI（CRS 專用，必須分清）**: for CRS, the **LTT p-value +
  Bonferroni** test is the *finite-sample risk-control guarantee*; the bootstrap
  CI on a fixed selected config is only *empirical stability*, **not** a
  guarantee. The two are reported separately and never conflated. CRS threshold
  grids are built on **calibration covariates only** (no evaluation covariate or
  label enters grid construction, risk testing, or operating-point selection).
- **Oracle gap**: every selective figure includes an oracle line bounding how
  much signal remains unexploited.

> 〔寫作備忘〕這五條是審查最愛攻的點（洩漏、輕量、顯著性、上界、guarantee≠CI），Method 章先把協定講死，
> Results 章就能省去反覆辯護。
