# Chapter: Method (草稿 v1)

> 寫作原則：論文主體用英文（學位論文慣例），中文註解標 framing/亮點機會（最終稿移除）。
> 對應 thesis framing：*Post-hoc Reliability Calibration for Frozen Referring Grounding Models*。
> 本章定義「凍結 base + 單一輕量 post-hoc belief policy」的形式化框架，所有實驗（C1–C4 + M4）皆為此框架的實例。

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
one of three actions:

- **answer** — emit the base prediction,
- **abstain** — declare no-target / refuse to answer,
- **(optionally) re-rank** — apply a grounding-specific candidate operation.

The thesis quantifies, across bases and datasets, *how reliably* `π` can be
calibrated from `g`'s by-products, *at what cost*, *how well it transfers across
bases*, and *where it provably fails* (the oracle gap / boundary).

> 〔framing〕主角是 π（policy），不是 g。所有貢獻都是「π 能從 g 的副產品學到多少可靠決策」。

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

## 3.6 Evaluation Protocol

- **Splits & leakage**: all thresholds / standardization / calibrator fits come
  from the calibration split only; test is evaluated once. Cross-base transfer
  never lets target-test statistics flow back to source training.
- **Cost**: inference cost is reported as **forward passes per query** plus
  measured throughput (GB10), not as an adjective (§5.8 Pareto).
- **Uncertainty**: all headline metrics carry **bootstrap 95% CIs** (≥1000
  resamples of the test set, calibrator fixed).
- **Oracle gap**: every selective figure includes an oracle line bounding how
  much signal remains unexploited.

> 〔寫作備忘〕這四條是審查最愛攻的點（洩漏、輕量、顯著性、上界），Method 章先把協定講死，
> Results 章就能省去反覆辯護。
