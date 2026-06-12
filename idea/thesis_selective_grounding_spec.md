---
type: spec-doc
aliases:
  - "Selective Grounding 實作規格"
  - "M0/M1 dump schema + baseline + protocol"
tags:
  - 研究主線
  - 規格
  - selective-prediction
  - referring-grounding
  - CLIP
  - belief
summary: "Selective / Uncertainty-Aware Referring Grounding 的五份可實作規格：related-work 差異表與 novelty claim、M0/M1 dump schema、C-min no-target baseline stack、B operator spec、split/calibration/threshold protocol。"
---

# Selective Grounding — 實作規格（5 份）

> 配套決策文件：[[thesis_selective_grounding_plan]]。本檔是把計畫落成可寫 code / 可寫論文的規格。
> 凡需查證但本 session 無法連網的條目，以 `〔待補〕` 標記，**不得用未經查證的引用充數**。

---

## 1. Related Work 差異表 + Final Novelty Claim

### 1.1 重要更正（VIRO 狀態）

舊計畫把 VIRO 寫成 *ICLR'26 withdrawn / unpublished*。**此描述已過期**：VIRO 已被 **CVPR 2026 接收**，必須當作**已發表競品**對待，不能再用「withdrawn」當作弱化它的理由。

- 正式引用（已查證 arXiv:2601.12781）：
  > Hyejin Park, Junhyuk Kwon, Suha Kwak, Jungseul Ok.
  > **VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension.**
  > arXiv:2601.12781 (v1 2026-01-19, v2 2026-03-20). Accepted to CVPR 2026.
- 影響：VIRO 與本研究在 *framing* 上高度重疊（abstention + no-target + verification），差異必須建立在**方法路線**而非「它沒發表」。
- VIRO 的具體賣點（查證自 abstract）：在 neuro-symbolic REC pipeline（LLM/VLM 把 query 拆成可執行 program）中，**每個 operator 內嵌輕量 verifier**（VIRO=Verification-Integrated Reasoning Operators），逐步驗證 object existence / spatial relation，解決「cascading error → high-confidence false positive」；以 **verification-aware abstention** 處理 no-target；report **61.1% balanced accuracy**（target-present + no-target 合併）、program failure rate ≤0.3%、可泛化到 egocentric 真實資料。abstract 未點名 RefCOCO/gRefCOCO 具體 split 數字。

### 1.1b 兩篇後加查證的競品（2026-06-11 GPT 紅隊第二輪補上）

- **True/False Verification REC（最危險的近鄰，威脅 B，不威脅主線）** — 已查證 arXiv:2509.09958：
  > Jeffrey Liu, Rongbin Hu. **Zero-Shot Referring Expression Comprehension via Vision-Language True/False Verification.** arXiv:2509.09958 (v1 2025-09-12, v3 2025-11-13).
  - 做法：把 REC 重構成 **box-wise visual-language verification** —— 用 COCO-clean generic detector（YOLO-World）給 proposals，再讓 general-purpose VLM 對**每個 region 獨立答 True/False**。**no REC-specific training、no fine-tuning**，支援 **abstention + multiple matches**。controlled study 主結論：**verification 顯著優於 selection-based prompting**，在 RefCOCO/+/g 超越 zero-shot GroundingDINO、甚至超越 trained GroundingDINO 與 GroundingDINO+CRG。
  - **為何危險**：它也是 zero-shot / no fine-tuning / verification / abstention，且明確證「verification > selection」。所以本研究的 **B（candidate-contrastive re-ranking）不能再賣「candidate verification 本身是新的」**。
  - **為何不致命（關鍵區辨）**：它**沒有** calibration / risk-coverage / belief policy；它沒有 gate，而是把整個 REC 換成「對每個 box 問 VLM 真假」的 workflow，且 verification 是丟給**外部 general-purpose VLM 重問**。本研究是「從 frozen base 自身副產品抽 uncertainty 訊號 → 輕量 calibrator → risk-calibrated 三態 gate」，且 B 的 verification 是**在原 base 分數空間裡做 ΔS residual**，不外掛 VLM。機制與評估語言都不同。
- **ReCoVERR（概念近、任務遠，當 selective-prediction 成熟度佐證）** — 已查證 arXiv:2402.15610：
  > Srinivasan, Hessel, Gupta, Lin, Choi, Thomason, Chandu. **Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning.** arXiv:2402.15610, ACL Findings 2024.
  - 做法：inference-time 演算法，VLM 低信心時不直接 abstain，改用 LLM 提相關問題、蒐集高信心 evidence，足夠才作答 —— 在 **VQAv2 / A-OKVQA** 上多答 20% 而不掉 accuracy。
  - **定位**：證明「selective prediction / coverage-risk 在 multimodal 已是成熟框架」，所以本研究**不可**把 risk-coverage 本身當 novelty；但它是 **VQA 不是 grounding**，非直接競品，引作 motivation 與「概念光譜」對照。

### 1.2 撰寫紀律

- 全文**禁用** "first" / "unique" / "novel framework"。改用可被審稿人核對的相對陳述：
  "we study", "we present a frozen-base, post-hoc …", "in contrast to X which Y, we Z"。
- 貢獻錨點維持三件、且都是**可被實驗證偽**的相對主張，不是 priority claim：
  1. **Frozen-base**：不更新任何 base 參數。
  2. **Post-hoc**：belief 由 base 免費/低成本副產品 + 輕量 calibrator 導出。
  3. **Risk-calibrated belief policy**：以 risk-coverage / calibration 量化的 ANSWER / ABSTAIN / INTERVENE 三態決策，並報 oracle gap。

### 1.3 差異表（取代舊表）

| 工作 | 任務 | 動 base？ | abstain / no-target | 機制路線 | 與本研究的關係 |
|---|---|---|---|---|---|
| CLIP-VG | REC | 是（curriculum fine-tune） | 否 | 把 base 練更準 | 可當本研究的 frozen base #1 |
| OWL-ViT | open-vocab detection | —（預訓練） | 否 | open-vocab detector | 可當 frozen base #2，驗 base-agnostic |
| RefFormer (NeurIPS'24) | REC | 是 | 否 | query-based grounding | 可當 frozen base #3（復現風險，列後） |
| LIHE / gRefCOCO 系 | GREC | 是 | 部分（顯式建 0/multi referent） | 專門架構 + 標籤監督 | 它把 no-target 當分類目標專訓；本研究以 post-hoc belief 在**凍結** base 上導出，可掛任意 base |
| HieA2G (AAAI'25) | GREC | 是 | 是（Adaptive Grounding Counter） | trained counter + 階層 align | trained counting head；本研究 no-target gate 不重訓、近零成本，定位為 post-hoc 對照 |
| SIMMC2.0 clarification | 對話式 grounding | — | 類似（多輪澄清） | 對話 loop + 使用者回覆 | 需互動；本研究單輪自動 abstain/消歧 |
| **VIRO** (Park et al., CVPR'26; arXiv:2601.12781) | REC + no-target | 否（凍結 LLM/VLM 組件） | 是（**verification-aware abstention**） | **LLM/VLM query→program 拆解 + operator 內嵌 verifier 逐步符號驗證** | framing 最近；**方法路線正交**：VIRO 走程式拆解 + operator-level 驗證器（驗 object existence / spatial relation）抑制 cascading error，本研究走 frozen base + 單一輕量 post-hoc calibrator，無程式合成、無 per-operator 符號驗證、base-agnostic、近零訓練。兩者可互補對照 |
| **True/False Verification REC** (Liu & Hu; arXiv:2509.09958) | REC | 否（zero-shot，no fine-tuning） | 是（abstention + multiple matches） | **YOLO-World proposals + general-purpose VLM 對每 box 獨立答 True/False** | **最近鄰、直接威脅 B**：它證「verification > selection」。但**無 calibration / risk-coverage / belief gate**，verification 外掛 VLM；本研究做 post-hoc 訊號校準 + 三態 gate，B 的 verification 在原 base 分數空間做 ΔS residual、不外掛 VLM。B 不主張 verification novelty |
| ReCoVERR (Srinivasan et al., ACL Findings'24; arXiv:2402.15610) | VQA（非 grounding） | — | 是（減少 over-abstention） | 低信心時 LLM 提問蒐 evidence 再決定答/棄 | 證 selective prediction 在 multimodal 已成熟 → risk-coverage 本身不可當 novelty；任務是 VQA 非 REC，引作 motivation/光譜對照 |
| **本研究** | REC + GREC | **否** | **是（核心）** | frozen base + post-hoc risk-calibrated belief policy | — |

> 表內每一列的「機制路線」是審稿人區辨本研究的關鍵欄，務必填實，不要只比「動不動 base」。

### 1.4 Final Novelty Claim（可貼進論文 intro / abstract）

> We study selective referring grounding under a **frozen base**: rather than retraining a grounding model to be more accurate, we attach a **post-hoc, risk-calibrated belief policy** that decides, per query, whether to **answer**, **abstain**, or run a grounding-specific **candidate-contrastive intervention**. The belief is derived from cheap by-products of any base model (candidate geometry, prompt-induced identity stability, relation/attribute residuals, cross-model agreement) plus a lightweight calibrator with no base updates. We quantify the policy on **risk–coverage** (REC) and on **no-target / generalized REC** (gRefCOCO), and we report the **oracle gap** to bound how much signal remains. In contrast to architecture-level GREC methods (LIHE, HieA2G) that train dedicated no-target/counting heads, and to neuro-symbolic pipelines that verify LLM-synthesized programs operator-by-operator (VIRO), our policy is a single base-agnostic, near-training-free calibrator.

差異一句話（防「只是加 threshold」與「跟 VIRO / True-False Verification 重複」）：

> Unlike a single confidence threshold, the policy is a learned multi-signal calibrator with a conditional, grounding-specific intervention; unlike VIRO's per-operator neuro-symbolic verification over LLM-synthesized programs, our belief comes from a **single lightweight post-hoc calibrator** on cheap base by-products, with no program decomposition or operator-level symbolic executors; and unlike box-wise True/False verification that **replaces** grounding with an external VLM's per-box queries, we **keep the frozen base** and calibrate its own uncertainty into a risk-controlled answer/abstain/re-rank policy, then quantify its transferability across bases and its oracle gap.

### 1.5 Framing 修正（2026-06-11 GPT 紅隊第二輪採納）

**核心轉變**：論文**不再賣成「我提出新的（verification-based）grounding 方法」**——那條線已被 VIRO（CVPR'26）與 True/False Verification REC（2509.09958）壓住。改賣成：

> **frozen referring grounding base 的 post-hoc reliability / calibration study**：研究「frozen zero-shot grounding base 是否暴露出可重用、grounding-specific 的不確定性結構，能否被 post-hoc 校準成可靠的 answer / abstain / re-rank 決策」，並量化其 **reliability、cost、transferability、oracle gap**，而非與 trained GREC 架構或 neuro-symbolic verification pipeline 比 accuracy。

**最安全的 thesis claim（可守版本，當前主錨）**：

> This thesis studies whether frozen zero-shot grounding models expose reusable, grounding-specific uncertainty signals that can be calibrated post hoc into reliable answer / abstain / re-rank decisions. Rather than competing with trained GREC architectures or neuro-symbolic verification pipelines, it quantifies the reliability, cost, transferability, and oracle gap of lightweight selective belief policies for REC/GREC.

**「輕量」紀律**：單純「比 VIRO 輕」**不算**貢獻。可主張的是「輕量 + risk-calibrated + cross-base transferable + oracle-gap quantified」整組。文中凡寫到效率，一律導向 §5.x 的 cost–risk–coverage Pareto，不可當形容詞單獨用。

**Thesis title 候選（暫不定案，待 M1 訊號 AUROC go/no-go 後拍板）**：
1. *Post-hoc Reliability Calibration for Frozen Referring Grounding Models*
2. *Risk-Calibrated Belief Policies for Frozen Zero-Shot Referring Grounding*
3. （保留原 *Selective / Uncertainty-Aware Referring Grounding* 作 fallback，但不作對外主標）

> ⚠️ 為何暫不定案：若 M1 發現 frozen-base uncertainty 訊號不 informative（與 BTS score bias 同源風險），題目要退到「系統性證明 CLIP grounding uncertainty 為何不可靠 + oracle 上界」，標題須再調。**不在訊號驗證前鎖死標題。**

### 1.6 最小可防守貢獻組合（4 條，GPT 紅隊第二輪採納）

主論文核心 = C1+C2+C3+C4；**B 不在核心**（成功則加分，見 §4.7）。

- **C1 — Grounding-specific uncertainty audit**：證明哪些 belief 訊號對 correctness / no-target 有資訊、哪些沒有。必備圖：AUROC(correctness)、AUROC(no-target)、calibration/ECE、failure taxonomy。（對應 §2 dump + M1 go/no-go）
- **C2 — Selective risk control**：learned/selected gate 在 RefCOCO risk-coverage 上優於 random / score-thr / margin-thr / entropy-thr，並報 oracle gap。（對應 Variant A + §5.4）
- **C3 — Post-hoc no-target gate**：gRefCOCO no-target 上輕量 gate 優於 forced-output / 各單訊號 threshold，附 oracle no-target，並用 HieA2G / VIRO / True-False Verification 做文獻定位。（對應 §3 C-min）
- **C4 — Cross-base / cost analysis**：同一 belief family 在 CLIP-VG / OWL-ViT 都 informative，**優先做 cross-base transfer**（§5.7），報 inference overhead，用 cost–risk Pareto 對照 VIRO（§5.8）。

> 對應關係：C1↔M1、C2↔Variant A、C3↔Variant C-min（M4 擴 full-GREC）、C4↔M3+成本分析。B(Variant)↔Chapter 5 上限。

> **M4 定位（紅隊裁決 2026-06-12，已落地）**：M4（full-GREC）**不是方法貢獻**，而是 C3 的延伸 **stress test / boundary analysis**，章節名 *Full-GREC Stress Test: The Boundary of Post-hoc Calibration*。結論=post-hoc calibration 能補 no-target abstention，但 multi-target exact-match（Pr@F1=1 的 T-acc）超出其能力（per-sample oracle τ 天花板僅 0.19–0.24，瓶頸=counting/set prediction 非 signal）。「ours≈conf-threshold」誠實承認不 cherry-pick。草稿見 [[chapter_m4_fullgrec_boundary]]。真正方法貢獻=C3+C4。

---

## 2. M0 / M1 Offline Dump Schema

**設計目標**：一次 dump 同時支撐 (a) RefCOCO correctness AUROC、(b) gRefCOCO no-target AUROC、(c) risk-coverage、(d) B intervention ablation、(e) cross-base transfer。

### 2.1 Granularity（每筆 row 是什麼）

一 row = **(sample, base_model, prompt_variant)** 三元組。

- `prompt_variant = "canonical"` 為主 row，承載 correctness / no-target / risk-coverage 主分析。
- 同義改寫 / discriminative / neutral prompt 各自為附 row（同 `sample_uid`、不同 `prompt_variant`），供 cross-prompt consistency 與 B operator 還原 ΔS。
- cross-base transfer：同一 `sample_uid` 跨多個 `base_model` 各一組 row；分析時以 `sample_uid` join。

存成 **JSONL（一 row 一行）** + 分區目錄 `dump/{dataset}/{base_model}/{split}.jsonl`。大張量（每候選的 per-prompt score 向量）可外掛 `.npz`，row 內存指標。

### 2.2 欄位表

#### (A) 識別與環境
| 欄位 | 型別 | 說明 |
|---|---|---|
| `sample_uid` | str | 全域唯一：`{dataset}:{split}:{image_id}:{ref_id}`，cross-base join key |
| `dataset` | str | `refcoco` / `refcoco+` / `refcocog` / `grefcoco` |
| `split` | str | `train` / `val` / `testA` / `testB`（gRefCOCO 用其官方 split 名） |
| `image_id` | int | 原始 COCO image id |
| `ref_id` | int | referring 標註 id（gRefCOCO 可為 no-target 標註 id） |
| `sent_id` | int | 句子 id（一 ref 多句時） |
| `expression` | str | referring expression 原文 |
| `image_path` | str | 相對路徑 |
| `image_w` / `image_h` | int | 影像尺寸（IoU / dispersion 正規化用） |
| `base_model` | str | `clip-vg` / `owl-vit` / `refformer` |
| `base_version` | str | checkpoint / commit，確保可復現 |
| `prompt_variant` | str | `canonical` / `paraphrase_k` / `disc_<phenotype>` / `neutral` |
| `dump_run_id` | str | dump 批次 id（追溯哪次 forward） |

#### (B) Base 輸出（候選）
| 欄位 | 型別 | 說明 |
|---|---|---|
| `candidates` | list[obj] | top-K 候選，每個 `{cand_id, bbox:[x,y,w,h], raw_score}`；K 固定（建議 K=10）並記於 meta |
| `K` | int | 實際候選數（base 可能 <K） |
| `pred_cand_id` | int | argmax 候選 id（standard 系統的輸出） |
| `pred_bbox` | list[float] | argmax bbox |
| `score_vec_ref` | str/null | 指向 `.npz` 的 key：本 row 各候選分數向量（cross-prompt 用） |

#### (C) Ground truth 與 correctness
| 欄位 | 型別 | 說明 |
|---|---|---|
| `is_no_target` | bool | gRefCOCO no-target 標註；RefCOCO 系恆 `false` |
| `num_referents` | int | GT referent 數（RefCOCO=1；gRefCOCO 可 0/1/多） |
| `gt_bboxes` | list[list] | GT 框（multi-target 為多個） |
| `gt_cand_ids` | list[int] | 候選中被 IoU≥0.5 對上的 GT 候選 id（可空=GT 不在候選內，重要負例） |
| `pred_iou` | float | `pred_bbox` 對最佳 GT 的 IoU |
| `correct@0.5` | bool | REC 正確性主標籤（IoU≥0.5 且非 no-target） |
| `best_cand_iou` | float | 全候選對 GT 的最佳 IoU（量「框集合是否含對的」，oracle 上界用） |

#### (D) Belief / uncertainty 訊號（M1 informativeness 篩選的核心）
| 欄位 | 型別 | 成本 | 說明 |
|---|---|---|---|
| `top1_score` | float | 免費 | argmax 絕對分 |
| `margin12` | float | 免費 | top1 − top2 |
| `score_entropy` | float | 免費 | softmax(scores) 的熵 |
| `spatial_dispersion` | float | 免費 | top-m 框中心的幾何離散度（除以 √(w·h) 正規化） |
| `cross_prompt_consistency` | float | 中 | 各 paraphrase row 的 top-1 一致比例 / 平均 IoU（由 paraphrase rows 聚合回 canonical） |
| `identity_stability_entropy` | float | 中 | 跨 prompt「哪個候選被選中」分布的熵 |
| `cross_model_agreement` | float/null | 中高 | 與另一 base top-1 的 IoU / 是否一致（cross-base join 後填） |
| `relation_residual` | float/null | 中 | ΔS(relation discriminative − neutral)；**RefCOCO+ 必為 null** |
| `attribute_residual` | float/null | 中 | ΔS(attribute discriminative − neutral)；RefCOCO+ 主用此項 |

> M1 go/no-go：對 (C) 的 `correct@0.5`（REC）與 `is_no_target`（gRefCOCO）分別算每訊號 AUROC。

#### (E) B intervention ablation 專用
| 欄位 | 型別 | 說明 |
|---|---|---|
| `phenotype` | str/null | `candidate_competition` / `spatial_ambiguity` / `semantic_absence` / `model_disagreement` / `none`（規則見 §4） |
| `disc_prompt` | str/null | 本 row（disc variant）所用 discriminative prompt 原文 |
| `neutral_prompt` | str/null | 對應 neutral prompt 原文 |
| `delta_scores` | list[float]/null | 各候選 ΔS = S(disc) − S(neutral) |
| `reranked_cand_id` | int/null | INTERVENE 後重排 top-1 |
| `reranked_iou` | float/null | 重排後 top-1 對 GT 的 IoU |
| `intervened` | bool | 本 sample 是否實際觸發 INTERVENE |

> ablation：比較 `pred_iou`(argmax) vs `reranked_iou`，在 `intervened=true` 子集上算 intervention lift；並記觸發率與平均額外 forward 次數。

#### (F) Protocol bookkeeping
| 欄位 | 型別 | 說明 |
|---|---|---|
| `calib_fold` | str | `calib_train` / `calib_val` / `test`（§5 切分，固定種子預先指派） |
| `seed` | int | 影響 fold / paraphrase 抽樣的種子 |

### 2.3 Meta（每個 dump 目錄一份 `meta.json`）
記錄：K、paraphrase 數、prompt 模板版本、base checkpoint hash、IoU 對應門檻（0.5）、softmax 溫度、dump_run_id、產生時間（由呼叫端傳入，腳本內不取系統時間以利重現）。

---

## 3. gRefCOCO C-min No-Target Gate — 正式 Baseline Stack

**任務**：判斷「圖中是否存在 referent」。輸出 abstain（no-target）或 grounding。**不能只比 forced-output base**。

### 3.1 Baseline 階梯

| # | Baseline | 規則 | 角色 |
|---|---|---|---|
| 0 | **Forced-output base** | 永遠輸出 argmax，從不 abstain | 下界（證 base 無 abstain 能力） |
| 1 | **Random abstain** | 以基準 abstain 率隨機棄答 | risk-coverage 最低參考線 |
| 2 | **Score-threshold** | `top1_score < τ_s` → no-target | 最常見稻草人 |
| 3 | **Margin-threshold** | `margin12 < τ_m` → no-target | 競爭型訊號對照 |
| 4 | **Entropy-threshold** | `score_entropy > τ_e` → no-target | 分布型訊號對照 |
| 5 | **Learned P(no-target)** | logistic / 小 MLP 吃 (D) 全訊號 → `P(no-target) > τ_p` | **本研究主方法** |
| 6 | **Oracle no-target gate** | 用 GT `is_no_target` 完美決定 | 上界（量 signal gap） |

- 2–4 是**單訊號** baseline；5 是**多訊號 learned** 方法；6 是上界。2–6 的所有 τ 一律在 calib_val 選定（§5），禁止 test 調參。
- **每條都掛在同一組 frozen base 上**（先 CLIP-VG，再 OWL-ViT 驗 base-agnostic），確保差異來自 gate 不是 base。

### 3.2 Related-work positioning（baseline ≠ 只有 threshold）

- **HieA2G (Wang et al., AAAI'25; arXiv:2501.01416)**：trained Adaptive Grounding Counter，屬「重訓 counting head」路線。列為**對照組（trained upper reference）**，誠實標註它有監督訓練、本研究 #5 無；比的是「post-hoc 近零成本能逼近多少」。各 split N-acc 已查證（見 §3.5）。
- **VIRO (Park et al., CVPR'26)**：neuro-symbolic verification 的 abstention（verification-aware abstention）。列為**對照路線**，標註其 LLM/VLM program 拆解 + per-operator verifier 的推論成本。**注意：VIRO 本身也凍結 base 組件**，所以「動不動 base」**不是**對 VIRO 的區辨點 —— 區辨點是「單一 post-hoc calibrator」vs「程式拆解 + 多個 operator-level 符號驗證器」。VIRO 各 split no-target 細數已查證（見 §3.4），可同表並列。
- **True/False Verification REC (Liu & Hu; arXiv:2509.09958)**：zero-shot box-wise VLM verification，支援 abstention/multiple matches，且證「verification > selection」。列為**對照路線**，標註其 verification 外掛 general-purpose VLM、無 calibration / risk-coverage。**主要影響在 B 不在 C**：C-min 的 baseline 階梯不受它威脅（它沒有 no-target gate 的 calibration 對照），但 B（§4）的 framing 必須因它退讓（見 §4.7）。若能取得其 no-target / abstention 數字可在 §3.5 後補表並列（其底是 YOLO-World+VLM，與本研究 CLIP-VG/OWL-ViT base 不同，需註明）。
- 三者都**不是**用來「打贏」，而是定位本研究在 trained-architecture ↔ post-hoc 光譜上的位置（HieA2G 全監督專訓 / VIRO 凍結 base+重型 neuro-symbolic / True-False 凍結 base+外掛 VLM verification / 本研究 凍結 base+輕量 post-hoc calibrator）。

### 3.3 指標
- **No-target AUROC**（gate 分數 vs `is_no_target`）：M1 第二 go/no-go。
- **GREC 官方**：N-acc、T-acc、Pr@(F1=1, IoU≥0.5)。
- **Risk-coverage**：把 abstain 決策當 selective prediction，畫 RC 曲線、AURC。
- 全部報 bootstrap CI（§5）。

### 3.4 VIRO 實測數字（已查證自全文 PDF，arXiv:2601.12781 v2）

> 評測協定：VIRO 用 **Balanced Accuracy = (TPR + TNR)/2** 同時量 no-target robustness 與 standard REC。
> - **TNR = N-acc**（no-target accuracy，在 **gRefCOCO no-target** 樣本上算 true-negative rate）。
> - **TPR = Acc@0.5**（在 **RefCOCO** target-present 樣本上算 IoU≥0.5 命中率）。
> - VIRO 把 gRefCOCO 的 no-target 樣本和 RefCOCO 的 target-present 樣本**混成一個平衡測試集**，按 TestA / TestB split 報。

**Table 2（no-target robustness + standard REC，TestA / TestB）**，TNR(gRef) 即各方法的 no-target accuracy：

| 類別 | 方法 | TestA Bal. | TestA TNR(gRef) | TestA TPR(Ref) | TestB Bal. | TestB TNR(gRef) | TestB TPR(Ref) |
|---|---|---|---|---|---|---|---|
| Fully-sup REC | Qwen2.5-VL-72B-AWQ✝ | 69.5 | 47.3 | 91.7 | 66.8 | 45.1 | 88.4 |
| Fully-sup (GREC) | GREC-MDETR-R101 | 62.0 | 34.5 | 89.6 | 56.2 | 31.0 | 81.4 |
| Fully-sup (GREC) | GREC-UNINEXT-R50 | 70.4 | 49.3 | 91.5 | 67.6 | 48.2 | 86.9 |
| Proposal-based | ReCLIP | 23.5 | **0.0** | 47.0 | 22.6 | **0.0** | 45.2 |
| Proposal-based | SS-CLIP | 33.3 | **0.0** | 66.5 | 27.5 | **0.0** | 54.9 |
| Proposal-based | GroundVLP | 30.7 | **0.0** | 61.3 | 21.8 | **0.0** | 43.5 |
| Detector-based | GLIP-L | 37.2 | 21.7 | 52.6 | 30.0 | 18.2 | 41.8 |
| Detector-based | GroundingDINO-T | 40.0 | 22.8 | 57.2 | 29.6 | 16.0 | 43.2 |
| Compositional | ViperGPT | 33.4 | 0.2 | 66.7 | 27.4 | 0.1 | 54.6 |
| Compositional | HYDRA | 35.2 | 7.5 | 62.8 | 34.7 | 7.0 | 62.4 |
| Compositional | NAVER | 33.8 | 3.4 | 64.2 | 30.0 | 1.8 | 58.2 |
| **Neuro-symbolic** | **VIRO (Ours)** | **61.1** | **50.2** | 71.9 | **56.9** | **52.9** | 60.8 |

✝ VIRO 附帶說明：Qwen2.5-VL 需額外加 negative instruction（"If there is no object, return []"）才有 no-target 能力；不加則 TNR 掉到 3.1%（TPR 94.7%）on TestA。

**Table 3（standard REC accuracy + 效率，Qwen2.5-72B-AWQ；RefCOCO/+/g）**，TPR=Acc@0.5，FR=failure rate%，Exc./Inc.=排除/含 program 失敗：

| 方法 | RefCOCO Exc./Inc. | RefCOCO+ Exc./Inc. | RefCOCOg Exc./Inc. | FR(g) |
|---|---|---|---|---|
| Qwen2.5-VL-72B-AWQ | 94.4 / 94.3 | 91.8 / 91.6 | 89.1 / 89.0 | 0.11 |
| GLIP-L | 52.6 / 52.6 | 48.6 / 48.6 | 52.6 / 52.6 | 0.00 |
| GroundingDINO-T | 57.2 / 57.2 | 57.6 / 57.6 | 59.5 / 59.5 | 0.00 |
| ViperGPT | 66.7 / 64.4 | 61.7 / 57.5 | 65.7 / 61.7 | 6.03 |
| HYDRA | 62.8 / 44.9 | 58.4 / 37.4 | 67.1 / 45.4 | 32.37 |
| NAVER | 64.2 / 60.3 | 60.1 / 55.6 | 68.4 / 55.8 | 9.74 |
| **VIRO (Ours)** | **71.9 / 71.9** | **63.3 / 63.3** | **66.6 / 66.3** | 0.30 |

**對本研究極有用的兩點證據（直接寫進 motivation / 差異論證）**：

1. **Proposal-based REC（ReCLIP / SS-CLIP / GroundVLP）的 no-target TNR = 0.0** —— VIRO 親自證實「forced-prediction 系統在 no-target 完全失能」。這正是本研究 §3.1 baseline #0（forced-output base）會被打爆的鐵證，可直接引用支撐「base 沒有 abstain 能力 → selective gate 有結構性價值」。
2. **VIRO 的 no-target 能力來自重型 pipeline**（LLM program + per-operator verifier，E2E 12.92 query/s vs GroundingDINO 0.20）。本研究主打「**單一 post-hoc calibrator 在凍結 base 上、近零成本**逼近多少 no-target accuracy」，對照軸天然成立：VIRO 是 heavy neuro-symbolic 上界參考，本研究是 light post-hoc。**注意 VIRO 用的是 GroundingDINO / Qwen-VL 當底，不是 CLIP-VG/OWL-ViT**，所以直接同表需註明 base 不同；公平比法是各自報「相對 forced-output base 的 N-acc 增益」。

### 3.5 HieA2G 實測數字（已查證自全文 PDF，arXiv:2501.01416）

> Wang et al., AAAI 2025。指標 = GREC 官方 **Pr@(F1=1, IoU≥0.5)** 與 **N-acc**（no-target accuracy）。HieA2G = ResNet101 backbone，在 RefCOCO/+/g + Flickr30K + gRefCOCO 合併預訓練後 finetune（**全監督、含 gRefCOCO no-target 標籤訓練**）。

**Table 1（gRefCOCO GREC，val / testA / testB）**：

| 方法 | val Pr | val N-acc | testA Pr | testA N-acc | testB Pr | testB N-acc |
|---|---|---|---|---|---|---|
| MCN✝ | 28.0 | 30.6 | 32.3 | 32.0 | 26.8 | 30.3 |
| VLT✝ | 36.6 | 35.2 | 40.2 | 34.1 | 30.2 | 32.5 |
| MDETR✝ | 42.7 | 36.3 | 50.0 | 34.5 | 36.5 | 31.0 |
| UNINEXT✝ | 58.2 | 50.6 | 46.4 | 49.3 | 42.9 | 48.2 |
| Ferret✱ | 54.8 | 48.9 | 49.5 | 45.2 | 43.5 | 43.8 |
| **HieA2G-R101** | **67.8** | **60.3** | **66.0** | **60.1** | **56.5** | **56.0** |

✝ = 被改造成可輸出多框（依 GREC, He et al. 2023）；✱ = MLLM (Ferret) 改造成 GREC。

**對本研究的定位用法**：

1. **HieA2G N-acc ≈ 56–60% 是「全監督 trained counting head」路線的 SOTA 參考線**。本研究 #5（learned P(no-target)，凍結 base + 輕量 calibrator）報 N-acc 時，HieA2G 是 trained upper reference，誠實說明「我們近零成本，差多少」而非聲稱打贏。
2. **可比性註記**：HieA2G 的 N-acc 算在完整 GREC 設定（no/single/multi-target 混合，五類 counting），與 §3.1 C-min 只做 no-target 二元 gate **設定不同**；同表需標明 HieA2G 是 full-GREC、本研究 C-min 是 no-target subset。要嚴格對齊需在 C 擴到 full-GREC（計畫 M4）後才公平並列。
3. **VIRO vs HieA2G vs 本研究 三點定位**：HieA2G = 全監督專訓架構；VIRO = 凍結 base + 重型 neuro-symbolic program 驗證；本研究 = 凍結 base + 單一輕量 post-hoc calibrator。三者構成「訓練成本 ↓、推論成本 ↓」光譜，本研究佔最輕量端 —— 這正是 base-agnostic + near-training-free 貢獻的座標。

---

## 4. B Operator 實作規格 — Belief-conditioned Candidate-Contrastive Re-ranking

> 紅隊修正：INTERVENE **不可**是「低 confidence 時多跑 prompt」（會被打成普通 TTA / prompt ensemble）。必須是 grounding-specific candidate-contrastive operator。

### 4.1 觸發條件（gate 進入 INTERVENE）
僅在 calibrator `p_correct` 落入「中段」且 **margin 小** 時觸發（高 p→ANSWER；全候選低分→ABSTAIN）。觸發後先做 phenotype 診斷。

### 4.2 Ambiguity Phenotype 判斷（四型，互斥，依序判定）

| Phenotype | 判定條件（皆用 calib_val 校過的門檻） | 直覺 |
|---|---|---|
| `semantic_absence` | 全候選 `top1_score` 低 且 `score_entropy` 高 | 圖裡可能根本沒有 → 偏 ABSTAIN/no-target |
| `model_disagreement` | `cross_model_agreement` 低（兩 base top-1 IoU 小） | 兩 base 不同意 → 需 contrastive 裁決 |
| `candidate_competition` | `margin12` 小 且 top-2 候選 IoU 互不重疊 | 兩個不同物在競爭 → attribute/relation 裁決 |
| `spatial_ambiguity` | `margin12` 小 且 top-m 框空間離散 `spatial_dispersion` 高 | 位置/關係詞決勝（**RefCOCO+ 不適用**） |
| `none` | 以上皆否 | 不介入，回 ANSWER |

> 判定順序：absence → disagreement → competition → spatial。落空才往下。phenotype 寫入 dump (E) `phenotype`。

### 4.3 Discriminative prompt 生成
從 expression 解析出**區辨屬性**，對 top-m 候選各生成一條 discriminative prompt：

- **competition（屬性差異）**：抽 expression 中的 attribute token（顏色/材質/類別形容詞），生成
  `"the {attribute} {noun}"`，逐候選打分。
- **spatial_ambiguity（關係）**（**僅 RefCOCO / RefCOCOg**）：抽空間/關係詞（left/on/behind…），生成
  `"the {noun} {relation} {anchor}"`，用候選對 anchor 的幾何關係驗證。
- **RefCOCO+**：資料集**禁止位置詞** → 一律走 **attribute-contrastive**，不得生成關係 prompt（否則違反資料集設定、訊號失真）。phenotype 即使判為 spatial 也降級為 competition 處理。
- **semantic_absence**：不生成 discriminative prompt，直接導向 ABSTAIN / no-target gate。

### 4.4 Neutral prompt 定義
同結構、抽掉區辨資訊的 base prompt，用來扣掉 base 的 per-object score prior（即 BTS 的 score bias 來源）：

- competition / spatial 對應的 neutral = `"the {noun}"`（只留名詞，去掉 attribute / relation）。
- neutral 對 top-m 同樣逐候選打分，得 `S(neutral)`。

### 4.5 Residual normalization（核心，防 score bias 復發）
對每候選 c：

```
ΔS(c) = S_disc(c) − S_neutral(c)
```

- 重排依 `ΔS` 而非絕對分 → 扣除「某候選天生分高」的 prior。
- 若多條 discriminative prompt（多屬性），取各條 ΔS 後 z-score 標準化再加總。
- INTERVENE 輸出 `reranked_cand_id = argmax_c ΔS(c)`，寫入 dump (E)。

### 4.6 退場條件
若 B 在 `intervened` 子集的 lift 不顯著（bootstrap CI 跨 0），依計畫降為 analysis/ablation，A+C 撐主論文。此判定也在 calib_val 上先看趨勢，test 只報最終數字。

### 4.7 B 的 framing 紀律（因 True/False Verification REC, arXiv:2509.09958）

2509.09958 已證「box-wise verification > selection-based prompting」並支援 abstention。因此 B **不得**主張以下任一：

- ❌「candidate verification / 對候選做區辨性驗證」是新的；
- ❌「verification 能修正 grounding 錯誤」是本研究發現；
- ❌ 把 B 當論文主貢獻去和 2509.09958 比 accuracy。

B 可主張且可守的範圍：

- ✅ B 是 **selective intervention**：只在 gate 觸發的歧義子集啟動，量化「在近 1× 推論成本下，post-hoc rescue 能救回多少 / 何時反傷（harm analysis）」；
- ✅ B 的 verification 在**原 frozen base 自己的分數空間**做 ΔS residual，**不外掛 general-purpose VLM**，與 2509.09958 的 workflow 機制不同（成本與依賴都更小）；
- ✅ B 的價值放在 **cost–risk 的 Pareto 位置**（§5.7），而非絕對 accuracy。

**定位**：B 是「成功則加分」的 Chapter 5 上限實驗。主論文核心 = **A（selective risk control）+ C（no-target calibration）+ calibration/oracle-gap + cross-base transfer**。B 效果普通時不硬推為主貢獻（與 §4.6 退場一致）。

---

## 5. Split / Calibration / Threshold Protocol

**鐵律：任何 τ、calibrator 參數、phenotype 門檻一律在 train/val 決定，test 只跑一次出最終數字。**

### 5.1 切分
- **RefCOCO/+/g**：用官方 train / val / testA / testB。
  - `calib_train` = 官方 **train**（訓 calibrator）。
  - `calib_val` = 官方 **val**（選所有 τ 與門檻、early stop、phenotype 校準）。
  - `test` = **testA / testB**（最終報告，分開報）。RefCOCOg 用 val/test。
- **gRefCOCO**：用其官方 train/val/test；no-target gate 的 τ_p 在 gRefCOCO val 選。
- dump 時即把每 row 的 `calib_fold` 依官方 split 固定指派（固定 seed），避免事後洩漏。

### 5.2 Calibrator 訓練
- 模型：logistic regression 或 ≤2 層小 MLP（數十～數百參數，**不碰 base**）。
- 特徵：dump (D) 全訊號；缺值（如 RefCOCO+ 的 relation_residual=null）以 mask + 指示位處理，不可用 0 混淆。
- **特徵分兩類（cross-base transfer 的關鍵設計，見 §5.7）**：
  - **base-specific（跨 base 不穩）**：raw `top1_score`、raw `margin12`、raw `score_entropy` —— 絕對分數尺度因 base 而異。
  - **base-normalized / grounding-structural（跨 base 可望可重用）**：rank-normalized margin、`identity_stability_entropy`(prompt 擾動)、`spatial_dispersion`、candidate-set entropy、`cross_model_agreement`、candidate recall upper bound、relation/attribute residual 的**rank**。
  - cross-base transfer 主實驗**只用 base-normalized 特徵**訓 calibrator；base-specific 特徵僅在 within-base 設定用，並單獨報「加了它們 transfer 掉多少」做 ablation。
- 標籤：REC 用 `correct@0.5`；no-target gate 用 `is_no_target`。
- 標準化參數（mean/std）只用 `calib_train` 統計，套用到 val/test，**禁止用 test 統計**。
- 多 seed（≥5）重訓，報平均 ± CI。

### 5.3 Threshold 選擇
- 所有 τ（score/margin/entropy/p_correct/p_no_target）在 `calib_val` 上依目標準則選：
  - selective：固定 target risk r*，選滿足 risk≤r* 的最大 coverage 之 τ（Coverage@target-risk）。
  - 或固定 target coverage，回報該點 risk。
- 兩種準則都**先在 val 定 τ，再凍結套到 test**。論文明確寫出準則與 r*/coverage 值。

### 5.4 報告指標
- **Risk–Coverage curve** + **AURC / E-AURC**（招牌圖，selective vs naive vs random vs oracle）。
- **AUROC**（correctness / no-target）。
- **ECE + reliability diagram**（calibrator 校準度）。
- **AUSE**（uncertainty 排序品質）。
- **Selective accuracy @ fixed coverage**、**Coverage @ target risk**。
- **GREC**：N-acc / T-acc / Pr@(F1=1, IoU≥0.5)。
- **Oracle gap**：每張主圖附 oracle 線，量「訊號還差多少」。

### 5.5 統計顯著性
- **Bootstrap CI**：對 test 樣本 resample（≥1000 次）求 AURC / AUROC / accuracy 的 95% CI。
- **多 seed**：calibrator / paraphrase 抽樣的隨機性以 ≥5 seed 平均，報 seed 間 std。
- 跨 base / 跨 baseline 的比較給 CI 重疊判讀；B 的 lift 以 `intervened` 子集 bootstrap CI 是否跨 0 作退場依據（§4.6）。

### 5.6 防洩漏 checklist（寫進論文 appendix）
- [ ] 所有 τ 來自 val，test 單次評估。
- [ ] 標準化統計來自 calib_train。
- [ ] paraphrase / discriminative prompt 模板在看 test 前凍結。
- [ ] phenotype 門檻在 val 校準。
- [ ] cross-base join 不洩漏 GT（cross_model_agreement 只用預測框 IoU，不用 GT）。
- [ ] cross-base transfer 的 calibrator 只用 base-normalized 特徵（§5.2），target base 的 test 統計不回流到 source base 訓練。

### 5.7 Cross-base transfer（**升格為主結果**，GPT 紅隊第二輪採納）

> 判準：**只做「多 base 各訓各測」不算 base-agnostic（會被打成 incremental）；必須做 train-on-A / test-on-B 的 transfer，才算「grounding uncertainty 有可重用結構」的科學主張。** 這也是對 VIRO / True-False Verification 唯一站得住的真區辨——它們都是 per-pipeline / per-VLM，無 transfer 主張。

**主表（必做，非加分）**：同一 belief policy（只用 base-normalized 特徵）在不同 frozen base 間轉移：

| train gate on | test on | AUROC(correct) | AURC | N-acc | ECE |
|---|---|---|---|---|---|
| CLIP-VG | CLIP-VG | — | — | — | — |
| OWL-ViT | OWL-ViT | — | — | — | — |
| CLIP-VG | OWL-ViT | — | — | — | — |
| OWL-ViT | CLIP-VG | — | — | — | — |
| （RefFormer 第三 base，可選加碼） | | | | | |

**結論的兩種寫法（依數據誠實選一）**：
- transfer 成立（即使 drop 一些）→ 主張「**grounding failure 有可觀察、跨 base 可重用的結構**」（強貢獻，撐 thesis novelty）。
- transfer 失敗 → 退為「policy is plug-in across bases, but **calibration remains base-dependent**」（誠實但弱），並把重心移回 within-base 的 A+C+oracle-gap。
- 對角線（within-base）vs 非對角線（cross-base）的差距本身就是一個科學結果（量「uncertainty 結構有多少是 base-共享 vs base-特有」）。

### 5.8 Cost–Risk–Coverage Pareto（讓「輕量」變成實驗主張）

> 「輕量」不可當形容詞。以一張 Pareto 表/圖把訓練成本、推論成本、reliability 一起報，回答「**在接近 1× 推論成本下，能吃到多少 VIRO/HieA2G 那類重型方法的 no-target / reliability benefit？**」

| 方法 | 訓練成本 | 推論成本(相對) | Risk / N-acc | Coverage | 備註 |
|---|---|---|---|---|---|
| forced-output base | 0 | 1× | 高風險 | 100% | 無 abstain |
| score threshold | 0 | 1× | 改善有限 | 下降 | naive |
| **learned belief policy（本研究主方法）** | 小 | 1×～1.x× | 改善 | 可控 | A + C-min |
| **+ candidate-contrastive B** | 小 | 1.x×～k× | 視 lift | 視觸發率 | selective intervention，只在歧義子集 |
| VIRO（文獻參考） | 0 base update / heavy pipeline | 高（E2E 12.92 q/s） | balanced 61.1 | — | program + per-operator verifier |
| HieA2G（文獻參考） | 高（全監督專訓） | 中 | N-acc 56–60 | — | trained counting head |

- 推論成本以 §2.3 dump 記的「平均額外 forward 次數 / query/s」量化；B 另報觸發率。
- VIRO/HieA2G 的成本欄是文獻定性對照（base 不同，不直接比絕對 accuracy，見 §3.2/§3.4/§3.5）。

---

## 待補清單
- ✅ VIRO 引用已查證並接入（arXiv:2601.12781 v2, CVPR'26, Park et al., POSTECH）。
- ✅ VIRO no-target / standard REC 各 split 細數已查證並接入（§3.4 Table 2/3，全文 PDF）。
- ✅ 查證副產品：VIRO 也評 **RefAdv**（adversarial OOD，Appendix A.6.4）與 **RefEgo**（video egocentric，含 no-target）；本研究若要加 OOD/egocentric robustness 章節可引這兩個 benchmark。
- ✅ HieA2G no-target 官方數字已查證並接入（§3.5 Table 1，arXiv:2501.01416，gRefCOCO val/testA/testB N-acc 56–60%）。
- ✅ 兩篇後加競品已查證接入（§1.1b/§1.3）：True/False Verification REC (arXiv:2509.09958)、ReCoVERR (arXiv:2402.15610)。
- `〔待補〕` True/False Verification REC 的 no-target/abstention 細數（abstract 只給定性，若要同表並列需翻全文 PDF；base 是 YOLO-World+VLM，與本研究不同需註明）。
- K（候選數）、paraphrase 數、softmax 溫度 → M0 跑通後回填 §2.3 meta。
