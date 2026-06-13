# 紅隊 briefing — CRS / Cross-Base Conformal Composition 主結果裁決

> 日期：2026-06-13  
> 專案：Selective Grounding 碩論主線  
> 本文件目的：請紅隊/外部顧問**對抗式審視**目前 CRS 主結果，判斷 formal validity、claim wording、實驗 protocol、圖表與還需要補的實驗。  
> 請不要附和；若你認為主張站不住，請直接指出 fatal flaw 與最小修補方案。

---

## 0. 最想請你裁決的重點（先看這裡）

目前主結果已跑完，看起來已足以把 thesis 從「reliability measurement」升級成「risk-controlled referring set construction」。但我最擔心以下 P0 問題：

1. **R1 目前是 conditional FNR（只在 answered target samples 上算），target abstention 只是 cost，不是 formal risk。這樣能不能主張「召回保證」？還是必須補第三風險 `target abstention ≤ γ`？**
2. **目前 operating point 是在 LTT-valid configs 中挑 `calib target-abstention ≤ 0.5` 下 set size 最小者。這個 selection rule 是否夠乾淨？要怎麼寫才不被說 cherry-picking？**
3. **目前腳本的 threshold grid quantiles 用了整個 split 的 scores（含 evaluation half 的 unlabelled scores），loss/p-value/selection 只用 calib。這是否構成 leakage / transductive calibration？是否必須改成 calib-only quantiles 重跑？**
4. **官方 val/testA/testB 各自再用 `ref_id` 奇偶切 calibration/evaluation。這對 split-conformal / LTT 的 exchangeability 是否可接受？是否需要 image-disjoint 或多 random split robustness？**
5. **Cross-base composition 是否足以算方法級貢獻，而不是「拿 OWL 當 gate、GDINO 當 detector 的簡單 ensemble / detector comparison」？該如何 framing？**

如果只能補一個實驗，我目前猜是：

> **三風險 LTT：R1≤0.3, R2≤0.3, R3(target abstention)≤0.5**，外加 calib-only grid rerun。

請紅隊判斷這是不是必要，或是否有更小但更關鍵的修補。

---

## 1. 背景：原主線為何不夠，為何轉 CRS

### 1.1 原 selective grounding 主線

原本 thesis 是：

> 對 frozen referring grounding base 做 post-hoc reliability / calibration study，而不是訓練新的 grounding model，也不跟 fully-trained GREC architecture 比 SOTA accuracy。

已完成的舊貢獻包括：

- **C1/C2：uncertainty signal 有效**  
  CLIP-VG / OWL-ViT 的 score entropy、cross-prompt consistency 等訊號能預測 grounding failure。
- **C3：gRefCOCO no-target gate**  
  OWL-ViT top score 對 no-target 有 AUROC 約 0.74–0.82。
- **C4：cross-base signal transfer**  
  raw consistency zero-shot transfer 在 CLIP-VG → OWL-ViT 上仍接近 native。
- **M4：full-GREC diagnostic**  
  post-hoc policy 可大幅改善 no-target abstention，但 multi-target exact-match T-acc 天花板很低，瓶頸是 FP / cardinality，不是 recall。

### 1.2 為何這不夠

紅隊與使用者判斷：C1–C4 + M4 比較像「measurement / diagnosis」，缺少方法級正面貢獻。尤其 M4 顯示：

- frozen base 的 candidate pool recall 不差；
- 但 exact F1=1 要求「所有 GT 命中且沒有多餘框」；
- 單一 threshold 無法每樣本選對 cardinality；
- post-hoc exact-match 幾乎無法救。

舊解讀是「死路」。新解讀是：

> 如果 exact-match point prediction 做不到，就改成輸出有 finite-sample guarantee 的 **box set**，把「做不到精確點估計」翻成「可控制風險與集合大小的 referring set construction」。

這就是 CRS。

---

## 2. 新主軸：CRS (Conformal Referring Set)

### 2.1 問題設定

資料：gRefCOCO / GREC style referring expression。

每個 query `(image, expression)` 可能是：

- **no-target**：沒有對應物件；
- **target-present**：有一個或多個 GT boxes；實務上 gRefCOCO target-present 很多是 multi-target。

Frozen base model 對 query 輸出候選框與分數：

```text
{(b_i, s_i)}_{i=1}^K
```

CRS policy 輸出一個 set：

```text
S(I, e) ⊆ candidate boxes
```

可為：

- 空集合：abstain / no-target；
- 單框；
- 多框。

### 2.2 要控制的風險

對 target-present 樣本，定義 GT coverage / FNR：

```text
FNR(S, G) = 1 - |{g ∈ G : ∃ b ∈ S, IoU(b, g) ≥ 0.5}| / |G|
```

對 no-target 樣本，定義 false selection：

```text
FP_NT(S) = 1[|S| > 0]
```

目前 composition 實驗實際控制的是：

- **R1：answered-target conditional FNR**  
  只在 target-present 且 gate 通過、真的輸出非 abstain 決策的樣本上計算 FNR。
- **R2：no-target false selection rate**  
  no-target 樣本上輸出非空集合的比例。
- **A：target abstention rate**  
  target-present 樣本被 gate 擋掉的比例。目前它是明碼 reported cost / selection constraint，不是 LTT formal risk。
- **Size：answered target samples 的平均 set size**。

> 這裡是最大紅隊點：R1 不是 unconditional target risk。若 target abstention 也應算 failure，必須改定義或補第三風險。

---

## 3. Cross-Base Conformal Composition：目前方法架構

### 3.1 核心洞察：兩個瓶頸由不同 base 控制

CRS 的保證代價分成兩個正交瓶頸：

| 瓶頸 | 需要什麼能力 | OWL-ViT | GroundingDINO |
|---|---|---:|---:|
| no-target / abstain gate | 判斷「該不該答」 | 較強，no-target AUROC 約 0.82 | 較弱，對很多 no-target 也給高信心框，AUROC 約 0.60 |
| target set selection | 在 target-present 樣本中選 compact boxes | 較弱，FNR≤0.3 需約 12 框（val） | 強，FNR≤0.3 只需約 2–3 框 |

因此沒有單一 base 同時強：

- OWL-ViT 適合做 **gate**；
- GroundingDINO 適合做 **box set**。

### 3.2 Policy 形式

目前 Cross-Base Composition policy：

```text
if OWL_top1_score ≤ τ:
    S = ∅
else:
    S = {GroundingDINO box b_i : GDINO_score_i ≥ λ}
```

也就是：

- **abstain gate**：OWL-ViT `top1_score`
- **set selection**：GroundingDINO candidate boxes + scores
- 兩個 base 都 frozen，不訓練，不 finetune，不合成 box。
- join key：`(ref_id, sent_id)`；OWL-ViT dump 與 GroundingDINO dump 在同一 gRefCOCO expression 上對齊。

### 3.3 Calibration / LTT protocol（目前腳本狀態）

主要腳本：

- 本機：`idea/crs_src/_compose_threesplit.py`
- spark：`~/selective-grounding/src/compose_threesplit.py`

目前 protocol：

1. 對每個官方 split `val / testA / testB` 各自讀 OWL-ViT 與 GroundingDINO dump。
2. 只取 canonical prompt variant。
3. 取兩 base 都有的 shared keys。
4. 在每個 official split 內用 `ref_id` parity 切：
   - even `ref_id` → calibration
   - odd `ref_id` → evaluation/test
5. 建 threshold grid：
   - `τ`：OWL gate score 的 quantiles，30 points，0 到 0.95
   - `λ`：box score 的 quantiles，30 points，0 到 0.99
   - total grid size `m = 30 × 30 = 900`
6. 對每個 `(τ, λ)`，在 calibration set 上算：
   - R1c = answered-target conditional FNR
   - R2c = no-target false selection
   - calib target abstention
   - calib answered set size
7. 用 Learn-then-Test / Hoeffding p-value + Bonferroni：

```text
p_R(config) = exp(-2 n_R (target_R - R_hat)^2), if R_hat < target_R else 1
valid iff p_R1 ≤ δ/m AND p_R2 ≤ δ/m
δ = 0.1
α = 0.3
β = 0.3
```

8. 在 valid configs 中挑 operating point：

```text
bright = {valid configs with calib target-abstention ≤ 0.5} if nonempty else all valid configs
choose config with minimum calib set size in bright
```

9. 固定 selected `(τ, λ)`，在 held-out evaluation half 上報：
   - R1
   - R2
   - target abstention
   - set size
10. Bootstrap CI：固定 config，只 resample evaluation half 500 次，報 median 與 95% CI。

### 3.4 需要紅隊注意的 protocol 細節

目前腳本有兩個可能被打的點：

#### A. threshold grid quantiles 用了整個 split 的 scores

`compose_threesplit.py` 中：

```python
sc_all = np.concatenate([r[sck] for r in recs])
lams = np.quantile(sc_all, np.linspace(0, 0.99, 30))
gpool = np.array([r[gate] for r in recs])
taus = np.quantile(gpool, np.linspace(0, 0.95, 30))
```

`recs` 是整個 split，包含 calibration 與 evaluation half。loss/p-value/selection 只用 calib，但 candidate threshold values 來自全 split 的 unlabelled scores。

請裁決：

- 這是否構成 leakage？
- 若是，是否只要改成 calib-only quantiles 重跑即可？
- 如果用 unlabelled test covariates 建 grid，在 conformal / LTT 下是否可視為 transductive but label-free，還是仍不建議？

#### B. target abstention 不是 formal risk

目前 `abst≤0.5` 是 selection constraint，而且只在 calib 上使用；test 上只是報出來。它不是 LTT p-value 控制的風險。

請裁決：

- 是否必須補 R3 = target abstention rate ≤ γ？
- γ 應設 0.5 還是更保守？
- 若補 R3，主結果是否仍大概率成立？目前 test abst：val 0.485 / testA 0.481 / testB 0.417，看起來 γ=0.5 有機會。

---

## 4. 實驗結果完整整理

### 4.1 Target-present subproblem：GDINO box set 明顯更 compact

這是先拆掉 no-target gate，只看 target-present 上，要達到 FNR≤α 所需 set size。

#### val detailed result

| α | OWL-ViT FNR | OWL-ViT size | GroundingDINO FNR | GroundingDINO size | shrink |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.094 | 29.7 | 0.106 | 5.6 | 約 5.3× |
| 0.20 | 0.189 | 18.6 | 0.199 | 3.3 | 約 5.6× |
| 0.30 | 0.285 | 12.1 | 0.287 | 2.4 | 約 5.1× |
| 0.40 | 0.375 | 8.1 | 0.347 | 1.9 | 約 4.3× |

Candidate-pool irreducible FNR：

| split | OWL-ViT floor | GroundingDINO floor |
|---|---:|---:|
| val | 0.062 | 0.007 |
| testA | 0.104 | 0.012 |
| testB | 0.142 | 0.017 |

Interpretation：GroundingDINO 幾乎涵蓋所有 GT boxes，所以在 target-present 子問題上，保證代價顯著低於 OWL-ViT。

#### 三 split target-present table

| base | split | floor | α=0.1 FNR / size | α=0.2 FNR / size | α=0.3 FNR / size |
|---|---|---:|---:|---:|---:|
| OWL-ViT | val | 0.062 | 0.094 / 29.7 | 0.189 / 18.6 | 0.285 / 12.1 |
| OWL-ViT | testA | 0.104 | 0.107 / 33.9 | 0.205 / 20.6 | 0.302 / 13.4 |
| OWL-ViT | testB | 0.142 | 0.142 / 33.9 | 0.183 / 27.3 | 0.296 / 15.8 |
| GroundingDINO | val | 0.007 | 0.106 / 5.6 | 0.199 / 3.3 | 0.287 / 2.4 |
| GroundingDINO | testA | 0.012 | 0.111 / 4.6 | 0.203 / 2.5 | 0.288 / 1.7 |
| GroundingDINO | testB | 0.017 | 0.090 / 6.9 | 0.184 / 3.5 | 0.269 / 2.2 |

---

### 4.2 單一 base multi-risk LTT 會退化

這是關鍵診斷：如果只用單一 base 同時做 gate + box selection，會出現退化操作點。

`RESULT_full_table.txt` 中的 best operating point（α=0.3, β=0.2）：

| base | split | R1 | R2 | target abstention | set size | # feasible |
|---|---|---:|---:|---:|---:|---:|
| OWL-ViT | val | 0.025 | 0.166 | 0.407 | 42.7 | 186 |
| OWL-ViT | testA | 0.060 | 0.146 | 0.567 | 43.9 | 203 |
| OWL-ViT | testB | 0.026 | 0.147 | 0.592 | 45.3 | 191 |
| GroundingDINO | val | 0.010 | 0.145 | 0.893 | 38.7 | 47 |
| GroundingDINO | testA | 0.003 | 0.161 | 0.770 | 23.3 | 125 |
| GroundingDINO | testB | 0.001 | 0.147 | 0.820 | 20.6 | 100 |

Interpretation：

- OWL-ViT 的 gate 可用，但 box set 爆大（40+ boxes）。
- GroundingDINO 的 boxes 強，但 gate 弱，靠高度 abstention 才守住 no-target。 
- 因此需要 cross-base factorization。

---

### 4.3 Cross-Base Composition 主結果（三 split 全量）

主要結果：`RESULT_compose_threesplit.txt`

Shared rows：

| split | shared keys | target-present | no-target |
|---|---:|---:|---:|
| val | 14229 | 5324 | 8905 |
| testA | 19200 | 14752 | 4448 |
| testB | 16063 | 11390 | 4673 |

Composition protocol：

- gate = OWL-ViT top1 score
- boxes/scores = GroundingDINO
- α=β=0.3
- LTT-valid configs + `calib target-abstention≤0.5` + minimum calib set size
- 500× bootstrap CI on evaluation half

| split | method | set size | R1 | R2 | target abstention |
|---|---|---:|---:|---:|---:|
| val | OWL-ViT only | 8.16 [7.78, 8.52] | 0.237 [0.223, 0.253] | 0.177 [0.164, 0.190] | 0.377 [0.359, 0.393] |
| val | **COMPOSE OWL gate + GD box** | **3.21 [3.10, 3.32]** | **0.193 [0.180, 0.207]** | **0.134 [0.123, 0.146]** | 0.485 [0.467, 0.503] |
| testA | OWL-ViT only | 9.14 [8.89, 9.40] | 0.257 [0.246, 0.269] | 0.207 [0.191, 0.225] | 0.443 [0.433, 0.454] |
| testA | **COMPOSE OWL gate + GD box** | **2.02 [1.97, 2.06]** | **0.257 [0.246, 0.267]** | **0.182 [0.167, 0.199]** | 0.481 [0.471, 0.492] |
| testB | OWL-ViT only | 7.07 [6.83, 7.30] | 0.235 [0.223, 0.246] | 0.235 [0.218, 0.251] | 0.417 [0.405, 0.429] |
| testB | **COMPOSE OWL gate + GD box** | **3.48 [3.35, 3.60]** | **0.169 [0.160, 0.178]** | 0.235 [0.218, 0.251] | 0.417 [0.405, 0.429] |

Interpretation：

- 三 split R1/R2 都低於 0.3。
- COMPOSE set size 顯著小於 OWL-ViT only，CI 完全分離。
- testA 最亮：2.02 boxes，幾乎等於 GT 平均基數。
- testB 仍成立：3.48 boxes，R1=0.169, R2=0.235。
- 但 COMPOSE target abstention 接近 0.5；這是必須明講的 cost。

主張應該謹慎寫成：

> Under the same LTT/Pareto protocol, cross-base composition reduces set size from 7–9 boxes to 2–3.5 boxes while maintaining both conditional FNR and no-target false-selection guarantees, at an explicit target-abstention cost of about 0.42–0.49.

不應把早期 partial 的「13× shrink」當最終主 claim；最終公平比較大約是 2.0×–4.5×。

---

### 4.4 2×2 gate × box ablation

主要結果：`RESULT_compose_ablation.txt`

目的：確認 improvement 不是「單純 GroundingDINO 比較強」，而是 gate 與 box 能力互補。

| split | gate | box | set size | R1 | R2 | target abstention |
|---|---|---|---:|---:|---:|---:|
| val | OWL | OWL | 8.15 | 0.238 | 0.177 | 0.376 |
| val | GD | GD | 3.22 | 0.207 | 0.215 | 0.795 |
| val | **OWL** | **GD (COMPOSE)** | **3.21** | **0.193** | **0.134** | 0.485 |
| val | GD | OWL (reverse) | 14.82 | 0.214 | 0.236 | 0.466 |
| testA | OWL | OWL | 9.15 | 0.258 | 0.208 | 0.443 |
| testA | GD | GD | 1.00 | 0.003 | 0.019 | 0.938 |
| testA | **OWL** | **GD (COMPOSE)** | **2.02** | 0.257 | 0.183 | 0.481 |
| testA | GD | OWL (reverse) | 12.10 | 0.254 | 0.258 | 0.493 |
| testB | OWL | OWL | 7.09 | 0.235 | 0.235 | 0.417 |
| testB | GD | GD | 1.00 | 0.040 | 0.091 | 0.873 |
| testB | **OWL** | **GD (COMPOSE)** | **3.48** | **0.169** | 0.235 | 0.417 |
| testB | GD | OWL (reverse) | 7.95 | 0.213 | 0.032 | 0.904 |

Interpretation：

- **val 是最乾淨的 factorization**：COMPOSE 同時拿到 GD box 的小集合與 OWL gate 的低 R2；reverse 又大又差。
- **testA/testB 顯示 GD gate 的風險**：GD gate 可以靠 extreme target abstention 讓表面 R1/R2 很低（例如 testA GD+GD abst=0.938，size=1.00），但這不是可用操作點。
- 因此更精確的主張是：

```text
OWL gate controls usable abstention / no-target trade-off.
GroundingDINO boxes control target-set compactness.
The useful region is not simply lowest R1/R2; it is low set size under bounded target abstention.
```

這也是為何可能需要 formal R3 或 Pareto curve。

---

## 5. 目前暫定 thesis claim

### 5.1 強 claim 版本（可能太強，請紅隊審）

> We introduce Conformal Referring Sets (CRS), the first post-hoc framework for risk-controlled box-set prediction in frozen referring grounding models. CRS controls target coverage and no-target false selection with finite-sample guarantees. We further propose Cross-Base Conformal Composition, which decomposes the guarantee cost into abstention and box-selection bottlenecks and composes two frozen bases: OWL-ViT for abstention and GroundingDINO for box selection. On gRefCOCO, this produces compact referring sets of 2–3.5 boxes across val/testA/testB while maintaining both risk guarantees.

### 5.2 較保守版本

> We study risk-controlled referring set prediction for frozen grounding bases. Rather than forcing an exact single/multi-box answer, CRS returns a calibrated set with explicit finite-sample risk control. We show that single-base policies expose a fundamental trade-off between no-target abstention and target-set compactness. A simple cross-base composition — OWL-ViT gate plus GroundingDINO box selection — achieves compact sets under the same risk-control protocol, while making target abstention an explicit cost.

### 5.3 請紅隊裁決

- 哪個版本比較安全？
- 「first」能不能說？還是只能說「to our knowledge」？
- 是否能說 finite-sample guarantees？若 R1 是 conditional，該怎麼精準措辭？
- 是否應避免「target coverage guarantee」而改成「conditional target-set FNR guarantee」？

---

## 6. 已知攻擊面與我的初步防守

### 6.1 攻擊：你只是 abstain 掉困難 target，所以 R1 看起來好

現況：這個攻擊部分成立。target abstention 目前是 cost，不是 formal guarantee。

初步防守：

- 明確報 target abstention 0.42–0.49；
- selection rule 限制 calib abst≤0.5；
- 補三風險 LTT，正式控制 `target abstention≤γ`。

請紅隊判斷：補 R3 是否必需？如果必需，γ=0.5 是否合理？

### 6.2 攻擊：grid 用 evaluation scores，有 leakage

現況：threshold grid quantiles 來自整個 split 的 scores，但 label/risk selection 只用 calibration。

初步防守：最好不要防守，直接改成 calib-only grid 重跑。因為這應該成本低，能消除疑慮。

請紅隊判斷：這是 fatal 還是 minor？是否必須重跑？

### 6.3 攻擊：val/testA/testB 各自切 calib/test，不是真正 held-out official test

現況：每個 official split 都被當成 exchangeable dataset，再 parity split 做 calibration/evaluation。這是 conformal 常見做法，但要講清楚不是 standard train→test benchmark。

初步防守：

- thesis 不是 SOTA benchmark，而是 post-hoc risk control study；
- formal guarantee 對每個 exchangeable split 內部成立；
- 可補 random split seeds 或 image-disjoint split。

請紅隊判斷：是否需要 image-disjoint 或 train/val calibration → testA/testB evaluation？

### 6.4 攻擊：這只是 detector comparison / two-model ensemble

現況：確實使用兩個 base；但方法 claim 不是「GDINO 比 OWL 準」，而是「風險瓶頸可分解，gate 與 box 由不同 base 控制，LTT 聯合校準」。

初步防守：

- 2×2 ablation 顯示 reverse 差、GD gate over-abstain；
- single-base LTT 都退化；
- composition 是 decision-layer conformal composition，不訓練、不改 detector。

請紅隊判斷：這個防守夠不夠？需要什麼圖或表讓它更像 method 而不是 ensemble？

### 6.5 攻擊：R1/FNR 不是 GREC 官方 Pr@(F1=1)

現況：CRS 不追求 exact F1=1；它把問題改成 risk-controlled set prediction。

初步防守：

- M4 已證 exact F1=1 對 frozen post-hoc threshold 幾乎不可能；
- CRS 提供可調風險集合，這是不同目標；
- 文中不可與 trained HieA2G/InstanceVG 直接比 Pr@(F1=1)。

請紅隊判斷：這種 metric pivot 是否合理？是否會被認為 dodge original task？

### 6.6 攻擊：bootstrap CI 被誤當保證

現況：formal guarantee 來自 LTT p-value；bootstrap CI 只是 empirical stability。

初步防守：寫作時分清楚：

- LTT：finite-sample risk-control selection；
- bootstrap：fixed config 下 evaluation metric uncertainty。

請紅隊判斷：目前表述是否容易混淆？

---

## 7. 請紅隊回答的具體問題

### P0 — Formal validity / protocol

1. **R1 定義問題**  
   目前 R1 是 answered-target conditional FNR。這樣是否還能稱為 target recall / coverage guarantee？如果不能，應改成什麼名稱？

2. **是否必須補 R3 = target abstention rate ≤ γ？**  
   若要補，γ 建議是多少？0.5 合理嗎？還是應該用 0.4 / 0.45 / split-specific？

3. **若補 R3，主張應如何改寫？**  
   從「雙保證」改成「三風險保證」？還是「雙風險 + bounded target abstention」？

4. **operating point selection 是否 valid？**  
   在 LTT-valid configs 中，以 calib abst≤0.5 再取最小 set size，是否會影響 finite-sample validity？需要把 `abst≤0.5` 也納入 p-value 檢定嗎？

5. **threshold grid 用全 split unlabelled scores 是否構成 leakage？**  
   如果是，改 calib-only quantiles 是否足夠？是否還需要固定 quantile grid before seeing data？

6. **ref_id parity split 是否合理？**  
   有沒有 same image / same object leakage 導致 exchangeability 被質疑？是否要補 image-disjoint split？

7. **testA/testB 的使用方式是否會被認為 tuning on test？**  
   我們在三個 official splits 都做 calibration/evaluation；是否應只把 val 作為 method development，testA/testB 作一次 final confirm？目前結果已經看過，寫作如何避免 hindsight bias？

8. **Bonferroni + Hoeffding LTT 是否適合這裡？**  
   Loss bounded [0,1]，但 R1 conditional 的 sample size `n1` 取決於 gate。這會不會有 adaptive sample selection 的問題？

### P0 — Main claim / novelty

9. **CRS 的 novelty 是否成立？**  
   「conformal + frozen referring-grounding + box-set + no-target」這個交叉點是否足夠新？還有哪些最相近工作必須補？

10. **Cross-Base Composition 是否足以作為 method contribution？**  
    還是會被視為簡單 ensemble / pipeline？若要讓它更像 method，應補什麼理論或實驗？

11. **該不該使用 “first” claim？**  
    建議 wording：
    - “the first”
    - “to our knowledge, the first”
    - “we study ...” 不說 first

12. **如何避免 detector comparison 批評？**  
    GroundingDINO 明顯更強。主張應如何聚焦在 risk decomposition / feasible region，而不是「換強 detector 就好」？

13. **是否應把 M4 exact-match 失敗作為 CRS 的動機？**  
    這樣是加分（誠實導出新問題）還是會讓 reviewer 覺得 metric pivot 是失敗後改題？

### P0 — 最小必要補實驗

14. 若只能再跑一個實驗，你建議是哪個？
    - A. 三風險 LTT：R1/R2/R3
    - B. calib-only grid rerun
    - C. 多 random split robustness
    - D. image-disjoint split
    - E. Pareto/feasible-region curves
    - F. 其他

15. 哪些實驗是 thesis defense 前一定要有，哪些可以放 appendix / 不做？

### P1 — 結果表與圖

16. 主文最應該放哪張主表？  
    我目前想放：三 split composition table（OWL only vs COMPOSE，size/R1/R2/abst + CI）。

17. 2×2 ablation 應放主文還是 appendix？  
    它證明 mechanism，但 testA/testB 有 GD gate over-abstain，表格可能較難解釋。

18. 是否需要 Pareto curve / feasible-region plot？  
    建議圖：x=target abstention，y=set size，點顏色=valid/invalid，四組 gate×box 分 panel。

19. target-present separability table 是否放主文？  
    它能清楚支撐 GD box 為何 compact，但可能像 detector comparison。

20. 是否應畫 cost-Pareto 而不是只報單點？  
    單點容易被說 cherry-pick；曲線能顯示 trade-off。

### P1 — Writing / framing

21. 建議章節順序是否合理？
    - M4 exact-match wall
    - CRS single-risk
    - multi-risk LTT
    - base bottleneck decomposition
    - cross-base composition
    - ablation / CI

22. 該如何命名這個方法？
    - Cross-Base Conformal Composition
    - Conformal Referring Sets
    - Risk-Controlled Referring Sets
    - 其他

23. `target abstention` 應叫 abstention 還是 target rejection / deferral？  
    因為 no-target 本來就應輸出空集，target abstention 才是 cost。

24. `R2` 應叫 no-target false selection 還是 hallucination risk？  
    哪個更符合 VLM / grounding literature？

25. 如何精準描述 guarantee？  
    例如：

```text
For a fixed calibration protocol and exchangeable samples, LTT selects configurations whose calibration risks pass finite-sample tests for conditional target FNR and no-target false selection. We report target abstention as an explicit cost.
```

這樣是否太弱？或剛好安全？

### P2 — Optional extensions

26. **label-efficiency P3 是否值得做？**  
    idea：新 base 只需少量 calibration labels 即可重新標定 guarantee。這會加分，還是會發散？

27. 是否需要第三個 base？  
    例如 Florence-2 / OWLv2 / DINO variants。我的直覺是不需要，避免變 detector comparison。

28. 是否需要更強 nonconformity score？  
    例如 box stability / cross-prompt consistency / score calibration。之前 P1 belief 只有弱正，是否還值得？

29. 是否需要和 trained GREC methods 同表？  
    我傾向只作定位，不直接比較，避免不公平。

30. 是否需要 human-facing examples / qualitative figure？  
    展示 COMPOSE 輸出 2–3 boxes，而 OWL only 輸出 8–10 boxes；是否能幫助說服？

---

## 8. 希望紅隊輸出的格式

請盡量逐條回答，並標註嚴重程度：

```text
[Formal validity]
- R1 conditional issue: fatal / major / minor / none
- Suggested fix:
- Required rerun:

[Protocol leakage]
- Grid quantile issue: fatal / major / minor / none
- Suggested fix:

[Main claim]
- Is CRS thesis-worthy? yes / no / only if ...
- Safe wording:
- Unsafe wording to avoid:

[Experiments]
- Must-do before writing:
- Nice-to-have:
- Do-not-do / avoid scope creep:

[Reviewer attack simulation]
- Most likely reviewer criticism #1:
- Best defense:
- If defense insufficient, minimal experiment:
```

請特別指出：

- 有沒有任何**會讓主結果無效**的 protocol bug；
- 有沒有任何**寫作措辭會被抓爆**；
- 哪些補實驗是必要，哪些只是焦慮型發散。

---

## 9. 檔案與重現位置

本地 paperwork：

- 一頁總結：[CRS_SESSION_SUMMARY.md](CRS_SESSION_SUMMARY.md)
- 方法章草稿：[chapter_crs_conformal_referring_set.md](chapter_crs_conformal_referring_set.md)
- 腳本備份：`crs_src/_*.py`
- 主圖：`figures/crs_money_setsize.png`

spark：

- repo：`~/selective-grounding`
- branch：`feature/crs-conformal-referring-set`
- dumps：`~/selective-grounding/dump/gdino_gref_{val,testA,testB}.jsonl`
- result files：
  - `dump/RESULT_tp_separability.txt`
  - `dump/RESULT_money_figure.txt`
  - `dump/RESULT_full_table.txt`
  - `dump/RESULT_compose_threesplit.txt`
  - `dump/RESULT_compose_ablation.txt`

最新本地 commit：

```text
0da1d99 docs(CRS): 全量 testB 完成三 split 主結果
```

---

## 10. 我目前的初步結論（請紅隊挑戰）

我目前認為：

1. CRS / Cross-Base Composition **足以作為 thesis 主結果**，因為它從 measurement 升級成 risk-controlled set construction。
2. 最大 formal 弱點是 **conditional R1 + target abstention 未 formal control**，應優先補三風險 LTT。
3. 最大 protocol 弱點是 **grid quantiles 用全 split scores**，應改 calib-only grid 重跑。
4. 最大 writing 弱點是不能誇大成「unconditional recall guarantee」或「13× shrink」。
5. 最安全主張是：

> CRS provides finite-sample control of conditional target-set FNR and no-target false selection, while explicitly bounding or reporting target abstention. Cross-base composition reveals and exploits a factorization of the guarantee cost: OWL-ViT is a better abstention gate, while GroundingDINO yields compact target sets.

請紅隊判斷這是否仍足夠強，或需要更強 formalization / additional experiments。
