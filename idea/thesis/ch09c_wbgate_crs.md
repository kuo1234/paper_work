# 第 9c 章　WB-Gate CRS：閘分數維度（第三主結果）

> 本章是三正交維度中的 **gate score** 軸。第 9 章 CRS 用單一 OWL top1 分數當棄答 gate；
> 本章問：把它換成一個**可解釋的 learned gate**，能否在保留 LTT 風險保證下改善 R2/R3 與可行性？
> 答案是肯定的——但**只在同分布成立**，cross-dataset 的 learned boundary 不轉移，這構成本章
> 最有研究價值的 negative finding。

## 9c.1 動機：單一分數的 gate 太弱嗎？

CRS 的 2×2 消融（表 9.3）證明 OWL top1 score 是最佳的 gate 訊號（無目標可分性 0.82）。
但「最佳的單一分數」不等於「最佳的 gate」。OWL 的無目標分數尾巴與 target 分數高度重疊，
任何單一閾值都切不乾淨。一個自然的問題：用多個 frozen 訊號餵一個 learned gate，能不能
比單一分數閾值更好地分離「該答 / 該棄」，從而在更嚴格的 budget 下進入可行域？

約束仍是 frozen：gate 的輸入特徵全部來自 frozen detector 的副產品，不更新任何 detector 權重；
learned 的只是 gate 這個薄薄的 decision 層。

## 9c.2 方法：21-dim 可解釋 white-box gate

**特徵（21 維，全 frozen，無 GT 洩漏）**：OWL×7（top1/margin/entropy/mean-topk 等）+
GDINO×7（對稱的分數統計）+ expr×5（純語法：長度、token 數等）+ cross×2（跨 detector 一致性）。
**禁用特徵**：no_target、n_gt、gt_boxes（任何 GT 衍生量），以及 decomposition 旗標
（會洩漏 has_target，見第 9b 章）。

**模型**：EBM（Explainable Boosting Machine，additive，`interactions=0`）——每個特徵一條可畫出的
shape function，gate 決策完全可解釋，符合「白箱」定位。

**防洩漏 3-way disjoint split**（本章最關鍵的協議）：val 內部以 `ref_id mod 3` 切成三段互斥子集——
gate_train（fit 模型）/ ltt_calib（搜門檻）/ test（評估），且 ref_id + image **雙重 disjoint**，
腳本內硬 assertion 強制。scaler 與模型只 fit gate_train。這修補了早期版本「在全 val 訓練後
又在 val 子集測試」導致 in-sample AUROC 0.99 的洩漏，修補後結果存活。

## 9c.3 主結果一：HB certificate（最大實質收益，免費）

在動 gate 之前，先換 certificate。把 Hoeffding 換成 Hoeffding–Bentkus（HB = min(Hoeffding, e·Binomial-tail)），
**同切分、同 grid、同 Bonferroni，只換 p-value**：

**表 9c.1　bound 對照（image-disjoint α=0.20）**

| bound | #feas | set size | status |
|---|---:|---:|---|
| Hoeffding | 22 | 5.07 | appendix sensitivity |
| **HB** | **69** | **3.72** | main certificate |
| Bernstein | 46 | 3.72 | appendix sensitivity |

可行區 ×3（22→69），set size 5.07→3.72，**完全免費**（不補任何 dump、不訓練）。
健全性四項全過（Bentkus 含 e 常數、p-value 單調、HB⊆Hoeffding superset、test 零違反）。
採用原則：主表用 HB，附錄報 Hoeffding/Bernstein 的 sensitivity。這是 certificate 維度的實例化，
與 gate、pool 正交。

## 9c.4 主結果二：WB-21 gate 讓 infeasible 變 feasible

協議：α=0.3、β=0.2、γ=0.3，HB certificate，3-split disjoint。rule 與 WB 同等套 HB：

**表 9c.2　WB-Gate 主表**

| method | gate | bound | #feas | certified UCB R1/R2/R3 | test R1/R2/R3 | set size |
|---|---|---|---:|---|---|---:|
| rule + HB | OWL top1 | HB | **INFEASIBLE** | — | — | — |
| **WB-21 + HB** | WB-21 EBM | HB | **115** | 0.220/0.032/0.089 | 0.159/0.016/0.075 | 3.68 |

**核心 claim**：單一 OWL top1 gate 在 α=0.3 三 split 全部 **infeasible**；換成 WB-21 learned gate
後 feasible（#feas=115）。certified UCB 全部守在 target 內，test empirical risk 更低
（R1/R2/R3 = 0.159/0.016/0.075）。這證明 gate 軸確實可改善可行性與 R2/R3 trade-off。

## 9c.5 主結果三：WB × Decomp 互補（兩維度疊加）

把 gate 維度（WB）與 pool 維度（Decomp）疊加。2×2：gate(OWL/WB) × pool(full/decomp)，
gate 特徵永遠用 full-pool，只換 box 候選池：

**表 9c.3　WB-Gate × Decomp 2×2**

| version | gate | pool | #feas | test R1 | test R3 | set size |
|---|---|---|---:|---:|---:|---:|
| Rule | OWL | full | INFEASIBLE | — | — | — |
| Decomp | OWL | decomp | **INFEASIBLE** | — | — | — |
| WB-Gate | WB | full | 115 | 0.159 | 0.075 | 3.68 |
| WB+Decomp | WB | decomp | 119 | 0.248 | **0.028** | **2.37** |

**互補鐵證**：decomp 候選池單獨配 OWL gate **仍 INFEASIBLE**——pool 改善需要 gate 改善才能被 CRS
安全使用。只有 WB+Decomp 同時動兩個維度，才把 set size 從 3.68 壓到 2.37（−36%）、R3 從 0.075 降到 0.028。
代價是 R1 從 0.159 升到 0.248（pool 聚焦 + 門檻趨嚴的 trade-off）。這正面回答「兩個改善會不會互相
抵銷」：它們攻不同 risk，疊加得到 Pareto 移動而非抵銷。

**query-adaptive λ（附錄）**：decomp box score 比 passthrough 高約 7×，故分組設不同 λ 可救回 R1
（0.208→0.173），但 R3 上升（0.106→0.196）——R1 與 R3 在此系統中對抗，三方法是同一 Pareto
frontier 的不同操作點。

## 9c.6 核心 negative finding：learned gate 不跨分布轉移

這是本章最重要、也最誠實的發現。固定門檻的 pass-rate 分析揭露：learned gate boundary
**不跨 gRefCOCO split 轉移**。

**表 9c.4　has-target pass rate（rule vs WB，三 tau 操作點一致）**

| split | rule pass\|tp | WB pass\|tp |
|---|---:|---:|
| val | 0.881 | 0.893 |
| testA | 0.861 | **0.388** |
| testB | 0.805 | **0.345** |

rule gate 跨 split 幾乎穩定（0.88→0.81），WB gate **崩塌**（0.89→0.35），三個 tau 操作點一致。
也就是說：**learned target-presence boundary 在 val 有效、在 testA/testB 不對齊**。
learned signal 比 frozen rule score 更受分布影響。

**真因經三輪排除**（這是本章的方法嚴謹度所在）：
- ❌ no-target prior shift：R2/R3 是 conditional risk，對群組比例免疫（resample 20–70%，range 0.001）。
- ❌ GDINO box scoring shift：GT-cover 分數 testA/testB 反而**更高**（val 0.145 / testA 0.173 / testB 0.159），box scoring 沒退化。
- ✅ **真因（兩機制並存）**：(1) OWL no-target 高分尾巴右移（傷 R2，q90 0.155→0.204）；(2) **WB learned boundary 不轉移（傷 R3，主導）**。

**封板定位（收緊 claim）**：WB-Gate = **in-distribution** interpretable gate improvement；
cross-dataset transfer 是 open limitation。同分布結果鐵打，跨資料集 WB 反而比無訓練 rule gate 更不穩。

## 9c.7 negative finding 的研究價值

> A learned gate improves in-distribution selective risk control but may overfit dataset-specific
> target-presence cues; the untrained rule gate is weaker in-distribution yet more robust to
> distribution shift. **學一個更強的可靠性訊號 ≠ 更可靠的跨分布轉移。**

這是 selective prediction 領域一個有意思的誠實教訓，呼應全篇主線「何時該信模型」：
正是當我們訓練出一個「更會判斷該不該信」的 gate 時，它對分布變化反而更脆。這不是修復對象，
而是一個有價值的 negative finding，放 discussion。

## 9c.8 其他負結果（界線，appendix）

- **Feature 擴張 = 沒用**：語法 compositional features AUROC +0.0002（null）；cross-prompt response
  profile AUROC +0.003 但 inner feature-selection 三 split 選三個不同 group → 增益被 split 雜訊淹沒，
  降 appendix。結論：gate 判斷力幾乎全來自 detector scores，feature 層近天花板。
- **Candidate-level utility = per-box trap**：per-box 7-dim EBM utility 取代 score threshold，
  R1 沒改善、set size 2–3× 大、per-box AUROC 僅 0.76。與 evidence-detector 同坑
  （per-box relevance ≠ set-level coverage）。確認 box selection 不可做 per-box classifier。
- **Multi-crop = 診斷否決**：R1 audit 顯示 98.8% query 池中每個 GT 都已有 box cover，
  但 40.7% cover GT 的框 score<0.1（被 λ 濾掉）。R1 瓶頸是 frozen GDINO 的 **scoring**，
  非候選池 coverage；multi-crop 只擴池不改分數 → 解錯問題，不值得跑。
- **Prior-shift = 推翻錯誤歸因**：推翻早期「val→testA/testB 失守 = prior shift」的歸因
  （R2/R3 對群組比例免疫）。

## 9c.9 小結

WB-Gate CRS 證明 gate score 是第三個正交可改善維度：可解釋的 21-dim white-box gate（EBM）
在同分布下把單一 OWL 分數 infeasible 的設定變 feasible，大幅改善 R2/R3，並與 Decomp 互補疊加。
HB certificate 是最大的免費收益。但 learned gate boundary cross-dataset 不轉移
（rule 0.88→0.81 穩 vs WB 0.89→0.35 崩），WB-Gate 的優勢限於 in-distribution——
這是本章最誠實、也最有研究價值的 negative finding。
