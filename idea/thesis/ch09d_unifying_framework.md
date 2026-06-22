# 第 9d 章　統一框架：三正交維度與 frozen detector frontier

> 前三章（9 / 9b / 9c）各自證明一個維度可改善。本章把它們收斂成一個 coherent system：
> 在完全 frozen 的 base detector 上，用 post-hoc、LTT-certified 的方式把 referring expression
> grounding 轉成 risk-controlled referring set，並系統性地測繪「候選池 / 閘 / 憑證」三個
> 可改善維度的可達邊界。

## 9d.1 一句話框架

> **不訓練任何 detector 權重。所有改善都在 frozen 輸出之上的 decision/calibration 層。**

系統把 grounding 轉成 risk-controlled referring set，並釘出三個可改善維度的可達邊界與天花板。

## 9d.2 系統的三個可改善維度

```
Image + Referring Expression
        │
        ▼
┌──────────────────────────┐
│  Frozen base detectors    │  OWL-ViT (gate signal) + GroundingDINO (boxes)
│  — 權重全凍結，不訓練      │  ← 能力天花板由此決定
└──────────────────────────┘
        │
   ┌────┴──────────────────────────────────┐
   │  維度 1: CANDIDATE POOL（第 9b 章）     │  ← Decomp CRS（主攻 R1）
   │    full-expr pool  /  decomp-union pool │
   └────┬──────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────┐
   │  維度 2: GATE SCORE（第 9c 章）         │  ← WB-Gate CRS（主攻 R2/R3/feasibility）
   │    OWL top1  /  WB-21 interpretable gate│
   └────┬──────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────┐
   │  維度 3: CERTIFICATE（第 9c 章）        │  ← HB-LTT（主攻 feasible region）
   │    Hoeffding  /  Hoeffding-Bentkus      │
   └────┬──────────────────────────────────┘
        │
        ▼
   Risk-controlled referring set  +  R1/R2/R3 certificate (or defer)
```

三個維度**正交且可獨立替換**，這是本論文的核心結構發現。

## 9d.3 三貢獻的分工（互不打架）

**表 9d.1　三維度分工**

| 維度 | 貢獻 | 改什麼 | 主攻 | 同分布結果 | 邊界 |
|---|---|---|---|---|---|
| pool | Decomp CRS | candidate pool | R1 / multi-target recall | R1 三 split 全 ≤ frozen；frontier rec@sz3 0.91 vs 0.69 | over-decomp 反例（n_gt=1 退步一半是自身錯） |
| gate | WB-Gate CRS | gate score | R2/R3 / feasibility | rule INFEASIBLE→WB feasible；R2 0.016 R3 0.075 | cross-dataset learned gate 不轉移（0.89→0.35） |
| cert | HB-LTT | bound | feasible region | #feas ×3，set size 5.07→3.72 | （免費，無邊界） |

**互補的鐵證**：
- decomp-pool 配 OWL-gate **仍 INFEASIBLE**；配 WB-gate 才 feasible → pool 改善需要 gate 改善才能被 CRS 安全使用。
- WB-Gate 主攻 R2/R3，R1 幾乎不動；Decomp 主攻 R1，R2/R3 持平 → 兩者攻不同 risk，疊加得 Pareto 移動而非互相抵銷。

## 9d.4 統一評估協議（所有貢獻共用）

- frozen base：OWL-ViT gate + GroundingDINO box，零訓練。
- 三風險 LTT：R1 answered-target FNR / R2 no-target false-sel / R3 target deferral。
- certificate：HB bound（主），Hoeffding/Bernstein（附錄 sensitivity）。
- 切分：calib/test 互斥；robustness = parity / random5 / image-disjoint。
- 防洩漏：calib-only grid；learned 模組三段互斥切分（gate_train/calib/test）。

任何 (pool, gate, bound) 組合都套同一協議，所以四個版本（Rule / Decomp / WB-Gate / WB+Decomp）
可在同一張表比較。

## 9d.5 整體 claim（三層）

**L1 base claim（CRS，第 9 章）**
> Frozen base + post-hoc LTT 即可把 grounding 轉成有限樣本風險保證的 referring set，無需訓練。

**L2 dimension claims（第 9b / 9c 章）**
> - Decomp：改善 candidate pool 在同分布下降低 R1（multi-target recall），且贏過所有 full-pool consolidation（NMS/top-K），非 matched 指標假象。
> - WB-Gate：可解釋 learned gate 在同分布下把 infeasible 設定變 feasible，大幅改善 R2/R3。
> - HB：更緊的 certificate 免費擴大 feasible region。

**L3 unifying claim（本章）**
> 三維度正交可組合；在 frozen detector 給定下，(pool, gate, bound) 三軸共同決定可達的
> risk-cost frontier。我們測繪了此 frontier 並釘出天花板。

## 9d.6 全景 limitation / negative findings（誠實層）

**表 9d.2　全景限制**

| 項目 | 內容 |
|---|---|
| 天花板 | frozen OWL+GDINO 的 scoring 決定上限；不碰 detector 的手段無法突破 |
| WB-Gate 泛化 | learned gate boundary cross-dataset 不轉移，比無訓練 rule gate 脆弱 |
| Decomp 反例 | n_gt=1 退步一半是 over-decomposition（自身錯），非全資料集模糊 |
| Decomp 成本 | training-free 但非 compute-free（每 target-present +1 VLM call） |
| 已否決 | candidate-level utility（per-box trap）、multi-crop（pool 非瓶頸）、prompt-template（不穩） |
| 已修正歸因 | cross-dataset 失守 ≠ no-target prior shift（受控實驗推翻） |

## 9d.7 天花板定性

整條線證明：**在不碰 frozen detector 的前提下，系統能力上限由 frozen OWL-ViT（gate）+
GroundingDINO（box scoring）決定。** 我們做的一切（白箱 gate、HB、Decomp、adaptive λ）都是在
榨乾這兩個 frozen 模型已產出的資訊，並把可達的 risk-cost frontier 完整測繪出來。

要抬此天花板，唯一有效方向是換更強的 frozen detector（DINO-X / GD-1.5，不違反 frozen，但 API-gated）。
multi-crop / 加 feature / candidate-level 等「不碰 detector」的手段已證明無法突破。

## 9d.8 future work（不在本研究範圍）

- 換更強 frozen detector（DINO-X / GD-1.5）：不違 frozen，抬天花板，API-gated。
- trained scoring head：僅作 upper-bound diagnostic，不進主線。
- WB-Gate cross-dataset 泛化修復（weighted conformal 等）：開新題，暫不做——這是第 9c 章
  negative finding 的自然延伸，可能獨立成一篇「learned reliability signal 為何不轉移」的研究。
- decomp over-decomposition 抑制（更保守的 VLM router）。

## 9d.9 小結

本論文的三條 contribution 不是三個獨立技巧，而是同一個 frozen selective grounding 系統的
三個正交維度。CRS 給出 base claim（frozen + LTT = 有保證的 referring set）；Decomp、WB-Gate、HB
分別在 pool、gate、certificate 三軸上把可達 frontier 往有用方向推；互補性論證證明它們攻不同 risk、
可疊加。我們不只給保證，還測繪了在 frozen detector 給定下的整個 risk-cost frontier，並誠實釘出
天花板與每個維度的邊界——這是本論文相對於任何單點方法的結構性貢獻。
