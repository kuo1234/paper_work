# M4 章節草稿 — Full-GREC Exact-Match Wall: The Motivation for Risk-Controlled Set Prediction

> 章節定位（紅隊裁決 2026-06-12，**2026-06-13 CRS 升級後重新定位 2026-06-15**）：
> 本節**不是方法貢獻**，是一個 **boundary / stress test**。但在 CRS 成為主台柱章之後，M4 的角色從
> 「post-hoc calibration 的**終點**（做不到，結束）」轉成「**動機/跳板**——exact-match wall 證明 point prediction 不足，
> 所以必須轉向 risk-controlled **set** prediction（CRS 章）」。同一批 M4 數據，敘事方向反轉：
> 不是「我們做不到 full-GREC」，而是「full-GREC exact-match 在 frozen base 上有一道可量化的牆，這道牆**正是 CRS 的存在理由**」。
> 本節回答：把 post-hoc point policy 推到完整 GREC（multi-target exact-match）會在哪裡撞牆、為什麼撞牆。
> **章序建議**：放在 gRefCOCO/GREC 章末，**緊接 CRS 章之前**（M4 撞牆 → CRS 解法），標題：
> **Full-GREC Exact-Match Wall: Why Frozen Post-hoc Point Prediction Is Not Enough**
> （中：Full-GREC exact-match 之牆——為何凍結 base 的 post-hoc 點預測不足，需轉向有保證的集合預測）
>
> 〔與 CRS 章的接口〕CRS 4.1 動機段直接引用本節的兩個 take-away：(1) per-sample oracle τ 的 T-acc 天花板僅 0.19–0.24
> → 單一信心閾值這個「動作」表達不了 per-sample counting；(2) recall 0.94 足夠、瓶頸在 FP 多餘框
> → raw score 把 TP/FP 在分數軸交織，任何**點估計**閾值都切不乾淨。CRS 的回應：不做點估計，改輸出**有保證的集合**。

---

## M4.1 動機與設定（Motivation & Setup）

C3 已證明 frozen base 的輸出訊號可被 post-hoc 校準出 **no-target abstention** 能力（base 本身結構上無此能力）。一個自然的壓力測試是：把同一套 post-hoc policy 推到 **完整 GREC**——不只判斷「該不該答」，還要在 target-present 時輸出**正確的框集合**（multi-target），並以 GREC 官方指標 **Pr@(F1=1, IoU≥0.5) / N-acc / T-acc** 評測，與 trained 架構（HieA2G、VIRO）在同一指標座標上定位。

**關鍵資料事實**：gRefCOCO 的 target-present 樣本**幾乎全是 multi-target**（val: 1499 no-target / 0 single / 2738 multi，多數 2 框）。因此 full-GREC 在此資料集上本質是 **set prediction / counting** 問題，而非單框定位。

**GREC 官方 Pr@(F1=1) 協定**（忠實移植自 gRefCOCO `mdetr/datasets/refexp.py`）：以信心閾值 τ 過濾預測框 → 與 GT 框做貪婪 IoU 匹配（非 Hungarian，IoU≥0.5）→ TP/FP/FN → `F1 = 2TP/(2TP+FP+FN)`；一個樣本須 **F1=1**（所有 GT 命中且無多餘框）才算正確；no-target 樣本須過濾後 **0 框** 才算正確。

OWL-ViT 原始候選分數偏低（multi-target top1 中位數 0.13），官方預設 τ=0.7 會濾掉所有框，故 τ 在 calib（ref_id 奇偶切分）重新標定，test 單次評估。

---

## M4.2 Policy 階梯與結果（Policy Ladder & Results）

格式 = **Pr@(F1=1) / N-acc / T-acc**（test split，95% bootstrap CI 見 M4.3）。

| policy | val | testA | testB |
|---|---|---|---|
| forced-output base（下界，永不 abstain） | 0.164 / 0.267 / 0.003 | 0.043 / 0.177 / 0.002 | 0.085 / 0.279 / 0.004 |
| conf-threshold（單一全域 τ） | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.938 / 0.035 |
| **P(no-target)+conf（本研究 post-hoc policy）** | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.939 / 0.035 |
| oracle（GT 完美 abstain） | 0.623 / 1.0 / 0.037 | 0.278 / 1.0 / 0.060 | 0.329 / 1.0 / 0.050 |
| per-sample oracle τ（天花板） | 0.682 / 1.0 / **0.189** | 0.409 / 1.0 / **0.230** | 0.460 / 1.0 / **0.235** |

對照線（trained，**base 不同，僅供座標定位，不主張可直接比較**）：HieA2G full-GREC Pr@(F1=1) val/testA/testB = 67.8 / 66.0 / 56.5（ResNet101 全監督，含 gRefCOCO 標籤訓練 counting head）。

**三點觀察：**

1. **Pr@(F1=1) 的提升幾乎全部來自 abstain，而非 multi-target 命中。** forced-output → policy 的提升（val 3.7×、testA 6×、testB 3.5×）對應的是 N-acc 從 0.18–0.28 升到 0.94–0.996，而 T-acc 幾乎不動。post-hoc policy 在「該不該答」維度有效，在「答幾個框」維度無效。

2. **「ours ≈ conf-threshold」——誠實承認，不 cherry-pick。** 三 split 的 two-stage P(no-target)+conf policy 與單一全域 conf-threshold 幾乎沒有差距（testA/testB 僅 N-acc 第三位小數差異）。原因：gRefCOCO no-target prevalence 高，加上 Pr@(F1=1) 這個 exact-match 指標，使得**全域信心閾值已吸收絕大部分可得的 gain**；learned no-target gate 在此指標下沒有額外拉開差距。這說明 full-GREC 的主要瓶頸**不在 no-target detection，而在 target-present 的 multi-target set prediction**。

3. **T-acc 天花板極低且非雜訊。** 即使給每個樣本完美的 per-sample τ（per-sample oracle，等同擁有 AUROC=1.0 的完美 counting 訊號），T-acc 天花板也只有 0.19–0.24。亦即：在這個 frozen base 上，multi-target exact-match 的失敗**不是因為缺乏好的 belief 訊號**，而是 action space 受限——「單一信心閾值」這個動作無法表達「該圖留 2 框、那圖留 5 框」的 per-sample counting。

---

## M4.3 為什麼失效：召回足夠，瓶頸在 FP（Diagnosis）

對 val target-present 樣本的診斷：

- **GT-coverage recall = 0.94**：九成樣本中，每個 GT 框都有某個預測框 IoU≥0.5 命中。**召回不是瓶頸。**
- **best achievable F1（per-sample 最佳 τ）平均僅 0.62，僅 18.8% 樣本能達 F1=1**：問題出在 **FP（多餘框）**——OWL-ViT 輸出一堆候選框，單一全域 τ 無法在每個樣本上恰好留下「正確數量」的框。

因此 full-GREC exact-match 本質是 **counting / set prediction** 問題：要破此須 per-sample 的框數決策（如 HieA2G 的 trained Adaptive Grounding Counter），而這已超出「凍結 base + 單一輕量 post-hoc calibrator」的範圍。

**Bootstrap 95% CI（leak-safe：calib 固定、只 resample test ≥1000 次）佐證此邊界主張為統計穩固，非抽樣雜訊：**

| split | T-acc point | 95% CI |
|---|---|---|
| val | 0.003 | [0.001, 0.006] |
| testA | 0.052 | [0.046, 0.057] |
| testB | 0.035 | [0.031, 0.040] |

T-acc CI 全部貼近 0（最高上界僅 0.057）→ 「frozen zero-shot detector 的 multi-target exact-match 超出 post-hoc calibration 能力」是一個有信賴區間支撐的結論。

---

## M4.4 結論與在 thesis 中的角色（Takeaway → CRS 的跳板）

> **post-hoc reliability calibration 可以補出 no-target abstention，但不能取代 trained counting head 或 set prediction module；full-GREC exact-match 暴露了 frozen-base post-hoc point policy 的 action-space 限制。**

這道牆不是論文的終點，而是**轉折點**。它劃出兩條路:
(a) 補一個 trained counting / set-prediction module（HieA2G 路線，但這就放棄了「凍結 base + 輕量 post-hoc」的賣點）;
(b) **不再強迫 exact single/multi-box 點預測，改輸出有分布無關保證的 box 集合**——這正是下一章 **CRS（Risk-Controlled Referring Sets）** 的解法。

M4 的兩個診斷數字直接成為 CRS 的設計依據:
1. **per-sample oracle τ 的 T-acc 天花板僅 0.19–0.24** → 「單一信心閾值」這個動作表達不了 per-sample counting。CRS 的回應:放棄點估計，輸出集合,並把「該留幾個框」的不確定性吸收進**集合大小**與**棄答**（CRS 的 R3 target deferral）。
2. **recall 0.94 足夠、瓶頸在 FP** → raw score 把 TP/FP 在分數軸交織,點閾值切不乾淨。CRS 的回應:用 LTT 對「漏檢率（answered-target FNR）」給有限樣本保證,而非追逐 F1=1 的 0/1 事件。

因此 M4 與 CRS 是**同一條 framing 的兩端**:M4 證明 frozen base 的 point prediction 撞牆（量化「缺的是 action space，不是 signal」）;CRS 證明換成 risk-controlled set prediction 後,同一批 frozen base 能在三風險保證下輸出 ≈GT 基數的 compact 集合。本研究不追 SOTA exact-match accuracy,而是量化 frozen base 的 reliability、cost、transferability、oracle gap,並在 point prediction 撞牆處給出有保證的集合預測替代方案。

> 〔誠實邊界保留〕M4 的結論本身不被 CRS 推翻:full-GREC **exact-match（F1=1）** 仍超出凍結 post-hoc 能力。
> CRS 不宣稱「解了 full-GREC」,而是**改變問題**——從 exact-match point prediction 換成 risk-controlled set construction。
> 這個 metric pivot 必須在兩章都寫白,避免被讀成「CRS 解了 M4 解不了的同一個問題」。

**圖**：`idea/figures/grec_ladder.png`（左：三 split policy ladder bar，含 forced-output / ours / oracle / per-sample ceiling / HieA2G 對照線；右：召回足夠但 exact-F1 低的 FP 瓶頸診斷）。

**對照定位（trained ↔ post-hoc 光譜）**：HieA2G（全監督專訓 counting head）/ VIRO（凍結 base + 重型 neuro-symbolic per-operator verifier）/ 本研究（凍結 base + 單一輕量 post-hoc calibrator）。三者在「訓練成本 ↓、推論成本 ↓」光譜上，本研究佔最輕量端；M4 誠實標明在此端，multi-target exact-match 是能力邊界，而非可由 calibration 跨越的目標。
