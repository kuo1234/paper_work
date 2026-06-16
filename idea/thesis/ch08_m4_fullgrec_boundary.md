# 第 8 章　M4：完整 GREC 的 exact-match 之牆

本章把事後點預測策略推到完整 GREC，定位它在何處、為何撞牆。這**不是方法貢獻，而是一個邊界／壓力測試**——但它在全篇敘事中是**轉折點**：exact-match 之牆證明點預測不足，正是下一章 CRS（風險受控集合預測）的存在理由。同一批數據的敘事方向因此不是「我們做不到完整 GREC」，而是「完整 GREC exact-match 在凍結基礎模型上有一道可量化的牆，這道牆正是 CRS 的動機」。

## 8.1 動機與設定

C3 已證明凍結基礎模型的輸出訊號可被事後校準出**無目標棄答**能力（基礎模型本身結構上沒有此能力）。一個自然的壓力測試是：把同一套事後策略推到**完整 GREC**——不只判斷「該不該答」，還要在 target-present 時輸出**正確的框集合**（multi-target），並以官方 **Pr@(F1=1, IoU≥0.5) / N-acc / T-acc** 評測，與 trained 架構（HieA2G、VIRO）在同一指標座標上定位。

**關鍵資料事實**：gRefCOCO 的 target-present 樣本**幾乎全是 multi-target**（val：1499 no-target／0 single／2738 multi，多數 2 框）。因此完整 GREC 在此資料集上本質是**集合預測／計數**問題，而非單框定位。

**GREC 官方 Pr@(F1=1) 協定**（忠實移植自 gRefCOCO `mdetr/datasets/refexp.py`）：以信心閾值 τ 過濾預測框 → 與 GT 框做貪婪 IoU 匹配（非 Hungarian，IoU≥0.5）→ 計 TP/FP/FN → `F1 = 2TP/(2TP+FP+FN)`；一個樣本須 **F1=1**（所有 GT 命中且無多餘框）才算正確；no-target 樣本須過濾後 **0 框**才算正確。OWL-ViT 原始候選分數偏低（multi-target top1 中位數 0.13），官方預設 τ=0.7 會濾掉所有框，故 τ 在 calib（ref_id 奇偶切分）重新標定，test 單次評估。

## 8.2 策略階梯與結果

格式為 **Pr@(F1=1) / N-acc / T-acc**（test split，95% bootstrap CI 見 §8.3）。

**表 8.1　完整 GREC 策略階梯（三個 split）**

| 策略 | val | testA | testB |
|---|---|---|---|
| forced-output base（下界，永不棄答） | 0.164 / 0.267 / 0.003 | 0.043 / 0.177 / 0.002 | 0.085 / 0.279 / 0.004 |
| conf-threshold（單一全域 τ） | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.938 / 0.035 |
| **P(no-target)+conf（本研究事後策略）** | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.939 / 0.035 |
| oracle（GT 完美棄答） | 0.623 / 1.0 / 0.037 | 0.278 / 1.0 / 0.060 | 0.329 / 1.0 / 0.050 |
| per-sample oracle τ（天花板） | 0.682 / 1.0 / **0.189** | 0.409 / 1.0 / **0.230** | 0.460 / 1.0 / **0.235** |

對照線（trained，**基礎模型不同，僅供座標定位，不主張可直接比較**）：HieA2G 完整 GREC Pr@(F1=1) val/testA/testB = 67.8 / 66.0 / 56.5（ResNet101 全監督，含 gRefCOCO 標籤訓練 counting head）。

**三點觀察：**

1. **Pr@(F1=1) 的提升幾乎全部來自棄答，而非多目標命中。** forced-output → 策略的提升（val 3.7×、testA 6×、testB 3.5×）對應的是 N-acc 從 0.18–0.28 升到 0.94–0.996，而 T-acc 幾乎不動。事後策略在「該不該答」維度有效，在「答幾個框」維度無效。

2. **「本研究 ≈ conf-threshold」——誠實承認，不挑數字。** 三個 split 的兩階段 P(no-target)+conf 策略與單一全域 conf-threshold 幾乎沒有差距（testA/testB 僅 N-acc 第三位小數差異）。原因：gRefCOCO 無目標比率高，加上 Pr@(F1=1) 這個 exact-match 指標，使得全域信心閾值已吸收絕大部分可得的 gain；learned no-target gate 在此指標下沒有額外拉開差距。這說明完整 GREC 的主要瓶頸**不在無目標偵測，而在 target-present 的多目標集合預測**。

3. **T-acc 天花板極低且非雜訊。** 即使給每個樣本完美的 per-sample τ（per-sample oracle，等同擁有 AUROC=1.0 的完美計數訊號），T-acc 天花板也只有 0.19–0.24。亦即：在這個凍結基礎模型上，多目標 exact-match 的失敗**不是因為缺乏好的信心訊號**，而是動作空間受限——「單一信心閾值」這個動作無法表達「該圖留 2 框、那圖留 5 框」的 per-sample 計數。

## 8.3 為什麼失效：召回足夠，瓶頸在 FP

對 val target-present 樣本的診斷：

- **GT-coverage recall = 0.94**：九成樣本中，每個 GT 框都有某個預測框 IoU≥0.5 命中。**召回不是瓶頸。**
- **best achievable F1（per-sample 最佳 τ）平均僅 0.62，僅 18.8% 樣本能達 F1=1**：問題出在 **FP（多餘框）**——OWL-ViT 輸出一堆候選框，單一全域 τ 無法在每個樣本上恰好留下「正確數量」的框。

因此完整 GREC exact-match 本質是**計數／集合預測**問題：要破此須 per-sample 的框數決策（如 HieA2G 的 trained Adaptive Grounding Counter），而這已超出「凍結基礎模型 + 單一輕量事後校準器」的範圍。

**Bootstrap 95% CI**（leak-safe：calib 固定、只 resample test ≥1000 次）佐證此邊界主張為統計穩固，非抽樣雜訊：

**表 8.2　T-acc 的 bootstrap CI**

| split | T-acc point | 95% CI |
|---|---|---|
| val | 0.003 | [0.001, 0.006] |
| testA | 0.052 | [0.046, 0.057] |
| testB | 0.035 | [0.031, 0.040] |

T-acc 的 CI 全部貼近 0（最高上界僅 0.057），「凍結零樣本偵測器的多目標 exact-match 超出事後校準能力」是一個有信賴區間支撐的結論。

## 8.4 結論：通往 CRS 的跳板

> **事後可靠性校準可以補出無目標棄答，但不能取代 trained counting head 或集合預測模組；完整 GREC exact-match 暴露了凍結基礎模型事後點預測的動作空間限制。**

這道牆不是論文的終點，而是轉折點。它劃出兩條路：(a) 補一個 trained counting／集合預測模組（HieA2G 路線，但這放棄了「凍結基礎模型 + 輕量事後」的賣點）；(b) **不再強迫 exact single/multi-box 點預測，改輸出有分布無關保證的框集合**——這正是第 9 章 CRS 的解法。

M4 的兩個診斷數字直接成為 CRS 的設計依據：

1. **per-sample oracle τ 的 T-acc 天花板僅 0.19–0.24** → 「單一信心閾值」這個動作表達不了 per-sample 計數。CRS 的回應：放棄點估計，輸出集合，把「該留幾個框」的不確定性吸收進**集合大小**與**棄答**（CRS 的 R3 目標棄答率）。
2. **recall 0.94 足夠、瓶頸在 FP** → raw score 把 TP/FP 在分數軸交織，點閾值切不乾淨。CRS 的回應：用 LTT 對「已作答目標漏檢率」給有限樣本保證，而非追逐 F1=1 的 0/1 事件。

因此 M4 與 CRS 是**同一條 framing 的兩端**：M4 證明凍結基礎模型的點預測撞牆（量化「缺的是動作空間，不是訊號」）；CRS 證明換成風險受控集合預測後，同一批凍結基礎模型能在三風險保證下輸出接近真實基數的緊緻集合。

需明寫的誠實邊界：M4 的結論本身不被 CRS 推翻——完整 GREC 的 **exact-match（F1=1）** 仍超出凍結事後能力。CRS 不宣稱「解了完整 GREC」，而是**改變問題**——從 exact-match 點預測換成風險受控集合建構。這個 metric pivot 在兩章都寫白，避免被讀成「CRS 解了 M4 解不了的同一個問題」。

對照定位（trained ↔ post-hoc 光譜）：HieA2G（全監督專訓 counting head）／ VIRO（凍結基礎模型 + 重型 neuro-symbolic per-operator verifier）／ 本研究（凍結基礎模型 + 單一輕量事後校準器）。三者在「訓練成本 ↓、推論成本 ↓」光譜上，本研究佔最輕量端；M4 誠實標明在此端，多目標 exact-match 是能力邊界，而非可由校準跨越的目標。
