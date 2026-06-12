# 紅隊 briefing — M4 full-GREC 定位裁決

> 背景：Selective Grounding 碩論，主線=**frozen referring grounding base 的 post-hoc reliability / calibration study**（不是新 grounding 方法，不與 trained 架構比 accuracy；量 reliability、cost、transferability、oracle gap）。錨 RefCOCO / gRefCOCO。四貢獻 C1–C4 已收斂，本文件只請你裁決 **M4（C3 擴到 full-GREC）該怎麼定位**。

---

## 一、四貢獻現況（讓你有脈絡，數字皆已跑出、三 split 齊備）

- **C1/C2（CLIP-VG，第一 base）**：frozen CLIP-VG 是 single-box regression（無候選分數）。用 cross-prompt consistency 當訊號，RefCOCO 三 split AUROC 0.71–0.72，risk-coverage AURC 全部約 random 的一半 → selective dominate。
- **C2（OWL-ViT，第二 base）**：OWL-ViT 有候選分數，score_entropy AUROC 0.68–0.69 三 split，RC 全 dominate random。
- **C4（cross-base transfer，已升為主結果）**：raw consistency 零參數、無 refit 直接 CLIP-VG→OWL-ViT，AURC 0.488 ≈ OWL-ViT native gate 0.489，比 random 降 15.9%。誠實 finding：多特徵 fitted gate 轉移較差，因 dispersion 權重 base-specific。
- **C3（gRefCOCO no-target gate，二元）**：frozen base 結構上無法 abstain（forced-output N-acc=0）；OWL-ViT max-score 當 P(target-present)。no-target AUROC 三 split 0.823 / 0.773 / 0.741，learned gate Bal-acc 0.758 / 0.704 / 0.689。**乾淨的正面結果。**

**→ 真正的方法貢獻是 C3（no-target 有效）+ C4（cross-base 轉移）。本文件問的是 M4。**

---

## 二、M4 是什麼（C3 從二元 no-target 擴到完整 GREC）

動機：C3 只做「答 / 棄答」二元 gate。要與 HieA2G（AAAI'25）/ VIRO（CVPR'26）/ GREC 官方 baseline 同表，必須報官方指標 **Pr@(F1=1, IoU≥0.5) / N-acc / T-acc**。

GREC 多目標設定：一句話可對應 0 個（no-target）、1 個、或多個（multi-target）物件。**關鍵資料事實：gRefCOCO target-present 樣本幾乎全是 multi-target**（val：1499 no-target / 0 single / 2738 multi，多數 2 框）。Pr@(F1=1) 要求「完全正確」：所有 GT 框命中（IoU≥0.5）且無多餘框，F1=1 才算對。

評測器已忠實移植官方（貪婪 IoU 匹配、非 Hungarian），self-test 通過。

---

## 三、M4 三 split 結果（格式 = Pr@(F1=1) / N-acc / T-acc，test split）

| policy | val | testA | testB |
|---|---|---|---|
| forced-output base（下界） | 0.164 / 0.267 / 0.003 | 0.043 / 0.177 / 0.002 | 0.085 / 0.279 / 0.004 |
| conf-threshold（單一全域 τ） | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.938 / 0.035 |
| **P(no-target)+conf（本研究主方法）** | 0.607 / 0.996 / 0.003 | 0.258 / 0.939 / 0.052 | 0.301 / 0.939 / 0.035 |
| oracle（GT 完美 abstain，target 共用 τ） | 0.623 / 1.0 / 0.037 | 0.278 / 1.0 / 0.060 | 0.329 / 1.0 / 0.050 |
| per-sample oracle τ（天花板） | 0.682 / 1.0 / **0.189** | 0.409 / 1.0 / **0.230** | 0.460 / 1.0 / **0.235** |

對照線（trained，**base 不同**，僅供定位）：HieA2G full-GREC Pr@(F1=1) val/testA/testB = 67.8 / 66.0 / 56.5（ResNet101 全監督 + 含 gRefCOCO 標籤訓練 counting head）。

---

## 四、我（執行者）對 M4 的診斷與主張

**診斷 1 — multi-target 的 T-acc 根本性無解，不是訊號弱：**
- per-sample oracle τ（給每個樣本完美閾值）天花板 T-acc 也只有 0.19–0.24。即使有 AUROC=1.0 的完美訊號去挑閾值，OWL-ViT 在 multi-target 還是答不對。
- 瓶頸不是召回：GT-coverage recall = 0.94（九成樣本每個 GT 都有預測框 IoU≥0.5 命中）。瓶頸是 **FP（多餘框）**。
- 根因：「單一信心閾值」這個動作本身無法表達「該圖留 2 框、那圖留 5 框」。要破此須 per-sample counting（≈ trained counting head），超出「凍結 base + 近零成本 post-hoc」範圍。

**診斷 2 — Pr@(F1=1) 的提升全來自 abstain，不是答對多框：**
- forced-output → policy 的 Pr 提升（val 3.7×、testA 6×、testB 3.5×）幾乎 100% 來自 N-acc（0.18–0.28 → 0.94），T-acc 幾乎沒動。

**診斷 3 — 「ours ≈ conf-threshold」防守不住：**
- 三 split 我的 two-stage P(no-target) gate 跟單一全域閾值幾乎沒差（testA/B 僅 N-acc 第三位小數差異）。
- 原因：gRefCOCO no-target 佔比高（val 0.63），「全部 abstain」這個懶惰策略本身就拿高分，全域閾值順手吃掉，掩蓋了 learned gate 的多訊號優勢。指標在此分布下獎勵懶惰。

**我的主張：** M4 不該當「方法貢獻」寫（會被打穿：multi-target 無效、no-target 又被資料偏置掩蓋 gate 價值）。應收成 **「reliability 上限 / 負結果診斷」**——第一個量化「frozen zero-shot detector 在 GREC exact-match 的硬天花板低到任何 post-hoc 訊號都救不起來」，支撐 thesis「量 reliability 與 gap、不比 accuracy」的主張。真正方法貢獻歸 C3 + C4。

---

## 五、請紅隊裁決的問題

1. **M4 當「負結果 / 邊界診斷」寫，對整篇是加分還是減分？** 一個主要章節是負結果，會不會讓審查覺得 contribution 被稀釋？還是「我們證明了為什麼任何 post-hoc 方法在此失效」本身是強結果？

2. **「ours ≈ conf-threshold」這個點怎麼救或怎麼防守？** 候選：(a) 改報 coverage–T-acc 曲線而非單點 Pr@(F1=1)；(b) 把 multi-target 移到 appendix、主文只留 no-target + 上限診斷；(c) 找一個讓 learned gate 拉開差距的 sub-slice（如低 no-target 比例子集）；(d) 其他。

3. **要不要為 M4 補強？** 例如：第三 base（GroundingDINO，更強的 detector，可能讓 multi-target T-acc 天花板更高、故事更完整）值不值得花時間？還是 M4 就此定稿、資源投去 bootstrap CI / cost-Pareto？

4. **整體 framing 風險**：四貢獻裡 C3/C4 正面、M4 偏負面、C1/C2 是機制驗證。這個組合作為碩論主結果，夠不夠？最大的攻擊面在哪？

> 請務必對抗式審視，不要附和。若認為我的主張錯了，直接說 M4 該怎麼定位。
