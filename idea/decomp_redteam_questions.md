# Decomposed CRS — 給紅隊的完整簡報 + 攻擊點清單（2026-06-18）

> 對象：外部紅隊 / GPT 紅隊（無持久記憶，請先讀 §0-§2 建立全景，再攻 §3）。
> 一句話：請盡力攻破「Decomposed CRS 是一個成立的論文第二主結果」這個主張。

---

# §0. 我們在做什麼（紅隊冷啟動必讀）

## 0.1 大背景：這是一篇碩士論文，主線立場已定

> **Frozen vision-language models 下的 post-hoc, risk-controlled referring sets。零訓練、零梯度、不改任何模型權重。**

不是在做新的 grounding 模型。是在做：**當凍結的偵測器不可靠時，如何用後處理的統計校準，輸出一個「帶風險保證的候選框集合」而非單一最佳框。**

## 0.2 任務（referring expression grounding / GREC）

輸入：一張圖 + 一句指稱語（如「穿藍衣站右邊的打者和空中的棒球」）。
輸出：框出所有被指稱的物件。難點：一句話可能指 **0 個（no-target）、1 個、或多個（multi-target）** 物件。資料集：gRefCOCO（val/testA/testB）。

## 0.3 第一主結果（已完成、已紅隊）：Cross-Base CRS

凍結 OWL-ViT（當 gate）+ 凍結 GroundingDINO（出框），用 Learn-then-Test (LTT) 聯合校準三個風險，給**有限樣本可證明的保證**：
- **R1** = answered-target FNR（在「有答案」的 target-present query 上的漏檢率）
- **R2** = no-target false selection（對 no-target query 卻吐了框的比率）
- **R3** = target deferral（target-present 卻棄答／吐空集的比率）

headline（α=β=0.3, γ=0.5）：set size val/testA/testB = 3.24/2.02/3.50，三風險 CI 上界 <0.3。2×2 ablation 證明是 **factorization 非 ensemble**（OWL gate 管棄答/no-target、GD box 管集合大小，拔一個就垮）。

## 0.4 第二主結果（本文焦點）：Expression-Decomposed CRS

**動機**：實測發現把**整句**餵 GDINO，在 multi-target 上 R1 天花板 0.27-0.34（被 16 次嘗試驗證），瓶頸不是定位、是「一句話混多個語義目標把 detector 搞混」。
**做法**：拆句 → 分頭定位 → 合併 → 接回 CRS。

---

# §1. 系統架構與資料流

```
影像 I + 指稱語 e
   │
   ▼
[1] Routing Module（v1 保守規則，非學習）
      ├─ no-target / single-target  → passthrough（不拆）
      └─ multi-target / 並列結構      → 拆成 sub-expressions
   │
   ▼
[2] Frozen 偵測器（GroundingDINO，零訓練，threshold=0）
      ├─ passthrough 案例：整句 e → full-expression 候選池（~45 框）
      └─ decomposed 案例：每個 sub-expr 各自當 query → 各自候選 → union（~6 框）
   │
   ▼
[3] Candidate Pool（目前只有 union，無 merge/filter）
   │
   ▼
[4] Risk-Controlled Decision Layer（CRS / LTT，純統計，非神經網路）
      ├─ OWL-ViT gate（控 no-target / 棄答）
      ├─ score threshold λ（控集合大小）
      └─ LTT 三風險聯合校準（calib-only quantile grid，Bonferroni 校正）
   │
   ▼
最終 referring set  /  defer（帶 R1/R2/R3 有限樣本保證）
```

關鍵設計決定：
- **拆句用 VLM（Qwen2.5-VL-7B），但只做語言拆解（prompt-only），不做視覺選框。** VLM-as-selector 已是負結果（見 §2.3）。
- **Routing 是 if-else 規則（v1 保守），不是學出來的 policy。** 刻意如此：學習式 routing 違反 frozen 立場，且我的學習式重排序嘗試全失敗。v1 保守規則正是 R2 安全保證的來源。
- **threshold=0**：與既有 baseline dump 同口徑（公平比較優先於速度）。

---

# §2. 關鍵實驗證據（紅隊攻擊的標的）

## 2.1 全量三 split LTT 主表（α=β=0.3, γ=0.5，parity split）

| split | Frozen CRS (sz/R1/R2/defer) | Decomp CRS (sz/R1/R2/defer) | n_feas | 改善 |
|---|---|---|---|---|
| val | 3.24 / 0.191 / 0.159 / 0.424 | 3.24 / **0.165** / 0.158 / 0.424 | 112/112 | R1↓，餘持平 |
| testA | 2.02 / 0.260 / 0.203 / 0.443 | **1.96** / **0.244** / 0.201 / 0.444 | 58/58 | R1↓ 且 size↓ |
| testB | 3.50 / 0.168 / 0.236 / 0.414 | **3.34** / 0.163 / 0.236 / 0.414 | 56/56 | size↓ |

= 跨 split 純帕累托改善，無 R2 爆、無 trade-off。

## 2.2 matched oracle 分析（決定性機制證據，繞過 LTT）

在「真被拆解」的 case 上（val 875 / testA 1243 / testB 974），各自 sweep λ 對齊到同一 avg pool size，比 recall：

| split | recall@size~3 full → decomp |
|---|---|
| val | 0.704 → 0.897（+19pp）|
| testA | 0.717 → 0.905（+19pp）|
| testB | 0.603 → 0.834（+23pp）|

機制解釋：per-part query 的 per-box 分數乾淨，threshold 時 GT-covering 框排前留得住；full-expr 分數被長句雜訊污染，同 size 留錯框。

## 2.3 已用實驗否決的方法（boundary analysis，prove decomposition 是對的路）

candidate scoring 重排序（負）、VLM-as-selector（11 次嘗試全負）、cardinality head（負）、evidence detector（CLIP-crop≈random、OWL-ViT 鎖死 0.40，負）。第 17 次嘗試 decomposition 才首次大勝。

## 2.4 subgroup 分析（matched recall@size~3）

- 長句受益更大：long(≥12詞) Δ +0.24 vs short Δ +0.18，三 split 一致（機制因果證據）
- 雙目標甜蜜點 Δ+0.24；n_gt≥3 遞減（testB 僅 +0.06）
- testA n_gt==1（61 例）Δ−0.148 → 逐案查證為 gRefCOCO「語言雙指稱 vs GT 單框」標籤歧義

## 2.5 我已自查排除的弱點（別重複攻這些）

- **no-target 全量零誤拆**：0/8905（val）、0/4448（testA）、0/4673（testB）→ R2 安全保證在全量成立
- **零 malformed**：所有拆解案例 n_parts≥2（val 840×2/28×3/6×4/1×5；testA/B 類似）
- 子集→全量：子集時的 R2 上升、testA n_feas=2/testB EMPTY 全證實為小樣本假象，全量消失

---

# §3. 攻擊點清單（請聚焦這些）

## P0 — 最可能一擊致命

### A. matched recall@size 是不是偽指標？⭐ 我最擔心
決定性證據（§2.2）是「同 avg pool size 下 decomp recall > full」。但 full pool ~45 框、decomp pool ~6 框，兩邊**各自 sweep 自己的 λ** 湊到 size~3，不是同一 operating threshold。
- **攻擊**：兩個 pool 的分數分布天差地別，「在 size~3 比 recall」會不會只是因為 decomp pool 本來就小而密、full pool 大而稀，這比較對 full 不公平？
- **求**：這比較的統計正當性站得住嗎？有沒有更公平的對齊（固定 recall 比 size、AU-recall-size-curve）？若 A 破，決定性證據垮。

### B. 只有 parity 單 split — 已知漏洞 ⭐
§2.1 全是 ref_id 奇偶（parity）split 的單一 operating point。既有 CRS 主線做過 parity/random5/image-disjoint 三模式 robustness，**decomp 還沒做**。
- **攻擊**：換 seed / image-disjoint split，decomp 還贏嗎？還是這 operating point 是 parity 專屬？

### C. R1 定義域被偷換？
R1 只在「非空答案」上算。decomp 讓更多 case 變非空。
- **攻擊**：decomp 是不是把難 case 從 defer 挪進 answered，讓 R1 看似沒爆但分母變了？（全量 val defer 持平 0.424 緩解此點，但需逐 split 確認 defer 沒暗中惡化。）

## P1 — 方法論 / 可比性

### D. threshold=0 的 full-expr baseline 是稻草人？
full-expr pool threshold=0 收滿 ~45 框（含大量 0.0x 垃圾框）。
- **攻擊**：給 full-expr 公平的 NMS / top-k 後處理，recall@size 會不會追上？我的 baseline 是不是「未後處理的裸 pool」稻草人？

### E. VLM 拆句成本沒算進 frozen 叙事？
decomp 每 multi-target query = 1 次 7B VLM 推論 + N 次 GDINO；full-expr 只 1 次 GDINO。
- **攻擊**：宣稱 frozen/便宜，但 inference compute 漲 2-4 倍 + 一個 7B VLM，這誠實嗎？（detector-forwards 成本軸還沒補。）

### F. 負信號「標籤歧義」是事後找補？
- **攻擊**：你只看 12 個 case 的 expression 就把 testA n_gt==1 的 −0.148 歸因標籤歧義。會不會 routing 真在某些結構誤拆，只是被雙目標大勝掩蓋？

## P2 — 敘事 / 定位（關乎論文結構）

### G. 與既有 compositional grounding 的差異夠嗎？
「拆句分頭 ground 再合併」非全新（scene-graph / modular grounding 早有）。
- **攻擊**：新意是「decomposition」還是「decomposition 接上 conformal risk control」？若是後者，decomp 本身只是 candidate-pool 工程，賣點該全壓 CRS/LTT 的 formal guarantee——那 decomp 還配當第二主結果嗎？

### H. 第二主結果 vs 一個 ablation？
decomp 整個掛在 CRS 框架（同套 LTT、同套三風險）。
- **攻擊**：這是「兩個主結果」還是「一個主結果 + 一個 candidate-pool ablation」？口試委員可能說 decomp 不夠格獨立成章。

---

# §4. 我自己的優先排序

- 最想聽紅隊意見：**A（matched 指標正當性）**、**B（多 split robustness，已知漏洞）**。
- **G/H** 關乎論文結構（decomp 算不算第二主結果）。
- **D/E/F** 我有部分答案，想被壓力測試。
- B 我打算自己先補（split_calib_test 已支援 random/imagedisj，一個指令能跑）。請紅隊集中火力在 A 和 G/H 這種我自己堵不掉的概念性弱點。

---

# §5. 紅隊回覆與裁決（2026-06-18）

> 總裁決：**Decomp CRS 站得住，但不能再把 matched recall@size 稱「決定性證據」。它是強 mechanism evidence，不是最終主證據。最終主證據＝full split + LTT 三風險 + robustness。**

## 逐點回覆

- **A（matched 指標）**：不是偽指標，但不夠當最終證據。它合理地比較「平均輸出成本相同時哪個 pool 更保留 GT-covering boxes」，但兩 pool 各自 sweep λ、分數分布不同，只能說「各自最佳 operating point 下 decomp 的 recall-size frontier 較好」，不能說「decomp 絕對更會 grounding」。→ §2.2 改名 **Matched Recall-Size Frontier Analysis**；補四指標：(1) recall@matched size (2) size@matched recall (3) AURC (4) **calib-selected λ 在 calib 選、eval 評**（最重要，堵偷看 eval sweep）。
- **B（split robustness）**：最該補的硬漏洞。補 parity(main)/random5(mean±std)/image-disjoint(robustness)，各報 R1/R2/R3/size/n_feas。方向一致即站得住；image-disjoint 變弱則誠實寫成 image-level generalization limitation。
- **C（R1 定義域）**：補 R_total = R3 + (1-R3)*R1，並把 answered/defer/target-failure 一起報，證明非靠改 deferral denominator 美化 R1。
- **D（baseline 稻草人）**：補 strong full baseline = full raw / full+NMS-merge / full+top-K(calib-tuned)，全接同套 CRS/LTT。decomp 連 full+NMS 都贏才穩。
- **E（成本）**：不能說 cheap。改 claim「training-free and weight-frozen, but with additional inference-time compute」，報 avg GDINO forwards/query + VLM parser calls/query。
- **F（負信號）**：標籤歧義不能只靠 12 例。對負增益 subgroup 抽 50-100 例標註錯誤類型（parser error / over-decomp / annotation ambiguity / detector miss / merge issue）報比例。
- **G（新穎性）**：不能主張「拆句 grounding」新（撞 compositional grounding / modular networks）。安全新穎性＝**decomposition as candidate-pool reconstruction for frozen, post-hoc, multi-risk controlled referring sets**，壓在三件組合：candidate-selection boundary analysis + decomposition-based pool reconstruction + CRS/LTT multi-risk guarantee。
- **H（章節定位）**：只停在 matched＝ablation；補完 full split LTT + robustness + R2 不爆 + full+NMS 仍輸 + subgroup＝正式第二主結果。章名用 **Expression-Decomposed CRS / Expression-Decomposed Candidate Pool Construction**。

## §9 待補實驗順序（1/3 最硬防線）
1. random5 + image-disjoint LTT robustness
2. recall-size curve / AURC / matched frontier 四指標
3. full-expression strong baseline（full + NMS / top-K / same consolidation）
4. overall target failure R_total = R3 + (1-R3)R1
5. cost table（VLM calls + detector forwards）
6. negative subgroup systematic failure audit

## 最終 claim（紅隊認可）
> Candidate selection alone fails to close the oracle gap. Expression decomposition changes the candidate-pool construction process and improves the recall-size frontier for compositional multi-target queries. When wrapped by CRS/LTT, it can reduce answered-target FNR or set size while preserving no-target and deferral risk control.

**禁用**：decisively solves the oracle gap / 決定性證據。

---

# §6. P1-E / P1-F 補完（2026-06-18）

## P1-E 成本表（決議）
routing 每 target-present query 都呼叫 VLM。誠實成本:VLM/q = 0.37/0.77/0.71(val/testA/testB),GDINO fwd/q = 1.06-1.08×。claim 改「training-free and weight-frozen, but with additional inference-time compute」,VLM/GDINO 分兩欄不合併(7B VLM forward 遠貴於 GDINO)。詳見 `decomp_cost_failaudit_result.md`。

## P1-F 失敗審計（決議,修正原推斷）
原「testA n_gt==1 −0.148 全是標籤歧義」(只看 12 例)**降調**:系統審計 testA n_gt==1 負增益 14 例 = ambiguity 50% + over_decomp 50%(後者因共用 head noun 無法與 ambiguity 切開,真 ambiguity 比例更高,故「至少 50%」是下界)。無 detector_miss、無 parser_error。**最終措辭**:decomposition 對 explicit multi-target 有益,對 single-target 標註下的多指稱 expression 有害(語言-標註粒度錯配),非偵測/解析失敗。

## 紅隊七項全補完狀態
| # | 項目 | 結果 |
|---|---|---|
| 1 降調 claim | ✅ | mechanism evidence,禁 decisive |
| 2 robustness | ✅ | 7/8 守,testB random5 trade-off |
| 3 frontier 四指標 | ✅ | size@rec0.8 半框,anti-cheat 守 |
| 4 strong baseline | ✅ | decomp R1 領先但非單點全勝(testA full+NMS) |
| 5 R_total | ✅ | R3 持平,非偷換分母 |
| 6 成本表 | ✅ | VLM 0.37-0.77/q,誠實 inference compute |
| 7 失敗審計 | ✅ | ambiguity ≥50%,原推斷降調 |
