# VIRO 深讀筆記 — Verification-Integrated Reasoning Operators

> 為碩論 selective grounding / CRS（Cross-Base Conformal Referring Set）整理。本篇是「no-target / abstention 軸」上最需要劃清界線的競品之一（另一篇為 True/False Verification, 2509.09958）。

---

## 一、書目資訊

- **標題**：VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension
  - 注意：縮寫 VIRO 在論文裡實際展開為 **Verification-Integrated Reasoning Operators**（標題用的是另一個 backronym 的全稱，但摘要與內文明確定義 VIRO = Verification-Integrated Reasoning Operators）。
- **arXiv**：2601.12781（v2，標頭顯示 `[cs.AI] 20 Mar 2026`）
- **作者 / 單位**：Hyejin Park、Junhyuk Kwon、Suha Kwak、Jungseul Ok（†通訊）— POSTECH（南韓浦項工科大學）
- **程式碼**：https://github.com/ml-postech/VIRO-neuro-symbolic-reasoning-with-verification
- **推測場合**：CVPR 2026 等級（neuro-symbolic compositional REC 主流會議圈，與 NAVER@ICCV、HYDRA 同梯）。
- **一句話定位**：在「LLM 把 query 拆成符號程式 → 逐步執行」的 compositional REC 流程裡，於**每個 operator 內嵌輕量驗證器**，驗證不通過就回傳空集合 ∅ 並提早終止，從而能顯式處理「圖中根本沒有 target」的 no-target case，不再被迫硬輸出一個框。

---

## 二、問題定義與動機

### 2.1 任務形式化（含 no-target）

傳統 REC 假設 query 描述的物件**一定存在**於影像中。VIRO 把輸出形式化為：

```
Y = B    若 I 中存在 target（B = (x,y,w,h) 像素座標框）
Y = ∅    否則（no-target，影像中沒有任何符合 query Q 的物件）
```

這正是 GREC / gRefCOCO 引入的「廣義 REC」設定（可能 0、1 或多個 target），但 VIRO 在本文只聚焦 **0-target（no-target）vs 1-target** 的二元軸，沒有處理 multi-target。

### 2.2 核心痛點：cascading error → forced prediction（強迫輸出）

論文點名 compositional / neuro-symbolic REC 的根本失效模式：

1. **中間步驟被假設為正確**：ViperGPT / VisProg / HYDRA / NAVER 這類方法把 query 拆成 object detection、attribute filtering、spatial reasoning 等步驟，但每一步的中間輸出（proposals、屬性匹配、空間關係）**被預設正確、直接往下傳，沒有顯式檢查**。
2. **OVD 的高信心 false positive**：open-vocabulary detector（GroundingDINO / GLIP）即使在圖中沒有該物件時，仍會對「視覺或語意相似」的區域給出高信心誤檢（object hallucination）。
3. **錯誤級聯（cascading error）**：上述 FP 沿 reasoning chain 一路傳播，最終在 no-target 情形下，pipeline 被「強迫」從這些 FP 中挑一個當答案 → 產生高信心的錯誤框。
4. **效率/擴展性問題**：(i) 不少系統把重量級 multimodal LLM 放在 inner loop → 高延遲；(ii) 程式生成與執行**耦合**，video / 多影像時要對每張影像重新生成程式 → 計算量隨影像數線性增長。

**Figure 1 的兩個例子**：query「the person to the left of the elephant」中圖裡根本沒大象 → 先前方法被迫輸出框；VIRO 的 FIND 發現「圖裡沒有 elephant」、或 FIND_DIRECTION 發現「person 並不在 elephant 左邊」→ 提早終止、回傳 no-target，而非幻覺一個答案。

### 2.3 三大貢獻

1. 把 **forced prediction** 指認為 compositional vision-language reasoning 的根本失效模式，並用 verification-integrated operators 對症。
2. 設計 **operator-level program**：每個 operator 同時「執行 reasoning」與「自我驗證」，用 uncertainty-based + logic-based 檢查抑制 error cascade，並達成顯式 no-target 偵測。
3. zero-shot 下在 gRefCOCO no-target split 展現強 robustness，並在標準 REC（RefCOCO/+/g、對抗 RefAdv、egocentric RefEgo）達 compositional baseline 中的 SOTA；藉「程式合成與執行解耦」實現 1-query–N-images 的良好擴展性。

---

## 三、方法詳解

### 3.1 整體：兩階段 neuro-symbolic pipeline

```
Q (自然語言 query)
  └─[Pre-execution stage]─> LLM 生成符號程式 P = (o1, o2, ..., oT)  →  Program Validator 文法校驗 / 自我修正
       └─[Execution stage]─> Interpreter 在影像 I 上逐步執行 operator，每步自我驗證；
              任何 operator 回傳 ∅ → 立即 early-exit → 該圖判為 no-target
              全部通過 → RESULT 把最終候選映成答案框 B
```

**關鍵設計**：解耦（decoupled）。程式只對 query 生成「一次」，可在 N 張影像上重複使用：`T_total = T_pre + N × T_exec`；而 HYDRA / NAVER 把合成與執行糾纏，每張圖都重生程式：`T_total = N × T_pre + N × T_exec`。

### 3.2 Verification Reasoning Operators (VROs)

定義一組有限的原語 operator 集合 𝒪。每個 operator **既執行一個 reasoning 動作，也驗證自己的輸出**；若驗證條件不滿足，回傳空集合 ∅（觸發整條 pipeline early-exit）。四類：

- **Identification**：`FIND`（偵測候選）、`PROPERTY`（依屬性 refine 實體）
- **Absolute spatial**：`LOCATE`、`SIZE`、`ORDER`、`ABSOLUTE_DEPTH`（絕對位置/尺度/排序/深度）
- **Relative spatial**：`FIND_DIRECTION`、`FIND_NEAR`、`FIND_INSIDE`、`RELATIVE_DEPTH`（多實體間空間關係）
- **Termination**：`RESULT`（把選定物件映入答案空間，程式必須以此結尾）

所有 operator 都設計成回傳「一組已驗證的 bounding box」或在條件不滿足時回傳 ∅。

#### 兩種驗證模組

**(A) Uncertainty Verification (UV) — 在 FIND operator 裡**

- `FIND(object_name=l)` 先叫 OVD `D`（GroundingDINO 或 GLIP）在影像上以標籤 `l` 產生候選 `{B_j}`。
- 為了過濾 OVD 的高信心 FP，加一個**輕量 CLIP-based 二元驗證**：對每個候選 crop 出區域 `I_j`，預先準備 K 個常見類別 `C = {c1...cK}` 當作 **負錨點（negative anchors）**，驗證分數是「target `l` vs 每個 `c_k` 一對一比較」的平均勝率：

```
S(l | I_j) = (1/K) Σ_k  exp(sim(I_j,l)/τ) / [ exp(sim(I_j,l)/τ) + exp(sim(I_j,c_k)/τ) ]
```

- 只有 `S(l|I_j) ≥ δ_l` 才接受該候選為 `l`。`δ_l` 可為固定門檻或**per-label 自適應門檻**。
- **門檻的取捨**：太接近 0.5 → TP 被誤殺或 FP 漏網；又因 CLIP 對訓練常見類別（person、car）有偏，固定門檻對各類別不公平。

**(B) Logical Verification (LV) — 在 FIND_DIRECTION operator 裡**

- `FIND_DIRECTION(object, reference_object, direction)`：對所有輸入候選做**幾何測試**，驗證每個 object proposal 是否相對於至少一個 reference_object 滿足指定的空間關係（左/右/上/下…）。不滿足者剔除；全空則回 ∅。
- 即「空間關係不是假設成立，而是逐一幾何檢查」。

### 3.3 自適應門檻校準（Adaptive Threshold，Appendix A.1.3）— **這是和 CRS 最易混淆、最需釐清的一段**

- 用 **ImageNet 當輔助校準資料**：每類取 5 張、共 5,000 張，用 GroundingDINO crop 出該類物件（記為 `D_A`），降低背景偏差。
- 對每個 target label `l`，在 `D_A` 上算出該類的驗證分數分布 `S(l | D_A)`。
- **Top-k% 選擇策略**：把分數排序，門檻 `δ_l` 取「最高分的 top-k%」對應的分數（k 通常設 10）。直覺是「捕捉 CLIP 對該類別在高信心預測時穩定達到的水準」，給一個 data-driven 的 per-label 門檻。
- 對 k 不敏感（Balanced Accuracy 穩定）。

> **重點**：這是一個**啟發式、per-label 的分位數門檻調參**，目的是抵銷 CLIP 的類別偏差。它**不是** distribution-free 的 conformal calibration，**沒有任何有限樣本的覆蓋保證或風險上界**；ImageNet 也只是用來估「CLIP 對某類的典型信心範圍」，不是用來控制 no-target 的錯誤率到某個 α。下面第五節會逐項對照。

### 3.4 Program Generation + Validation（Appendix A.2）

- **生成**：用 LLM 做 few-shot（仿 VisProg），`P = LLM(Q | m)`。主用 **Qwen2.5-72B-Instruct-AWQ**（強 code-gen，與 ViperGPT 公平比較）；也可用 GPT-4o/mini、Llama3.1。
- **驗證器（Validator）**：強制嚴格文法 `VAR = OP(ARG=..., ...)`，結尾必為 `FINAL_RESULT = RESULT(object=VAR)`。檢查 syntax、變數追蹤、引數型別、operator 約束、輸出格式。失敗→回診斷訊息給 LLM 重生（最多 5 次）。這使 program failure rate 壓到 ≤0.3%（相對 HYDRA/NAVER 動輒 20–35% 失敗率）。

### 3.5 是否 training-free / backbone

- **是 training-free / zero-shot**：所有元件皆現成預訓練模型，無 REC fine-tune（自適應門檻只用 ImageNet 統計，不訓練）。
- **Backbone 組合**：OVD = GroundingDINO-T 或 GLIP-L；驗證/屬性 = CLIP（ViT-H/14，輕量替代 ViT-L/14）；深度 = DepthAnything v2；程式生成 = Qwen2.5-72B（或 GPT-4o/Llama3.1）。

---

## 四、實驗

### 4.1 資料集與指標

- **No-target**：**gRefCOCO no-target split**（只取「圖中無對應物件」的 query；負例被約束成「與圖語意相關」以避免 trivial negative）。
- **標準 REC**：RefCOCO / RefCOCO+ / RefCOCOg（TestA=person、TestB=object）。
- **對抗語言**：RefAdv（Ref-Hard 的結構依賴 OOD 子集）。
- **影片/第一人稱**：RefEgo（Ego4D，含 target-present 與 no-target frame，天然對應 1-query–N-images）。
- **可擴展到 VQA**：GQA（附錄）。

**指標（關鍵，與 CRS 對比重點）**：

- **Balanced Accuracy = (TPR + TNR)/2**。
  - **TPR = TP/(TP+FN)**，等同標準 REC 的 **Acc@0.5**（IoU>0.5）。
  - **TNR = TN/(TN+FP)**，即 **no-target accuracy（N-acc）**。
- 混淆矩陣定義：TP=有 target 且 IoU>0.5；TN=無 target 且正確判 ∅;FP=無 target 卻輸出框;FN=有 target 卻判 no-target 或 IoU<0.5。
- RefEgo 另用 **mSTIoU** 與 **ACC@0.5+n**（frame 層級，target-present 看 IoU>0.5、no-target 看正確判 ∅）。

> **重要觀察（對 CRS 切割直接相關）**：VIRO **並未採用 GREC 官方的 `Pr@(F1=1)` / N-acc / T-acc 這套三元 GREC metric**。它把 gRefCOCO 重新框成一個**二元分類**問題（target-present vs no-target），只用 TPR/TNR/Balanced Accuracy，且 **no-target 與 target-present 是用不同 split 各自評估後合併**（gRefCOCO no-target split 量 TNR、RefCOCO 量 TPR）。也就是說，VIRO 的「no-target 評估」本質是 **點估計的分類正確率**，沒有覆蓋/風險的統計保證描述。

### 4.2 主結果

**Table 2（gRefCOCO no-target + RefCOCO TestA/TestB）— Balanced Acc / TNR / TPR**：

| 類別 | 方法 | TestA Bal.Acc | TestA TNR | TestA TPR | TestB Bal.Acc | TestB TNR | TestB TPR |
|---|---|---|---|---|---|---|---|
| 全監督 | Qwen2.5-VL-72B-AWQ† | 69.5 | 47.3 | 91.7 | 66.8 | 45.1 | 88.4 |
| 全監督 | GREC-MDETR-R101 | 62.0 | 34.5 | 89.6 | 56.2 | 31.0 | 81.4 |
| 全監督 | GREC-UNINEXT-R50 | 70.4 | 49.3 | 91.5 | 67.6 | 48.2 | 86.9 |
| Proposal | ReCLIP | 23.5 | **0.0** | 47.0 | 22.6 | **0.0** | 45.2 |
| Proposal | SS-CLIP | 33.3 | **0.0** | 66.5 | 27.5 | **0.0** | 54.9 |
| Proposal | GroundVLP | 30.7 | **0.0** | 61.3 | 21.8 | **0.0** | 43.5 |
| Detector | GLIP-L | 37.2 | 21.7 | 52.6 | 30.0 | 18.2 | 41.8 |
| Detector | GroundingDINO-T | 40.0 | 22.8 | 57.2 | 29.6 | 16.0 | 43.2 |
| Compositional | ViperGPT | 33.4 | 0.2 | 66.7 | 27.4 | 0.1 | 54.6 |
| Compositional | HYDRA | 35.2 | 7.5 | 62.8 | 34.7 | 7.0 | 62.4 |
| Compositional | NAVER | 33.8 | 3.4 | 64.2 | 30.0 | 1.8 | 58.2 |
| **本文** | **VIRO** | **61.1** | **50.2** | 71.9 | **56.9** | **52.9** | 60.8 |

要點：proposal-based 因「必選一框」TNR≈0；compositional baseline 雖有 reasoning 但 TNR 仍只有個位數（ViperGPT≈0.1–0.2、NAVER 1.8–3.4）。VIRO 把 TNR 拉到 50% 以上、且 zero-shot 下 Balanced Acc 逼近甚至超越部分全監督 GREC 模型（如 GREC-MDETR），但其 TPR（71.9/60.8）明顯低於全監督（88–91）——即它用「敢棄答」換了 no-target robustness，代價是 target-present recall 下降。

**Table 3（標準 REC 效率）**：VIRO program failure rate ≤0.3%（HYDRA/NAVER 為 28–36%）；execution runtime 最低（Exec. 0.71s/query），E2E 12.92s（多半是 72B LLM 的 pre-execution）。Acc@0.5（TestA）RefCOCO 71.9 / RefCOCO+ 63.3 / RefCOCOg 66.6，在 compositional baseline 中最高。

**Table A2（gRefCOCO no-target，純 TNR）**：VIRO 在 GroundingDINO 下 Val/TestA/TestB = 56.5/50.2/52.9，遠勝 NAVER(3.0/3.4/1.8)、HYDRA(8.6/7.5/7.0)、ViperGPT(≈0.3)，甚至勝過全監督 GREC-UNINEXT(50.6/49.3/48.2)。

**Ablation（Table 5）**：Detector-only(Bal 40.0) → +Operators(56.8) → +LV(57.0) → +UV fixed(58.8) → +UV adaptive(**61.1**)。可見 no-target robustness 的主要增益來自 compositional operators 本身 + UV 的 adaptive 門檻（TNR 從 22.8→50.2）；LV 貢獻較小。OVD 門檻取 0.2（偏高 recall）。

**RefEgo（Table 4）**：VIRO mSTIoU 22.8、ACC@0.5+n 51.9，在 zero-shot 中最佳，ACC@0.5+n 甚至超過全監督 MDETR+BH(51.1)。

**Forced prediction（Table A1）**：把 abstention 關掉、強迫輸出時，VIRO 標準 REC TPR 也具競爭力（如 RefCOCO TestA 75.0–75.7），說明棄答不是靠犧牲 grounding 換來的。

---

## 五、與 CRS 的逐項 diff（最重要）

> CRS = 我的碩論：Cross-Base Conformal Referring Set。OWL-ViT 當 gate（控 abstention/no-target）+ GroundingDINO 出 box（控 compact set），用 **LTT（Learn-then-Test）做 distribution-free 校準**，在 α=β（含 R3 三風險 Bonferroni）下對 **recall + abstention 同時給有限樣本風險上界**，輸出一個**conformal referring SET**（多框集合，帶覆蓋保證），錨 gRefCOCO/RefCOCO、用官方 GREC metric。

| 維度 | VIRO | CRS（本論文） |
|---|---|---|
| **核心機制** | symbolic 程式 + operator 級「驗證」（CLIP 二元判別 + 幾何邏輯檢查），驗證不過就回 ∅ | post-hoc **conformal composition**：對 frozen base 的分數做分布無關校準，輸出風險受控的集合 |
| **no-target 判定依據** | operator 回傳空集合即 early-exit；本質是「CLIP 分數 < 自適應門檻」或「幾何關係不成立」 | OWL-ViT gate 在 LTT 校準下，當且僅當無候選通過風險受控門檻時 **abstain**；abstention 是一個**被風險上界約束的決策** |
| **是否有統計保證** | **無**。Top-k% / ImageNet 門檻是啟發式調參，**沒有有限樣本覆蓋或風險上界**；TNR 是事後點估計，沒有「保證 N-acc ≥ 1−α」這類 claim | **有**。LTT 給 **distribution-free, finite-sample** 的風險上界；R1/R2（含 R3）CI 上界皆 <0.3，可寫成「P(風險 > α) ≤ δ」 |
| **pipeline 結構** | **single pipeline**（單一 OVD + 單一 CLIP 驗證 + DepthAnything），所有 operator 都掛在這條鏈上 | **cross-base composition**：OWL-ViT 與 GroundingDINO 是**兩個獨立 base**，分工（gate vs box），2×2 ablation 證 factorization 非 ensemble |
| **輸出型態** | 單一 box B 或 ∅（**point prediction**，二元） | **referring SET**（0..K 個框的集合）+ 覆蓋保證；本質是 set-valued prediction |
| **是否有 referring set / 覆蓋保證** | **無 set、無覆蓋保證**；只輸出一個框或棄答 | **有**：set size 受控（α=β=0.3 下 val/testA/testB ≈ 3.21/2.02/3.48 框），且 size CI 與純 OWL 分離 |
| **評估指標** | Balanced Acc / TPR / TNR（自定二元混淆矩陣，**未用官方 GREC Pr@(F1=1)**），no-target 與 present 分 split 各自量 | **官方 GREC metric**（Pr@(F1=1) / N-acc / T-acc）+ 風險 CI + size CI；no-target 在同一統計框架內 |
| **multi-target** | 不處理（只 0/1 target 二元軸） | GREC 廣義設定，可涵蓋 0/1/多 target（set 天然支援多框） |
| **訓練/校準資料** | ImageNet（5k 張）估 per-label CLIP 門檻——**用於去 CLIP 偏差，非控錯誤率** | gRefCOCO/RefCOCO 校準集——**用於 LTT 風險控制**（calib-only grid、修 leakage） |
| **是否 training-free** | 是（zero-shot，無 REC fine-tune） | 是（post-hoc、frozen base，無 fine-tune）——**這點兩者相同，更需在「保證」軸劃界** |
| **跨 base 轉移** | 無此 claim（換 OVD 是 robustness 比較，非轉移保證） | C4：raw consistency zero-shot 跨 base 轉移≈native；composition 跨 base 校準 |

**一句話差異**：VIRO 和 CRS 都「post-hoc、training-free、針對 no-target/abstention、都用 CLIP/GroundingDINO」，表面高度重疊；**但 VIRO 的 abstention 是一個沒有統計保證的啟發式門檻決策、輸出單框、用自定二元指標**，而 CRS 的 abstention 是**分布無關、有限樣本風險上界約束下的集合值決策（conformal referring set），用官方 GREC metric 評估**。VIRO 是「更聰明地猜要不要棄答」，CRS 是「以可證明的風險上界保證棄答/召回」。

---

## 六、論文該怎麼引用 / 切割（精準切割文字）

VIRO 直接踩在 CRS 的 no-target/abstention 軸上，是 related work 與 motivation 都必須點名、且要明確劃清「保證 vs 啟發式」的一篇。建議：

### 6.1 Related Work 段落（中性引介 + 切割）

> 「近期 neuro-symbolic REC 已開始正視 no-target / forced-prediction 問題。VIRO [cite] 在 compositional pipeline 的每個 operator 內嵌輕量驗證（CLIP-based uncertainty verification 與幾何 logical verification），當驗證失敗即回傳空集合並提早終止，從而顯式偵測 no-target，並在 gRefCOCO no-target split 上大幅提升 no-target accuracy（TNR）。然而 VIRO 的棄答決策來自一個**啟發式的 per-label CLIP 門檻**（以 ImageNet top-k% 分位數校準以抵銷 CLIP 類別偏差），其 no-target 正確率僅以**事後點估計（Balanced Accuracy / TNR）**報告，**並未提供任何有限樣本的覆蓋或風險保證**；其輸出亦為單一框或空集合的 point prediction，而非帶覆蓋保證的集合。」

### 6.2 與本文（CRS）的定位差異（緊接著寫）

> 「相對地，本文不修改 base 模型、也不依賴啟發式門檻，而是在 frozen base 之上以 Learn-then-Test 做**分布無關、有限樣本**的校準，使 abstention（no-target）與 recall 同時受**可證明的風險上界**約束，並輸出一個帶覆蓋保證的 **conformal referring set**。換言之，VIRO 與 True/False Verification [cite] 解決的是『**如何更可靠地判定要不要棄答**』，而本文解決的是『**棄答與召回的錯誤率如何被統計上界所保證**』；兩者在『verification 機制是否提供分布無關保證』、『輸出是 point prediction 還是 set with coverage』、『評估是否在官方 GREC 風險框架內』三點上正交。」

### 6.3 防審稿人「你和 VIRO 差在哪」的一段（rebuttal-ready）

> 三條硬切割，逐條釘死：
> 1. **保證類型**：VIRO 的 adaptive threshold 是去偏的啟發式，無 finite-sample 保證；CRS 用 LTT 給 `P(risk > α) ≤ δ`（R1/R2/R3 Bonferroni）。
> 2. **輸出語意**：VIRO 輸出單框/∅；CRS 輸出 size 受控、覆蓋受控的 referring set（size CI 與純 OWL 分離）。
> 3. **架構**：VIRO 是 single pipeline；CRS 是 cross-base composition（OWL gate × GD box），2×2 ablation 證明是 factorization 而非 ensemble。
> 另可補：VIRO 用自定二元 TPR/TNR（未用官方 GREC Pr@(F1=1)/T-acc），且不處理 multi-target；CRS 全程在官方 GREC metric 與 multi-target set 下評估。

### 6.4 引用時務必避免的誤述

- 不要說 VIRO「校準了 no-target 風險」——它只是調了一個去偏門檻。
- 不要說 VIRO 用 conformal / 覆蓋保證——它完全沒有。
- 不要把 VIRO 的 TNR 當成「保證值」——那是 split 上的點估計。
- 注意：VIRO 用的是 gRefCOCO 的 **no-target split + 自定 Balanced Acc**，不是官方 GREC 三元指標——比較數字時不可直接和 CRS 的 Pr@(F1=1) 並列，需註明指標口徑不同。

---

## 七、個人評價

**優點**

- 把「forced prediction / cascading error」這個 compositional REC 的痛點講得很清楚，Figure 1 與 ablation（Detector-only 40.0 → +UV adaptive 61.1）把 no-target 增益歸因得乾淨。
- operator-level 驗證 + program validator 的工程價值高：program failure rate ≤0.3% vs HYDRA/NAVER 20–35%，是實打實的可靠性提升。
- decoupled 設計在 1-query–N-images / video（RefEgo）上的擴展性論證紮實，這是對機器人視覺搜尋等落地場景真正有用的角度。
- training-free、backbone 都是現成模型，復現門檻相對低（雖然要 72B LLM）。

**侷限（也正是 CRS 的切入空間）**

- **abstention 無統計保證**：整套 no-target 判定建立在一個啟發式 per-label 門檻上，沒有覆蓋/風險上界，TNR 是事後點估計。一旦分布偏移（RefEgo 上 TNR 與 RefCOCO 不同），它沒有任何形式保證能撐住——這正是 conformal 框架的價值所在。
- **單框輸出、不處理 multi-target**：在真正的 GREC（0/1/多 target）設定下表達力受限；CRS 的 set-valued 輸出在這點上是結構性優勢。
- **指標自定**：用 Balanced Acc / TNR 而非官方 GREC Pr@(F1=1)，雖然直觀，但與 GREC 主線文獻不完全可比，也讓「no-target 到底控到多嚴」缺乏統一口徑。
- **TPR 明顯低於全監督**（71.9 vs 88–91）：敢棄答的代價是 recall 掉，但因為沒有風險框架，使用者無法在「我要 recall ≥ 某值」與「no-target 錯誤 ≤ 某值」之間做有保證的取捨——CRS 的 α/β 雙旋鈕正好補這個洞。

**對 CRS 的威脅評估**：VIRO 是「同軸最近競品」，會搶走「我們是第一個正視 compositional REC no-target」這種 novelty 措辭，但**完全沒碰到 CRS 的核心 novelty（distribution-free finite-sample 保證 + conformal referring set + cross-base composition + 官方 GREC 風險評估）**。只要在 related work 與 rebuttal 把「啟發式門檻 vs 可證明風險上界」「point prediction vs set with coverage」「single pipeline vs cross-base」三刀切清楚，VIRO 對 CRS 的 abstention novelty 不構成實質威脅，反而是極好的 motivation 對照組與 baseline 候選（可在 GREC metric 下重評它，凸顯它沒有保證）。
