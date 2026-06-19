# VLM-as-Decision for Risk-Controlled Referring Sets — 11 次探索的完整存檔

> 日期：2026-06-16　分支：feature/bts-poc-v4　硬體：spark GB10
> 資料：gRefCOCO，spark `~/selective-grounding/dump/`，模型 Qwen2.5-VL-3B / 7B
> 目的：誠實記錄「為 CRS 候選做更好的選擇 / 用 VLM 當決策層」這條延伸路線的**完整負結果**，供徵詢外部意見（GPT）與防止未來重走。
> **定位**：這不是「失敗」的紀錄，而是一個被 11 層證據釘死的**邊界**——「在 gRefCOCO multi-target 上，沒有任何 per-candidate 打分/選框範式（frozen 訊號或 VLM）能吃掉 oracle gap」。這個「別人也做不到」的事實本身就是研究價值，且回頭強化了 CRS「不追求選準、改追求有保證涵蓋」的設計哲學。

---

## 0. 背景：這條路要解的問題

已完成的 Frozen CRS（Cross-Base Conformal Referring Sets）有一個明確、有保證可達、但未被吃掉的 headroom：

| | R1（answered-target FNR，size=k） |
|---|---|
| 現況：raw GDINO score top-k | 0.24–0.31 |
| **oracle 排序（同 size=k）** | **0.006–0.019** |

CHECK 1b（前一份 handoff `cardinality_candidate_feasibility_handoff.md`）證明：候選池裡**存在**能把 R1 壓到接近零的框，只是排不出來。這個 0.27→0.007 的 gap 就是目標。

本文件記錄：用各種訊號源（免費 by-products + VLM）嘗試吃掉這個 gap 的 **11 次嘗試，全部失敗**。

---

## 1. 第一階段：免費訊號（6 次，零 GPU，見前份 handoff）

| # | 訊號 | per-query AUROC | 接回 top-k R1 | 結論 |
|---|---|---|---|---|
| 1 | oracle-k 上界 | — | raw 0.24–0.31 / oracle 0.006–0.019 | gap 在 ranking 不在 cardinality |
| 2 | frozen→k 可學性 | — | — | cardinality 難學、天花板低 |
| 3 | candidate global/pooled AUROC | 0.73–0.76 | — | **pooled 陷阱，假 GO** |
| 4 | learned p_match 接 top-k | — | 0.30–0.39（比 raw 差） | naive top-k 傷覆蓋 |
| 5 | per-query AUROC | learned 0.73 / raw 0.69 | — | 排序真有 gain 但來自反多樣性特徵 |
| 6 | + cross-base agreement + paraphrase（免費） | 0.77–0.79 | 0.38（仍比 raw 0.27 差） | AUROC↑ 但覆蓋不動 |

**第一階段結論**：免費訊號的 per-query AUROC 天花板 ≈ 0.79，但**接回 top-k 的 R1 永遠壓不過 raw score 的 0.27**。核心病灶：最具資訊量的特徵（overlap_count 擁擠度）反多樣性——把同一個 GT 周圍的密集框排前面，漏掉孤立的 GT。AUROC（排序）≠ set coverage（覆蓋不同 GT）。

---

## 2. 第二階段：VLM（5 次，GPU，本文件重點）

第一階段封死「免費訊號」後，唯一還沒堵死、且理論上能分辨「不同 GT」（語義）的訊號源是 VLM。所有 VLM 嘗試都在 **val 的 eval 半（ref_id 奇）前 400 個 target-present query** 上做，與 CRS 切分一致。

### 設計演進與「不要二次詢問」的紀律

使用者明確要求：**VLM 要當決策層本身，不是事後逐框 verify（那是 True/False Verification 2509.09958 的浪費做法）。** 因此設計朝「一次 forward 看全局 + 所有候選」演進。

| # | 做法 | per-query AUROC / R1 | 結論 |
|---|---|---|---|
| 7 | 裸 crop 逐框問 yes/no | AUROC 0.46 | crop 失去上下文（重蹈 BTS CLIP-crop≈random 覆轍）；且是逐框=二次詢問，違反紀律 |
| 8 | Set-of-Marks（整圖畫紅框標號，問「哪一個號碼」），抽號碼 logit | AUROC 0.52 | 仍隨機 |
| 9 | Multi-label SoM（問「列出所有符合號碼」），抽生成機率 | AUROC 0.47 | 仍隨機 |
| 10 | **診斷 5 query**（人眼看標註圖 + VLM 文字回答） | argmax 5中4 | **揭穿關鍵矛盾** |
| 11a | VLM 生成號碼集合**直接當預測集合**（3B） | R1 0.42 / prec 0.52 / size 3.55 | 比 raw 全面差 |
| 11b | 同上**換 7B** | R1 0.42 / prec 0.52 / size 2.92 | **與 3B 幾乎相同，排除「模型太小」** |

### 診斷（#10）揭穿的關鍵矛盾——必看

5 個 query 人眼檢查：VLM 的 **argmax（最高分號碼）5 個對 4 個**（答 5/7/8/8 都在 GT 集合裡）。一個 argmax 80% 準的模型，AUROC 不該是 0.47。矛盾的解釋：

> gRefCOCO 的 multi-target 幾乎全是**「A and B」複合指稱**，GT 動輒 `[1,2,3,4,5]`、`[1,3,5,6,7,9]` 一次對好幾個框。
> - 「選一個號碼」的問法本身錯了：答案是一**組**，不是一個。argmax 能對（押中組裡某一個），但完整集合召回差。
> - 抽機率（#8/#9）：機率質量被分散/autoregressive 衰減，扭曲，AUROC 隨機。
> - 直接用生成集合（#11）：precision 僅 0.52，**VLM 選的框一半是錯的**，7B 也一樣。

### VLM 失敗的根因（看實例就懂，非 harness bug）

實際 expression 範例（解析正確、人眼可驗）：
- `right bottom person and right guy above child` — GT[1,2,3,4,5]，7B答[5,6,7,8,9]，只對1
- `the bottom right slice of sandwich and the spoon next to the bowl of soup` — GT[1,3,5,6,7,9]，7B答[6,7]，對2漏4
- `PARTIAL PARTS OF THE TABLE ABOVE THE SOUP and food` — GT[1,8]，7B答[8,9]

這些是**刁鑽的複合空間指稱 + 標註噪音（大小寫混亂、partial parts 這種模糊描述）**，連人都要想。3B→7B 無改善（R1 0.415→0.420，precision 0.524→0.523）證明：**這不是模型容量問題，是「VLM 看標號圖選框」這個範式對 gRefCOCO multi-target 本質不友善。**

---

## 3. 最終判決

**11 次嘗試（6 免費訊號 + 5 VLM 含 7B），沒有一個贏過 GDINO raw score 的 R1≈0.27。**

> **在 gRefCOCO multi-target 上，用「為候選框打分 / 選框」這個範式吃掉 oracle gap（0.27→0.03），無論訊號來自 frozen by-products 還是 VLM（含 7B），都做不到。** 此結論有 11 層證據，非常硬。

這個邊界回頭**強化 CRS 的價值**：CRS 不靠「選對框」，而用 LTT 給**有保證的集合**。在「沒人能把候選選準」這個被驗證的現實下，CRS「不追求選準、改追求有保證地涵蓋三風險」的設計哲學是正確的——這正是「別人做不到、我們用不同框架繞過」的論文價值點。

---

## 4. 唯一還沒試、且方向正交的角度（給 GPT 評估）

11 次全都在問「**選哪個框**」（candidate selection / scoring）。但有一個正交、且利用 VLM 真正獨特能力的角度從未試過：

> **瓶頸可能不在「選框」，而在「這句話指了幾個、指什麼」的語言理解。**
> gRefCOCO 的 multi-target = 「A and B」複合句。也許 VLM 該做的不是看標號圖選框，而是先**把 expression parse 成子指稱**（"right bottom person and right guy above child" → ["right bottom person", "right guy above child"]），**每個子指稱再各自交給 frozen detector 定位**，最後 union 成集合接回 CRS/LTT。
> - 這利用 VLM 真正強、detector 完全沒有的能力：**語言結構分解**，而非視覺選框（後者 detector 本就會）。
> - 它把「為什麼 raw score 選不準」的根因（一個 query 含多個語義不同的目標，detector 用單一 expression embedding 無法同時對齊）直接拆解掉。
> - 風險：(a) 子指稱定位仍是單目標 REC，可能落回 CLIP-VG/GDINO 的單目標能力（那是已知可用的 0.84）；(b) parse 錯誤會傳播；(c) 需驗證「子指稱 union」在 CRS 三風險下的表現。

**想請教 GPT**：
1. 這個 expression-decomposition 角度，是否比「candidate selection」更可能吃掉 gap？還是會落回 detector 的單目標天花板、換湯不換藥？
2. 11 次 candidate-selection 負結果，本身夠不夠成為論文裡一個有價值的 finding（「per-box scoring/VLM selection 對 compositional referring 無效，need decomposition」）？還是只是 supporting 的 ablation？
3. 有沒有第三條我們沒想到的、能利用「別人做不到」這個 gap 的路？（例如：不選框、不分解，而是改變 CRS 的候選池建構本身？）

---

## 4b. Expression-Decomposition 上界檢查（2026-06-16，零 GPU，強 GO）

回應 §4 的正交角度，先用現有 dump 測 decomposition 的**理論上界**（不用 VLM）。設計三條線夾出 decomposition 能買到什麼：

- **L1 raw top-k**（現況）：單一 ranking 一次選 k 個。
- **L3 完美分解 + 分數定位**：對每個 GT，在「覆蓋它的候選框」裡挑 GDINO **分數最高**的。模擬「decomposition 告訴你該找幾個子指稱，定位仍只用 detector 分數」。
- **L2 完美分解 + 完美定位**：對每個 GT 挑 IoU 最高的框（絕對上界）。

| | val | testA | testB |
|---|---|---|---|
| pool 對每個 GT 召回率 | 0.994 | 0.989 | 0.981 |
| L1 raw top-k R1（現況） | 0.261 | 0.239 | 0.308 |
| **L3 完美分解+分數定位 R1** | **0.006** | **0.011** | **0.019** |
| L2 完美分解+完美定位 R1 | 0.006 | 0.011 | 0.019 |

**關鍵發現（推翻原本「子定位會落回弱 detector」的擔憂）**：

> **L3 = L2。** 一旦知道「該找幾個子指稱、各自是什麼」，即使定位仍用 GDINO 原始分數，R1 也到 0.006——跟完美定位一樣。原因：detector 分數分不開的是「多個 GT 混在一起時誰是誰」，不是「定位單一目標」。decomposition 把「一次選 k 個」拆成「k 個獨立單目標選擇」，每個只需選一個，detector 分數在單目標子問題內就夠用。

這正是前 11 次 candidate-selection 失敗根因的**鏡像**：raw top-k 爛（0.26）不是因為框不好（pool 對每 GT 召回 0.98–0.99），而是單一 ranking 被擁擠度帶向同一個 GT。decomposition 的結構性改變直接解掉這點。

**與 candidate-selection 上界的決定性差別**：candidate-selection 的 oracle 上界需要「知道哪個框對」這個作弊資訊；decomposition 的上界只需要「把句子拆成幾個子指稱」這個 **VLM 真正會做的事（純語言解析）**，拆完定位用現成 detector 分數即可。

→ **decomposition 上界 = 強 GO**（gain 0.23–0.29 可達，且有結構性理由相信可達）。唯一剩的不確定性 = VLM 拆解品質（下一個檢查）。腳本：`dump/decomp_ceiling.py`。

---

## 4c. VLM Expression-Decomposition 品質檢查（2026-06-16，7B 純文字，GO）

讓 7B 對 300 個 multi-target query 做**純語言**拆解（不看圖，只拆句子成子指稱 JSON list）。

| 指標 | 數字 |
|---|---|
| malformed（無法 parse） | **0 / 300** |
| n_parts == n_gt | **0.893** |
| n_parts within±1 | **0.983** |
| 實例人眼品質 | 8/8 合理 |

（`corr=0.000` 是假象：val 幾乎全 n_gt=2 無變異，相關係數算不出；看 n_parts==n_gt 0.89 與實例。）

**關鍵對比——同一批句子，VLM 從災難變乾淨**：
- 「right bottom person and right guy above child」：看標號圖選框（#11）→ 選框全錯；純語言拆解 → `['right bottom person','right guy above child']` ✅
- 「PARTIAL PARTS OF THE TABLE ABOVE THE SOUP and food」（大小寫亂、模糊）→ 正確拆成 `['PARTIAL PARTS OF THE TABLE ABOVE THE SOUP','food']` ✅

**證明分工假設**：VLM 不會「看圖選框」（視覺定位是 detector 的事），但**會「拆語言」**（語言結構是 VLM 的事）。前 11 次失敗是逼 VLM 做不擅長的；這次讓它做擅長的，一次就成。腳本：`decomp_quality.py`。

**decomposition 兩道門都過**：上界（§4b，R1 可達 0.006）+ 拆解品質（§4c，within±1 0.98）。這是 candidate-selection 從未到過的位置。剩最後一塊：end-to-end（VLM 拆 → 每個子指稱各自跑 detector → union → 實際 R1），需對子指稱重跑 GDINO。

---

## 4d. Expression-Decomposed CRS — End-to-End Pilot（2026-06-16，7B+GDINO，★突破★）

真實流程：7B 拆 expression → 每個子指稱當新 query 餵 GDINO 重跑 → union 候選 → 量 R1/size。val eval 半（ref_id 奇）200 個 multi-target query。

| 方法 | R1（漏檢） | set size |
|---|---|---|
| **full-expr top-k（現況範式）** | **0.335** | 2.00 |
| decomp top-1/子指稱 | **0.172** | 2.06 |
| **decomp top-2/子指稱** | **0.075** | 4.11 |
| oracle 上界（§4b） | 0.006 | — |

checkpoint 全程穩定（50/100/150/200：d2 R1 = 0.060/0.065/0.070/0.075）。avg n_gt 2.00，avg n_parts 2.06。

**意義**：
- **decomp top-1**：同 set size（2.06 vs 2.00），R1 從 0.335 砍半到 0.172——不增大集合、純靠 decomposition 選框的免費改善。
- **decomp top-2**：R1 壓到 0.075（降 78%，逼近 oracle 0.006），代價 size 翻倍到 4.11——正是 CRS/LTT 該收的：union 後交三風險校準 trim。

**17 次嘗試的轉捩點**：前 11 次 candidate-selection + 5 次 VLM-selection 沒一個贏過現況；第 17 次（decomposition）**首次大幅贏過**。機制乾淨：VLM 拆語言（品質 0.98）+ detector 單目標定位 + CRS 收風險。

**三道門全過**：上界（L3=L2，0.006 可達）→ 拆解品質（within±1 0.98）→ end-to-end（0.34→0.075）。證明瓶頸從來不是「定位」，是「單一 query 混多個語義目標」；full-expression 餵 detector 的 R1 天花板（0.27–0.34，被 16 次驗證）由 decomposition 直接繞過。

**注意（誠實邊界）**：此 pilot 用即時重跑 GDINO（threshold=0），其 full-expr 基線 R1=0.335 與主 CRS dump 的 0.27 不同口徑，**只能同腳本內相對比較**；尚未接 CRS/LTT 三風險校準、尚未測 no-target（R2）、尚未測 testA/testB、尚未處理 parse error 傳播。這些是 pilot→完整方法的待辦。腳本：`decomp_e2e.py`。

## 4e. Decomposed CRS — 接回 LTT 三風險主表（2026-06-18，★GO★）

> **紅隊裁決校準 claim（2026-06-18）**：matched recall@size 是 **mechanism evidence（Matched Recall-Size Frontier）**，**不是最終主證據**。最終主證據＝full split + LTT 三風險 + robustness。
> 安全 claim：「Candidate selection alone fails to close the oracle gap. Expression decomposition changes the candidate-pool construction process and improves the recall-size frontier for compositional multi-target queries. When wrapped by CRS/LTT, it can reduce answered-target FNR or set size while preserving no-target and deferral risk control.」
> **禁用**：「decisively solves the oracle gap」「決定性證據」。
> 紅隊 §9 待補實驗（見 `decomp_redteam_questions.md` + 任務清單）：robustness / recall-size 四指標 / strong full+NMS baseline / R_total / 成本表 / 失敗審計。

整夜 dump `gdino_gref_val_decomp.jsonl`（10405 行 = no_target 8905 passthrough + tp 抽樣 1500，其中 **255 真被拆解**）完成後，跑 `src/decomp_maintable.py`（共用 `crs_protocol.py`，calib-only grid，α=β=0.3 γ=0.5），在 `set(owl)&set(decomp)` 共同 key 上公平比較 Frozen vs Decomposed。

**val 主表（共同 key）：**

| 指標 | Frozen CRS | Decomp CRS | 方向 |
|---|---|---|---|
| set size | 3.24 [3.15,3.36] | 3.23 [3.06,3.41] | 持平 |
| R1（漏檢） | 0.191 [0.179,0.203] | **0.162 [0.142,0.183]** | ↓ ~15% |
| R2（誤含） | 0.159 [0.146,0.171] | 0.209 [0.196,0.222] | ↑（仍守 β=0.3）|
| defer | 0.424 [0.405,0.443] | **0.299 [0.264,0.331]** | ↓ ~30% |
| n_feas | 112 | 84 | — |

主表三疑慮（R1 CI 重疊、R2 反常上升、只 255/10405 真拆卻全域位移）由 **matched oracle 分析（`src/decomp_matched.py`，繞過 LTT，僅在 255 真拆 case 上比較）** 全數解消：

**同輸出 size 下 recall 大幅勝出（Matched Recall-Size Frontier，mechanism evidence，非最終主證據）：**

| 輸出 size | full-expr recall | decomp recall | Δ |
|---|---|---|---|
| ~2 | 0.639 | **0.811** | +17pp |
| ~3 | 0.697 | **0.899** | +20pp |
| ~4 | 0.741 | **0.938** | +20pp |

- **機制**：per-part GDINO query 給的 per-box 分數乾淨，threshold 時 GT-covering 框排前留得住；full-expr 分數被雜訊污染，同 size 留錯框。**與 LTT 無關，是 decomposition 本身功勞** → caveat #3（全域位移）否決。
- **R2 上升解釋**：LTT 看到 recall 變好 → 重新最優化到較鬆 operating point（換 defer 42%→30% 大降），花掉部分 R2 預算但守 β=0.3。划算交易，非缺陷 → caveat #2 解消。
- oracle pool ceiling 0.989→0.946（修剪 4.3pp），但那是 full pool 塞 45 框（threshold=0）下的天花板，實務無意義；可用 size 2-4 decomp 完勝。

**判讀＝GO（mechanism + 子集 LTT 支持）**。testA/testB decomp dump 已啟動整夜跑（`run_dd3_test.sh`，依序 testA→testB 1500），完成後跑 maintable 做跨 split 確認。

### 跨 split 確認（2026-06-18，testA/testB dump 完成後）

testA/testB decomp dump 跑完（testA 5948 行/138 真拆、testB 6173 行/123 真拆）後，主表與 matched 分裂成兩個必須分開講的故事：

**① 方法層（matched oracle，繞過 LTT）= 跨 split 全 GO，零例外：**

| split | size~3 full recall | size~3 decomp recall | Δ |
|---|---|---|---|
| val | 0.697 | 0.899 | +20pp |
| testA | 0.743 | 0.889 | +15pp |
| testB | 0.602 | 0.845 | +24pp |

size~2/~4 也全部同向勝出（testB size~2 達 +25pp）。**decomposition 本身的有效性跨三 split 穩固。**

**② 校準層（LTT 主表）在小子集上力不從心：**

| split | Frozen R1 | Decomp R1 | n_feas | 狀態 |
|---|---|---|---|---|
| val (10405行) | 0.191 | 0.162 | 84 | ✅ |
| testA (5948行) | 0.260 | 0.189 | **2** | ⚠️ 點估對但可行配置剩 2 |
| testB (6173行) | 0.168 | EMPTY | — | ❌ feasible 空 |

testA/testB 點估方向對（R1 降、defer 降），但 LTT 找不到（或幾乎找不到）同時守三風險的合法配置。**matched 證明這不是方法失效，是統計力問題**：子集太小（testB calib 砍半後 no-target 撐不起三風險聯合可行區間）。val 勉強撐住、更小的 testA/testB 就空了。

**結論**：方法層跨 split 全 GO（核心主張坐實）；校準層需要更大 calib set → 下一步必須跑**全量 dump**（解 GDINO threshold=0 慢，handoff 已列）。子集本就是 pilot。

腳本：`src/decomp_maintable.py`（主表）、`src/decomp_matched.py`（matched，吃 split 參數：`python src/decomp_matched.py {val,testA,testB}`）。

### 全量 val 結果（2026-06-18，875 真拆 vs 子集 255）— ★全量 LTT GO★

全量 dump（threshold=0，與 baseline 同口徑）val 完成後（14229 行、875 真拆），val-only LTT + matched：

| 指標 | Frozen CRS | Decomp CRS 全量 | 子集時 |
|---|---|---|---|
| set size | 3.24 | 3.24（持平）| 3.23 |
| R1（漏檢）| 0.191 | **0.165** ↓ | 0.162 |
| R2（誤含）| 0.159 | **0.158（持平！）** | 0.209 |
| defer | 0.424 | 0.424（持平）| 0.299 |
| n_feas | 112 | **112（健康）** | 84 |

**全量解決了子集唯一瑕疵**：子集時 R2 升 0.159→0.209（當時解釋為 LTT 換鬆 operating point 的代價）；**全量 R2 完全持平 0.158** → 證明那個 R2 上升是小樣本雜訊，非方法固有代價。全量 decomp = **純帕累托改善**（同 size/defer/R2，R1 白降 0.026），零 trade-off。matched 875 case 三 size 全勝（size~3: 0.704→0.897 +19pp），與子集幾乎一致，機制大樣本下穩定。

testA/testB 全量 dump 仍背景跑中（testA 19200 行、testB 16063 行，串跑 ~4-5h）。完成後重跑 maintable，預期 n_feas 由 EMPTY/個位數變健康（如 val）。

### ★全量三 split 定論（2026-06-18）★ — Decomposed CRS 跨 split LTT 純帕累托改善（最終主證據待 robustness 補全）

全量 dump 三 split 全完成（val 875 / testA 1243 / testB 974 真拆，threshold=0 同口徑）。完整三 split LTT + matched：

**LTT 主表（α=β=0.3, γ=0.5）：**

| split | Frozen sz/R1/R2/defer | Decomp sz/R1/R2/defer | n_feas | 方向 |
|---|---|---|---|---|
| val | 3.24/0.191/0.159/0.424 | 3.24/**0.165**/0.158/0.424 | 112/112 | R1↓餘持平 |
| testA | 2.02/0.260/0.203/0.443 | **1.96**/**0.244**/0.201/0.444 | 58/58 | R1↓且size↓(雙贏) |
| testB | 3.50/0.168/0.236/0.414 | **3.34**/0.163/0.236/0.414 | 56/56 | size↓R1微降 |

**三 split n_feas 全健康（112/58/56=與 frozen 同）**——子集時 testA n_feas=2 / testB EMPTY 完全消失，證實純小樣本統計力問題、非方法失效。全量下 decomp = **乾淨帕累托改善，零例外**：val R1 白降；testA R1 降且 size 更小；testB size 降。無任何 split R2 爆或 trade-off。子集所有瑕疵（R2 升、可行崩潰）全是小樣本假象。

**matched oracle 三 split（Matched Recall-Size Frontier，mechanism evidence，size~3）：** val 0.704→0.897 / testA 0.717→0.905 / testB 0.603→0.834（+19~23pp），大樣本與子集幾乎一致，機制穩固。

**結論：Decomposed CRS 跨 split 全量 LTT 純帕累托改善，是論文第二主結果的核心數字；但「最終主證據」地位需補齊 robustness（random5/image-disjoint）+ strong full+NMS baseline + R_total（紅隊 §9）。** 子集 pilot 備份在 `gdino_gref_{sp}_decomp_sub.jsonl`，全量在 `gdino_gref_{sp}_decomp.jsonl`。

### Subgroup 分析（2026-06-18，全量真拆 case，matched recall@size~3）

`src/decomp_subgroup.py`，繞過 LTT（分組後樣本太少 LTT 不穩），在真拆 case 上按 query 特性分組比 full vs decomp recall@size~3。三個論文級洞察：

**① 長句受益更大（機制因果證據，三 split 一致）：**

| split | short(<12詞) Δ | long(≥12詞) Δ |
|---|---|---|
| val | +0.181 | +0.241 |
| testA | +0.198 | +0.240 |
| testB | +0.136 | +0.253 |

句子越長 decomposition 收益越大 → 直接佐證「長句最易把 detector 搞混、最需要拆」。

**② 雙目標是甜蜜點，n_gt≥3 收益遞減（界定有效邊界）：**

| split | n_gt==2 Δ | n_gt≥3 Δ |
|---|---|---|
| testA | +0.239 | +0.163 |
| testB | +0.246 | +0.064 |

目標越多 → 子指稱越多 → union pool 越雜 → per-part 分數乾淨的優勢被稀釋。

**③ 唯一負信號已澄清＝標籤歧義非方法缺陷：** testA n_gt==1（61 例，佔 5%）decomp −0.148。逐案檢視這些 expression（如 "person in black pants far left **and** man in white shirt"、"right half of sandwich **and** sandwich on left"）發現它們語言上明明是雙指稱、routing 拆 2-part 正確；是 gRefCOCO 把它們標成 n_gt==1（GT 單框）造成「語言指稱數 vs GT 框數」錯配。非 routing 誤拆。

腳本：`src/decomp_subgroup.py`（吃 split 參數）。

---

## 5. 復現
- 第一階段腳本（spark `~/selective-grounding/dump/`）：`cardinality_check.py` `candidate_check.py` `check4.py` `check5.py` `check6.py`
- VLM 階段（spark `~/selective-grounding/`）：`vlm_check.py`(#7) `som_check.py`(#8) `som_multi.py`(#9) `som_debug.py`(#10) `som_setpred.py`(#11a 3B) `som_setpred_7b.py`(#11b 7B)
- 環境：`.venv`（uv 建，torch 2.12+cu130、transformers 5.11、accelerate 1.14）；VLM 權重 HF Qwen2.5-VL-3B/7B-Instruct
- 影像：`data/coco/images/train2014/`；切分 ref_id parity（eval=奇）
