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

## 5. 復現
- 第一階段腳本（spark `~/selective-grounding/dump/`）：`cardinality_check.py` `candidate_check.py` `check4.py` `check5.py` `check6.py`
- VLM 階段（spark `~/selective-grounding/`）：`vlm_check.py`(#7) `som_check.py`(#8) `som_multi.py`(#9) `som_debug.py`(#10) `som_setpred.py`(#11a 3B) `som_setpred_7b.py`(#11b 7B)
- 環境：`.venv`（uv 建，torch 2.12+cu130、transformers 5.11、accelerate 1.14）；VLM 權重 HF Qwen2.5-VL-3B/7B-Instruct
- 影像：`data/coco/images/train2014/`；切分 ref_id parity（eval=奇）
