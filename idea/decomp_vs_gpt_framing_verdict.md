# 對兩份 GPT 研究重構建議的回應與實況對照

> 給 GPT 看的回饋。你（GPT）先後提了兩份建議：
> (A) **Adaptive Risk-Controlled Target Acquisition** — 跳脫 CRS 框架的研究重構（大 framing + adaptive 系統藍圖）。
> (B) **Risk-Controlled Referring Set Stack Optimization** — 保留既有成果下的 stack 優化（冷靜版，已自砍 adaptive/robot/learned）。
> 這份文件用「我手上已有的真實實驗證據」逐一對照兩份建議，標明哪些採納、哪些反對，反對的依據是什麼。
> 目標：留住兩份 framing/計畫的好點子，擋掉會讓我畢不了業的系統擴張。
> 結構：§0 實況 → §1-§5 對 (A) 的裁決 → §6 對 (B) 的裁決 → §7 兩份合併淨增量 → 附錄。

---

## 0. 我目前的真實處境（你提建議時不知道的上下文）

這是一篇**碩士論文**，主線立場已定且已被多輪紅隊收斂：

> **Frozen vision-language models 下的 post-hoc reliability / risk-controlled referring sets。零訓練、零梯度、不改任何模型權重。**

我**已經有兩個做完/接近做完的主結果**，不是空想：

### 主結果 1：Cross-Base Conformal Referring Sets（CRS）— 已完成、紅隊過
- OWL-ViT gate + GroundingDINO box，用 Learn-then-Test (LTT) 聯合校準三風險：
  - R1 = answered-target FNR（漏檢）
  - R2 = no-target false selection（誤含）
  - R3 = target deferral（棄答）
- 全量 val/testA/testB，α=β=0.3, γ=0.5 下 headline set size = 3.24 / 2.02 / 3.50，三風險 CI 上界皆 <0.3。
- 2×2 ablation 證明這是 **factorization 非 ensemble**：OWL gate 管棄答/no-target、GD box 管集合大小，拔掉任一個就垮。
- 已過 P0+P1 紅隊（修 calibration leakage、Bonferroni 三風險校正、三種 split 模式 robustness）。
- **這是我最硬的 formal contribution，有可證明的有限樣本保證。**

### 主結果 2：Expression-Decomposed CRS — 今天剛跨 split 確認，全量跑中
- 機制：VLM（Qwen2.5-VL，凍結，只 prompt）把複合指稱拆成子指稱 → 各子指稱單獨餵 GDINO → union 候選 → 接回 CRS/LTT。
- **今天（2026-06-18）的決定性證據**（matched oracle 分析，繞過 LTT，同輸出 size 比 recall）：

  | split | size~3 full-expr recall | size~3 decomp recall | Δ |
  |---|---|---|---|
  | val | 0.697 | 0.899 | **+20pp** |
  | testA | 0.743 | 0.889 | **+15pp** |
  | testB | 0.602 | 0.845 | **+24pp** |

  size~2 / size~4 也全部同向勝出，**三 split 零例外**。
- LTT 主表在子集上：val 健康（n_feas=84, R1 0.191→0.162, defer 0.42→0.30），testA/testB 因子集太小統計力不足（n_feas=2 / EMPTY）——已啟動全量 dump 補上，純統計力問題，非方法失效（matched 已證）。

### 我已經有的「負結果」（很重要，下面會用到）
為了到達上面的成果，我試過並**用實驗否決**了一串方法：
- candidate scoring / p_match 重排序 → 吃不掉 oracle gap（負）
- VLM-as-selector（VLM 直接選框）→ 11 次嘗試無一勝過現況（負）
- cardinality head（預測該選幾個）→ 負
- evidence detector（CLIP-crop / OWL-ViT 做 per-object 證據）→ CLIP-crop≈random、OWL-ViT 鎖死 0.40，per-object score prior 不可校正（負）

**第 17 次嘗試（decomposition）才首次大幅勝出。** 這串負結果不是浪費，是 boundary analysis。

---

# 第一部分：對 (A) Adaptive Risk-Controlled Target Acquisition 的裁決

## 1. 你的 framing 升級：採納（但只在寫作層）

你最有價值的主張：

> 不要把問題定義成「improve grounding accuracy」，而是「risk-controlled target acquisition under frozen VLMs」。

**這個我完全採納**——但要講清楚它的性質：**這是 thesis 的敘事 framing，不是新的工作。** 它把我已經做的事（set-valued output + decomposition + post-hoc calibration + defer）統一在一個更大氣、也更誠實的傘下。零額外實驗。

具體採納項：
- **論文總 framing**：用「Risk-Controlled Target Acquisition with Frozen Vision-Language Models」當 thesis 級標題；CRS + Decomposed CRS 是其下兩個具體 instantiation。
- **負結果資產化**（你的 §12.2）：把我那串負結果（candidate scoring / VLM-selector / cardinality / evidence detector）寫成「為什麼非得用 set-valued + decomposition + post-hoc」的論證，而非失敗紀錄。這點我很認同，會寫進 thesis。

---

## 2. 你的「Adaptive Decision Policy」（§5/§6）：反對實作，理由是證據

你把 routing 畫成系統大腦：診斷不確定性 → 在 6 個 action 間自適應選擇。我必須指出三個**基於實況**的問題：

### 2.1 我沒有 adaptive policy，我有一條 if-else——而且那是刻意的
我的 routing 是 **v1 保守規則**：`no-target / single → 不拆；multi-target → 拆`。
- 實測誤拆率 0.000（150 個 no-target 全不拆）。
- **這條保守規則正是 R2（誤含）安全保證的來源**——因為 decomposition 不碰 no-target，R2 才不會爆。
- 如果照你的 framing 去**訓一個學習式 routing policy**，我會：(a) 違反整條主線「frozen / 零訓練 / post-hoc」的立場；(b) 重開一個學習問題（而我手上 candidate scoring / VLM-selector 等學習式重排序全是負結果，沒理由相信 learned routing 會成功）。
- **裁決**：保留 v1 保守規則，不升級成 ML policy。所以最終 framing 我傾向**拿掉「Adaptive」這個字**，避免名實不符被口試攻擊。

### 2.2 「Adaptive」會招來「你的 policy 學在哪？」的攻擊
你 §16 最推薦的名字是「Adaptive Risk-Controlled Referring Sets」。我同意「Risk-Controlled Referring Sets」，但**反對「Adaptive」**——因為我沒有可學習的 policy，標題掛 adaptive 等於給審查者一個現成的攻擊點。

---

## 3. 你的 6 個 action 逐一裁決（用實況）

| Action | 你的定位 | 我的真實狀態 | 裁決 |
|---|---|---|---|
| `ANSWER_SET`（CRS 集合）| 核心 | ✅ 已完成、formal guarantee、紅隊過 | **主結果 1** |
| `DECOMPOSE` | 一個 action | ✅ 跨 split +15~24pp，全量跑中 | **主結果 2** |
| `DEFER` | 一個 action | ✅ 已內建於 CRS（R3）| 已有，非新工作 |
| `VERIFY`（VLM set-level）| 一個 action | ⚠️ 概念，未實作；且 **VLM-as-selector 已是負結果** | future work，不碰 |
| `ASK_CLARIFICATION` | 一個 action | ❌ 純概念，需人機介面 | future work，不碰 |
| learned routing | 全文賣點 | ⚠️ 只有 v1 if-else（刻意）| 見 §2，不升級 |

### 特別說明 VERIFY（你的 §8）
你建議「VLM 不做 selector，改做 set-level verifier」。但我已經有 **VLM-as-selector 的負結果（11 次嘗試）** 和 **evidence detector 的負結果（CLIP-crop≈random、OWL-ViT 鎖死 0.40，per-object score 不可校正）**。把同一個被否決的元件換名字叫「verifier」再做一次，風險高、回報不明。**頂多 future work 一句話，不開新章。**

---

## 4. 你的 risk-cost / human-check 模擬（§9-10）：謹慎，不是免費

你說「只做 offline simulation 畫 Pareto 就好，很便宜」。兩點修正：
- 我**已經有** cost-risk Pareto（`crs_pareto.png`），不是從零開始。
- 你提的多成本項（defer cost + verification cost + decomposition cost + human cost…）需要為每項定一個 **cost 係數**，這些是**沒有 ground truth 的自由參數**。口試會問「為什麼 c_human = 這個值」。除非係數能 justify，否則不開——現有的 size-as-human-cost 單一口徑已足夠且好辯護。

---

# 第二部分：對 (B) Risk-Controlled Referring Set Stack 的裁決

## 5. 對 (B) Risk-Controlled Referring Set Stack 的裁決

(B) 比 (A) 好很多：它自己就內建了我對 (A) 的擋箭牌（§1「不推翻已完成」「不開 learned policy / robot / human-in-the-loop」、§8「不建議做的優化」）。所以對 (B) 我不是擋擴張，而是**排優先序、剔除看似便宜其實是坑的項**。

它的核心框架——把 CRS 升級成「candidate-pool construction + risk-control stack」——我同意，**而且我已經在做**（decomposition 就是 candidate-pool reconstruction）。問題是它 §3-§7 列的五個優先級全做又是另一篇論文。逐一裁決：

| (B) 的優先級 | 它的定位 | 我的裁決 | 理由 |
|---|---|---|---|
| §3 Decomposed CRS 做成正式第二主結果 | 第一優先 | ✅ **正在做，最高優先** | 全量 dump 跑中就是這件事；它列的必補項（三 split / R1R2R3 / set size / subset / runtime）正是我的接手清單 |
| §4 Conservative decomposition gate | 第二優先 | ✅ **已完成** | 我的 v1 保守 routing 就是這個，誤拆率 0.000；它也說「rule-based 別用 learned」，與我對 (A) 的立場一致 |
| §6 Subgroup / 分組風險分析 | 強烈建議 | ✅ **採納，CP 值最高的新工作** | 唯一我會主動加的。便宜（不需新模型）、直接強化論文、正面回答「decomposition 何時有效」 |
| §5 Object-centric candidate pool | 第三優先 | ⚠️ **降為 future work，且預期是負的** | 見 §5.2 |
| §7 Cost-aware Pareto | 第五優先 | 🔸 **已有一半，補一軸即可** | crs_pareto.png 已存在；補「detector forwards」軸好防守，別擴成多成本 |

### 5.1 Subgroup 分析（B §6）是最值得採納的，但有一個時序陷阱
按 no-target / single / multi / long / conjunction / high-n_gt 分組報 R1R2R3，能把 matched 的 +15~24pp 拆解成「贏在哪種 query」。**陷阱**：目前 decomp dump 只有 ~255/138/123 真拆 case，再切 6-10 subgroup 每組剩個位數，CI 寬到無意義。**必須等全量 dump 完成後再做**，順序：全量 dump → LTT 主表健康 → subgroup。現在做＝得到一堆不可靠小樣本。

### 5.2 Object-centric pool（B §5）是「看似互補、實則撞負結果」的坑
它建議除拆子指稱外，再用名詞 prompt（"person"/"guy"/"child"）抓物件補召回。兩個問題：
- 它 §5.4 自己承認「會增 false positives，R2 惡化就放 appendix」——等於先承認可能沒用。
- 更關鍵：matched 顯示 decomp pool oracle ceiling 已 0.90-0.95，**瓶頸從來不是召回不足，是同 size 留對框**。加 object pool 把更多框塞進池子，恰恰惡化 R2 / 增大 size，方向與我的優勢相反。**裁決：future work，且預期負。**

### 5.3 Cost 模型（B §7）的克制是對的
它建議「只用 set size + detector forwards，別碰 c_human/c_verify 自由參數」——完全正確，與我對 (A) §4 的批評一致。我已有 risk-vs-size Pareto，要補的只有「detector forwards」一軸（decomp 多跑 N 次 detector 是真實 compute 成本，好量化好防守）。全量後順手做。

---

## 6. 兩份合併後的淨增量（關鍵結論）

把 (A)(B) 兩份所有建議過濾掉「已完成 / future work / 撞負結果」後，**對一篇能畢業的碩論，真正的淨增量新工作只有一件**：

> **全量 dump 完成後，做 subgroup 分析（B §6）。**

其餘全部歸位：
- **已完成**：CRS 主結果、v1 conservative gate、risk-vs-size Pareto、decomposition 本體。
- **正在做**：全量 dump（Decomposed CRS 正式第二主結果）。
- **順手補**：Pareto 加 detector-forwards 軸。
- **future work（不動手）**：VERIFY/VLM-verifier、ASK_CLARIFICATION、learned/adaptive routing、object-centric pool、robot、multi-cost 模型。
- **純寫作（採納 framing）**：thesis 級 framing 升級、負結果資產化。

兩份 GPT 文件的关系：
- **(A) adaptive**：好 framing、壞計畫。採標題敘事，擋系統擴張。
- **(B) stack**：好計畫、仍偏多。採 §3/§4/§6，剔 §5，§7 補一軸。它是 (A) 的冷靜版。

---

## 7. 我想請你（GPT）幫的具體事

基於上面的實況，我**不需要**更大的系統藍圖。我需要你協助把現有成果寫好：

1. **幫我打磨 thesis framing 段落**：用「Risk-Controlled Target Acquisition with Frozen VLMs」為傘，把 CRS（主結果1）+ Decomposed CRS（主結果2）+ defer 收進去，**但不宣稱有 adaptive learned policy**。
2. **幫我把負結果寫成 boundary analysis 論述**：candidate scoring / VLM-selector / cardinality / evidence detector 四個負結果，如何共同論證「為什麼必須 set-valued + decomposition + post-hoc calibration」。
3. **幫我寫一節克制的 future work**：把 VERIFY / ASK_CLARIFICATION / learned routing / object-centric pool / robot / multi-cost Pareto 收進去，講清楚為什麼它們是 future work 而非本論文範圍（連結到既有負結果與 frozen 立場）。
4. **標題定案建議**：在「Risk-Controlled Referring Sets for Frozen Grounding Models」與「Risk-Controlled Target Acquisition with Frozen VLMs」之間給我利弊。

**請不要**再提：learned/adaptive routing policy、把 VLM-selector 改名重做、需要新自由參數的 cost 模型、object-centric pool 當主線、robot demo、human-in-the-loop 系統。這些會讓一篇能畢業的論文變成做不完的計畫。

---

## 附錄：一句話總結我的立場

> (A) 的 framing 是好的「論文故事架構師」，(B) 是冷靜的「研究計畫」但仍偏多。
> framing 升級照單全收（零實驗）；系統擴張一個都不動手；(B) 只採 subgroup 一項新工作。
> 我現在最大的風險不是「不夠大」，是「再開新坑導致畢不了業」——framing 讓它顯得大就夠了。
