---
title: "論文導讀：Conformal Prediction Sets for Instance Segmentation"
author: "整理：kuo（碩論 CRS 主線比對用）"
date: "2026-06-18"
---

# Conformal Prediction Sets for Instance Segmentation — 深度導讀

## 0. 書目

- **標題**：Conformal Prediction Sets for Instance Segmentation（標題已核對無誤）
- **作者**：Kerri Lu（MIT LIDS / EECS）、Dan M. Kluger（MIT IDSS）、Stephen Bates（MIT LIDS / EECS）、Sherrie Wang（MIT MechE / IDSS）
- **arXiv**：2602.10045v1 [cs.CV]，提交日 2026-02-10
- **程式碼/資料**：<https://github.com/conformal-instance-segmentation/conformal-instance-segmentation-code>
- **關鍵字**：conformal prediction、instance segmentation、IoU 保證、set cover、Learn-Then-Test (LTT)、Conformal Risk Control (CRC)
- **與本人碩論的關聯**：這是與 CRS（Conformal Referring Set）**同型保證、鄰近任務**的論文。CRS 是「referring set 內 ≥1 個 box 命中 GT（recall 保證）+ no-target abstention」，本文是「instance set 內 ≥1 個 mask 與 GT 高 IoU」。Stephen Bates 是 LTT / RCPS / conformal 理論大將，本文又明確宣稱**勝過 LTT 與 CRC**，因此必須逐項比對 nonconformity 設計與「set ≥1 命中」的保證結構。

---

## 1. 問題定義與動機

### 1.1 任務設定

- 輸入 `X = (I, (z1, z2))`：一張影像 `I ∈ R^{W×H×C}` 加一個**像素座標查詢點** `(z1, z2)`。
- 該查詢點落在某個 ground-truth 物件實例上，GT 以二值遮罩 `Y ∈ {0,1}^{W×H}` 表示。
- 有一個 instance segmentation 模型 `f(X, T)`，吃輸入 `X` 與**可調參數 `T`**，輸出二值遮罩預測 `Ŷ = f(X, T)`。調 `T` 即可產生一族多樣化的預測（例：SAM 的 mask index、watershed threshold、connected-component 前的 logit threshold）。
- 品質度量：`IoU(Y, Ŷ) = |Y∩Ŷ| / |Y∪Ŷ|`，完美預測 IoU=1。

### 1.2 動機（為什麼需要這個）

現有 instance segmentation 模型「平均表現好」，但**沒有原則性的不確定性量化**：

1. 模型輸出的 logits / confidence **未經統計校準**，也**不保證任一預測遮罩接近 GT**。例如 SAM 對每個 query 吐 3 個 mask + 預測 IoU，但**不保證任一個 mask 正確、也不保證它預測的 IoU 準**。
2. 既有的 conformal segmentation 方法（Davenport 2024；Mossina & Friedrich 2025）**只會在「單一模型預測 mask」上做修改**（膨脹 dilation、邊界擴張），只能表達**局部邊界不確定性**，無法表達真正主宰實務的**結構性不確定性（structural uncertainty）**——例如「相鄰區域到底是一個物件還是該拆成多個」（田地該合還是該分、細胞重疊、車輛遮擋）。
3. 要處理結構性不確定性，confidence set 內必須包含**質性不同的 mask**，而不是同一個 mask 的小擾動。

→ 本文目標：給定影像 + 像素查詢，輸出一個**多樣化的 instance 預測 confidence set**，並帶可證明保證：**集合內至少有一個預測與 GT 的 IoU 高於門檻 τ**，機率 ≥ 1−α。

---

## 2. 方法詳解

### 2.1 核心保證（要記住的一句話）

對新的測試點 `(X_test, Y_test)`（與 calib 同分布、IID），給定使用者指定的 IoU 目標 `0<τ<1` 與誤差率 `0<α<1`，演算法輸出 confidence set `C_{α,τ}(X_test)`，滿足：

> 以機率 ≥ 1−α，集合內**存在**某個遮罩 `ŷ ∈ C_{α,τ}(X_test)` 使得 `IoU(Y_test, ŷ) > τ`。

這正是「set 內 ≥1 命中」的 **marginal、existential（存在型）** 保證——和 CRS 的 recall 保證是同一個邏輯骨架（後面詳比）。

### 2.2 nonconformity score 怎麼設計？（重點）

**關鍵差異：本文不是傳統「單一 nonconformity 分數做 split-conformal 取分位數」的形式，而是把不確定性集合化約成一個 Set Cover 問題。** 流程（Algorithm 1）：

1. 先選定可調參數 `T` 的 `k` 個格點 `{t_1, ..., t_k}`。動機：**沒有任何單一 `T` 對所有 query 都最好**——某些設定對某些 query 成功、對另一些失敗（Figure 2）。
2. 對每個 calib 樣本 `(X_i, Y_i)`、每個參數值 `t_j`，計算遮罩 `Ŷ_ij = f(X_i, t_j)` 與其品質 `ρ_ij = IoU(Y_i, Ŷ_ij)`。
   - **這裡的「nonconformity 訊號」本質就是 IoU 本身**：對每個 `t_j` 蒐集「被它覆蓋（cover）的 calib 點集合」
     `S_j = { i : IoU(Y_i, f(X_i, t_j)) > τ }`，即在 `t_j` 下預測 IoU 超過 τ 的 calib 點索引。
3. **找最小子集 `J_{α,τ} ⊆ [k]`**，使得這些參數覆蓋的 calib 點聯集 `∪_{j∈J} S_j` 至少涵蓋 `(1−α)n` 個 calib 點：
   `J_{α,τ} = argmin_{J⊆[k]} { |J| : |∪_{j∈J} S_j| ≥ (1−α)n }`。
4. 對測試影像，用選出的每個 `t_j` 產生預測 `f(X_test, t_j)`，組成 set `C_{α,τ}(X_test) = { f(X_test, t_j) : j∈J_{α,τ} }`。

**因此「nonconformity 設計」可以這樣理解**：傳統 conformal 是「對 score 取 (1−α) 分位數當門檻」；本文則是把校準目標重述為「在 calib 集上，挑一組最小參數集合，使其聯集的 hit 覆蓋率 ≥ 1−α」。它用的不是一個純量分數，而是「每個參數 `t_j` 在 calib 集上產生的 hit/miss 二元指示向量 `g_j(X_i,Y_i) = 1[IoU>τ]`」，再對這些向量做 set cover。校準的統計合法性靠大數法則（後述 Lemma C.1 / Prop C.1）。

> Set Cover 是 NP-hard：實作上先跑多項式時間的 **greedy set cover** 得初始解 `J'`，再對更小基數的子集 brute-force 精煉到真正最小（greedy 解通常很小所以可行；若太大則略過精煉步）。

### 2.3 自適應集合大小：去重複（Algorithm 2）

`|J_{α,τ}|` 是固定的，但很多 `f(X_test, t_j)` 其實**幾乎一樣**。於是用使用者門檻 `η`（兩 mask IoU>η 視為重複）做去重，得 `C_{α,τ,η}(X_test)`：最小子集使每個被丟棄的 mask 都與保留集中某 mask IoU>η。**這讓集合大小隨 query 難度自適應**（簡單 query → 集合小；模糊 query → 集合大）。

> 去重等價於圖論的 **Dominating Set** 問題（也 NP-hard），同樣靠「集合小所以可 brute-force」化解。

**去重會破壞原保證**，因此要**重新校準 IoU 門檻**：對每個 calib 點，計算去重後集合內的最佳 IoU `s_i = max_{ŷ∈C_{α,τ,η}(X_i)} IoU(Y_i, ŷ)`，取 `s_i` 的 α-quantile 當新門檻 `θ̃`。新保證（Theorem 2.1，漸近型）：

> `liminf_n P( max_{ŷ∈C_{α,τ,η}} IoU(Y_test, ŷ) ≥ θ̃ ) ≥ 1−α`。

實驗中 `θ̃ ≳ τ`（接近或大於），代表去重幾乎不傷原保證；`θ̃>τ` 發生在覆蓋超過 `(1−α)n` 時的 over-coverage。

### 2.4 為什麼宣稱勝過 LTT 與 CRC？（本文最重要的論證，跟 CRS 直接相關）

本文與 LTT / CRC 同屬「靠調可調參數做 conformal」，但 **LTT/CRC 對每個輸入只輸出單一參數值、單一預測**。本文列出三個勝點：

1. **可行性（feasibility）更廣**：LTT/CRC 需要「存在單一參數值在 calib 上達到足夠覆蓋」。但 instance seg 裡，**常常不存在單一 `T` 能對 (1−α) 比例的 calib 點都產生高 IoU**——此時 LTT/CRC 直接 **回傳錯誤 / 空集**。本文改找**多個參數的集合**，只要「聯集中至少一個參數命中」即可，因此在更多情境可行。（附錄 E + Figure E.1 用田地實例證明：目標 80% 覆蓋，最佳單一參數 `T=0.24` 只到 ~67%，單參數方法注定達不到；本文的多參數集合可達 ~80%。）
2. **不需要單調 loss**：CRC（及前身 RCPS, Bates 2021）要求 loss 對參數**單調或近單調**。但 **IoU loss 不單調**——調大 `T` 既可能改善也可能惡化 IoU。本文的（漸近版）方法**不需要單調 loss**，所以能直接用 IoU-based loss。
3. **自適應 + 多樣性**：LTT/CRC 永遠吐單一 mask，無法表達結構性不確定性；本文吐**大小自適應、形狀質性不同**的集合。
4. **vs.「把 LTT 套到冪集」的天真做法**：理論上可把 LTT 的參數空間擴成 `2^{t_1..t_k}` 的冪集（`2^k` 元素），但那個超大參數空間 LTT 原論文未探討；本文提供了具規範性的演算法與實作。

### 2.5 有限樣本版本（Appendix H, Algorithm 3）

主文用的是**漸近保證**版（靠 WLLN）。附錄 H 給**有限樣本保證**版：

1. 把 calib 隨機切兩半 `n_1, n_2`。
2. 用 `n_1` 跑 greedy 對 `k` 個參數**排序**（高到低優先序 `L`，每步選覆蓋最多「尚未覆蓋」點的參數）。
3. 用 `n_2` 跑 **CRC** 找最小 `λ̂`，使 `L[1:λ̂]` 的前綴參數集合構成合法 conformal set。為了讓 CRC 能用，去重程序 `UNIQUE*` 特別設計成 `C_λ ⊆ C_{λ+1}`（巢狀），使 loss `l_i(λ)=1[max IoU ≤ τ]` 對 `λ` **非遞增 → 單調**，滿足 CRC 前提。
4. 有限樣本保證（Theorem H.1）：`P( max_{ŷ∈C_λ̂} IoU(Y_test, ŷ) > τ ) ≥ 1−α`。

漸近版通常較緊（集合更小），因為可 brute-force 找最小 set cover、最小去重子集；有限樣本版受限於前綴形式 `L[1:λ]`，無法保證最小。**論文實驗一律用漸近版。**

### 2.6 校準合法性（證明骨架，供本人寫 related work 引用）

- **Assumption 1**：calib 與 test IID。
- **Assumption A.2（可行性）**：`P(max_j IoU(Y, f(X,t_j)) > τ) > 1−α`，可實證檢查；不成立時 Algorithm 1 報錯，使用者要調大 α / 調小 τ / 擴大參數空間。
- **Lemma C.1**：`J^{(n)}_{α,τ}` 漸近落在合法集合 `A_{α,τ}`（靠 WLLN，誤選不合法子集或報錯的機率 → 0）。
- **Prop C.1 / Theorem 2.1**：由上推出漸近 ≥ 1−α 覆蓋。去重後保證靠**可交換性 + α-quantile**（Fact 2.15(ii)，Angelopoulos 2025a）。

---

## 3. 實驗

### 3.1 資料集與模型（Table 1 / Appendix I）

| 應用 | 資料集 | 模型 | 可調參數 T | calib / test 像素 |
|---|---|---|---|---|
| 田地切割 Field delineation | Fields of The World（法國，350 圖 → 250 calib / 100 test 圖）| FoTW U-Net + watershed | watershed threshold（0~1，細格）| 2476 / 917 |
| 細胞切割 Cell segmentation | Cellpose（52 圖 → 30/22）| Cellpose-SAM | cell extent threshold（−5~5，步長 1）| 925 / 659 |
| 車輛偵測 Vehicle detection | Cityscapes（200 圖 → 100/100）| SAM | T1=mask index∈{1,2,3}、T2=prob threshold（0~1 步長 0.05）| 105 / 121 |

每張圖隨機抽像素查詢點（只保留落在目標 instance 內者）。`η=0.9` 統一用於去重。

### 3.2 (α, τ) 怎麼選

對每個任務畫「每點在所有 `T` 下可達的最大 IoU 的累積分布」（Figure D.1），定出 base model 的**性能天花板**。選在 Pareto frontier 上的最嚴可行配置：

- 田地：α=0.2, τ=0.7（最模糊、保證最弱）
- 細胞：α=0.2, τ=0.75
- 車輛：α=0.1, τ=0.8（SAM 最強）

> 重要立場：**conformal 只能「認證」base model 的可靠度，無法「超越」它的天花板。** 田地連 90% 信心 + IoU>0.8 都做不到（最佳 T 下仍 >10% 的點 IoU<0.8）。

### 3.3 主結果（Table 2：覆蓋率）

| 應用 | τ | θ̃ | 目標覆蓋 (1−α) | 本文 conformal 覆蓋 | LTT/CRC baseline 覆蓋 |
|---|---|---|---|---|---|
| 田地 | 0.7 | 0.696 | 0.8 | **0.797** | 0.663 |
| 細胞 | 0.75 | 0.760 | 0.8 | **0.835** | 0.803 |
| 車輛 | 0.8 | 0.842 | 0.9 | **0.843** | 0.661 |

- 本文覆蓋率**接近目標 1−α 且顯著高於 feasible LTT/CRC baseline**。
- 田地 66.3%→79.7%、車輛 66.1%→84.3% 大幅提升（救回被單參數合併/拆錯、或被 SAM top-1 漏掉的物件）；細胞僅 80.3%→83.5%（因細胞誤差多為「邊界局部」而非結構性）。
- 注意：車輛雖達 0.843、未到目標 0.9，但這反映 base model 天花板（θ̃=0.842，且 baseline 僅 0.661）。

### 3.4 自適應集合大小（Figure 4）

去重後多數集合 ≤3 個 mask，模糊時擴大。田地最大到 5（對應 under/over-seg 假說）；細胞多塌成 1~2（有些 size=0，因該像素被判為非細胞）。

### 3.5 對比 morphological dilation baseline（Appendix F, G）

- Dilation 只保證「GT 被膨脹 mask 包住」，**不保證高 IoU**。田地上膨脹 D=2 像素，只有 **54.4%** test 點 IoU>0.7（本文 79.7%）。
- Dilation 傾向 **undersegmentation**（為了包住 GT 把 mask 撐大）。用 Persello & Bruzzone (2009) 的 over/under-seg 分數量化（Appendix G）：兩者 over-seg 都低，但 dilation 的 under-seg 明顯較差；本文因吐多個預測，總有一個避開 under-、一個避開 over-。

### 3.6 限制（作者自陳，Discussion）

1. 保證是 **global / marginal**：每個輸入用同一個 IoU 門檻，簡單與模糊 query 一視同仁（無 conditional / adaptive 門檻）。
2. **依賴「可調參數」**：若沒有能誘發多樣性的參數，方法難用。
3. 組合最佳化（set cover、去重）在大集合時可能很貴。
4. 現有模型不被訓練去「產生多個合理替代」，未來工作可朝 diversity-promoting training。

---

## 4. 與 CRS 的逐項 diff 表（本人碩論比對核心）

| 面向 | 本文（Conformal Instance Segmentation） | 本人 CRS（Conformal Referring Set） |
|---|---|---|
| **任務 / query 型態** | instance segmentation，**像素座標查詢**（point prompt） | referring expression comprehension，**自然語言查詢** |
| **輸出單元** | instance **遮罩 mask** 的集合 | **box** 的 referring set |
| **核心保證結構** | set 內 **≥1 個 mask 與 GT IoU > τ**，機率 ≥ 1−α（existential/marginal recall）| set 內 **≥1 個 box 命中 GT**（IoU 門檻）的 recall 保證，機率高 |
| **「≥1 命中」邏輯** | **相同骨架**：都是「集合內存在一個高品質元素」的 existential 保證，都靠 marginal coverage | **相同骨架** |
| **集合怎麼生成** | 對**單一模型**掃可調參數 `T` 的格點，挑最小參數子集（**set cover**）| **跨 base 組合**：OWL-ViT gate（管 abstention/no-target）+ GroundingDINO box（管 compact set），LTT 聯合校準 |
| **校準框架** | 漸近版用 set cover + WLLN；有限樣本版用 **CRC**（明確聲稱勝過 LTT/CRC 的單參數做法）| **LTT**（Learn-Then-Test），用 calib-only grid、Bonferroni 多風險校正 |
| **nonconformity 訊號** | 每個參數 `t_j` 的 hit 指示向量 `1[IoU>τ]`，化約成 set cover 覆蓋率 | OWL-ViT score / consistency 等做 abstention 判斷 + box recall |
| **abstention / no-target** | **無**。每個 query 都假設落在某 instance 上，沒有「該棄答 / 無目標」機制（細胞偶有 size=0 是模型判非細胞，非設計的 abstention 保證）| **有**。gRefCOCO no-target，明確的 abstention 雙保證（recall + abstain）|
| **多模型 / cross-base** | **單模型**（同一個 f 調參數），明說「考慮 base 的 superset」是補救手段而非主軸 | **cross-base composition** 是主結果；factorization（gate vs box）非 ensemble |
| **set 大小自適應** | 有（去重 Algorithm 2，隨 query 難度變）| 有（α=β=0.3 下平均 3.21/2.02/3.48 框）|
| **保證型態** | 漸近 + 有限樣本兩版 | 有限樣本（LTT/RCPS 路線）|
| **conditional / per-input** | 明說是 **global** 限制，待未來工作 | 同樣是 marginal（共同弱點）|

---

## 5. 這篇對 CRS 的威脅評估與借鏡

### 5.1 「set ≥1 命中」保證會威脅 CRS 的 recall-guarantee 新穎性嗎？

**結論：不直接構成威脅，但會壓縮「保證骨架本身」的可宣稱新穎性，必須在寫作上明確區隔。**

- **保證骨架（existential recall）並非 CRS 獨創、也非本文獨創**：兩者都是「集合內 ≥1 高品質元素，marginal 機率 ≥1−α」。事實上 Angelopoulos 2025b 的 LTT 早就對 instance seg 給過 recall/coverage 保證。所以**「我們保證 set 內 ≥1 命中」這句話本身不能當成 CRS 的核心賣點**——本文的存在會讓審稿人說「這是 conformal recall 保證的標準形式」。
- **CRS 真正的差異化（仍站得住）有三點，這篇都沒做**：
  1. **語言 query（REC）而非像素 query**：本文是 point-prompt，CRS 是 referring expression。語言語意歧義 + grounding 是不同問題類別。
  2. **abstention / no-target 雙保證**：本文**完全沒有** no-target / 棄答機制。CRS 在 gRefCOCO 上同時校準 recall 與 abstention，這是本文不覆蓋的軸。
  3. **cross-base composition + factorization**：本文是單模型掃參數；CRS 是 OWL gate × GroundingDINO box 的跨 base 分工，且有 2×2 ablation 證明是 factorization 非 ensemble。
- **因此威脅是「修辭層」而非「方法層」**：CRS 不應把賣點定位成「我們有 set ≥1 命中保證」（會被這篇與 LTT 蓋過），而要定位成「**在語言 grounding 上，把 recall 與 abstention 用跨 base 分工聯合校準**」。本文剛好是強力的「同型保證、鄰近任務」對照，能襯托 CRS 的 query 型態與 abstention 差異。

### 5.2 nonconformity 設計可否借鏡？

- **可借鏡 1：set cover 視角當作 LTT 的替代敘事**。本文把「多參數聯集覆蓋 (1−α)」講成 set cover，並嚴格論證「為何單參數 LTT/CRC 不可行 / IoU loss 非單調」。CRS 用 LTT，但若 CRS 也碰到「單一 threshold 無法同時保 recall+abstain」的情境，可借這套 set cover / 多參數聯集論證，強化「為何需要組合而非單一門檻」的動機。
- **可借鏡 2：去重 + 重新校準 IoU 門檻（θ̃）的兩階段技巧**。CRS 輸出 box set 時若也想做「去近似重複 box → 重新校準命中門檻」，這個 Algorithm 2 的「去重後用 calib 上 best-IoU 的 α-quantile 重新定門檻」是乾淨可移植的模式（保證仍靠 exchangeability）。
- **可借鏡 3：可行性 frontier（α–τ 天花板）做法**。Figure D.1 用「每點可達最大 IoU 的累積分布」畫 base model 天花板、定 Pareto (α,τ)。CRS 的「R1/R2 CI 上界 + size CI」報法已類似，但這篇的「天花板histogram → 選最嚴可行配置」可作為**呈現方式的對照範例**，強化 CRS 報「我們選的是 frontier 上最嚴點」的論述。
- **不宜照搬**：本文的核心是「單模型掃參數 + set cover」，這與 CRS 的 cross-base 哲學相反，**不要把 CRS 改成單模型掃參數**（會丟掉 factorization 賣點）。借的是論證與校準技巧，不是架構。

### 5.3 related work 怎麼寫（具體建議）

1. 在「conformal for segmentation / detection」段落**必引此文**，定位為「**像素 query 的 instance segmentation 版同型 recall 保證**」，並點出三個與 CRS 的差異（語言 query、abstention、cross-base）。
2. 借用本文的「LTT/CRC 單參數不可行 + IoU loss 非單調」論證，當作「**為何 CRS 需要 LTT 多閾值 / 跨 base 而非單一門檻**」的 supporting citation。
3. 把 Davenport 2024、Mossina & Friedrich 2025（dilation）也順帶引為「只保證包含、不保證高 IoU」的反例，與 CRS 的「命中（hit）保證 vs 包含（containment）保證」對齊——這個「hit vs containment」的區分，CRS 可直接借來強化自己的保證語意。
4. Bates 是共同作者：CRS 已用 LTT（Bates 系），引此文能讓「CRS 的 conformal 理論血統」更完整、審稿友善。

---

## 6. 個人評價

**優點**

- 問題切得準：把「結構性不確定性」（合/分歧義）與「邊界局部不確定性」分開，並指出既有 dilation 法只能做後者，是很乾淨的 motivation。
- set cover 視角優雅：把「多參數聯集覆蓋」對應到經典 NP-hard 問題，再用 greedy+brute-force 化解，務實。
- 「hit（高 IoU）保證 vs containment（包住）保證」的區分有洞見，對整個 conformal segmentation 文獻是有價值的概念釐清。
- 漸近 + 有限樣本兩版都給、證明完整（WLLN + exchangeability），理論扎實（Bates 風格）。
- 誠實：反覆強調「conformal 只認證、不超越 base model 天花板」，車輛達不到 0.9 也照實報。

**侷限 / 可挑戰處**

- 保證是 **marginal/global**，沒有 conditional（per-input）門檻——這是與 CRS 共享的弱點，審稿人會問。
- **強依賴「存在能誘發多樣性的可調參數 T」**。一旦模型沒有這種旋鈕（很多端到端模型沒有），方法就難用；這比 CRS 的 cross-base 組合更受限。
- 沒有 abstention / out-of-distribution / no-target 處理——所有 query 都假設落在某 instance 上，實務上「點到背景」沒有保證語意。
- 組合最佳化（set cover + dominating set 去重）靠「集合小」才可行，scalability 是隱憂（作者自承）。
- 資料規模偏小（test 像素 數百級），雖然 conformal 對 calib size 敏感、有限樣本版有形式保證，但實證涵蓋面不算大。

**對本人的 take-away（一句話）**：這是 CRS 最該引、也最該主動區隔的「同型保證、鄰近任務」論文。它證明「set ≥1 命中」是 conformal 的標準骨架（所以 CRS 不靠這句話取勝），同時它**缺 abstention、缺 cross-base、是像素而非語言 query**——這三點正好是 CRS 的差異化護城河。技術上值得偷學的是「去重 + 重新校準 θ̃」與「α–τ 可行性 frontier」兩個乾淨技巧，以及「為何單參數 LTT 不夠」的論證模板。
