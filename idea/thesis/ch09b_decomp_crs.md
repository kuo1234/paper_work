# 第 9b 章　Decomp CRS：候選池維度（第二主結果）

> 本章是三正交維度中的 **candidate pool** 軸。CRS（第 9 章）以 frozen base 的原生候選池為輸入；
> 本章證明：在**完全不碰 detector、不訓練**的前提下，把候選池從「full-expression pool」
> 換成「decomposition-union pool」，可在同一 LTT 協議下**純賺 R1（多目標漏檢）**而不付代價。

## 9b.1 動機：候選池是一個獨立可改善的維度

第 9 章的 CRS 把風險控制代價因式分解為 gate（該不該答）與 box（答得準）。但在 box 軸之前，
還有一個更上游的瓶頸：**候選池本身**。GroundingDINO 對一條完整 referring expression 跑一次前向，
產生的候選框集合，未必涵蓋 multi-target query 的所有 GT——尤其當 expression 描述多個語意子部件時
（「the man in red **and** the woman beside him」），單次整句 forward 容易偏重其中一個子部件。

這定義了第二個正交維度：**在 gate 與 box 都不變的情況下，能不能用更好的候選池降低 R1（已作答目標漏檢率）？**
關鍵約束是不能違反 frozen——任何改善都必須來自 frozen detector 已能產出的資訊，不更新任何權重。

## 9b.2 方法：VLM-routed decomposition union pool

**流程**（training-free，weight-frozen）：

1. 對 target-present query，用一個 frozen VLM（Qwen2.5-VL-7B）作為 **router**，判斷 expression 是否含多個可拆解的指稱子部件；若是，輸出子部件清單。
2. 每個子部件各自餵 GroundingDINO 跑一次前向，得到 per-part 候選框。
3. 把整句 forward 的候選池與各子部件的候選池取 **union**，得到 decomposition-union pool。
4. **其餘完全沿用第 9 章 CRS 協議**：同樣的 OWL gate、同樣的 GDINO box scoring、同樣的三風險 LTT、同樣的 calib-only grid 與 min-size 選點。**只換候選池這一個變因**。

設計重點：decomposition 只擴充候選池，不改任何分數、不改 gate、不改 certificate。因此它與 gate 維度（第 9c 章）、certificate 維度正交，可獨立疊加。

**防洩漏的關鍵**：router 只對 target-present query 觸發會造成「decomposed 旗標 100% 蘊含 has_target」的標籤洩漏，因此 decomposition 旗標**不可**作為 gate 特徵；本章 decomposition 只用於候選池建構，gate 仍只看 frozen detector 分數（與第 9 章一致）。

## 9b.3 主結果：三 split 純賺 R1

協議：a=b=0.3、g=0.5（與第 9 章 CRS 同協議），parity split，bootstrap CI。**只有候選池不同**。

**表 9b.1　Decomp CRS vs Frozen CRS（同協議，只換候選池）**

| method | split | pool | R1 | R2 | set size | n_feas |
|---|---|---|---:|---:|---:|---:|
| Frozen CRS | val | full | 0.191 | 0.159 | 3.24 | 112 |
| Frozen CRS | testA | full | 0.260 | 0.203 | 2.02 | 58 |
| Frozen CRS | testB | full | 0.168 | 0.236 | 3.50 | 56 |
| **Decomp CRS** | val | decomp | **0.165** | 0.158 | 3.24 | 112 |
| **Decomp CRS** | testA | decomp | **0.244** | 0.201 | 1.96 | 58 |
| **Decomp CRS** | testB | decomp | **0.163** | 0.236 | 3.34 | 56 |

**結論**：Decomp 的 R1 在三個 split **全部 ≤ Frozen**（0.165/0.244/0.163 vs 0.191/0.260/0.168），
而 R2 與 set size **持平**（差異在小數第三位或 set size <0.2 框）。也就是說，換更好的候選池
**單方向降低多目標漏檢，不付任何其他風險或集合大小的代價**。這正是「候選池維度可獨立改善」的鐵證。

## 9b.4 硬防線：三道紅隊補強

主結果之外，本章對三個最可能的攻擊各補一道防線。

### 9b.4.1 Robustness（三切分模式）

Decomp R1 在 parity / random5 / image-disjoint 三種切分下均 ≤ Frozen。val random5：
Decomp R1 0.164±0.003 vs Frozen 0.191±0.007。**誠實標註**：testB random5 下 Decomp 變異較大
（R1 0.226±0.053、set size 2.55±0.65）——某些 seed 壓小集合但升高 R1，是一個不穩點，記入 limitation。

### 9b.4.2 Strong full-expression baselines（堵 matched 假象）

最強的質疑是：Decomp 的 R1 優勢只是「拿擴大的池跟未經整理的 full 池比」的 matched 假象。
為此，我們讓 full-expression pool 也吃各種 consolidation（NMS、top-K），在 val 上正面對比：

**表 9b.2　Decomp vs 強化版 full-pool baseline（val）**

| method | R1 | set size |
|---|---:|---:|
| full raw (threshold=0) | 0.191 | 3.24 |
| full + NMS@0.5 | 0.211 | 2.31 |
| full + NMS@0.7 | 0.174 | 2.99 |
| full + top-10 | 0.241 | 2.64 |
| full + top-20 | 0.237 | 2.69 |
| **Decomp** | **0.165** | 3.24 |

Decomp 的 R1 **低於所有** full-pool consolidation（NMS@0.5/0.7、top-10/20），且此結論三 split 一致。
這證明 R1 優勢來自候選池**涵蓋了 full-pool 整理不出來的 GT**，而非 matched 指標假象——硬防線成立。

### 9b.4.3 Frontier（truly-decomposed 子集的可達邊界）

把分析限縮到 router **真正觸發拆解**的子集（val n=875），畫 recall–size frontier：

**表 9b.3　Truly-decomposed 子集的 frontier（val n=875）**

| pool | rec@sz2 | rec@sz3 | rec@sz4 | size@rec0.8 | AURC |
|---|---:|---:|---:|---:|---:|
| full | 0.589 | 0.686 | 0.753 | 5.48 | 0.690 |
| decomp | **0.823** | **0.913** | **0.937** | **2.08** | **0.875** |

在真正需要拆解的 query 上，decomp 池在 set size=3 時的召回 0.913 vs full 0.686，
達到 recall 0.8 所需集合大小 2.08 vs 5.48 框。**Anti-cheat**：用 calib 選 λ、在 disjoint eval
評估（calib n=419 / eval n=456），target size~3.0 時 decomp eval rec=0.912 vs full 0.682，
結論在 held-out 上維持，非校準集過擬合。

## 9b.5 兩項誠實限制

**1. Over-decomposition（自身錯，非全資料集模糊）。** 失敗審計（P1-F failaudit）顯示，
testA 上 n_gt==1 的 negative-gain case（n=14）中，約 **50% 是標註本身的歧義（annotation ambiguity）、
50% 是 over-decomposition**——router 把一個單目標 expression 錯誤拆成多部件，反而引入多餘候選、
升高 R1。這修正了早期「退步全是標註模糊」的過度宣稱（原宣稱有一半是錯的）。抑制 over-decomposition
（更保守的 VLM router）是明確的 future work。

**2. Training-free 但非 compute-free。** Decomp 不更新任何權重，但有額外推論開銷。
成本量化（P1-E cost）：

**表 9b.4　Decomp 推論成本（vs frozen full-expression）**

| Method | VLM calls / query (val/testA/testB) | GDINO forwards / query | Avg set size |
|---|---|---|---|
| Frozen (full-expr) | 0 / 0 / 0 | 1.00 / 1.00 / 1.00 | 3.24 / 2.02 / 3.50 |
| Decomp (VLM-routed) | 0.37 / 0.77 / 0.71 | 1.06 / 1.08 / 1.07 | 3.24 / 1.96 / 3.34 |

VLM router 對**每個 target-present query** 呼叫一次（actual cost；val 0.37 / testA 0.77 / testB 0.71
calls/query，差異反映各 split 的 target-present 比例），但真正觸發拆解的只有約 6% 的 query
（平均拆 2.1 部件），故 GDINO forward 只多 1.06–1.08x。誠實 claim：**training-free 且 weight-frozen，
但有額外 inference-time compute**——不能宣稱 cheap，只能宣稱不訓練。

## 9b.6 小結

Decomp CRS 證明 candidate pool 是一個**正交且可獨立改善**的維度：在 gate、box scoring、
certificate 全部不變、不碰任何 detector 權重的前提下，VLM-routed decomposition union pool
讓 R1 三 split 全降而 R2/set size 持平。三道硬防線（robustness、strong baseline、frontier）
堵住 matched 假象與過擬合質疑；兩項誠實限制（over-decomposition、compute cost）標明邊界。
這是論文的第二主結果，也是統一框架（第 9d 章）中 pool 軸的實例化。
