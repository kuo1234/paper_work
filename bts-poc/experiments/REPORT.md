# BTS PoC — 實驗報告（誠實版）

最後更新：2026-06-05（v7 partial-obs + ctx bottleneck 已實作，待跑遠端 matrix）

---

## 1. 研究目標（提醒）

把 meta-RL 的 task belief 推論，從「測試時要先互動」解放成「**免互動的多模態規格**」：
規格 = **prior**，觀察 = **likelihood**，讓 policy 對 belief 做 **risk-aware** 行為（BTS）。

本 PoC 用 toy GridWorld 驗證三個現象，決定值不值得搬到 CALVIN：
- **P1**：規格歧義 → 初始 belief entropy 更高（prior 有作用）
- **P2**：觀察進來 → belief entropy 下降（likelihood 有作用）
- **P3**：belief / risk-aware 版 > single-point baseline（belief 對**決策**有價值）

---

## 2. 最終結論（一句話）

**P1、P2 真實且穩健（5 seeds 驗證）。P3 在 fully-observable toy（v6）與 partial-obs + ctx-bottleneck toy（v7）下皆未通過統計檢驗。**

核心推論鏈「spec=prior → observation=likelihood → belief 正確收斂」成立；
但「belief 對決策有額外價值、勝過單點」這條**沒有**在此 toy env 站住。

**v7 關鍵新發現（經三輪診斷定錨）**：P3 不過的真因**不是 belief、不是 multi-modal、不是 covariate shift、不是 unseen 泛化**，而是**最底層的 policy 學不起來**——模型連 seen task、餵 ground-truth history，下一步動作也只對 55%（測試 D，§6.9）。P3 需要一個本身能可靠導航的 policy backbone 為前提，toy 的「座標 + 小 transformer + 最短路 BC」先天不提供。**建議收掉 toy P3、帶 take-away 上 CALVIN。**


---

## 3. 定量結果（v6，5 seeds × {single, dual}，為準）

來源：`experiments/summary.json`

| 現象 | single (n=5) | dual (n=5) | 判讀 |
|---|---|---|---|
| P1 gap | `+1.023 ± 0.044` | `+0.986 ± 0.039` | ✅ 強且穩 |
| P2 slope | `−0.027 ± 0.012` | `−0.035 ± 0.010` | ✅ 穩定下降 |
| P2 drop | `+0.441 ± 0.021` | `+0.535 ± 0.061` | ✅ 穩 |
| **P3 return_gap** | `+0.011 ± 0.097` | `−0.014 ± 0.072` | ❌ 均值≈0、變號 |

- dual return_gap 每 seed：`[−0.011, −0.104, +0.075, +0.033, −0.063]` → 標準差 >> |均值|，典型「無效應 + 噪音」。
- **ablation**：dual 未優於 single，第二 hint 額外貢獻 `−0.025`（無因果）。

---

## 4. 為什麼一度誤以為 P3 成立（重要教訓）

單 seed 的演進看起來像穩定上升：

| 版本 | P3 指標（單 seed） |
|---|---|
| v3 | success_gap `+0.014` |
| v4 | return_gap `+0.054` |
| v5 | return_gap `+0.092` |

**這個上升趨勢是 seed 噪音造成的假象。** train.py 在 v6 前根本沒設 torch seed（且 DataLoader shuffle=True），每次訓練的 weight 初始化與 batch 順序都不同；單一 run 的 return_gap 標準差約 0.07–0.10，和「看起來的進展」同量級。多 seed 一跑，均值塌回 ≈ 0。

**教訓**：單 seed + 小 n_eval 的差異，在沒有跨 seed 變異估計前，不能當成效應。先做這個 de-risk，避免在昂貴的 CALVIN 上才發現 P3 站不住。

---

## 5. P3 失敗的根因診斷

在這個 **fully-observable + behavior cloning + greedy argmax** 的 toy env：

1. 物件身分（color/shape/pos）從 t=0 就全在 observation 裡，不是漸進揭露。
2. single baseline 的 policy 雖只吃 argmax one-hot，但它的 transformer `ctx` 一樣看得到所有 hint token → **不需要顯式 belief 也能解出 target**。
3. 第二個 hint（v5）沒改變這點：兩個模型最終都能從 ctx 解出該往哪走，belief 的顯式表徵是**冗餘**的。
4. v4 的不對稱懲罰是 eval-only、不進 loss，無法在訓練時塑造「belief-aware 才有的保守行為」。

→ 在 belief 可被 ctx 繞過的設定下，P3 結構上難成立。

---

## 6. 下一步選項（未定，待決策）

要讓 P3 真正有舞台，需要讓 belief **在結構上不可繞過**：

- **A. partial observability**：物件身分非 t=0 全可見，必須靠 belief 跨時間累積。最可能讓 P3 在 toy 成立，但動到 env 較大。
- **B. risk-aware 進 training objective**：CVaR / worst-case over belief 進 loss（非 eval-only），讓 belief 在優化目標裡真正被用到。
- **C. 直接上 CALVIN**：承認 P1/P2 已支持核心鏈，P3 在 fully-observable toy 難成是預期內；CALVIN 的 partial-observable + 高維才是 P3 的舞台。風險：CALVIN 昂貴，若 P3 仍不成立代價高。

判斷：A 或 B 是「在便宜環境把 P3 先做出來」的穩健路線；C 是賭更大但更接近最終目標。建議優先 A（partial observability），它最直接對應「belief 不可繞過」這個缺失條件。

---

## 6.5 v7 de-risk：partial-obs + ctx bottleneck（已實作，待跑遠端）

採 A，並追加一個 v6 沒處理的更深層根因。

**第二根因（v6 沒抓到的）**：不只 env fully-observable，連 **model 架構**都讓 belief 可繞過——`models/transformer.py` 中 belief 模型（`TinyBTS`）與 single baseline（`SinglePointBaseline`）的 policy head **都吃完整的 `ctx`**（transformer 表徵），只差 belief channel 是 full dist vs argmax。只要 `ctx` 在，揭露的資訊對兩者都進得來，belief 是冗餘的。這正是 P3 失敗的同一個機制。

**v7 兩條一起做**：
- **(A) partial observability**：env 加 `observe_object_identity` flag（預設 True = fully-obs，byte-identical 重現 v6）。False 時鄰格揭露（Manhattan ≤ 1，sticky），物件位置永遠可見、只遮 color/shape + 加 seen bit。逼 belief 跨時間累積。
- **(D) 限制 single 的 ctx**：`SinglePointBaseline` 改吃 `[nav_feat, point_onehot]`（nav_feat = 最後有效 obs token 的位置資訊，identity-free），**不再吃 ctx**，使 belief channel 成為唯一的 task-identity bottleneck。對應 method_spec §6.3「belief vs 單點」核心 ablation。belief 模型 `TinyBTS` 不動。

**不改的東西**：oracle posterior 不改（保證 P1/P2 監督目標 byte-identical、必續成立）；reward / training loss 不改（B = 不對稱懲罰進 loss 是後續步驟，只在 A+D 訊號不足時才做）。

**注意**：因改了 `SinglePointBaseline` 架構，**v6 的 single checkpoint 失效**，fully-obs control 須用新 code 重訓（與 partial-obs 同條件對照）。

**本地驗證（7 項全過，不含 GPU 訓練）**：fully-obs byte-identical、obs_dim 差 = n_objects、reveal 在 data-gen 與 eval 逐步重現、t=0 遮罩 == (Manhattan≤1)、oracle 不變、nav_feat 不洩漏被遮身分、端到端 pipeline 跑通。2-epoch smoke 早期方向符合預測：`dual_full return_gap −0.083` vs `dual_partial +0.045`（無統計意義，僅 sanity）。

### Pre-registration（跑遠端前先承諾，避免重蹈 v6 單 seed 自欺）

> **P3 通過條件（5 seeds）**：`dual_partial return_gap` 的 `mean − 1×std > 0`，且每 seed 同號 ≥ 4/5。
> **對照要求**：`dual_full return_gap` 仍 ≈0（落在 v6 區間 ±0.1 內），否則效應無法歸因於 partial-obs + bottleneck。
> 兩者同時成立才記 P3 通過。否則記未通過，**不再事後調參找訊號**——若 A+D 不過，結論是 P3 需要的不只 partial-obs，而是高維/長 horizon（→ 走 C/CALVIN）。

### v7 定量結果（5 seeds × {single,dual} × {full,partial}，已跑完）

來源：`experiments/summary.json`（2026-06-05 遠端）

| 組別 | P1 gap | P2 drop | P3 success_gap | **P3 return_gap（主判準）** |
|---|---|---|---|---|
| single_full | `+1.023 ± 0.044` | `+0.441 ± 0.021` | `+0.078 ± 0.049` | `−0.103 ± 0.102` |
| single_partial | `+0.774 ± 0.675` | `+0.460 ± 0.103` | `+0.040 ± 0.033` | `−0.184 ± 0.085` |
| dual_full（control） | `+0.986 ± 0.039` | `+0.535 ± 0.061` | `+0.039 ± 0.043` | `−0.100 ± 0.082` |
| **dual_partial（headline）** | `+0.565 ± 0.696` | `+0.482 ± 0.118` | `+0.042 ± 0.052` | **`−0.117 ± 0.079`** |

**P3 判讀（依 pre-registration，未事後調參）**：
- dual_partial return_gap 每 seed `[−0.088, −0.207, −0.169, −0.118, −0.002]` → **5 seed 全負**。
- pre-reg 判準 `mean − 1σ = −0.196 > 0`？ → **未通過**。
- `partial − full = −0.017`：partial-obs 不但沒讓 belief 勝出，反而略差。
- 這次的負是**方向一致的負**（非 v6 的均值≈0+變號），比 v6 更可信——是「belief 反而吃虧」，不是「無效應+噪音」。

**P1/P2 仍成立**，但 partial 的 **P1 gap 變異暴增**（`±0.675`/`±0.696`）：partial-obs 下某些 seed 的 belief entropy 行為不穩。oracle 不變的設計讓 P1/P2 核心鏈未被破壞（符合預期）。

## 6.6 P3 不過的真因診斷（路徑 2 的起點）

**最強線索**：v7 per-seed 顯示 belief 模型 **success_gap 多為正**（更常走到 target）但 **return_gap 全負**（累積 return 更差）。

| | belief 平均 return | single 平均 return |
|---|---|---|
| dual_full | ≈ −0.49 | ≈ −0.39 |
| dual_partial | ≈ −0.49 | ≈ −0.38 |

→ belief 模型「**走到了，但走得低效**」：寬 belief 下猶豫/繞路/多踩中途懲罰；single argmax 一個 task 後**果斷直衝**，效率高、偶爾錯。

**推論**：P3 不過**不是 belief 沒舞台**（partial-obs 已給了），**而是 policy 沒被教會怎麼用寬 belief**。method_spec §3.4 主張「belief 寬時 risk-aware 行為（info-gathering / CVaR 保守）」，但**目前 training 完全沒有 risk 項**——belief 寬時 policy 沒被訓練成「值得探索才探索、不確定時走對多個高機率任務都不差的動作」，所以寬 belief 對 rollout 是**負擔不是資產**。這正是 v4 就點出的「不對稱懲罰是 eval-only、不進 loss」（§5 第 4 點）未解的後果。

**路徑 2 = 在動 CALVIN 前，先驗證「補上 risk-aware 訓練目標（B）後，belief 是否從負擔變資產」。** 若補 B 後 dual_partial return_gap 翻正 → P3 的真因確認、novelty 站穩；若仍不過 → 才認定 toy 不夠、走 C/CALVIN。

> ⚠️ **下方 6.7 的實測診斷推翻了本節（6.6）的「belief 走到了但低效」假設。** 保留本節記錄推論過程，但結論以 6.7 為準。

## 6.7 實測診斷：真因是 rollout 崩壞（covariate shift），不是 belief（路徑 2 結果）

用 `experiments/diagnose_p3.py` 分解 rollout 結局（success / wrong_object / timeout）+ 步數，對 dual_partial 與 dual_full 各跑 seed1（return_gap 最負）。**結果推翻 6.6 假設**：

**dual_partial seed1**（189 歧義 episodes）：
| 模型 | success | wrong_object | timeout |
|---|---|---|---|
| belief | 12.7% | **55.6%** | 31.7% |
| single | 14.3% | 32.3% | **53.4%** |

**dual_full seed1**（對照，無揭露問題）：
| 模型 | success | wrong_object | timeout |
|---|---|---|---|
| belief | 11.6% | **58.2%** | 30.2% |
| single | 9.0% | 34.9% | **56.1%** |

**三個顛覆性發現**：
1. **success_rate 全部只有 9–14%**——連 fully-obs + dual-hint（task 完全可由 hint 決定）都只有 ~10%。低 success **不是 partial-obs 造成的**（full/partial 數字幾乎一樣），是更根本的 **BC policy 自由 rollout 崩壞**。
2. **belief return 更負的真因 = 衝去踩 wrong_object（−1.0 重罰）55–58%**，不是「走到了但低效」。belief 太衝動、往錯物件衝；single 太保守、龜縮吃較輕的 timeout step penalty（53–56%）。兩者用不同方式爛。
3. **rollout 崩壞蓋過了 belief vs single 的任何差異**——P3 的訊號被淹沒。

**真根因**：BC 在 multi-modal 資料（同 spec 配多個 task）下 **塌成平均動作**——往幾個候選 target 的中間/錯誤方向走，一出手就常踩錯。teacher-forcing 訓練看不出來，自由 rollout 才暴露 covariate shift。**這正是 innovation_notes 第 25 點（C-BeT / From Play to Policy）的核心警告**：uncurated/multi-modal 資料 BC 必塌成平均動作，需 multi-modal generation（k-means+offset / diffusion）才不崩。toy 撞上同一面牆。

**修正後的路徑**：P3 不過的真因**既不是 belief 沒舞台、也不是缺 risk 項**，而是**底層 policy class（單峰 BC + greedy argmax）撐不住 multi-modal 規格**。在這個崩壞被修好前，belief vs single 的比較沒有意義——任何 risk-aware loss（B）都會被 ~85% 的 rollout 失敗噪音淹沒。

## 6.8 再診斷：連 EXACT 規格都崩壞 → 真因是 BC 導航本身，非 multi-modal（路徑 2 續）

6.7 推論「multi-modal 塌成平均動作」。兩個更細的判別測試**再次推翻它**：

**測試 A — decoding（greedy vs sample，dual_full s1）**：
| | GREEDY success | SAMPLE(T=1) success |
|---|---|---|
| belief | 11.6% | 12.7%（幾乎沒變）|
| single | 9.0% | 20.6%（翻倍）|

→ sampling 救得了 **single 的 argmax 卡死 timeout**（部分次因），但**救不了 belief**（12.7% 仍極低）。decoding 不是主因。

**測試 B — EXACT vs AMBIGUOUS 規格（belief, dual_full s1）**：
| 規格 | success | wrong_object |
|---|---|---|
| AMBIGUOUS | 12.0% | 58.0% |
| **EXACT（零歧義、零 multi-modal）** | **17.5%** | **47.0%** |

→ **決定性**：連 EXACT（task 完全確定、無任何 multi-modality）的 rollout success 也只有 17.5%、踩錯 47%。**這排除了 multi-modal 假設**——問題在更基礎的層級：**BC policy 自由 rollout 根本不會穩定導航到指定物件，連單一明確 target 都做不到。**

**真根因（最終）= BC compounding error / covariate shift**：teacher-forcing 訓練每步看專家軌跡上的 state；自由 rollout 一走偏就進入訓練沒見過的 state，誤差滾雪球。`generate.py` 的 15% inject_stay 噪音遠不足以覆蓋 rollout 的 state 分布。

**對「修 toy policy」的修正**：multi-modal head（C-BeT / diffusion）**不是病灶**——EXACT 沒有 multi-modality 卻照樣崩。真正要修的是 **rollout 穩定性**：
- (i) **DAgger / 強化 noise-injection**：讓訓練資料覆蓋偏離態（最便宜，先試）。
- (ii) closed-loop / on-policy 修正（較重）。
- multi-modal head 對 EXACT 零幫助，列為非優先。

**對整體決策的意涵**：toy 用「離線 BC + 最短路專家」這個 data/policy 組合，先天會在自由 rollout 崩壞，使 P3（需自由 rollout 比較 return）無法在此設定觀測。這本身是強結論——**P3 的舞台需要一個 rollout-穩定的 policy class**，而 CALVIN 的 backbone（Octo / diffusion action head + 大規模 play data）天生具備。toy 已問出「需要什麼」。

## 6.9 第三輪診斷：DAgger 無效 → 真因是「連 seen task 都學不會」（路徑 2 最終）

依使用者指示先試最便宜的 DAgger-lite（closed-loop 專家重標偏離態，`generate.py --dagger-noise 0.3` + env `expert_next_action`）。**又一次推翻 6.8 的 covariate-shift 假設**：

**測試 C — DAgger-lite（fully-obs dual s1, noise=0.3）**：EXACT rollout success `17.5% → 14.0%`（**沒救，略降**）。

→ covariate shift 若是主因，DAgger 應拉高 success。沒拉高 → 不是主因。再往下挖：

**測試 D — teacher-forced 1-step action accuracy（餵 ground-truth history 預測下一動作）**：
| 資料 | TF action acc | rollout success |
|---|---|---|
| **SEEN tasks（train set）** | **55.1%** | 19.3% |
| UNSEEN green_triangle（test） | 52.4% | 15.3% |

→ **決定性最終發現**：模型**連訓練看過的 task、餵 ground-truth history，下一步動作也只對 55%**。seen 與 unseen 幾乎一樣（排除「unseen-task 不泛化」假設）。55% per-step → 5 步路徑全對機率 `0.55^5 ≈ 5%`，與觀測到的 ~10–19% rollout success（多條等長路徑放寬）吻合。

> 補充：v6/v7 報的 `test_action_loss ≈ 16`（vs train 0.78）是因 test=held-out `green_triangle`，模型對 unseen task 給「自信但錯」的 logits。但測試 D 證明**問題不在 unseen——seen task 的 TF acc 也只有 55%**。

**真根因（最終定錨）= policy 本身學不起來**，不是 belief、不是 multi-modal、不是 covariate shift、不是 unseen 泛化。在這個架構/設定下，模型無法從 normalized 座標可靠學出「哪個方向縮短到 target 的距離」這個空間函數。可疑點：(i) action 只從最後 token 的 `ctx` 預測，agent↔target 相對向量要靠小 transformer 從座標算出來，太難；(ii) belief KL（λ=0.5）可能與 action loss 競爭；(iii) 座標表徵（normalized float）對方向推理不友善。

**對 P3 與下一步的最終意涵**：
- P3 在此 toy 結構上**不可能觀測到**——belief vs single 的差異需要一個會導航的 policy 為前提，但 policy 連 55% per-step 都到不了，訊號被淹沒。三輪診斷（belief 低效→multi-modal→covariate shift）逐一被便宜測試推翻，最終定位到最底層的 policy 學習能力。
- 這是 toy 階段能得到的**最有價值結論**：核心鏈 P1/P2（belief 推論）真實穩健；但要驗證 P3（belief 對決策有價值），**需要一個本身就能可靠導航/操作的 policy backbone**——正是 CALVIN 的 Octo / diffusion action head + 大規模 play data 提供、而「toy 座標 + 小 transformer + 最短路 BC」先天不提供的。
- **建議：收掉 toy 的 P3 嘗試，帶三個明確 take-away 上 CALVIN**：(1) P1/P2 機制已驗證可移植；(2) P3 需要 rollout-穩定且能學會 control 的 backbone（非自製小 policy）；(3) belief head + risk-aware + calibration 應疊在成熟 backbone 上，而非在學不會導航的玩具上比較。

## 6.10 找到並修復一個真 bug：專家 BFS 穿過錯物件（但非主因）

回頭檢查專家資料品質，發現一個真 bug：`shortest_path_actions` 的 BFS **不把非 target 物件當障礙**，最短路常直接穿過錯物件格 → 觸發 `wrong_object`(−1.0) 提前失敗。量測：**dual-hint 下 33.7% 的「專家」軌跡自己就踩錯物件**，等於 1/3 訓練資料在教模型「走進錯物件」這個會被重罰的動作。

**已修復**：給 `shortest_path_actions` 加 `blocked` 參數，`expert_trajectory` / `expert_next_action` 傳入非 target 物件位置繞過。修後專家成功率 **66% → 97.7%**、wrong_object **33.7% → 0.1%**。

**但修了不是主因**：用乾淨資料重生 + 重訓（fully-obs dual s1），結果幾乎沒變：
| 資料 | TF acc | rollout success |
|---|---|---|
| SEEN(train) EXACT | 50.6% | 15.0% |
| UNSEEN(test) EXACT | 48.8% | 15.5% |
| SEEN(train) AMBIGUOUS | 69.4% | 13.5% |
| UNSEEN(test) AMBIGUOUS | 68.9% | 13.5% |

三個確認：
1. **SEEN ≈ UNSEEN**（50.6 vs 48.8）→ 再次排除 unseen 泛化問題。
2. **EXACT(50%) < AMBIGUOUS(69%)** 反常，但與導航能力無關：EXACT 軌跡極短（mean_len 3.6 vs AMB 13.3），per-step 分母小、起手步分佈被起點固定+BFS tie-break 嚴重傾斜（first-step up 624 vs left 138）。
3. **寬鬆判準**（pred ∈ 最優動作集合，on-policy）EXACT 仍只 **23.8%** → 排除「模型選了另一條等長最優路」的 tie-break 假設。模型是真的走錯方向/繞遠路。

**結論不變、但更穩固**：bug 真實且值得修（已修），但**不是 50% TF acc 的主因**。乾淨資料、seen task、餵 ground-truth history，模型仍學不會「往 target 走」。**這是「toy 設計」層級的問題**——見下節對「toy 是否太簡單」的最終判讀。

## 6.11 對「toy 是否太簡單」的最終判讀

不是「太簡單」，而是**設計與目標錯配**。逐項拆解：

- **太簡單會長怎樣**：模型輕鬆 100% success，P3 因「天花板效應」看不出 belief 價值。**這不是現況**（success 僅 10-19%）。
- **現況是「對小 policy 太難 + 規則互斥」**：(a) 從 normalized 座標學「方向 = f(agent_pos, target_pos)」對 4-layer tiny transformer 是硬的空間推理；(b) reward 結構（踩錯物件 −1.0 即死）與最短路專家在修 bug 前互斥；(c) BFS tie-break 讓監督訊號對「等價最優動作」有任意性。
- **更深的錯配**：toy 想驗證的 P3 是「belief 對**決策**有價值」，但它把「決策」實作成一個**需要從零學會空間導航**的子問題。導航本身的學習難度**淹沒**了 belief 的訊號——這不是 belief 假設錯，是**評測載體把兩個正交的難點綁在一起**。

**所以答案**：toy 不是太簡單，是**把「belief 推論」和「空間導航學習」耦合在同一個學不起來的小 policy 上**。P1/P2（純 belief 推論、不需 rollout）正因如此仍穩健；P3（需要會導航的 policy）才崩。修法不是「讓 toy 更難」，而是**解耦**——要嘛換一個導航不需從零學的設定（給相對座標/方向特徵、或用會導航的 backbone），要嘛直接上 CALVIN（backbone 已會 control，belief 疊上去）。





---


## 6.12 E0 / P3-oracle-decision：環境**有** decision value，但完整分布未必必要（外部建議第一步）

外部審閱（POMDP/QMDP/PBVI 視角）建議：在改 representation 或加 risk-loss 前，先做**不訓練的 oracle P3**，回答「環境到底有沒有 decision value」。實作 `experiments/oracle_p3.py`：三個 agent 共用同一套「最大化 belief 下期望 return」決策規則 + 同一個 BFS low-level controller，唯一差異是決策用的 belief。env 真實 step 結算實際 return（不用估計值報結果）。

**結果（2000 歧義 episodes, seed0, 本地 CPU）**：
| agent | success | avg_return |
|---|---|---|
| **belief**（完整分布） | **96.6%** | **+0.864** |
| single（argmax 單點） | 43.5% | −0.156 |
| single+H（argmax + entropy 純量, E6） | 96.6% | +0.864 |

- **belief − single return_gap = +1.02**（巨大），succ_gap = +0.53。
- **環境確實有 decision value**：用完整 belief 會「先收集 hint 再賭」幾乎不踩錯；argmax 單點一半機率賭錯吃 −1.0。→ **排除「環境沒 decision value」這個最壞可能**。neural P3 失敗確定是 learning/representation，不是環境。

**但關鍵警訊（E6 的價值）**：**single+H 完全追平 belief（gap=0.000）**。在此 toy，「argmax + 一個 entropy 純量」就拿到全部好處，**不需要完整分布**。因為決策只需「我夠不夠確定（要不要先收集 hint）」，一個 entropy 純量就夠；多峰分布的**形狀**不被需要。

**對 novelty 的意涵（比 P3 過不過更重要）**：method_spec 主張「需要**完整分布** belief」，但此 toy 上 entropy 純量即足夠 → reviewer 會問「full distribution 贏在哪？」。要讓「完整分布 > entropy 純量」，環境需要**形狀依賴的多峰歧義**：例如 ≥3 個分得很開的候選、entropy 相同但最優 hedge 動作因分布形狀而不同（單一純量無法區分）。**這是下一步設計的真正標的**，比單純修導航更關鍵。

## 6.13 反例測試：在改 env 前先證「完整分布 > entropy 純量」有結構來源（不訓練、本地秒跑）

延續 6.12 標的，但**不直接改 env 再跑 E0**（會浪費一輪 GPU、且改錯方向也可能假性通過）。先做**便宜判別測試**：純算地證明存在一種任務結構，使公平版 single+H **資訊上必然**輸給 belief。實作 `experiments/counterexample_p3.py`，沿用 env reward 常數，不依賴訓練、不碰遠端。

**公平版 single+H 的定義（修正 6.12 的洩漏）**：`oracle_p3.py:170` 的 single+H 在高 entropy 時**直接退化成完整 belief `b_raw`**——它其實偷看了完整分布，難怪追平。真正公平的 single+H 決策只能吃 `(argmax 物件身分, entropy 純量 H)` 兩個量，**看不到 mass 落在哪幾個候選、各多少**。本測試一律用此公平版，且給它**後見之明最大優勢**（每個 `(argmax,H)` 特徵桶可選桶內平均 EV 最高的固定動作）。連這種「最佳可能的 single+H」都輸，才算證成。

**兩輪負結果（重要，揭露 E0 追平的真根因）**：
1. **動作=賭單一候選物件** → 0 碰撞對。±1.0 對錯獎勵主導、step 成本太小，**最優動作幾乎永遠=argmax 物件**，連 single 都未必輸。
2. **動作=hedge 點、中央起點、寬鬆 horizon** → 0 碰撞對。belief 最優 hedge 對全部盤面都是「待在 START 原地等揭曉」（先移動的成本＞收益）。
   → **真相**：toy 的自然設定（去某物件、中央起點、寬鬆步數）下，完整分布**真的**不比 argmax 多買到東西。E0 追平不是 baseline 寫巧，是**任務結構沒給分布形狀任何發揮空間**。

**v3 成功模型 = 時限壓力下須 pre-commit**：真實「完整分布 > entropy」機制 = 揭曉前就得朝機率質量集中處預移。模型：target 身分在第 `t_reveal` 步才揭曉，agent 揭曉前須移動到 pre-commit 點 `p`，揭曉後從 `p` 走向真 target，**剩餘步數不足則 timeout 失敗**。最優 `p` 是 mass 幾何（加權 facility-location）的函數；entropy 相同但 mass 偏不同群 → 最優 `p` 方向相反。

**robustness sweep（防呆，判別真結構 vs 人造窄帶）**：掃 BUDGET（揭曉後剩餘步數）：

| BUDGET | 碰撞對 | belief | single | single+H | b−s+H |
|---|---|---|---|---|---|
| 2 | 77 | +0.337 | +0.260 | +0.311 | **+0.026** |
| 3 | 40 | +0.619 | +0.564 | +0.615 | **+0.004** |
| 4 | 30 | +0.644 | +0.564 | +0.622 | **+0.023** |
| 5 | 0* | +0.805 | +0.613 | +0.785 | **+0.020** |
| ≥6 | 0 | +0.950 | +0.61→0.93 | +0.950 | **+0.000** |

`belief − single+H > 0` 在 **BUDGET∈{2,3,4,5}** 穩定成立（非刀刃窄帶）。代表碰撞對：兩盤面 **argmax 物件 `(6,6)`、entropy `0.983` 完全相同**，唯一差別是次要 mass 落在右群(R2) 或左群(L1)，belief 最優預移方向**相反**（右 `(3,5)` vs 左 `(2,1)`）；single+H 只有兩個純量 → 對兩盤面必給同一預移點 → 必輸其一。

**兩個結論**：
1. ✅ **「完整分布 > entropy 純量」有乾淨、可複現的結構來源**：時限壓力 + pre-commit + 多峰質量。novelty 命題站得住。
2. ⚠️ **同時精確界定了命題邊界**：BUDGET≥6（步數寬鬆）時 gap 歸零、碰撞歸零——**這就是 E0 追平的精確解釋**（原 env horizon=20 對 7×7 太寬鬆，落在無效區）。誠實的 novelty 命題應收窄為「**belief 在『時限緊、須預先 commit、質量多峰』的決策下優於 argmax+entropy 純量**」，而非「普遍更優」。這版命題反而更強：它能解釋強 baseline 何時、為何追平。

**對 env 改造的明確指引（反例導出，非拍腦袋）**：(1) **揭曉延遲**——target 身分延後到第 ~3 步而非 t=0 全給；(2) **收緊 horizon**——把步數預算壓進 budget≈2–5 緊區；(3) **兩群多峰幾何**——候選分空間上分得開的兩群，次要 mass 偏向決定最優預移方向。三者缺一，分布形狀即失效。

執行：`python experiments/counterexample_p3.py`（本地秒跑，exit 0=反例成立）。**下一步**：據此進 plan mode 規劃 `gridworld.py` 改造，approve 後再動 code、再跑 E0。

## 6.14 E0 pre-commit：把反例機制移植進真實 env，belief 嚴格勝公平版 single+H（3 seeds，本地 CPU）

§6.13 在抽象模型上證成；本節把同機制移植進**真實 env**，用真實 `env.step` 結算驗證它不是抽象 artifact。**範圍嚴格收窄**：不改 env 動態、不重生資料、不碰 GPU、不破壞 P1/P2。實作為 `oracle_p3.py` 的 opt-in `--pre-commit` 路徑（legacy 路徑 byte-identical 保留）。機制純在決策迴圈複製：target 身分在 `t<t_reveal` 不可知；揭曉前須朝質量集中處預移；揭曉後衝真 target；`horizon=t_reveal+budget`，剩餘步數不足則 timeout 失敗。三 agent 共用同凍結盤面與 BFS controller，唯一差異是決策用的 belief；single+H 為**公平版**（只吃 `(argmax_pos, entropy)`，後見之明每桶選交集 reachable 格中平均 EV 最高者）。

**關鍵正確性修正（診斷推翻第一版）**：初版 sweep 出現 `single+H > belief`（反常）。單盤面診斷發現根因——`ev_precommit` 用「BFS 路長 + 曼哈頓式 EV」估值，**沒算「預移路徑穿過 wrong-object 致命格提前死」**，與真實 `env.step` 結算嚴重背離（某盤面 belief 預期 EV +0.95、實際 −1.02）。修法：`settle_ev` 改成**對每個候選 z 當真 task，直接呼叫 `run_episode_precommit` 用真實 env.step 跑、再 mass 加權**——EV 與結算共用同一套邏輯，徹底杜絕背離。三 agent 的 p* 選擇與 single+H 桶評分全改用 `settle_ev`。**延續方法論：先單盤面診斷找根因，不盲調參數。**

**結果（t_reveal=3, spec=ambiguous, 3 seeds, budget sweep）**：

| budget | belief | single | single+H | **b−s+H** (s0/s1/s2) |
|---|---|---|---|---|
| 2 | +0.86 | +0.65 | +0.59 | **+0.279 / +0.255 / +0.294** |
| 3 | +0.96 | +0.76 | +0.67 | **+0.301 / +0.249 / +0.277** |
| 5 | +0.96 | +0.79 | +0.76 | **+0.198 / +0.235 / +0.195** |
| 8 | +0.97 | +0.58 | +0.85 | **+0.111 / +0.175 / +0.111** |

**三個結論**：
1. ✅ **機制成功移植進真實 env**：belief（≈+0.97 近滿分）在**全部 budget、全部 3 seed** 嚴格勝公平版 single+H，符號穩定、量級一致（非單 seed 假象，延續 v6 教訓）。緊預算 gap 最大（+0.25~+0.30）。
2. ⚠️ **與抽象反例的誠實差異**：§6.13 抽象模型 budget≥6 gap **歸零**；真實 env **不歸零**（budget 8 仍 +0.11）。根因——真實 env 有 **wrong-object 致命格**，即使步數寬鬆，single+H 用「同桶單一固定點」仍會讓某些盤面預移撞雷，而 belief 每盤面客製化能避開。即「分布的價值在真實 env 比抽象模型**更持久**」（時限壓力 + 避雷雙重來源）。
3. 🎯 **novelty 命題在真實 env 站穩且更強**：method_spec「需要完整分布」不再只是抽象斷言——在真實 env 的 oracle 層，完整 belief 對決策有**不可被 (argmax,entropy) 純量替代**的價值，且此價值穩健跨 seed、跨 budget。E0 原本的「single+H 追平」確認為**舊 single+H 偷看完整分布 + horizon 過寬**雙重假象，已被公平版 + 緊預算推翻。

執行：`python experiments/oracle_p3.py --pre-commit --t-reveal 3 --budget-sweep --spec-mode ambiguous --n-episodes 300 --seed 0`（本地 CPU，分鐘級）。**範圍未含**：神經 E1–E3、資料重生、env 動態改造——延後到此 oracle 證據確立後再評估（見下「明確延後」）。

## 6.15 policy 為何學不起來：三個判別測試定錨「屬性-物件 binding」（本地 CPU）

路徑2 測試 D 顯示真實 transformer 連 SEEN task + teacher-forcing 單步 action acc 僅 55%。為精確找出瓶頸，用**同容量 memoryless MLP**做三個便宜判別測試（本地秒級、不碰 GPU、不碰主 pipeline），逐一排除假設。腳本 `experiments/factor{1,2,3}_*.py`。

| 測試 | 假設 | 設計（唯一變因） | 結果 | 判定 |
|---|---|---|---|---|
| 因素1 | 座標表徵 | ABS 絕對座標 vs REL 相對Δ（**給 target one-hot**） | 0.874 vs 0.873（gap −0.001） | ❌ 座標非瓶頸 |
| 因素2 | identity inference | 真實 vectorize_obs+spec，給/不給 target one-hot | 0.620 vs 0.629（gap −0.009） | ❌ 自推 target 沒問題 |
| 因素3 | multi-stage 切換 | 單一 MLP vs 每階段專屬 MLP | +0.015 | ❌ 切換非主因 |

**決定性發現（因素3）**：連 **phase2 專屬 MLP**（兩 hint 全揭露、target 完全無歧義、任務退化成「走到一個已知物件」）也只 **0.53**；而因素1 同樣「走到已知 target」但**直接給 target one-hot** 卻 0.87。差別就在：phase2 模型須從 obs 裡「揭露的 hint 屬性」**比對**出「盤面上哪個物件符合」，而 `vectorize_obs` 把 hint 屬性與物件身分編在**不同欄位**。

**最終真因 = 屬性-物件 binding（關係綁定）**：flat 單步向量無法承載「屬性↔物件」跨欄位綁定 → 即使 target 確定也只 53%。這精確解釋 **oracle（符號式 `compatible_tasks_given_state` 比對）96% vs neural（flat 向量學 binding）55%** 的落差。**與 belief 機制無關，是 flat 表徵的結構性極限。**

**四輪定錨總結**（座標→identity→切換→binding）：toy 的小 transformer 從 flat 向量學不出符號式屬性綁定，這是 P3（需自由 rollout 比 return）在 toy 觀測不到的底層原因。

**對下一步的意涵**：(1) 救 toy 需 **entity-centric/slot 表徵或 cross-attention**（讓物件能 query 屬性），非小修；(2) CALVIN 的 **Octo/diffusion backbone 本就用 entity/patch tokens + cross-attention**，天生解 binding；(3) 再次支持「P3 舞台在 CALVIN 不在 toy」。**robot 資料補充**：人類 play data（CALVIN/RT-X/Bridge）天生含「偏離後修正」軌跡，腳本最短路專家沒有——但 binding 是主瓶頸，修正資料為次要。

執行：`python experiments/factor1_repr_test.py`、`factor2_identity_test.py`、`factor3_stage_test.py`（皆本地 CPU、秒～分鐘級）。

```bash
# 在 SSH 遠端 venv
cd /home/p76141495/bts-poc && . .venv/bin/activate
# v7 完整 matrix：{single,dual} hint × {full,partial} obs × 5 seeds ≈ 40 訓練
SEEDS="0 1 2 3 4" OBS_MODES="full partial" bash experiments/run_matrix.sh
python experiments/summarize.py       # 印 P1/P2/P3 mean±std + P3 恢復 headline（dual_partial vs dual_full）+ pre-reg 判準
# 縮小版 smoke：SEEDS="0" N_EPISODES=300 EPOCHS=1 OBS_MODES="full partial" bash experiments/run_matrix.sh
```

