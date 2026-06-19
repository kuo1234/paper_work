# Decomp CRS 成本表 (P1-E) + 負增益失敗審計 (P1-F) 結果（2026-06-18）

## P1-E：誠實成本表（`src/decomp_cost.py`）

關鍵發現:routing 在**每個 target-present query 都呼叫 VLM**(判斷要不要拆),no-target 直接 passthrough 0 次。所以 VLM 成本可從 `no_target` 欄位精確還原,不只是 decomposed 比例下界。

| split | 總query | target-present(VLM) | decomposed | GDINO fwd/q | VLM/q 實際 | fwd倍數 |
|---|---|---|---|---|---|---|
| val | 14229 | 5324 (37.4%) | 875 | 1.065 | 0.374 | 1.065× |
| testA | 19200 | 14752 (76.8%) | 1243 | 1.078 | 0.768 | 1.078× |
| testB | 16063 | 11390 (70.9%) | 974 | 1.070 | 0.709 | 1.070× |

n_parts 分布(decomposed): val 2→840/3→28/4→6/5→1; testA 2→986/3→256/4→1; testB 2→832/3→142

### 成本表（論文用）

| Method | Train cost | Model update | VLM calls/q | GDINO fwd/q | Avg set size (v/A/B) |
|---|---|---|---|---|---|
| Frozen (full-expr) | none | frozen | 0 | 1.00 | 3.24/2.02/3.50 |
| Decomp (VLM-routed) | none | frozen | 0.37/0.77/0.71 | 1.06/1.08/1.07 | 3.24/1.96/3.34 |

**claim（紅隊認可）**:training-free and weight-frozen, **but with additional inference-time compute**。誠實點:
- GDINO forward 倍數溫和(1.06-1.08×,因只 6-16% 真拆且多 n_parts=2),但**這遮蔽了 VLM 負擔**——一個 7B VLM forward 遠比一次 GDINO 貴,故 VLM/GDINO 分兩欄不合併。
- VLM/q 報「實際」(每 target-present)非「decomposed 比例」(下界 0.06)。testA/testB 偏高因 target-present 占比高。

## P1-F：負增益 subgroup 失敗審計（`src/decomp_failaudit.py`）

負增益 = decomp pool oracle recall **嚴格低於** full pool 的真拆 case。啟發式自動分類(可信度標註):

| split | 負增益數 | 佔真拆% | n_gt==1負增益 | ambiguity | over_decomp | detector_miss |
|---|---|---|---|---|---|---|
| val | 71 | 8.1% | 0 | — | — | — |
| testA | 103 | 8.3% | 14 | 7 (50%) | 7 (50%) | 0 |
| testB | 144 | 14.8% | 33 | 16 (48.5%) | 17 (51.5%) | 0 |

**方法學修正**:testA n_gt==1 負增益全部 dec_recall==0.0,但這是「decomposition 把雙指稱拆成兩個單物件 prompt,每 part 框到另一物件、都不覆蓋唯一 GT 框」——是 ambiguity 表現成零覆蓋,**非 detector 失明**。故 n_gt==1 改語言模式優先分類。

### 誠實結論（修正原推斷）

原本只看 12 例斷言「全是標籤歧義」**過度概括,降調為**:
1. testA n_gt==1 負增益**確認約 50% 是明確 annotation_ambiguity**(兩側不同名詞雙指稱、GT 單框)——原推斷核心機制成立,但只佔一半。
2. 另一半歸 over_decomposition(LOW conf),但這是啟發式盲區:`second from right bottle and the center bottle`、兩個 plant 這類**共用 head noun 的同類兩實例**,人讀也是雙指稱(也是 ambiguity),只是表面共用名詞讓規則無法區分。→ over_decomp 桶**很可能高估**,真 ambiguity 比例只會更高。
3. **無 detector_miss、無 parser_error**:傷害是拆解把單框 GT 對應的雙指稱語言拆散,非偵測漏框或解析亂拆。

**論文措辭(取代舊版)**:decomposition 對 explicit multi-target(n_gt>1)有益,但對 **single-target 標註下的多指稱/同類兩實例 expression**(語言指兩物、GT 只標一框)有害——本質是拆解放大「語言-標註粒度錯配」,非偵測/解析失敗。可信度:ambiguity「至少 50%、實際更高」是站得住的下界陳述。
