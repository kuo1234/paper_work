# CRS Session 進度總結（2026-06-13，自主推進階段）

> 你離開前下的 /goal：「繼續找出**強框架 + 亮眼正面操作點**的解決辦法，自行決定，別問」。
> **已達成。** 以下是回來後一眼看全貌的總結。詳細記憶在 `crs-pivot-conformal-referring-set.md`。

---

## 一句話結論

找到了核心 invention：**Cross-Base Conformal Composition** —— 兩個 frozen base 各司其職
（OWL-ViT 分數判斷「該不該答」、GroundingDINO 候選負責「答得準」），用 Learn-then-Test
聯合校準出**召回+棄答雙保證**。全量 val/testA/testB 三 split 完成後，composition 在
α=β=0.3 下輸出 **3.21 / 2.02 / 3.48 框**，皆壓到接近 GT 基數且 R1/R2 全部守住；
bootstrap CI 與 2×2 ablation 證明不是小樣本僥倖。這是「強框架 + 亮眼正面操作點」。

---

## 證據鏈（全部 commit + 存記憶）

| # | 結果 | 數字 | 狀態 |
|---|---|---|---|
| 新穎性 | conformal × frozen referring-grounding set selection 是空白 | 四象限文獻交叉點無人做 | ✅ |
| 單風險 CRS validity | nominal α vs empirical FNR 對角線追蹤 | 三 split 成立 | ✅ |
| 多風險 LTT | 天真序貫校準失敗 → Learn-then-Test 聯合保證 | 三 split BOTH OK | ✅ |
| **集合可分性** | GDINO target 子問題集合 vs OWL-ViT | **小 5.4×**，召回下限 0.004 vs 0.062 | ✅ |
| **★Composition 亮眼操作點★** | OWL gate + GDINO box，雙保證 | **三 split 集合 3.21 / 2.02 / 3.48 框**，R1/R2 全部守 α=β=0.3 | ✅ 三 split |
| Bootstrap CI | config 固定只 resample test 500× | val/testA/testB 集合 CI 分別 **[3.10,3.32] / [1.97,2.06] / [3.35,3.60]**，R1/R2 CI 上界皆 <0.3 | ✅ 三 split |
| **2×2 ablation** | gate×box 對調 | OWL gate 控可用棄答/誤選、GDINO box 控小集合；GD gate 在 testA/B 退化成 over-abstain | ✅ 三 split |
| **三 split 覆現** | val/testA/testB 全量 | GDINO rows 14229 / 19200 / 16063，RESULT_*.txt 全部產出 | ✅ 完成 |

---

## 為什麼這次不一樣（對比舊 C3+C4+M4）

- 舊四貢獻是 **reliability measurement**（測量訊號多 informative）→ 你判定「絕對不夠」。
- CRS 是 **reliable set construction with guarantees**（建構有 distribution-free 保證的集合）→ 方法級。
- M4 那道牆（multi-target exact-match 做不到）從「失敗/邊界」翻成「保證代價的根本刻畫 + 用 composition 突破」。

## 護城河（躲過你紅隊三地雷）
1. 非 TTA：兩 base 都 frozen、不訓練、不合成框，只在 decision layer 做 conformal。
2. 非 detector comparison：不是比誰準，是**組合兩 base 互補強項**（ablation 證實）。
3. 不撞 True-False Verification (2509.09958)：那是單答案 verify，這是集合層雙保證 + 計數。
4. 接回你的 C4 cross-base 主軸（訊號結構跨 base 可用）。

---

## 最後完成狀態 / 下一步

- **spark 流水線已完成**：`~/selective-grounding/dump/gdino_gref_{val,testA,testB}.jsonl`
  rows = 14229 / 19200 / 16063；`dump/RESULT_*.txt` 全部產出
  （tp_separability / money_figure / full_results / compose_threesplit / compose_ablation）。
- **最終三 split composition（α=β=0.3）**：
  - val：COMPOSE sz=3.21 CI[3.10,3.32], R1=0.193, R2=0.134；純 OWL sz=8.16。
  - testA：COMPOSE sz=2.02 CI[1.97,2.06], R1=0.257, R2=0.182；純 OWL sz=9.14。
  - testB：COMPOSE sz=3.48 CI[3.35,3.60], R1=0.169, R2=0.235；純 OWL sz=7.07。
- **判讀**：三 split 都成立，CRS = 碩論主結果；testA 最亮（2.02 框≈GT 基數），testB 也守雙保證且集合 <3.6。
- **2×2 ablation 更新判讀**：val 是最乾淨對角線；testA/testB 顯示 GD gate 會靠 over-abstain 作弊式壓低風險，
  反而強化「OWL gate 才是可用棄答軸、GDINO box 才是小集合軸」的 factorization 主張。
- **下一步**：把 [chapter_crs_conformal_referring_set.md](chapter_crs_conformal_referring_set.md) 改成英文主結果版本，
  補主圖/表（composition 三 split + ablation + cost Pareto），再做 P3 label-efficiency 或直接寫作。

## 關鍵檔案
- 章節：`idea/chapter_crs_conformal_referring_set.md`（主方法章）、`idea/chapter_method.md`（框架）
- 腳本：`idea/crs_src/_*.py`（spark `~/selective-grounding/src/` 同名無底線）
- 主圖：`idea/figures/crs_money_setsize.png`
- 記憶：`crs-pivot-conformal-referring-set.md`（5 個 gate 全紀錄）
- spark 分支：`feature/crs-conformal-referring-set`
