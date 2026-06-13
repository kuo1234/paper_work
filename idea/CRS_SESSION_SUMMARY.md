# CRS Session 進度總結（2026-06-13，自主推進階段）

> 你離開前下的 /goal：「繼續找出**強框架 + 亮眼正面操作點**的解決辦法，自行決定，別問」。
> **已達成。** 以下是回來後一眼看全貌的總結。詳細記憶在 `crs-pivot-conformal-referring-set.md`。

---

## 一句話結論

找到了核心 invention：**Cross-Base Conformal Composition** —— 兩個 frozen base 各司其職
（OWL-ViT 分數判斷「該不該答」、GroundingDINO 候選負責「答得準」），用 Learn-then-Test
聯合校準出**召回+棄答雙保證**，在 gRefCOCO 達到 **≈GT 基數的精準集合（~3 框）**，
有 bootstrap CI 與 2×2 ablation 背書。這是「強框架 + 亮眼正面操作點」。

---

## 證據鏈（全部 commit + 存記憶）

| # | 結果 | 數字 | 狀態 |
|---|---|---|---|
| 新穎性 | conformal × frozen referring-grounding set selection 是空白 | 四象限文獻交叉點無人做 | ✅ |
| 單風險 CRS validity | nominal α vs empirical FNR 對角線追蹤 | 三 split 成立 | ✅ |
| 多風險 LTT | 天真序貫校準失敗 → Learn-then-Test 聯合保證 | 三 split BOTH OK | ✅ |
| **集合可分性** | GDINO target 子問題集合 vs OWL-ViT | **小 5.4×**，召回下限 0.004 vs 0.062 | ✅ |
| **★Composition 亮眼操作點★** | OWL gate + GDINO box，雙保證 | **集合 3.2 框**（純 base ~40），R1=0.20/R2=0.13 守住 | ✅ val |
| Bootstrap CI | config 固定只 resample test 1000× | 集合 CI [3.09,3.45]，R1/R2 CI 上界 <0.3 | ✅ val |
| **2×2 ablation** | gate×box 對調 | COMPOSE 雙軸最佳/reverse 雙軸最差（對角線）→ 互補真機制 | ✅ val |
| 三 split 覆現 | testA/testB | 自動流水線跑中 | ⏳ ~2.5h |

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

## 仍在自動進行 / 回來後可決策

- **流水線**（spark PID 3465513）：GDINO testA→testB dump → 自動跑 RESULT_*.txt
  （tp_separability / money_figure / full_results / compose_threesplit / compose_ablation）。
- 回來檢查 `~/selective-grounding/dump/RESULT_*.txt`，確認亮眼操作點在 testA/testB 也成立。
- **若三 split 都成立**：CRS = 碩論主結果，進入紮實寫作（章節草稿已備：
  `chapter_crs_conformal_referring_set.md` 含 4.5b 核心 invention 節）。
- **可選深化**：P3 cross-base 校準 label-efficiency（C4 限制 → label-efficiency 正面命題）。

## 關鍵檔案
- 章節：`idea/chapter_crs_conformal_referring_set.md`（主方法章）、`idea/chapter_method.md`（框架）
- 腳本：`idea/crs_src/_*.py`（spark `~/selective-grounding/src/` 同名無底線）
- 主圖：`idea/figures/crs_money_setsize.png`
- 記憶：`crs-pivot-conformal-referring-set.md`（5 個 gate 全紀錄）
- spark 分支：`feature/crs-conformal-referring-set`
