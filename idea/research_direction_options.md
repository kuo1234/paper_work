---
type: decision-doc
aliases:
  - "research_direction_options"
  - "研究方向決策"
tags:
  - 研究主線
  - 決策
summary: "可行方案、四個必釘定義、五個方案、風險與對策的決策文件。"
---
# 研究方向：可行方案決策文件

> 狀態：**探索中，尚未定案**。本文件收錄所有可行方案、待決定的定義、權衡與風險，供日後回頭挑選。
> 相關檔案：`paper/PEARL.md`、`paper/VariBAD.md`、`innovation_notes.md`（第 9–14 節為 VariBAD/belief idea）。
> 最後更新：2026-06-01

---

## 0. 一句話定位

把 **meta-RL 的 task inference**（從互動軌跡推任務 latent）換成 **language inference**（從語言推同一個 latent），讓兩者在同一 latent space 對齊，於是測試時**只給語言就能 zero-shot 指定任務**。

- 方向一（完整題目）：robot play data + 極少量弱語言標註 + offline meta-RL + 文字→task embedding 的 zero-shot 泛化。
- 方向二（核心機制）：meta-learning × zero-shot。**方向二是方向一能成立的關鍵技術**，不是另一個獨立題目。

**可行性結論：機制上沒有矛盾，且 LUMOS / T2DA 各自驗證了相鄰的一半，本研究是把這座橋接起來。**

---

## 1. 概念橋樑（為何兩個方向是一體）

meta-RL（PEARL / VariBAD）核心：

```
context inference:  q(z | τ)        從少量互動軌跡推任務 latent
policy:             π(a | s, z)
```

zero-shot 機制：把「軌跡推論」換成「語言推論」並對齊：

```
play data:          q(z | τ)        訓練：用 play trajectory 學任務 latent
weak language:      g(z | l)        用少量弱標註讓語言映到同一 latent
對齊:               g(l) ≈ q(τ)     描述同一任務的 l 與 τ 在 latent 上要接近
測試(zero-shot):    新指令 l → g(l) → π(a | s, g(l))    無 target demo / 無 gradient / 無 env interaction
```

VariBAD 額外貢獻：latent 是 **belief / task uncertainty**，於是「語言指定任務」有乾淨接口——文字當 **task belief 的 prior / interface**，而非直接當 command token。學術故事比 Octo/OpenVLA 的「語言直接 condition policy」更清楚。

---

## 2. 文獻地景（LUMOS、T2DA 已精讀查證，詳見 `paper/LUMOS.md`、`paper/T2DA.md`）

| 工作 | 做了哪幾塊 | 缺的那塊 |
| --- | --- | --- |
| LUMOS | play data + <1% 弱語言 + world model + zero-shot 語言操控 + sim→real（CALVIN 勝 HULC，已查證） | 無 offline meta-RL task belief，語言直接 condition policy，無 unseen-task 泛化框架 |
| T2DA | offline meta-RL + natural language supervision | 偏 supervision，非 play data 主軸 |
| Zero-shot task specification 那篇 | foundation models 做 task specification | 非 meta-RL latent / 非 play data |
| Octo / OpenVLA | 大規模 generalist language-conditioned robot policy | 語言直接 condition，無 meta-RL belief latent |

**空位（candidate novelty）：把「play data + 極少弱語言 + offline meta-RL belief latent + zero-shot task embedding」四塊整套一起做，且以 meta-RL/BAMDP latent 作為語言與行為的橋。**

---

## 3. 四個必須釘死的定義（每個都附選項）

### 定義 A — 跨任務變什麼？

| 選項 | 說明 | 對設計的影響 |
| --- | --- | --- |
| A1 Reward/goal 變 | 同動作不同目標（導航到不同目標、推不同目標物） | 只需 reward decoder，較單純，貼近 VariBAD MuJoCo |
| A2 Dynamics 變 | 不同物體質量/摩擦/操作方式 | 需 transition decoder，latent 要表示動力學，較難但區隔度高 |
| A3 兩者混合 | reward 與 dynamics 都變 | 最貼近真實 robot，latent inference 最難 |

### 定義 B — zero-shot 的嚴格界線（最易被 review 攻擊）

zero-shot 必須**同時**滿足：測試時(a)無 target task demonstration、(b)無 gradient update、(c)無新 task 的 env interaction。
偷吃任一項 → 變成 few-shot / fine-tune，賣點失效。守住 `innovation_notes.md` 第 9 節風險欄。

### 定義 C — 泛化目標

| 選項 | 說明 | 難度 |
| --- | --- | --- |
| C1 Unseen instruction | 看過的任務、沒看過的語言講法（組合/語言泛化） | 低，適合第一篇先證機制 |
| C2 Unseen task | 沒看過的任務本身（真 task 泛化） | 高，stronger result |
| C3 分階段 | 先 C1 證機制，再 C2 當延伸 | 工作量大 |

建議：第一篇別同時宣稱 C1+C2，先講清楚做哪個。

### 定義 D — offline 下的 action selection

純 offline（資料固定）下 VariBAD 的「Bayes-optimal 探索」**沒得探索**。
本研究的 offline 版本更像：offline 學好 `q(z|τ)`、`g(z|l)`、conditional policy，而非學探索策略。
→ 別在 offline 設定主打 BAMDP 探索賣點（會被指出空洞）；可改主打 **belief latent 作為語言-行為對齊接口**。

---

## 4. 可行方案清單（依野心由小到大）

### 方案 1：最小可行驗證（推薦作為第一步）
- 設定：A1（reward/goal 變）+ C1（unseen instruction）+ offline。
- 機制：play→`q(z|τ)`、language→`g(z|l)`、對齊 loss、`π(a|s,z)`。
- 賣點：證明「語言可映到 meta-RL latent 並 zero-shot 指定任務」。
- 風險最低，最可能跑出來，適合先建 pipeline 與正面結果。

### 方案 2：belief-conditioned（VariBAD 風味）
- 在方案 1 上把 latent 改成 belief（含 uncertainty），policy 吃 posterior 參數 (μ,σ) 而非單一 sample。
- 加「含未來的重建 ELBO」當 auxiliary（innovation 第 10 節）。
- 賣點：文字當 task belief prior，理論定位比直接 condition 更清楚。

### 方案 3：dynamics 泛化版
- 設定改 A2/A3，需 transition decoder，hidden-wind toy task（innovation 第 2 節）正好可當中間難度 benchmark。
- 賣點：latent 表示動力學，區隔「reward latent vs dynamics latent」inference 難度。

### 方案 4：弱標註極限（強 ablation 主軸）
- 系統性減少語言標註量，量出「降到多少會崩」的曲線。
- 這本身就是一張好圖，也是對 LUMOS「能做」之外的科學貢獻。

### 方案 5：完整題目（C2 unseen task + A3 + belief + 弱標註）
- 四塊整套，最強但風險最高，建議作為終極目標而非起點。

---

## 5. 主要風險與對策

| 風險 | 說明 | 對策 |
| --- | --- | --- |
| 對齊 `g(l)≈q(τ)` 在極少弱標註下不穩（成敗關鍵） | 標註太弱 → latent 對不準 → zero-shot 崩 | 方案 4：先用較多標註證機制，再逐步減量畫崩潰曲線當 fallback 結果 |
| policy 忽略 z 退化成普通 BC | latent 沒被用到 | actor-z sensitivity 診斷（innovation 第 7 節）、latent intervention |
| zero-shot 定義被質疑偷吃資料 | review 常見攻擊點 | 嚴守定義 B，論文明列三條界線 |
| offline 下 BAMDP 探索賣點空洞 | 資料固定無探索 | 改主打 belief latent 作對齊接口（定義 D） |
| novelty 與 LUMOS/T2DA 重疊 | 可能被指增量 | 待查證兩篇細節後，明確列差異表 |
| play data 缺 task diversity | latent 只學到 behavior style 非 task semantics | 設計時確保任務多樣性；用 true latent 對齊度診斷（innovation 第 3 節） |

---

## 6. 與現有 PEARL 重現專案的銜接

- 現有 toy task（hidden goal + wind，innovation 第 2 節）可直接當本研究的 debugging benchmark。
- innovation_notes 第 9–14 節（VARIBAD-style belief、含未來重建 ELBO、encoder 梯度解耦、task-belief-as-state、時間固定 embedding、OOD 診斷）都是本方向的可用零件 / ablation。
- 下一步若要推進，先從**方案 1 的 pipeline 草圖 + 與 LUMOS/T2DA 的差異查證**開始最穩。

---

## 7. 待辦（回頭時從這裡接續）

1. 決定定義 A / C（目前未定）。
2. 查證 LUMOS、T2DA 實際作法與限制，補第 2 節差異表。
3. 選一個方案（建議方案 1 起步），畫方法架構圖。
4. 把選定方案的 idea 對應成可跑實驗 + baseline + 評估指標。
