---
type: poc-plan
aliases:
  - "PoC 實作計畫"
  - "BTS PoC"
tags:
  - 研究主線
  - zero-shot
  - meta-rl
  - 實作
  - PoC
summary: "BTS 最小可跑 PoC 的完整實作計畫：自製 GridWorld 多任務 + 歧義語言規格 + in-context transformer + 顯式 belief head，單卡（A6000/DGX Spark）數小時可驗證三現象。"
---

# PoC 實作計畫：Belief-from-Specification（BTS）最小驗證

> 上游：[[method_spec|完整方法]]｜[[zero_shot_proposal|提案]]｜[[related_work_checklist|查證]]
> 目標讀者：你自己（之後照這份開 repo、寫 code、跑實驗）
> 硬體：RTX A6000（48GB）或 NVIDIA DGX Spark — 兩者對 PoC 都遠遠夠用（PoC 模型 < 5M 參數，單卡數小時）。

---

## 0. PoC 的唯一目標：證明「核心鏈」會動

不做完整系統，只證明這條因果鏈成立：

> **歧義規格 → 寬 belief → risk-aware 行為 → 隨觀察收斂 → 表現勝過單點 baseline**

要看到的**三個現象**（這就是 PoC 的成功判準）：

- **現象 1（歧義 → 寬 belief）**：給歧義規格時，belief 的 entropy 顯著高於精確規格。
- **現象 2（觀察 → 收斂）**：隨著環境觀察進入 context，belief entropy 單調下降、任務成功率上升。
- **現象 3（belief 有用）**：risk-aware（吃整個 belief）版本在歧義規格下，成功率顯著高於「單點 argmax」baseline（= 模擬 T2DA 式做法）。

**三個現象都成立 → 核心假設驗證 → 才進到 CALVIN、圖片、多模態的完整版。**

刻意**不**在 PoC 階段碰：CALVIN（太重）、圖片規格、真實機器人、diffusion。先用最便宜的環境把想法跑通。

---

## 1. 為什麼 PoC 不直接上 CALVIN

CALVIN 要跑 PyBullet 模擬、影像 encoder、長 horizon，光裝環境和重現 baseline 就要好幾天，且 GPU/時間成本高。**PoC 的精神是「用最小的環境，最快證明想法對不對」**——如果連在玩具環境都做不出三現象，上 CALVIN 也是白費。所以 PoC 用**自製的輕量 GridWorld**，CALVIN 留到第二階段。

這個選擇也呼應 [[method_spec]] §8 與 [[research_direction_options|innovation_notes 第 2 節的 hidden-goal toy task]]——你專案裡本來就有玩具環境的思路。

---

## 2. 環境設計：Ambiguous-Spec GridWorld（自製，~150 行）

### 2.1 基本環境
- **N×N 格子世界**（PoC 用 7×7）。agent 從固定起點出發。
- **物件**：盤面上放幾個有「顏色 × 形狀」屬性的目標物（例如 red-square、blue-square、red-circle、blue-circle，放在不同格子）。
- **動作**：上下左右 + stay（5 個離散動作）。
- **任務 z**：「走到某個指定屬性的物件上」。例如 z = "go to red square"。任務集合就是所有 (顏色,形狀) 組合（PoC：4 個屬性物 → 但任務可定義成更細，見下）。
- **reward**：到達指定物件 +1，每步 -0.01，episode 上限 H=15 步。
- **狀態**：agent 座標 + 盤面物件配置（PoC 用低維向量即可，不需影像 → 省掉影像 encoder，聚焦核心）。

### 2.2 任務分布（多任務 meta-RL 的「分布」）
- 每個 episode 隨機抽：盤面物件配置 + 一個目標任務 z。
- 任務空間 = 屬性組合（顏色 ∈ {red, blue, green}, 形狀 ∈ {square, circle, triangle}）→ 9 種任務。
- **train/test 切分**：留幾個「組合」不出現在訓練（測 unseen-task 組合泛化），例如訓練不給 "green triangle"，測試才出現 → 對應 [[method_spec]] §2.2 的泛化軸。

### 2.3 歧義語言規格（PoC 的核心，§4.1 的玩具版）
這是整個 PoC 最關鍵的設計。每個任務 z 配一個語言規格 `c`，分兩種模式：

- **精確規格**：`"go to the red square"` → 只對應 1 個任務。
- **歧義規格**：刻意省略一個屬性，對應多個任務：
  - `"go to the red object"` → 對應 {red square, red circle, red triangle}（3 個任務）
  - `"go to a square"` → 對應 {red square, blue square, green square}
  - `"go to an object"` → 對應全部（最模糊）

**關鍵實作**：每個規格 `c` 都附一個 ground-truth 任務集合 `Z(c)`（哪些任務符合這個描述，用屬性規則自動算——玩具環境不需 task detector，規則就能算）。這個 `Z(c)` 就是 belief 的監督目標（[[method_spec]] §4.3 的 oracle 後驗，在玩具環境是精確可算的）。

語言怎麼進模型：PoC **不用大型 text encoder**，直接把規格表示成「屬性 one-hot + 缺失標記」的小向量（例如 [color: red, shape: UNKNOWN]），或用一個小 embedding table。這樣省掉 LLM、聚焦在 belief 機制本身。（第二階段才換成真的 text encoder。）

---

## 3. 模型架構（PoC 版，< 5M 參數）

對應 [[method_spec]] §3，但全部縮小：

```
輸入序列：[spec_token] ++ [(s_1, a_1), (s_2, a_2), …, (s_t, ·)]
           ↓
      小型 causal Transformer（4 層、4 heads、d_model=128）
           ↓ ↓
   belief head        policy head
 (softmax over 9      (5 個離散動作的
  個 task prototype)    logits，吃 belief)
```

### 3.1 三個元件
- **Spec encoder**：規格小向量 → linear → 1 個 spec token。
- **In-context transformer**：causal mask，吃 spec token + 歷史 (state, action) tokens。**PoC 用 DPT 式**：在離線多任務軌跡上做 supervised next-action prediction。
- **Belief head（類別後驗，主版本）**：從 transformer 對 spec/history 的表徵，輸出對 9 個 task prototype 的 softmax `b_t`。entropy 直接可算。
- **Policy head**：吃 `(當前 state 表徵, b_t)` → 5 個動作 logits。risk-aware 版本：對 belief 下高機率的幾個任務，取「對它們都不差」的動作（PoC 用簡單版：用 belief 加權各任務的 Q/動作偏好）。

### 3.2 PoC 刻意簡化的地方
- 不用影像（用低維 state）→ 省掉 vision encoder。
- 不用大型 LLM（用屬性向量）→ 省掉 text encoder。
- belief head 先只做類別版（最好解釋、entropy 好算）；高斯/ensemble 版留 ablation。
- 動作離散 → 不用連續控制那套。

---

## 4. 資料生成（離線多任務軌跡 + 歧義規格配對）

這是 PoC 要寫的第一段 code，也是「製造歧義」的關鍵（[[method_spec]] §4.1 玩具版）。

### 4.1 收集專家/近最優軌跡
- 對每個 (盤面配置, 任務 z)，用 **A\* 或 BFS 最短路** 算出到目標物的最優動作序列（玩具環境不需訓 RL，規則就能產生專家軌跡）。
- 收集大量 `(盤面, z, 最優軌跡 τ)`。

### 4.2 配上歧義規格（核心步驟）
- 對每條軌跡的任務 z，**隨機抽一個相容的規格 `c`**（可能是精確的、也可能是歧義的，且歧義規格 `c` 的 `Z(c)` 必須包含 z）。
- 結果：**同一個歧義規格 `c`（如 "go to red object"）會配到不同的 z（red square / red circle / red triangle）的軌跡**。
- 這就是逼出「模糊→寬 belief」的機制：模型看到 "go to red object" + 還沒觀察時，無法確定是三個 red 物件中的哪一個 → 必須保持寬 belief；看到 agent 往某個方向走、或盤面某些線索後才收斂。

### 4.3 資料規模（PoC）
- 9 任務 × 多種盤面配置 × 多種規格 → 約 10k–50k 條軌跡就夠（玩具環境，生成極快，CPU 幾分鐘）。

---

## 5. 訓練（對應 [[method_spec]] §4 的 PoC 版）

### 5.1 Loss（先做兩個，risk/cal 留後面）
- **L_IC（主）**：supervised next-action prediction（給 spec + history，預測最優動作）。
- **L_belief**：belief head 的 softmax `b_t` 對齊「當前 context 下仍相容的任務分布」`p*(z|c, o_{1:t})`（玩具環境精確可算：t=0 時是 `Z(c)` 上均勻，t 增加時排除被觀察否證的任務後 renormalize）。用 KL 或 cross-entropy。
- 先 `L = L_IC + λ·L_belief`（λ 從 0.5 起調）。risk-aware loss（CVaR）與 calibration 等三現象出來後再加。

### 5.2 訓練設定
- optimizer AdamW、lr 3e-4、batch 256、約 50k–100k steps。
- 單卡 A6000 估計 **1–3 小時**跑完（模型小、資料小）。DGX Spark 更快。

---

## 6. 評估：怎麼量測三現象

### 6.1 現象 1（歧義 → 寬 belief）
- 在 test 規格上，分別給「精確規格」與「歧義規格」，量 **t=0 時 belief 的 entropy**。
- **預期**：歧義規格的 entropy 明顯高；且歧義程度越高（缺越多屬性）entropy 越高。
- 圖：x = 規格歧義程度（對應任務數 |Z(c)|），y = belief entropy。應單調上升。

### 6.2 現象 2（觀察 → 收斂）
- 給歧義規格，跑 rollout，記錄每步 `belief entropy(t)` 與 `是否最終成功`。
- **預期**：entropy 隨 t 下降；成功率隨「容許的觀察步數」上升。
- 圖：x = timestep，y = belief entropy（下降曲線）；另一張 x = 容許觀察步數，y = 成功率（上升曲線）。

### 6.3 現象 3（belief 有用 vs 單點 baseline）
- **baseline（模擬 T2DA 式單點）**：同架構，但拔掉 belief 分布，t=0 直接 argmax 一個任務後照做（不收斂、不 risk-aware）。
- **我們的**：吃整個 belief、risk-aware、隨觀察收斂。
- **預期**：在歧義規格上，我們的成功率顯著高於 single-point baseline；在精確規格上兩者相近（證明「好處來自處理歧義」）。
- 表：精確 vs 歧義 × (ours vs single-point) 的成功率 2×2。

### 6.4 加分（若三現象都成立，順手做）
- **calibration**：belief entropy 是否能預測成功率（reliability diagram）。
- **unseen 組合**：測試 "green triangle" 這種訓練沒出現的組合,看 belief 與成功率。

---

## 7. 硬體與工程規劃

### 7.1 你的硬體怎麼用
- **A6000（48GB）**:PoC 全程單卡就夠。模型 <5M 參數、batch 256、序列短(H≤15),記憶體用不到 2GB。**這台拿來跑 PoC 與之後 CALVIN 的中小實驗都行。**
- **DGX Spark**:如果是指 GB10 那台(統一記憶體、ARM),適合「開發/原型」與中小模型;PoC 完全沒問題。等到第二階段要跑 CALVIN + 影像 + 大 transformer + 多 seed 平行,DGX 的記憶體與多卡會比較有感。
- **建議**:PoC 在 A6000 上開發(x86 + CUDA 生態最順),DGX 留給第二階段的長時間/多 seed 訓練。

### 7.2 軟體 stack
- Python 3.10+、PyTorch 2.x、CUDA。
- 不需 PyBullet（玩具環境自己用 numpy 寫）。
- 套件極少：torch、numpy、einops（選用）、wandb（記錄三現象的曲線）。
- repo 結構建議:
  ```
  bts-poc/
    envs/gridworld.py        # 環境 + 任務分布 + 規格規則
    data/generate.py         # A* 收軌跡 + 配歧義規格 + 算 Z(c)/oracle 後驗
    models/transformer.py    # in-context transformer + belief head + policy head
    train.py                 # L_IC + L_belief
    eval/phenomena.py        # 量三現象、畫圖
    configs/poc.yaml
  ```

### 7.3 估時
- 寫環境 + 資料生成:1–2 天。
- 寫模型 + 訓練:1–2 天。
- 跑實驗 + 畫三現象圖:0.5–1 天(訓練本身才幾小時)。
- **整個 PoC 約 1 週內可得到「三現象成立與否」的明確答案。**

---

## 8. 決策點（PoC 跑完後）

| 結果 | 判讀 | 下一步 |
| --- | --- | --- |
| 三現象都成立 | 核心假設驗證 ✓ | 進第二階段：換 CALVIN、加圖片規格、換真 text/vision encoder、加完整 baseline（[[T2DA]]/[[LUMOS]]/DPT） |
| 現象 1 不成立（belief 不變寬） | 「製造歧義」的資料設計有問題（同 c 沒配到夠多樣的 z），或 belief head collapse | 檢查 §4.2 配對、加強 L_belief 權重、改用 ensemble head |
| 現象 2 不成立（觀察進來 belief 不收斂） | transformer 沒在用 context 觀察，或 oracle 後驗算錯 | 檢查 causal mask、檢查 §5.1 的 p* 計算、看 attention |
| 現象 3 不成立（belief 沒贏單點） | risk-aware 機制無效，或玩具任務太簡單（單點剛好夠） | 加大歧義程度（更模糊規格）、強化 risk-aware（CVaR）、確認 baseline 真的吃虧 |

---

## 9. 與完整方法的對應（PoC 是哪一塊的縮影）

| 完整方法（[[method_spec]]） | PoC 對應 | PoC 簡化 |
| --- | --- | --- |
| 多模態規格（圖+語言） | 只做語言 | 圖片留第二階段 |
| 真 text/vision encoder | 屬性向量 | 省 LLM |
| CALVIN | 自製 GridWorld | 省 PyBullet/影像 |
| in-context transformer | 小 transformer | 4 層 |
| 顯式 belief head（3 選 1） | 類別後驗 | 高斯/ensemble 留 ablation |
| 4 個 loss | L_IC + L_belief | risk/cal 後加 |
| oracle 相容後驗（task detector） | 屬性規則精確算 | 玩具環境免 detector |
| 3 泛化軸 | unseen 組合 | env 泛化留 CALVIN |

PoC 保留了**最核心的因果鏈**（歧義規格→belief→收斂→risk-aware），只把「重工程」的部分（影像、LLM、CALVIN）抽掉。這樣若三現象成立，就有把握那條鏈在完整版也會動。

---

## 10. 一句話總結

PoC = 用一個自製的「歧義語言 GridWorld」+ 小型 in-context transformer + 類別 belief head，在單卡數小時內驗證「歧義規格→寬 belief→隨觀察收斂→risk-aware 勝單點」這條核心鏈;三現象成立才進 CALVIN 完整版,避免一開始就燒大量工程在不確定會不會動的想法上。
