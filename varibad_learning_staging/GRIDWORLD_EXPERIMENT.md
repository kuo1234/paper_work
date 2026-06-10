# GridWorld 實驗:VariBAD 是否學到接近 Bayes-optimal exploration？

本目錄依照 [`spec.md`](spec.md) 實作 **Phase 0(環境)+ Phase 1(hard-coded baselines)**,用**純 Python、零依賴**先把可信的對照標準建好。在訓練 neural network(RL² / VariBAD)之前,先確認:任務定義正確、且有上下界把「好壞」框出來。

> 程式在 `src/varibad_gridworld/`。原本的 bandit toy(`src/varibad_toy/`)保留不動。

---

## Phase 0:環境(BAMDP,spec §3–4)

| 項目 | 設定 |
|------|------|
| Grid | 5×5,row 0 在上、row 4 在下 |
| 起點 | **左下角 `(4,0)`** |
| goal | 從 **前三排 15 格**(`allowed_goals`)uniform sample,對 agent 隱藏 |
| actions | up / right / down / left / **stay**(5 個;撞牆留原地) |
| reward | goal **+1**、非 goal **−0.1**(依 **next state** 判定) |
| 不終止 | 踩到 goal **不結束 episode** → 可 `stay` 一直收 +1 |
| H | 每 **15 steps** reset 位置(MDP horizon) |
| task | 同一 rollout 內 **goal 不變** |
| rollout | 一次 BAMDP rollout = **N 個 episodes**(N=4 → H⁺=60) |

### 核心:H 與 H⁺

- **H = 15**:單一 episode horizon,每 15 步位置 reset 回左下角。
- **H⁺ = N × H**:整個 BAMDP rollout horizon(N=4 → 60)。
- **最重要:position 每 episode reset,但 belief 跨 episode 保留**。所以 agent 用前面 episode「找」goal、後面 episode「利用」goal。`stay` 之所以重要,正是因為 goal 不終止——找到後最佳行為是停在 goal 上把剩餘步數換成 +1。

spec §17.1 最易寫錯的地方(我已寫測試守住):**episode reset ≠ task reset**,`reset_episode()` 絕不重抽 goal。

環境 API 照 spec §4:`reset_task / reset_episode / step / get_state_onehot`。
對應:[envs/gridworld.py](varibad_learning_staging/src/varibad_gridworld/envs/gridworld.py)

### belief(精確後驗,限制在 allowed_goals)

任務空間小 → 用精確後驗:goal 在「`allowed_goals` 內尚未踩過的格」上均勻分布。entropy = `log2(候選數)`,找到 goal 歸 0。這是之後 neural encoder 的「正確答案」對照。
對應:[utils/belief.py](varibad_learning_staging/src/varibad_gridworld/utils/belief.py)

---

## Phase 1:四個 hard-coded baselines(spec §5)

> 為什麼先做?**沒有 baseline,VariBAD 訓練出來不知道好不好。** 這四條線把範圍框出來。

| Baseline | 角色 | 行為 |
|----------|------|------|
| **Random** | 最低基準 | 隨機 5 動作,不用 belief |
| **Oracle** | **上限(≠ Bayes-optimal)** | 知道 goal,直奔→stay。**注意:Oracle 不需探索,所以不是 Bayes-optimal,只是 upper bound** |
| **Posterior Sampling** | Thompson sampling | 從後驗抽猜測 goal 走過去,踩錯重抽 |
| **Bayes-like Search** | 接近 Bayes-optimal 的參考 | 沿**固定蛇形 search_order** 系統性掃描,找到 stay |

對應:[baselines/](varibad_learning_staging/src/varibad_gridworld/baselines/)

---

## 實際結果(300 rollouts, H=15, N=4, H⁺=60)

```
Method                AvgReturn     Ep1     Ep2  FirstGoal  Success%  Redundant
-------------------------------------------------------------------------------
Oracle                    42.22   10.56   10.56       4.04      100%       0.00
Bayes-like Search         37.68    6.01   10.56       8.55      100%       0.25
Posterior Sampling        35.48    4.69    9.68      11.38      100%       2.18
Random                    -5.06   -1.27   -1.26      28.48       37%      37.44
```

這張表(spec §18 六欄)就是回答原始問題的工具,對齊 spec §12 的每一條預期:

- **排序 Oracle > Bayes-like Search > Posterior Sampling > Random** ✅
- **Ep1 vs Ep2**:Bayes/Posterior 的 Ep1 偏低(在探索),Ep2 跳到接近 Oracle → **meta-learning 曲線**(belief 跨 episode 保留)。Oracle 則全程平(無學習,因為一開始就知道)。
- **First Goal Step**:Oracle 4.0(直奔)< Bayes 8.6 < Posterior 11.4 < Random 28.5。
- **Redundant visits(探索效率,spec §6.4/§12.3)**:Oracle 0(不探索)、**Bayes 0.25 vs Posterior 2.18 → 系統性搜尋的重複訪問少約 9 倍**。這正是「systematic search 比 posterior sampling 更有效探索」的量化證據。
- **Success%**:Oracle/Bayes/Posterior ≈ 100%,Random 只有 37%。

> Redundant visits 只計**找到 goal 之前**的重複訪問(探索階段);找到後 exploit 時重走最短路是必要的,不算浪費。

### step-by-step(Bayes-like Search)

```
episode 0: return=+10.60 (reached goal at t=4)   ← 系統性掃描後找到
episode 1: return=+10.60 (reached goal at t=4)   ← 記得 goal,直奔
episode 2: return=+10.60 (reached goal at t=4)
episode 3: return=+10.60 (reached goal at t=4)
```

---

## 怎麼跑

```bash
cd varibad_learning_staging

# 四條 baseline 比較表(spec §18)
PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --tasks 300 -N 4

# 加上某個 policy 的 ASCII rollout 視覺化
PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --visualize bayes_like_search -N 4
PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --visualize posterior_sampling -N 6

# 測試
PYTHONPATH=src python -m unittest discover -s tests
```

參數:`--tasks`(rollout 數)、`-N/--episodes`(每 rollout episode 數,4 或 6)、`--seed`、`--visualize <policy>`。

---

## 目錄結構(照 spec §14)

```
src/varibad_gridworld/
├── envs/gridworld.py          # GridWorldTask (有狀態 BAMDP env)
├── baselines/                 # random / oracle / posterior_sampling / bayes_like_search / rl2_adapter
├── models/rl2.py              # RL² (GRU + policy/value head)
├── trainers/                  # a2c.py / train_rl2.py
├── utils/                     # belief / rollout / metrics / visualization / posterior_viz / torch_rollout
├── configs/gridworld.py       # spec §15 config (Python dict)
└── experiments/               # run_baselines.py / run_posterior_tracking.py
```

---

## Phase 2:Posterior tracking 視覺化(spec §11,零依賴)

把 Phase 0 的 exact `GoalBelief` 沿一個 rollout 追蹤,**把「不確定性如何下降」畫出來**。這是理解 Bayesian RL 的橋樑,也是之後「對照 neural encoder 學到的 posterior 對不對」的 ground truth。

```bash
PYTHONPATH=src python -m varibad_gridworld.experiments.run_posterior_tracking --seed 7
```

輸出兩個東西:

**1. entropy 隨時間(ASCII sparkline)** — entropy = `log2(候選格數)`:
```
####****===_________________________________________________
start=3.91   end=0.00   first_goal_step=10
```
看它**階梯狀下降**(每排除一格、不確定性少一點)、找到 goal 那刻**崩到 0**。

**2. posterior heatmap 隨時間** — `@` 濃淡 = P(goal 在此),`G` = 真 goal:
```
t=0  entropy=3.91          t=12 entropy=0.00
@ @ @ @ @                  . . . . .
G @ @ @ @                  G . . . .
@ @ @ @ @       ──►        . . . . .
. . . . .                  . . . . .
. . . . .                  . . . . .
```
從均勻一片 `@` 收斂到單一 `G`——這就是 belief 在做的事。

對應:[utils/posterior_viz.py](varibad_learning_staging/src/varibad_gridworld/utils/posterior_viz.py)、[experiments/run_posterior_tracking.py](varibad_learning_staging/src/varibad_gridworld/experiments/run_posterior_tracking.py)

> exact belief 沒有 Gaussian variance,所以這裡用 **entropy** 當對應量(spec §11.2 的 variance curve 的零依賴版)。

---

## Phase 3:RL²(GRU + A2C,spec §7 & §9)

第一個**真正的 neural meta-RL baseline**。RL² 不顯式建模 posterior,改用 **RNN hidden state** 做隱式的 online task inference。

**架構**(spec §7):
- 每步輸入 `x_t = [state onehot(25), prev_action onehot(5), reward(1), done(1)] = 32 維`
- `GRU(32 → 128)`;policy/value head 吃 `concat(hidden, state_onehot)`
- **關鍵(spec §17.2):hidden state 跨 episode 不清空,只在新 task 才清**。done flag 告訴 GRU episode 邊界,但記憶連續——這樣它才能「記得前面 episode 找到的 goal」。
- 訓練:**A2C**(gamma=0.95, value_coef=0.5, entropy_coef=0.01, lr=1e-3),整個 H⁺=60 當單一 horizon 算 return(所以它會為了後面 episode 的 reward 而在前面探索)。

```bash
# 訓練(CPU 約 6–16 分鐘,看 updates)
PYTHONPATH=src python -m varibad_gridworld.trainers.train_rl2 --updates 1500 --eval-tasks 100
# 把訓練好的 RL² 加進比較表
PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --tasks 200 -N 4 --rl2 rl2_policy.pt
```

**學習曲線**(從隨機 → 學會探索+利用,1500 updates / CPU 約 16 分鐘):
```
update    avg_return  success%   per-episode
   1        -6.00        0%      [-1.5 -1.5 -1.5 -1.5]   ← 跟 Random 一樣
 300        +5.80       49%      [+2.6 -0.3 +2.5 +1.0]   ← 開始找得到 goal
 900       +26.32      100%      [+3.9 +5.8 +8.3 +8.4]   ← 學會探索後利用
1400       +29.06       88%      [+4.9 +8.1 +8.5 +7.6]
```

`avg_return` 從 −6 爬到 +25~29、success 0% → ~100%。**per-episode 數字上升**(ep1 探索 → ep2+ 利用)代表它學到了 within-task adaptation——這正是 meta-RL 的核心(spec §12.4)。

放進比較表(deterministic eval, 200 tasks):
```
Method                AvgReturn     Ep1     Ep2  FirstGoal  Success%  Redundant
Oracle                    41.74   10.44   10.44       4.15      100%       0.00
Bayes-like Search         37.12    5.82   10.44       8.70      100%       0.24
RL2                       22.78    3.96    7.55      10.25       76%      13.37
Posterior Sampling        34.90    4.56    9.47      11.70      100%       2.34
Random                    -5.11   -1.32   -1.27      28.13       34%      38.23
```

怎麼讀(誠實版):
- RL² 的 **Ep1=3.96 → Ep2=7.55**:明顯的 meta-learning——第一個 episode 探索,第二個 episode 用學到的東西做得更好。**這就是這個實驗要看到的核心現象**。
- 但 RL² 的 raw return(22.78)這個訓練預算下**還沒追上 Posterior Sampling**(34.90)。原因:RL² 非常 sample-hungry,而這是 CPU 教學規模的 run(1500 updates),不是論文規模。再訓練更久、調 entropy/lr,還會更好。
- redundant visits 高(13.37):RL² 沒有顯式 belief,探索路徑不像 Bayes-like Search 那麼有系統——這正凸顯了「隱式記憶 vs 顯式 posterior」的差別,也是 VariBAD 要改進的地方。

對應:[models/rl2.py](varibad_learning_staging/src/varibad_gridworld/models/rl2.py)、[trainers/train_rl2.py](varibad_learning_staging/src/varibad_gridworld/trainers/train_rl2.py)、[baselines/rl2_adapter.py](varibad_learning_staging/src/varibad_gridworld/baselines/rl2_adapter.py)

> RL² 的限制(對比之後的 VariBAD):hidden state **不可解釋**——你沒辦法把它拿出來畫成 posterior heatmap。Phase 4 的 VariBAD 會有明確的 `q(m|τ)`,可以直接跟 Phase 2 的 exact belief 對照。這就是 VariBAD 比 RL² 多出來的價值。

---

## Phase 4:Simplified VariBAD(spec §8–9)— 專案目標

這是回答最初問題的關鍵階段。VariBAD 相對 RL² 多出的東西:**顯式的 variational posterior `q(m|τ)`**——可以把學到的 belief 畫出來,跟 Phase 2 的 exact belief 並排對照。RL² 的 hidden state 做不到這件事。

**三個元件**(spec §8,維度精確對齊):
- **Encoder** `q(m|τ)`:`fc(32→40) → GRU(40→64) → μ(5), logσ²(5)`,latent dim 5
- **Policy**(Design A):吃 `[state(25), μ(5), logσ²(5)]` → MLP → action/value。**吃 posterior 參數而非取樣 latent**,所以是 uncertainty-aware
- **Reward decoder**:`[s, a, s', m] → P(reward=+1)`,BCE。訓練它**逼 latent 帶有 goal 資訊**

**VariBAD 的精髓**(spec §9.3):對多個 context time t,encoder 只看 `τ_{:t}` → 取樣 `m_t` → decoder 用 `m_t` 預測**整段 trajectory** 的 reward。用部分歷史重建整段任務 = 逼 encoder 推斷 goal 在哪。

**ELBO** = reconstruction BCE + β·KL(對 N(0,I)),β=0.1。

```bash
# 訓練(device-agnostic;有 CUDA 版可加 --device cuda)
PYTHONPATH=src python -m varibad_gridworld.trainers.train_varibad --updates 3000 --beta 0.01 --eval-tasks 100
# 關鍵交付:VariBAD 學到的 posterior vs exact belief 並排
PYTHONPATH=src python -m varibad_gridworld.experiments.run_varibad_posterior --varibad varibad_policy.pt --seed 7
# 比較表(含 VariBAD/RL² 列)
PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --tasks 200 --rl2 rl2_policy.pt --varibad varibad_policy.pt
```

### 訓練結果(β=0.01, 3000 updates, CPU)

```
update    a2c    recon    kl   avg_return  success%   ep_returns
   1    -0.737   0.670  0.02     -6.00        0%      [-1.5 -1.5 -1.5 -1.5]
 800     4.805   0.356  0.65     +7.13       56%      [+1.4 +1.6 +1.2 +2.9]
1400     3.397   0.165  1.68    +10.07      100%      [+3.9 +1.4 +1.8 +2.9]
2400     4.292   0.184  1.41    +14.97       76%      [+3.1 +4.0 +3.9 +3.9]
3000     3.440   0.210  2.08    +13.05       69%      [+3.1 +3.4 +3.2 +3.3]
```

**reconstruction loss 從 0.67(接近亂猜)掉到 ~0.16–0.21** → reward decoder 確實學到了 task structure(latent 帶有 goal 資訊)。avg_return −6 → 約 +13~15,success 0% → 70–100%。

### posterior 對照(本 phase 核心)

`run_varibad_posterior` 把 VariBAD 解碼出的 goal map 跟 exact belief 並排:

```
        EXACT belief          VariBAD learned
t=2     @ @ @ @ @             = - - + -
        G @ @ @ @             G : = + =
        . @ @ @ @             @ - + + =      ← goal cell (1,0) 一直亮著 G
```

可以看到 VariBAD 的 decoder **抓到了 goal 大致位置**(goal cell 持續活躍),但**沒有像 exact belief 那樣乾淨地排除已訪格**。

### 誠實的結論:接近 Bayes-optimal 了嗎?

```
Method                AvgReturn     Ep1     Ep2  FirstGoal  Success%  Redundant
Oracle                    41.74   10.44   10.44       4.15      100%       0.00
Bayes-like Search         37.12    5.82   10.44       8.70      100%       0.24
RL2                       22.78    3.96    7.55      10.25       76%      13.37
VariBAD                   15.30    3.36    4.30       8.99       70%      15.31
Posterior Sampling        34.90    4.56    9.47      11.70      100%       2.34
Random                    -5.11   -1.32   -1.27      28.13       34%      38.23
```

**部分接近,但這個 CPU 教學規模的 run 還沒到。** 誠實評估:
- ✅ **機制有學到**:reward decoder reconstruction loss 大幅下降、posterior heatmap 抓到 goal 位置——VariBAD 的核心 pipeline 確實在運作。
- ✅ **比 Random 好非常多**,學到了基本探索。
- ⚠️ **還沒贏過 Posterior Sampling**,但 per-episode `Ep1=3.36 → Ep2=4.30` 有小幅上升——表示有一點 cross-episode adaptation,只是遠不如 Bayes-like / Posterior Sampling 穩定。
- ⚠️ **posterior 不如 exact belief 銳利**,variance 曲線後期甚至上升(輕微 posterior collapse 跡象)。

這正是有教育意義的地方:**VariBAD 不是免費的**。它比 RL² 多了可解釋的 posterior,但要逼近 Bayes-optimal 需要更多訓練、調 β、調 entropy bonus(論文用 16 process × 數千 updates,我們是單 process CPU)。**這個實驗讓你親眼看到「學到機制」與「逼近最優」之間的差距**——而且因為有 exact belief 當 ground truth,你能精確指出它差在哪。

> 改善方向(已附 `--beta` 等旗標可調):降低 β 避免 posterior collapse、加長訓練、加大 batch、或用 GPU 跑更多 updates。

---

## 全部跑一遍

```bash
cd varibad_learning_staging
PYTHONPATH=src python -m varibad_gridworld.experiments.run_baselines --tasks 300       # Phase 1
PYTHONPATH=src python -m varibad_gridworld.experiments.run_posterior_tracking --seed 7 # Phase 2
PYTHONPATH=src python -m varibad_gridworld.trainers.train_rl2 --updates 1500           # Phase 3
PYTHONPATH=src python -m varibad_gridworld.trainers.train_varibad --updates 3000 --beta 0.01  # Phase 4
PYTHONPATH=src python -m varibad_gridworld.experiments.run_varibad_posterior --varibad varibad_policy.pt
PYTHONPATH=src python -m unittest discover -s tests                                     # 52 tests
```

> 註:Phase 0–2 零依賴;Phase 3–4 用 PyTorch(`pip install -e .[neural]`)。程式 device-agnostic,有 CUDA 版 PyTorch 可加 `--device cuda`。遵守 spec §17(hidden 跨 episode 不清、policy 不看 goal、用 A2C)。
