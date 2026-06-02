# GridWorld 實驗：VariBAD 是否學到接近 Bayes-optimal exploration？

這份實驗把原本的 bandit lab 搬到 **GridWorld**，目的只有一個：

> 讓「Bayes-optimal exploration（貝氏最優探索）」這件抽象的事，變成**看得見、算得出來**的東西。

在 bandit 裡你只有兩根拉桿，看不出「探索」的空間感。在 GridWorld 裡，agent 必須**實際走過去**才知道隱藏 goal 在哪——於是「為了降低不確定性而主動去走訪未知區域」這個 Bayes-optimal 的招牌行為，就會直接畫在地圖上。

---

## 1. 任務設定

- **5×5 GridWorld**，goal 隨機藏在某一格（不會是起點）。
- agent 從 `(0,0)` 出發，動作為上下左右；撞牆 = 留在原地。
- **只有踩到 goal 那一格才知道它是 goal**（reward +1、episode 結束）。其他格 reward 0。

關鍵：goal 位置是**隱藏的任務**（hidden task），唯一獲取資訊的方式是「去踩」。這正是 meta-RL / Bayes-adaptive RL 的核心情境。

對應檔案：[gridworld.py](varibad_learning_staging/src/varibad_toy/gridworld.py)

---

## 2. 什麼是 Bayes-optimal exploration？

**Belief（信念）= 對隱藏任務的後驗機率分布。**

這個 grid 的後驗超級乾淨：goal 在「尚未踩過的每一格」上是**均勻分布**。每踩到一個不是 goal 的格子，就把那格機率歸零、其餘格重新均分。entropy（不確定性，單位 bits）= `log2(候選格數量)`，會隨著探索一路下降，踩到 goal 的瞬間掉到 0。

對應檔案：[grid_belief.py](varibad_learning_staging/src/varibad_toy/grid_belief.py)

**Bayes-optimal 行為**：在「最大化期望累積 reward」=「最小化期望找到 goal 的步數」的目標下，最優策略是——

> 沿著一條短路徑，**依序走訪尚未排除的候選格**；一旦踩到 goal，就直奔（其實已經到了）。

直覺上為什麼是這樣：
1. 既然 goal 均勻分布，每一格「是 goal」的機率一樣，所以**先檢查近的格**最划算（同樣的資訊、更少的步數）。
2. 每一步同時做兩件事：**往候選格前進（探索）** + **縮小不確定性（資訊增益）**，沒有浪費。
3. 找到後 belief 崩塌成單點，策略自動從「探索」切換成「利用」。

這就是一個策略內含完整的 **explore → exploit** 弧線。`BayesOptimalGridPolicy` 用「永遠走向最近的候選格」(nearest-candidate sweep) 實作它——對 uniform prior 而言這是 Bayes-optimal 的良好近似（嚴格的最優搜尋路徑是 NP-hard，但近候選優先在這種規則網格上幾乎總是最優）。

對應檔案：[grid_policies.py](varibad_learning_staging/src/varibad_toy/grid_policies.py)

---

## 3. 三個策略（拿來對照）

| 策略 | 行為 | 角色 |
|------|------|------|
| `bayes_optimal` | 走向最近候選格、系統性掃描 | **上限 / oracle 下界** |
| `greedy` | 只盯住某個固定候選格走，不主動掃描 | 「不會探索」長什麼樣 |
| `random` | 隨機亂走 | 最弱對照（沒有 belief） |

跑一下就能在地圖上看到差別。`bayes_optimal` 會畫出漂亮的蛇行（boustrophedon）掃描：

```
S . . . .
. . . . .
. . . . .
_ _ _ * .
```

（`.` = 走過、`*` = agent、`G` = goal、`_` = 還沒看過）——它一行一行系統性地排除，這就是最優探索的視覺簽名。

---

## 4. 如何量化「有多接近 Bayes-optimal」？

demo 結尾會把三個策略**跑在同一批隨機任務上**（公平比較），印出：

```
policy          mean_steps_to_goal   regret_vs_bayes   found
bayes_optimal                12.22             +0.00    100%
greedy                       14.62             +2.40    100%
random                       49.35            +37.13     97%
```

- `mean_steps_to_goal`：平均花幾步找到 goal（越小越好）。
- `regret_vs_bayes` = 該策略步數 − bayes_optimal 步數。**越接近 0，代表探索效率越接近最優。**
- 每個 episode 也會印 `oracle(shortest)` = 若**早就知道** goal 的最短步數（Manhattan 距離），這是絕對下限；`steps_to_goal − oracle` 就是「不知道 goal 所付出的探索成本」。

這張表就是回答原始問題的工具：**把一個學到的策略丟進來，看它的 regret 離 0 多近，就知道它是否真的學到接近 Bayes-optimal 的探索。**

對應檔案：[grid_rollout.py](varibad_learning_staging/src/varibad_toy/grid_rollout.py)、[grid_demo.py](varibad_learning_staging/src/varibad_toy/grid_demo.py)

---

## 5. 怎麼跑

```bash
cd varibad_learning_staging

# 看某個策略一步一步怎麼走 + 結尾比較表
PYTHONPATH=src python -m varibad_toy.grid_demo --policy bayes_optimal --episodes 3 --seed 0
PYTHONPATH=src python -m varibad_toy.grid_demo --policy greedy --episodes 3 --seed 0
PYTHONPATH=src python -m varibad_toy.grid_demo --policy random --episodes 3 --seed 0

# 測試
PYTHONPATH=src python -m unittest discover -s tests
```

可調參數：`--size`（grid 大小）、`--episodes`（視覺化幾局）、`--compare-episodes`（比較表用幾局）、`--seed`、`--max-steps`。

---

## 6. 下一步：接上真的 neural VariBAD

目前 belief 是**精確算出來的**（因為任務空間小），policy 直接讀 belief 物件。完整的 VariBAD 則是：

1. **學一個變分 encoder** `q(m | trajectory)`（通常是 RNN）取代 `GridBeliefEncoder`，把歷史壓成 latent。
2. **學一個 RL policy**（小 MLP）吃 `belief.latent()`（目前已輸出固定長度向量：flatten 機率圖 + entropy）。

接口已經預留好了——`grid_policies.py` 裡的 `BeliefConditionedPolicy` Protocol 定義了 `act_from_latent(state, latent, rng)`。你只要：

1. 實作一個吃 `latent()` 的神經網路 policy。
2. 用 PPO / A2C 在 `run_grid_episode` 上訓練它。
3. 把訓練好的 policy 丟進**第 4 節那張比較表**，直接讀 `regret_vs_bayes`。

如果學到的 policy 的 regret 趨近 `bayes_optimal` 的 0，就證明了 **VariBAD 確實學到接近 Bayes-optimal 的探索**——這正是這個實驗要回答的問題。

> 註：本階段刻意維持**零依賴純 Python**（不引入 PyTorch/numpy），先把 exact 對照與量測框架建好。要進入 neural 訓練階段時再加依賴。
