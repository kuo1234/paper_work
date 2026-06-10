以下是 **VariBAD GridWorld 實驗開發 Spec**。目標不是一開始完整重現論文所有結果，而是做一個可以逐步擴充的實驗框架，讓你清楚比較：

[
\text{Random} \rightarrow \text{Oracle} \rightarrow \text{Posterior Sampling} \rightarrow \text{RL}^2 \rightarrow \text{Simplified VariBAD}
]

這份 spec 以論文的 GridWorld 設定為基準：5×5 grid、未知 goal、agent 從左下角開始、每 15 steps reset、goal task 固定、reward 為非 goal (-0.1)、goal (+1)，並以 BAMDP horizon 評估多個 episode 內的 online return。論文也用 GridWorld 展示 VariBAD 的 posterior belief、reward prediction、latent space 變化與接近 Bayes-optimal 的探索行為。

---

# VariBAD GridWorld Experiment Spec

## 1. 實驗目標

本實驗的主要目標是建立一個小型、可控、可視覺化的 meta-RL benchmark，用來理解 VariBAD 的核心概念：

1. agent 進入新 task 時不知道 goal 在哪裡；
2. agent 必須透過 interaction 推斷 task；
3. agent 的行為應該根據 task uncertainty 做 exploration / exploitation trade-off；
4. policy 不只是根據 state 行動，而是根據：

[
\pi(a_t \mid s_t, q_\phi(m \mid \tau_{:t}))
]

5. 訓練後的 agent 在 meta-test 不做 gradient update，只靠 encoder + policy forward pass 完成 online adaptation。

---

# 2. 實驗分階段

不要一開始直接做完整 VariBAD。建議分成 5 個階段。

| 階段      | 名稱                    | 目的                                                      |
| ------- | --------------------- | ------------------------------------------------------- |
| Phase 0 | GridWorld Environment | 先確認任務定義正確                                               |
| Phase 1 | Hard-coded Baselines  | 建立 performance 上下界                                      |
| Phase 2 | Posterior Tracking    | 手工 belief update，理解 Bayesian RL                         |
| Phase 3 | RL² Baseline          | 做 recurrent meta-RL baseline                            |
| Phase 4 | Simplified VariBAD    | 實作 encoder + latent posterior + reward decoder + policy |

---

# 3. Environment Spec

## 3.1 Grid size

使用：

[
5 \times 5
]

座標系統建議：

```text
row = 0 at top
row = 4 at bottom
col = 0 at left
col = 4 at right
```

起點：

[
s_{\text{start}} = (4,0)
]

也就是左下角。

---

## 3.2 Goal distribution

每個 task 對應一個 hidden goal position：

[
g_i \sim p(g)
]

goal 在 task 內固定。

也就是同一個 BAMDP rollout 裡，即使 episode reset，goal 位置不變。

### 建議 allowed goal cells

論文描述 goal 不在起點附近，而是在灰色區域中隨機選。為了簡化，你可以先設定：

```python
allowed_goals = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
]
```

也就是前三排都可能是 goal。

這樣 goal 數量是：

[
15
]

如果你想更接近 Figure 1，可以改成只有上方某個灰色區域，但初版不用太糾結。

---

## 3.3 State

最簡單 state representation：

[
s_t = (row_t, col_t)
]

神經網路輸入時建議用 one-hot：

[
s_t \in {0,1}^{25}
]

例如：

```python
state_id = row * 5 + col
state_onehot = one_hot(state_id, 25)
```

---

## 3.4 Action space

離散動作，共 5 個：

| action id | action |
| --------: | ------ |
|         0 | up     |
|         1 | right  |
|         2 | down   |
|         3 | left   |
|         4 | stay   |

轉移 deterministic。

如果撞牆，位置不變。

---

## 3.5 Reward function

reward 設定：

[
r_{t+1}
=======

\begin{cases}
+1, & s_{t+1} = g \
-0.1, & s_{t+1} \neq g
\end{cases}
]

注意：reward 建議根據 **next state** 是否為 goal 給。

也就是 action 後移動到 goal 才拿 (+1)。

---

## 3.6 Episode horizon 與 BAMDP horizon

MDP episode horizon：

[
H = 15
]

BAMDP horizon：

[
H^+ = N \times H
]

建議先用：

[
N = 4
]

所以：

[
H^+ = 60
]

論文 Section 5.1 文字中有一個不一致處：寫 (H=15)，但又寫 (H^+=4H=45)。數學上 (4\times15=60)，Appendix B 也使用 (H^+=60)。實作時建議採用 **4 episodes × 15 steps = 60 steps**。

---

## 3.7 Reset rule

每 15 steps reset 回起點：

```python
if step_in_episode == 15:
    state = start_state
```

但 task 不重抽，goal 不變。

也就是：

```text
episode 1: same goal
episode 2: same goal
episode 3: same goal
episode 4: same goal
```

這點非常重要，因為 agent 應該利用前面 episode 的資訊。

---

# 4. Environment API Spec

建議實作 class：

```python
class GridWorldTask:
    def __init__(self, grid_size=5, goal=None, allowed_goals=None):
        self.grid_size = grid_size
        self.goal = goal
        self.allowed_goals = allowed_goals
        self.start_state = (4, 0)
        self.state = self.start_state
        self.step_count = 0

    def reset_task(self):
        # sample new goal
        # reset state and counters
        pass

    def reset_episode(self):
        # same goal, reset position only
        pass

    def step(self, action):
        # return next_state, reward, done, info
        pass

    def get_state_onehot(self):
        pass
```

### reset_task()

用於 meta-training / meta-test 抽新任務。

```python
def reset_task(self):
    self.goal = random.choice(self.allowed_goals)
    self.state = self.start_state
    self.step_count = 0
    return self.get_state_onehot()
```

### reset_episode()

用於同一 task 內 episode reset。

```python
def reset_episode(self):
    self.state = self.start_state
    return self.get_state_onehot()
```

### step(action)

```python
def step(self, action):
    next_state = transition(self.state, action)
    reward = 1.0 if next_state == self.goal else -0.1
    self.state = next_state
    self.step_count += 1

    episode_done = (self.step_count % H == 0)

    if episode_done:
        self.reset_episode()

    return obs, reward, episode_done, info
```

info 建議包含：

```python
info = {
    "goal": self.goal,
    "state": self.state,
    "step_in_episode": self.step_count % H,
    "global_step": self.step_count
}
```

訓練 policy 時不要把 goal 給 agent，只能用來 debug / evaluation。

---

# 5. Baseline Spec

## 5.1 Random Policy

目的：最低基準。

```python
action = random.choice([0,1,2,3,4])
```

評估：

[
J_{\text{random}}
=================

\frac{1}{K}
\sum_{k=1}^{K}
\sum_{t=0}^{H^+-1} r_t
]

---

## 5.2 Oracle Policy

目的：performance upper bound。

Oracle 知道 goal 位置。

策略：

1. 用 shortest path 走到 goal；
2. 到 goal 後 stay；
3. reset 後再用 shortest path 回 goal；
4. 到 goal 後 stay。

這代表 optimal policy with privileged goal information。

注意：Oracle 不是 Bayes-optimal，因為它不需要探索。

它是上界：

[
J_{\text{Bayes-optimal}} \leq J_{\text{Oracle}}
]

---

## 5.3 Posterior Sampling Baseline

目的：模擬 Thompson sampling / posterior sampling。

維護 possible goals：

```python
possible_goals = set(allowed_goals)
```

每個 episode 或每次 sample 時：

```python
sampled_goal = random.choice(possible_goals)
```

策略：

1. sample 一個 possible goal；
2. 走 shortest path 到 sampled goal；
3. 如果到達後 reward 不是 (+1)，排除該位置；
4. 重新 sample；
5. 如果找到 true goal，之後 exploit。

### belief update

每次 agent 到達某 cell 並觀察 reward：

```python
if reward != 1.0 and state in possible_goals:
    possible_goals.remove(state)
```

若 reward = 1：

```python
found_goal = state
possible_goals = {state}
```

---

## 5.4 Hand-coded Bayes-like Search

目的：接近 Bayes-optimal 的 deterministic search baseline。

這不是嚴格 Bayes-optimal planning，但可以做成「系統性掃描」。

策略：

1. 預先設計一條掃描路徑，覆蓋 allowed goal cells；
2. 走過每個 possible goal cell；
3. 找到 goal 後 stay；
4. reset 後直接回 goal。

例如：

```python
search_order = [
    (2,0), (2,1), (2,2), (2,3), (2,4),
    (1,4), (1,3), (1,2), (1,1), (1,0),
    (0,0), (0,1), (0,2), (0,3), (0,4),
]
```

這個 baseline 用來觀察「系統性探索」通常會比 posterior sampling 更有效。

---

# 6. Metrics Spec

## 6.1 Average online return

主要 metric：

[
\mathbb{E}*{M\sim p(M)}
\left[
\sum*{t=0}^{H^+-1} r_t
\right]
]

實作：

```python
avg_return = np.mean(total_returns_over_tasks)
```

---

## 6.2 Return per episode

因為 BAMDP rollout 包含多個 episodes，所以也要記錄：

[
R^{(1)}, R^{(2)}, R^{(3)}, R^{(4)}
]

你會看到：

* Episode 1：主要探索；
* Episode 2–4：如果已找到 goal，應該 exploitation；
* 好的 meta-RL 方法應該 episode 1 就不要太差，episode 2 開始快速接近 oracle。

---

## 6.3 Steps to find goal

記錄第一次找到 goal 的 timestep：

```python
first_goal_step
```

如果整個 BAMDP rollout 都沒找到：

```python
first_goal_step = None
```

---

## 6.4 Redundant visits

記錄 agent 重複訪問已排除 cell 的次數。

這用來衡量 exploration efficiency。

```python
redundant_visits = number of visits to known non-goal cells
```

Posterior sampling 通常會比 Bayes-like search 有更多重複路徑。

---

## 6.5 Success rate

定義：

```python
success = found_goal_within_first_episode
```

或：

```python
success = found_goal_within_H_plus
```

建議兩個都記。

---

# 7. RL² Baseline Spec

## 7.1 目的

RL² 是 VariBAD 的重要 baseline。

它不顯式建模 posterior，而是用 RNN hidden state 做 online adaptation。

形式：

[
h_t = GRU(h_{t-1}, x_t)
]

[
\pi(a_t \mid s_t, h_t)
]

---

## 7.2 Input

每一步輸入給 RNN：

[
x_t = [s_t, a_{t-1}, r_t, d_t]
]

其中：

| 變數                      | 維度 |
| ----------------------- | -: |
| (s_t) one-hot           | 25 |
| (a_{t-1}) one-hot       |  5 |
| (r_t) scalar            |  1 |
| (d_t) reset / done flag |  1 |

總維度：

[
25 + 5 + 1 + 1 = 32
]

第一步時：

```python
prev_action = zero vector
prev_reward = 0
done = 0
```

---

## 7.3 Network

```python
class RL2Policy(nn.Module):
    def __init__(self):
        self.gru = nn.GRU(input_size=32, hidden_size=128)
        self.policy_head = nn.Linear(128 + 25, 5)
        self.value_head = nn.Linear(128 + 25, 1)
```

policy input 可以用：

```python
policy_input = concat(state_onehot, hidden_state)
```

輸出：

```python
action_logits = policy_head(policy_input)
value = value_head(policy_input)
```

---

## 7.4 Training algorithm

初版建議用 **A2C**，比 PPO 簡單。

A2C loss：

[
L_{\text{policy}}
=================

-\log \pi(a_t\mid s_t,h_t) A_t
]

[
L_{\text{value}}
================

(V(s_t,h_t)-R_t)^2
]

[
L_{\text{entropy}}
==================

-\mathcal{H}(\pi)
]

總 loss：

[
L
=

L_{\text{policy}}
+
c_v L_{\text{value}}
--------------------

c_e \mathcal{H}(\pi)
]

建議係數：

```python
value_coef = 0.5
entropy_coef = 0.01
gamma = 0.95
lr = 1e-3
```

---

# 8. Simplified VariBAD Spec

## 8.1 目的

實作 VariBAD 的核心機制：

[
\tau_{:t}
\rightarrow
q_\phi(m\mid \tau_{:t})
\rightarrow
\pi_\psi(a_t\mid s_t,q_\phi(m\mid \tau_{:t}))
]

初版只做 reward decoder，不做 transition decoder。

---

## 8.2 Encoder

Encoder 使用 GRU。

輸入：

[
x_t = [s_t, a_{t-1}, r_t, d_t]
]

同 RL²。

輸出 posterior parameters：

[
\mu_t, \log\sigma_t^2
]

建議 latent dim：

[
d_m = 5
]

這也和論文 GridWorld 使用的 task embedding size 一致。

### Network

```python
class VariBADEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, latent_dim=5):
        self.fc = nn.Linear(input_dim, 40)
        self.gru = nn.GRU(input_size=40, hidden_size=hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_seq, h=None):
        z = F.relu(self.fc(x_seq))
        out, h = self.gru(z, h)
        mu = self.mu_head(out)
        logvar = self.logvar_head(out)
        return mu, logvar, h
```

---

## 8.3 Reparameterization

從 posterior sample latent：

[
m_t = \mu_t + \sigma_t \epsilon
]

其中：

[
\epsilon \sim \mathcal{N}(0,I)
]

實作：

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
m = mu + eps * std
```

policy 可以有兩種設計：

### Design A：policy 吃 posterior parameters

```python
belief_input = concat(mu, logvar)
```

這比較符合 VariBAD 的 uncertainty-aware policy。

### Design B：policy 吃 sampled latent

```python
belief_input = m
```

這比較像 PEARL / posterior sampling。

建議先用 **Design A**：

[
\pi(a_t\mid s_t,\mu_t,\log\sigma_t^2)
]

---

## 8.4 Policy Network

輸入：

[
[s_t, \mu_t, \log\sigma_t^2]
]

維度：

[
25 + 5 + 5 = 35
]

輸出：

* action logits；
* value estimate。

```python
class VariBADPolicy(nn.Module):
    def __init__(self, state_dim=25, latent_dim=5):
        input_dim = state_dim + latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(32, 5)
        self.value_head = nn.Linear(32, 1)
```

---

## 8.5 Reward Decoder

Reward decoder 用來訓練 latent (m) 是否包含 task information。

輸入：

[
[s_i, a_i, s_{i+1}, m_t]
]

輸出：

[
\hat{p}(r_{i+1}=1)
]

因為 GridWorld reward 只有 goal / non-goal，可以用 binary classification。

label：

```python
label = 1 if reward == 1.0 else 0
```

Network：

```python
class RewardDecoder(nn.Module):
    def __init__(self, state_dim=25, action_dim=5, latent_dim=5):
        input_dim = state_dim + action_dim + state_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, s, a, s_next, m):
        x = torch.cat([s, a, s_next, m], dim=-1)
        logits = self.net(x)
        return logits
```

Loss：

[
L_{\text{reward-decoder}}
=========================

BCEWithLogitsLoss(\hat{r}, r_{\text{binary}})
]

---

## 8.6 KL Loss

Posterior：

[
q_\phi(m\mid \tau_{:t})
=======================

\mathcal{N}(\mu_t,\sigma_t^2)
]

初版可以先用 standard normal prior：

[
p(m)=\mathcal{N}(0,I)
]

KL：

[
KL(q\Vert p)
============

-\frac{1}{2}
\sum_j
\left(
1+\log\sigma_j^2-\mu_j^2-\sigma_j^2
\right)
]

實作：

```python
kl = -0.5 * torch.sum(
    1 + logvar - mu.pow(2) - logvar.exp(),
    dim=-1
)
```

進階版再改成論文提到的 previous posterior prior：

[
p_t(m) = q_\phi(m\mid \tau_{:t-1})
]

但初版不建議一開始做，會增加 implementation complexity。

---

## 8.7 ELBO Loss

論文中的 ELBO：

[
ELBO_t
======

\mathbb{E}*{q*\phi(m\mid \tau_{:t})}
[
\log p_\theta(\tau_{:H^+}\mid m)
]
-

KL(q_\phi(m\mid \tau_{:t})\Vert p(m))
]

實作時我們最小化 negative ELBO：

[
L_{\text{VAE}}
==============

L_{\text{reconstruction}}
+
\beta L_{\text{KL}}
]

建議：

```python
beta = 0.1
```

或先用：

```python
beta = 1.0
```

若 posterior collapse，再降低。

---

# 9. Training Loop Spec

## 9.1 Rollout collection

每次 meta-batch 抽多個 tasks。

建議：

```python
num_processes = 16
num_episodes_per_task = 4
H = 15
H_plus = 60
```

每個 task 收集：

```python
trajectory = [
    {
        "s": s_t,
        "a": a_t,
        "r": r_{t+1},
        "s_next": s_{t+1},
        "done": done_t,
        "mu": mu_t,
        "logvar": logvar_t,
    }
]
```

---

## 9.2 Policy update

用 A2C。

計算 returns：

[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
]

Advantage：

[
A_t = G_t - V_t
]

Policy loss：

[
L_\pi = -\log \pi(a_t\mid s_t,\mu_t,\log\sigma_t^2) A_t
]

Value loss：

[
L_V = (V_t - G_t)^2
]

Entropy loss：

[
L_H = -\mathcal{H}(\pi)
]

總 RL loss：

[
L_{\text{RL}}
=============

L_\pi
+
0.5L_V
------

0.01H
]

---

## 9.3 VAE update

從 trajectory buffer sample 一批 trajectories。

對每個 trajectory，取多個 context time (t)。

簡化版可以先用：

```python
context_times = [0, 5, 10, 15, 30, 45, 60]
```

對每個 (t)：

1. encoder 只看 (\tau_{:t})；
2. 得到 (\mu_t,\log\sigma_t^2)；
3. sample (m_t)；
4. decoder 用 (m_t) 預測整段 trajectory 裡所有 transition 的 reward。

也就是：

```python
for t in context_times:
    mu_t, logvar_t = encoder(context_prefix)
    m_t = reparameterize(mu_t, logvar_t)

    for i in range(H_plus):
        pred_reward = decoder(s_i, a_i, s_next_i, m_t)
```

這會逼 encoder 用 partial trajectory 推斷整個 task。

這正是 VariBAD 的關鍵。

---

# 10. Evaluation Spec

每隔固定 update 數做 evaluation。

建議：

```python
eval_tasks = 100
deterministic_policy = True
```

記錄：

1. average total return over (H^+)；
2. return per episode；
3. first goal step；
4. success rate；
5. trajectory path；
6. posterior mean / variance over time；
7. decoder reward prediction heatmap。

---

# 11. Visualization Spec

## 11.1 Trajectory visualization

畫 5×5 grid：

* 起點標記 S；
* true goal 標記 G；
* agent path 用箭頭或數字標記；
* visited cells 上色。

你要看的是：

1. agent 是否系統性搜尋；
2. 是否重複走已排除 cell；
3. 找到 goal 後是否 stay；
4. reset 後是否直接回 goal。

---

## 11.2 Posterior variance curve

畫：

[
\frac{1}{d_m}
\sum_j \sigma_{t,j}^2
]

對 timestep。

預期：

* 一開始 variance 高；
* 探索時逐漸下降；
* 找到 goal 後快速下降；
* reset 後不應該重新變高，因為 task 沒變。

---

## 11.3 Latent mean curve

畫每個 latent dimension 的：

[
\mu_{t,j}
]

對 timestep。

預期：

* 前期變化較大；
* 找到 goal 後穩定；
* 不同 goal 對應不同 latent pattern。

---

## 11.4 Reward prediction heatmap

對每個 timestep (t)，用當前 posterior (q(m\mid\tau_{:t})) 預測每個 cell 是 goal 的機率。

做法：

對每個 cell (c)，構造：

```python
s = some previous state
a = action that moves into c
s_next = c
```

或簡化成讓 decoder 接：

[
[s_{\text{dummy}}, a_{\text{dummy}}, s_{\text{cell}}, m_t]
]

初版 decoder 如果依賴 (s,a,s')，你需要小心構造 transition。

更簡單的替代方案是另外做一個 auxiliary goal decoder：

[
p(g \mid m)
]

輸出 25 維 goal probability。

這不是原始論文必要設計，但非常有利於 debug。

---

# 12. Expected Results

## 12.1 Random

預期：

* return 很低；
* often cannot find goal；
* path 無結構。

---

## 12.2 Oracle

預期：

* upper bound；
* 每個 episode 都快速到 goal；
* 到 goal 後 stay。

---

## 12.3 Posterior Sampling

預期：

* 比 random 好；
* 但會有重複路徑；
* 如果 sample 到遠處錯誤 goal，會浪費很多 steps；
* online return 明顯低於 Bayes-like systematic search。

---

## 12.4 RL²

預期：

* 訓練好後可以學會探索；
* 但 hidden state 不可解釋；
* reset 後可能不穩；
* latent / uncertainty 無法直接分析。

---

## 12.5 Simplified VariBAD

預期：

* return 高於 random；
* 接近或高於 posterior sampling；
* reward decoder 可以逐步排除 non-goal cells；
* posterior variance 找到 goal 後下降；
* path 有系統性探索傾向。

---

# 13. Success Criteria

## Minimum success

做到以下即可算第一版成功：

1. GridWorld environment 正確；
2. Random / Oracle / Posterior Sampling baseline 可跑；
3. 可畫每個 baseline 的 average return；
4. Posterior Sampling 明顯優於 Random；
5. Oracle 明顯最高。

---

## Intermediate success

做到以下算第二版成功：

1. RL² 可以訓練；
2. RL² 在 4 episodes 內 return 高於 posterior sampling；
3. RL² 能在找到 goal 後 reset 回來繼續 exploit。

---

## Full simplified VariBAD success

做到以下算第三版成功：

1. encoder 輸出 posterior mean / variance；
2. reward decoder loss 下降；
3. posterior variance 隨資料增加下降；
4. reward prediction heatmap 能排除 visited non-goal cells；
5. policy return 高於 posterior sampling；
6. agent 找到 goal 後轉成 exploitation。

---

# 14. Recommended Directory Structure

```text
varibad_gridworld/
│
├── envs/
│   └── gridworld.py
│
├── baselines/
│   ├── random_policy.py
│   ├── oracle_policy.py
│   ├── posterior_sampling.py
│   └── bayes_like_search.py
│
├── models/
│   ├── rl2.py
│   ├── varibad_encoder.py
│   ├── varibad_policy.py
│   └── reward_decoder.py
│
├── trainers/
│   ├── train_rl2.py
│   ├── train_varibad.py
│   └── a2c.py
│
├── utils/
│   ├── rollout.py
│   ├── buffer.py
│   ├── metrics.py
│   └── visualization.py
│
├── configs/
│   ├── gridworld.yaml
│   ├── rl2.yaml
│   └── varibad.yaml
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_rl2.py
│   └── run_varibad.py
│
└── README.md
```

---

# 15. Config Spec

建議用 YAML 管理參數。

```yaml
env:
  grid_size: 5
  start_state: [4, 0]
  episode_horizon: 15
  num_episodes_per_task: 4
  reward_goal: 1.0
  reward_non_goal: -0.1
  allowed_goals:
    - [0, 0]
    - [0, 1]
    - [0, 2]
    - [0, 3]
    - [0, 4]
    - [1, 0]
    - [1, 1]
    - [1, 2]
    - [1, 3]
    - [1, 4]
    - [2, 0]
    - [2, 1]
    - [2, 2]
    - [2, 3]
    - [2, 4]

training:
  algorithm: a2c
  gamma: 0.95
  lr_policy: 0.001
  lr_vae: 0.001
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  num_processes: 16
  updates: 5000

model:
  latent_dim: 5
  encoder_hidden_dim: 64
  rl2_hidden_dim: 128
  policy_hidden_dim: 32
  decoder_hidden_dim: 32

vae:
  beta_kl: 0.1
  context_times: [0, 5, 10, 15, 30, 45, 60]
  decoder_type: reward_only
```

---

# 16. 實作順序

## Week 1：Environment + Baselines

### 要做

1. GridWorld environment；
2. Random policy；
3. Oracle policy；
4. Posterior sampling；
5. Bayes-like search；
6. baseline evaluation；
7. trajectory visualization。

### 產出

```text
baseline_returns.png
baseline_paths.png
baseline_metrics.csv
```

---

## Week 2：RL²

### 要做

1. RL² model；
2. A2C training loop；
3. recurrent rollout；
4. hidden state 不在 episode reset 時清掉；
5. evaluation per episode return。

### 產出

```text
rl2_learning_curve.png
rl2_eval_paths.png
rl2_return_per_episode.png
```

---

## Week 3：Simplified VariBAD

### 要做

1. VariBAD encoder；
2. Gaussian posterior；
3. reward decoder；
4. VAE loss；
5. policy conditioning on posterior parameters；
6. A2C + VAE joint training；
7. posterior visualization。

### 產出

```text
varibad_learning_curve.png
posterior_variance.png
latent_mean.png
reward_prediction_heatmap.png
varibad_eval_paths.png
```

---

## Week 4：Comparison + Report

### 要做

1. 比較所有方法；
2. 整理 metrics；
3. 畫 Figure 1-like plot；
4. 分析 failure cases；
5. 寫報告。

### 產出

```text
comparison_table.md
experiment_report.md
figures/
```

---

# 17. 最重要的實作注意事項

## 17.1 episode reset 不等於 task reset

這是最容易寫錯的地方。

錯誤寫法：

```python
if episode_done:
    sample_new_goal()
```

正確寫法：

```python
if episode_done:
    reset_position_only()
```

goal 必須在整個 BAMDP rollout 內固定。

---

## 17.2 RNN hidden state 不能在 episode reset 清掉

對 RL² 和 VariBAD encoder 來說，hidden state / posterior 要跨 episode 保留。

因為 agent 應該記得前面 episode 找到的資訊。

只有 task reset 時才清空 hidden state。

---

## 17.3 Policy 不可以看到 true goal

true goal 只能用於：

* reward 計算；
* oracle baseline；
* evaluation；
* visualization。

訓練 RL² / VariBAD 時，policy input 不能包含 goal。

---

## 17.4 先不要做 transition decoder

GridWorld transition deterministic 且與 task 無關。

初版做 transition decoder 幫助不大。

reward decoder 更直接，因為 task 差異主要在 reward function。

---

## 17.5 先用 A2C，不要一開始上 PPO

PPO 比較穩，但 implementation 複雜。

你現在目標是理解 VariBAD pipeline。

A2C 足夠。

---

# 18. 最終比較表格式

最後報告建議放這張表。

| Method             | Avg Return | Ep1 Return | Ep2 Return | First Goal Step | Success Rate | Redundant Visits |
| ------------------ | ---------: | ---------: | ---------: | --------------: | -----------: | ---------------: |
| Random             |            |            |            |                 |              |                  |
| Oracle             |            |            |            |                 |              |                  |
| Posterior Sampling |            |            |            |                 |              |                  |
| Bayes-like Search  |            |            |            |                 |              |                  |
| RL²                |            |            |            |                 |              |                  |
| Simplified VariBAD |            |            |            |                 |              |                  |

這張表會直接回答：

> VariBAD 是否真的比 posterior sampling 更有效探索？
> RL² 是否可以學到類似行為？
> VariBAD 的 posterior 是否真的有解釋性？

---

# 19. 最小可行版本範圍

你第一版只需要做到：

```text
GridWorld + Random + Oracle + Posterior Sampling + Bayes-like Search
```

先不要寫 neural network。

這會讓你確認：

1. 任務設計正確；
2. posterior sampling 確實可能浪費探索；
3. systematic search 確實比較接近 Bayes-optimal；
4. online return metric 設定正確。

等這些 baseline 跑出合理圖之後，再進 RL² 和 VariBAD。

---

# 20. 結論

這個 spec 的核心是：**先建立一個你完全掌控的 GridWorld BAMDP，再逐步把 meta-RL 組件加上去。**

最推薦的實作順序是：

[
\text{Environment}
\rightarrow
\text{Baselines}
\rightarrow
\text{RL}^2
\rightarrow
\text{Reward Decoder}
\rightarrow
\text{Simplified VariBAD}
\rightarrow
\text{Posterior Visualization}
]

不要先追求完整論文 reproduction。先讓自己親眼看到：

[
\text{uncertainty decreases}
]

以及：

[
\text{policy changes from exploration to exploitation}
]

這樣 VariBAD 的概念才會真正變成你的實驗能力。
