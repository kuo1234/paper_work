對，你這個修正是對的。現在不應該急著設計 zero-shot。你現在最該做的是一個 **few-shot meta-RL 對照實驗**，目標不是追求最強結果，而是看清楚：

> **MAML、RL²、PEARL 這三種方法，分別把「快速適應新任務」這件事交給哪個機制來完成？**

PEARL 論文其實也正是用這個角度比較：MAML 屬於 gradient-based meta-RL，RL² 屬於 recurrent / context-based meta-RL，PEARL 則把 task inference 拆成 probabilistic latent context，並用 off-policy actor-critic 來提升 sample efficiency。論文中也明確指出 PEARL 在 meta-test 時會先用 prior 探索，再用收集到的 trajectory 更新 context posterior；而 MAML、RL² 則是它用來比較的主要 meta-RL baseline。

---

# 一、三個方法本質上在優化什麼？

先用一句話抓住三者差異：

| 方法        | 它學的是什麼？                                                                | 適應新任務靠什麼？                    | 最核心的優化對象                              |
| --------- | ---------------------------------------------------------------------- | ---------------------------- | ------------------------------------- |
| **MAML**  | 一組容易被微調的初始 policy parameters                                           | test-time gradient update    | policy initialization                 |
| **RL²**   | 一個會「用歷史經驗更新內部狀態」的 RNN policy                                           | RNN hidden state             | recurrent adaptation procedure        |
| **PEARL** | 一個能從 context 推論 task latent variable 的 encoder，加上一個 conditioned policy | posterior inference over (z) | task inference encoder + actor-critic |

這三個方法都在做 meta-learning，但「meta」的東西不同。

---

# 二、MAML：優化的是「好微調的初始化」

MAML 的核心不是直接學一個在所有任務都好的 policy，而是學一組參數：

[
\theta
]

使得它在新任務 (T_i) 上，只要用少量資料做一兩步 gradient update，就可以變成適合該任務的 policy：

[
\theta_i' = \theta - \alpha \nabla_\theta L_{T_i}(\theta)
]

然後 MAML 真正優化的是：

[
\min_\theta \sum_i L_{T_i}(\theta_i')
]

意思是：

> 我不要求 (\theta) 本身馬上解好每個任務，而是要求它「經過少量資料更新後」能解好每個任務。

所以 MAML 的 few-shot adaptation 是靠 **gradient descent**。

---

## MAML 在你的實驗中應該觀察什麼？

你要看的是：

1. **未更新前表現如何？**

   這是 (K=0) 或 before adaptation。

2. **收集 1 條 trajectory 後，做 1 次 gradient update，表現提升多少？**

3. **多做幾次 gradient update，是否繼續變好？**

MAML 的典型現象應該是：

```text
before adaptation: 普通或偏弱
after 1 update: 明顯提升
after 2-5 updates: 繼續提升或趨於穩定
```

所以你的實驗如果要看懂 MAML，就一定要畫：

[
\text{return before update} \rightarrow \text{return after 1 update} \rightarrow \text{return after 2 updates}
]

而不是只看 final return。

---

# 三、RL²：優化的是「RNN 裡的學習演算法」

RL² 的想法完全不同。它不在 test-time 做 gradient update。

它把 agent 跑過的歷史經驗餵給 RNN，例如：

[
(s_t, a_{t-1}, r_{t-1}, d_{t-1})
]

RNN hidden state 會變成一種任務記憶：

[
h_t = f_\theta(h_{t-1}, s_t, a_{t-1}, r_{t-1}, d_{t-1})
]

policy 是：

[
\pi_\theta(a_t \mid s_t, h_t)
]

所以 RL² 學到的不是一個顯式更新公式，而是一個「內化在 RNN hidden state 裡的 adaptation rule」。

它在 meta-training 時會被訓練成：

> 第 1 個 episode 可以探索，第 2 個 episode 利用第 1 個 episode 的 reward feedback 做得更好，第 3 個 episode 再更好。

---

## RL² 在你的實驗中應該觀察什麼？

你要看的是 episode-to-episode 的 improvement：

```text
episode 1: 還不知道任務，只能探索
episode 2: RNN hidden state 已經看過 reward feedback，應該變好
episode 3: 如果任務資訊更清楚，應該再提升
```

所以 RL² 的測試不能每個 episode 都 reset hidden state。

錯誤測法：

```text
每條 trajectory 都 reset RNN hidden state
```

這樣等於把 RL² 的 adaptation 能力砍掉。

正確測法：

```text
同一個 test task 中，連續跑 K 條 trajectories，RNN hidden state 不重置
```

你要畫：

[
R_1, R_2, R_3, ..., R_K
]

其中 (R_1) 是第一條 trajectory 的 return，(R_2) 是第二條 trajectory 的 return。

RL² 的典型現象應該是：

```text
episode 1: 探索
episode 2: 根據 hidden state 改善
episode 3+: 持續改善或穩定
```

---

# 四、PEARL：優化的是「task inference + off-policy control」

PEARL 的設計跟前兩者又不同。

它不靠 gradient update，也不靠 RNN hidden state。它用一個 encoder：

[
q_\phi(z \mid c)
]

從 context 裡推論目前任務的 latent variable (z)。

context 是過去收集到的 transitions：

[
c = {(s, a, r, s')}
]

然後 policy 和 critic 都 conditioned on (z)：

[
\pi_\theta(a \mid s, z)
]

[
Q_\theta(s, a, z)
]

也就是說，PEARL 把 adaptation 拆成兩件事：

1. **task inference**：encoder 根據 context 推論 (z)。
2. **control**：policy 根據 state 和 (z) 選 action。

論文中特別強調，PEARL 的 actor 和 critic 可以用 replay buffer 裡的 off-policy data 訓練，而 encoder 的 context sampling 則要小心處理，避免和 meta-test 時的 on-policy adaptation distribution 差太多。它的 Algorithm 1 也明確把 context batch 和 RL batch 分開抽樣。

---

## PEARL 在你的實驗中應該觀察什麼？

你要看的是：

1. 沒有 context 時，從 prior sample (z)，表現如何？
2. 收集 1 條 trajectory 後，用 context 更新 posterior，表現是否提升？
3. context 越多，posterior 是否更準，return 是否上升？
4. PEARL 是否比 MAML / RL² 更省 meta-training samples？

PEARL 的典型現象應該是：

```text
trajectory 1: 從 prior z 探索
trajectory 2: 用 c 更新 q(z|c)，策略變得更對任務
trajectory 3+: posterior 更集中，表現穩定提升
```

尤其在 sparse reward 裡，PEARL 的 probabilistic context 可以做 posterior sampling，形成比較連貫的探索。PEARL 論文中的 sparse 2D navigation 實驗就是用這點展示它能透過 posterior sampling 做 structured exploration。

---

# 五、所以你的第一個小實驗應該設計成什麼？

我建議你不要一開始碰 MuJoCo。先做一個 **2D Goal Navigation**。原因是：

1. 任務簡單。
2. 可以視覺化 trajectory。
3. MAML、RL²、PEARL 的 adaptation 差異會很明顯。
4. 之後可以自然延伸到 sparse reward、zero-shot、OOD generalization。

---

# 六、第一個實驗：2D Goal Navigation Few-Shot Meta-RL

## 6.1 任務設定

agent 是一個 2D point mass。

狀態：

[
s_t = (x_t, y_t)
]

動作：

[
a_t = (\Delta x_t, \Delta y_t)
]

每個 task 是一個不同 goal：

[
T_g: \text{move to goal } g = (g_x, g_y)
]

goal 從某個分布取樣，例如圓周上或方形區域中。

---

## 6.2 Reward 設計

初學階段先用 dense reward，不要一開始用 sparse reward。

Dense reward：

[
r_t = -|p_t - g|_2
]

其中：

[
p_t = (x_t, y_t)
]

意思是離 goal 越近 reward 越高。

也可以加 action penalty：

[
r_t = -|p_t - g|_2 - 0.01|a_t|_2^2
]

但第一版可以先不要加，避免干擾分析。

---

## 6.3 Task split

例如：

| Split            | 數量 | 用途               |
| ---------------- | -: | ---------------- |
| train tasks      | 50 | meta-training    |
| validation tasks | 10 | 調 hyperparameter |
| test tasks       | 20 | meta-testing     |

每個 task 對應一個 goal location。

例如：

```text
train goals: 隨機取 50 個點
val goals: 取 10 個沒看過的點
test goals: 取 20 個沒看過的點
```

這就是基本 few-shot meta-learning setting。

---

# 七、三個方法在同一個任務上怎麼跑？

你要讓三個方法看到同一組 train tasks、val tasks、test tasks。

否則比較不公平。

---

## 7.1 MAML 訓練流程

對每個 meta-iteration：

1. sample 一批 tasks：

[
T_i \sim p_{\text{train}}(T)
]

2. 對每個 task，用目前 policy 收集 support trajectories。
3. 用 support trajectories 做 inner-loop policy gradient update：

[
\theta_i' = \theta - \alpha \nabla_\theta L_{T_i}^{support}(\theta)
]

4. 用更新後的 policy (\theta_i') 收集 query trajectories。
5. 用 query loss 更新原始初始化 (\theta)。

MAML 優化的是：

```text
讓 policy 初始化 θ 經過少量 gradient update 後，在該任務上表現好
```

---

## 7.2 RL² 訓練流程

對每個 meta-iteration：

1. sample 一批 tasks。
2. 對每個 task，連續跑多個 episodes，例如 3 或 5 條 trajectories。
3. RNN hidden state 在同一個 task 內不要 reset。
4. task 換掉時才 reset hidden state。
5. 用 PPO / A2C 之類 on-policy RL 更新整個 RNN policy。

RL² 優化的是：

```text
讓 RNN policy 從前幾個 episodes 的 state-action-reward history 中學會辨識任務
```

---

## 7.3 PEARL 訓練流程

PEARL 流程比較複雜，但在概念上是：

1. 每個 train task 有自己的 replay buffer。
2. 從目前 context (c) 推論：

[
z \sim q_\phi(z \mid c)
]

3. 用：

[
\pi_\theta(a \mid s, z)
]

收集資料放進 replay buffer。

4. 訓練時分開抽：

   * context batch：給 encoder 推論 (z)
   * RL batch：給 SAC actor / critic 更新

5. encoder 用 critic loss + KL loss 訓練。

6. actor / critic 用 off-policy SAC 更新。

PEARL 優化的是：

```text
讓 encoder 能從少量 context 推論任務，讓 actor/critic 在給定 z 時能有效控制
```

---

# 八、測試 protocol：這是最重要的公平性設計

你的測試要統一成：

[
K = 0, 1, 2, 3, 5
]

條 adaptation trajectories。

對每個 test task，記錄每個 (K) 下的 return。

---

## 8.1 MAML 的測試

對每個 test task：

### (K=0)

不做 gradient update，直接用 (\theta) 評估。

### (K=1)

1. 用 (\theta) 收集 1 條 support trajectory。
2. 做一次 gradient update 得到 (\theta')。
3. 用 (\theta') 評估 query return。

### (K=2,3,5)

用更多 support trajectories 做 inner update，再評估。

---

## 8.2 RL² 的測試

對每個 test task：

1. reset RNN hidden state。
2. 在同一個 task 中連續跑 5 條 trajectories。
3. 不做 gradient update。
4. 不 reset hidden state。
5. 記錄每一條 trajectory 的 return。

所以：

|               (K) | RL² 對應含義                                |
| ----------------: | --------------------------------------- |
| 0 / first episode | 沒有任務歷史，只靠初始 hidden state                |
|                 1 | 已看過 1 條 trajectory 的 reward feedback    |
|                 2 | 已看過 2 條 trajectories                    |
|                 5 | 已累積 5 條 trajectories 的 recurrent memory |

嚴格來說，RL² 的 (K=0) 對應「第一條 trajectory 開始時」。實務上你可以把 first trajectory return 記為 (R_0)，第二條記為 (R_1)。

---

## 8.3 PEARL 的測試

對每個 test task：

1. 初始化 context：

[
c = \emptyset
]

2. 第 1 條 trajectory：

[
z \sim q_\phi(z \mid \emptyset)
]

通常等同於從 prior sample。

3. 收集 trajectory 後加入 context：

[
c \leftarrow c \cup D_1
]

4. 第 2 條 trajectory：

[
z \sim q_\phi(z \mid c)
]

5. 重複直到 (K=5)。

PEARL 的 (K) 就是 context 裡已經累積幾條 trajectories。

---

# 九、主要結果圖：Adaptation Curve

你第一個小實驗最該產生的圖是這張：

```text
Average Test Return
^
|                         PEARL
|                    _____
|              RL² _/
|         MAML ___/
|
+------------------------------------>
   K=0     K=1     K=2     K=3     K=5
   number of adaptation trajectories
```

這張圖的意義：

* (K=0)：完全還沒適應。
* (K=1)：看過 1 條 trajectory 後。
* (K=2)：看過 2 條 trajectories 後。
* (K=5)：few-shot adaptation 後。

你不是只比較最後誰最高，而是比較：

1. 誰一開始比較好？
2. 誰適應最快？
3. 誰最穩？
4. 誰需要最多資料？
5. 誰的 meta-training sample efficiency 最好？

---

# 十、你應該記錄的指標

## 10.1 Adaptation speed

看從 (K=0) 到 (K=1) 的提升：

[
\Delta R_1 = R_1 - R_0
]

如果 MAML 的 (\Delta R_1) 很大，表示它真的學到「好微調的初始化」。

如果 RL² 的 (\Delta R_1) 很大，表示它真的用 hidden state 學會從 reward feedback 適應。

如果 PEARL 的 (\Delta R_1) 很大，表示 encoder 能用少量 context 推論任務。

---

## 10.2 Final few-shot performance

看：

[
R_5
]

表示給足 5 條 trajectory 後，誰能適應得最好。

---

## 10.3 Meta-training sample efficiency

這對 PEARL 很重要。

你要畫第二張圖：

```text
Average Test Return after K=2 adaptation
^
|
|
+------------------------------------>
   environment steps during meta-training
```

這張圖比較：

* MAML 需要多少 samples 才學起來？
* RL² 需要多少 samples 才學起來？
* PEARL 是否因為 off-policy replay 更省資料？

PEARL 論文的主要實驗就是用 test-task performance vs meta-training samples 來展示 PEARL 相比 MAML、RL² 等方法有顯著 sample efficiency 優勢，並指出 PEARL 能透過 off-policy data 達到 20–100 倍 sample efficiency improvement。

---

## 10.4 Stability

你至少跑 3 個 random seeds，最好 5 個。

記錄：

[
\text{mean} \pm \text{std}
]

因為 meta-RL variance 很大，尤其 MAML 和 RL² 都可能不穩。

---

# 十一、初學階段最推薦的實驗順序

不要一口氣做完整 PEARL 論文那種 MuJoCo benchmark。你可以照下面順序。

---

## Phase 1：2D Goal Navigation，dense reward

目的：理解三個方法 adaptation 機制。

設定：

```text
state = agent position
action = movement direction
task = goal location
reward = negative distance to goal
horizon = 50
train tasks = 50 goals
test tasks = 20 goals
K = 0, 1, 2, 3, 5
```

這個階段你要看：

| 方法    | 你應該觀察到的現象                     |
| ----- | ----------------------------- |
| MAML  | 做 gradient update 後 return 上升 |
| RL²   | 第二條 trajectory 開始比第一條好        |
| PEARL | context 增加後 (z) 更準，return 上升  |

---

## Phase 2：2D Goal Navigation，sparse reward

目的：觀察探索差異。

reward 改成：

[
r_t =
\begin{cases}
1, & |p_t - g| < \epsilon \
0, & \text{otherwise}
\end{cases}
]

這時候你會看到：

| 方法    | 預期                                            |
| ----- | --------------------------------------------- |
| MAML  | 如果探索不到 reward，gradient signal 很弱              |
| RL²   | 需要學到 trial-and-error exploration              |
| PEARL | probabilistic (z) + posterior sampling 可能更有優勢 |

PEARL 論文的 sparse 2D navigation 就是在這裡展示 posterior sampling 的價值：agent 一開始從 prior sample task hypothesis，收集 trajectory 後更新 posterior，再逐漸更精準地探索。

---

## Phase 3：Half-Cheetah-Vel

目的：對齊正式 meta-RL benchmark。

每個 task 是不同 target velocity：

[
v^*
]

reward 大致是：

[
r_t = -|v_t - v^*| - \text{control cost}
]

這個階段比較接近 PEARL 論文中的 MuJoCo benchmark。論文也使用 Half-Cheetah-Vel、Ant-Goal-2D、Walker-2D-Params 等任務來比較 PEARL、MAML、RL² 等方法。

---

# 十二、你第一個小實驗的最小版本

如果只做一個最小可行版本，我建議：

## Environment

2D point navigation。

## Methods

先不要一次全部實作很完整，可以分階段：

1. Multi-task PPO baseline
2. MAML-PPO
3. RL²-PPO
4. PEARL-lite 或 PEARL-SAC

PEARL 最難，可以最後做。

---

## Dataset / task distribution

```text
train goals: 40
val goals: 10
test goals: 20
horizon: 50
max adaptation trajectories: 5
reward: dense reward
```

---

## Main plot

```text
x-axis: adaptation trajectories K
y-axis: average test return
curves: MAML, RL², PEARL
```

---

## Secondary plot

```text
x-axis: meta-training environment steps
y-axis: test return after K=2
curves: MAML, RL², PEARL
```

第一張圖看 few-shot adaptation。
第二張圖看 sample efficiency。

---

# 十三、三者在實驗中你應該怎麼解讀

## 如果 MAML 表現好

代表：

```text
任務之間存在可以共享的 policy initialization。
新任務只需要少量 gradient update 就能適應。
```

你可以說 MAML 優化的是：

> gradient-adaptable representation / initialization。

---

## 如果 RL² 表現好

代表：

```text
任務資訊可以從 trajectory history 中被 RNN 記住。
RNN hidden state 學到了一種 implicit learning algorithm。
```

你可以說 RL² 優化的是：

> recurrent memory-based adaptation procedure。

---

## 如果 PEARL 表現好

代表：

```text
任務可以被少量 transition context 推論成 latent variable z。
policy conditioned on z 可以有效控制。
off-policy replay 提升 meta-training sample efficiency。
```

你可以說 PEARL 優化的是：

> probabilistic task inference + off-policy control。

---

# 十四、最核心的比較表

你可以把這張表當成你的實驗設計中心。

| 比較面向                    | MAML                           | RL²                     | PEARL                    |                    |         |
| ----------------------- | ------------------------------ | ----------------------- | ------------------------ | ------------------ | ------- |
| adaptation 形式           | gradient update                | RNN hidden state update | posterior inference      |                    |         |
| test-time 是否更新參數        | 是                              | 否                       | 否                        |                    |         |
| 是否需要 trajectory history | 需要，用來算 gradient                | 需要，餵給 RNN               | 需要，作為 context            |                    |         |
| 訓練方式                    | on-policy meta-policy gradient | on-policy recurrent RL  | off-policy SAC + encoder |                    |         |
| 學到的東西                   | 好微調初始化                         | 內隱學習演算法                 | task latent inference    |                    |         |
| few-shot 表現重點           | update 後提升多少                   | episode 間是否變好           | context 增加後是否變好          |                    |         |
| sample efficiency       | 通常較低                           | 通常較低                    | 通常較高                     |                    |         |
| 初學實驗觀察重點                | (\theta \to \theta') 的提升       | (h_0 \to h_K) 的提升       | (q(z                     | \emptyset) \to q(z | c)) 的提升 |

---

# 十五、你現在應該先做哪個版本？

我建議你的第一版研究問題寫成：

> 在簡單 2D goal navigation few-shot meta-RL 任務中，比較 MAML、RL²、PEARL 三種 adaptation mechanism：gradient update、recurrent memory、probabilistic context inference，在不同 adaptation trajectories 數量下的 test-time return 與 meta-training sample efficiency。

這個問題很乾淨，而且完全符合你現在的階段。

你的第一階段不要碰 zero-shot。先把這三件事看懂：

1. **MAML：為什麼更新一次參數後會變好？**
2. **RL²：為什麼第二條 trajectory 會比第一條好？**
3. **PEARL：為什麼 context 多了之後 (z) 會更有用？**

等這個實驗跑通後，再把 (K=0) 拿出來單獨分析，才自然進入 zero-shot 問題。
