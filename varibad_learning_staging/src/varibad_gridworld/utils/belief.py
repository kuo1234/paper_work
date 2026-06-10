from __future__ import annotations

from math import log2

Cell = tuple[int, int]


class GoalBelief:
    """Exact Bayesian posterior over the hidden goal (restricted to allowed cells).

    Task ``m`` = which cell is the goal, with a uniform prior over
    ``allowed_goals`` (spec section 3.2 -- the top three rows, NOT the whole
    grid). The observation model is deterministic: stepping on a non-goal cell
    rules it out, stepping on the goal identifies it. So the exact posterior stays
    uniform over the allowed cells not yet visited.

    This is the "correct answer" that a learned VariBAD encoder will later be
    compared against, and ``latent()`` is the fixed-length vector a neural policy
    would consume.
    """

    def __init__(self, allowed_goals: list[Cell]) -> None:
        self.allowed_goals = list(allowed_goals)
        self.ruled_out: set[Cell] = set()
        self.known_goal: Cell | None = None

    def candidate_cells(self) -> list[Cell]:
        """Allowed cells that could still be the goal."""

        if self.known_goal is not None:
            return [self.known_goal]
        return [c for c in self.allowed_goals if c not in self.ruled_out]

    def update(self, visited_cell: Cell, was_goal: bool) -> None:
        """Posterior update after observing the reward at ``visited_cell``."""

        if was_goal:
            self.known_goal = visited_cell
        elif visited_cell in self.allowed_goals:
            self.ruled_out.add(visited_cell)
        # visiting a non-allowed cell teaches nothing (it was never a candidate).

    def probs(self) -> dict[Cell, float]:
        if self.known_goal is not None:
            return {self.known_goal: 1.0}
        candidates = self.candidate_cells()
        if not candidates:
            return {}
        p = 1.0 / len(candidates)
        return {c: p for c in candidates}

    def entropy_bits(self) -> float:
        """Uncertainty over the goal, in bits. Uniform over N -> log2(N)."""

        return sum(-p * log2(p) for p in self.probs().values() if p > 0.0)

    def latent(self, grid_size: int) -> list[float]:
        """Fixed-length belief vector: flattened grid posterior + entropy.

        A future neural policy can consume this unchanged. Cells outside
        ``allowed_goals`` are simply 0 in the map.
        """

        probs = self.probs()
        flat = [probs.get((r, c), 0.0) for r in range(grid_size) for c in range(grid_size)]
        flat.append(self.entropy_bits())
        return flat

    def copy(self) -> "GoalBelief":
        b = GoalBelief(self.allowed_goals)
        b.ruled_out = set(self.ruled_out)
        b.known_goal = self.known_goal
        return b

'''
`belief.py` 定義的是 GridWorld 裡「agent 目前相信 goal 在哪裡」的精確 belief state，也就是不用神經網路學，直接用 Bayesian posterior 算出正確答案。

核心檔案：[`belief.py`](c:/Users/kuo/Desktop/paperwork/varibad_learning_staging/src/varibad_gridworld/utils/belief.py:8)

`GoalBelief` 的假設很簡單：

- hidden task `m` = goal 是哪一格。
- 一開始 goal 在 `allowed_goals` 裡均勻分布。
- 如果踩到某格但不是 goal，且那格本來可能是 goal，就把它排除。
- 如果踩到 goal，就直接知道答案。
- 所以 posterior 永遠是「剩下候選格子的均勻分布」，或「已知 goal 的 100% 分布」。

幾個方法的意思：

`__init__(allowed_goals)`

建立 belief：

```python
self.allowed_goals = list(allowed_goals)
self.ruled_out = set()
self.known_goal = None
```

也就是：所有合法 goal 都可能，還沒有排除任何格子，也還不知道真正 goal。

`candidate_cells()`

回傳目前還可能是 goal 的格子。

如果已經知道 goal：

```python
return [self.known_goal]
```

否則回傳 `allowed_goals` 中尚未被排除的格子。

`update(visited_cell, was_goal)`

這是 belief 更新的核心。rollout 裡每走一步會呼叫：

```python
belief.update(next_state, was_goal=info["on_goal"])
```

如果 `was_goal=True`，代表踩到了 goal：

```python
self.known_goal = visited_cell
```

如果不是 goal，但踩到的是合法候選格，就排除它：

```python
self.ruled_out.add(visited_cell)
```

如果踩到不在 `allowed_goals` 的格子，belief 不變，因為那格本來就不可能是 goal。

`probs()`

把目前 belief 轉成機率分布：

- 已知 goal：`{goal: 1.0}`
- 還不知道：所有 candidate 平分機率
- 沒有 candidate：回傳 `{}`，這理論上比較像保護性處理

例如如果 allowed goals 有 15 格，已經排除 3 格，剩 12 格，每格就是 `1/12`。

`entropy_bits()`

計算 belief 的不確定性，單位是 bits。

因為 posterior 是均勻分布，所以直覺上：

```text
候選格越多，entropy 越高
知道 goal 後，entropy = 0
```

例如：

- 16 個候選：`log2(16) = 4 bits`
- 4 個候選：`2 bits`
- 1 個候選：`0 bits`

`latent(grid_size)`

這個方法把 belief 轉成固定長度向量，方便未來給 neural policy 或 VariBAD encoder 比較。

它做兩件事：

1. 產生一個 flattened grid posterior map  
   每個 grid cell 對應 goal 在那裡的機率。
2. 最後 append entropy。

所以如果 `grid_size = 5`，輸出長度是：

```text
5 * 5 + 1 = 26
```

前 25 個是每格的 goal posterior，最後 1 個是 entropy。

`copy()`

複製一份 belief，避免模擬或 policy 評估時改到原本 belief。這對「假設我下一步走某格會得到多少資訊」這類 planning 很有用。

一句話總結：`belief.py` 是 GridWorld 裡的「正確 belief oracle」。它根據 agent 探索過哪些格子，精確維護 goal 的 posterior，並提供候選格、機率、entropy，以及神經網路可吃的 latent vector。'''