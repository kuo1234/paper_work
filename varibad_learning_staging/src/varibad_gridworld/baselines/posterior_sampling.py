from __future__ import annotations

from random import Random

from ..envs.gridworld import GridWorldTask
from ..utils.belief import GoalBelief
from ._common import action_toward

Cell = tuple[int, int]


class PosteriorSamplingPolicy:
    """Thompson / posterior sampling (spec section 5.3).

    Sample one guessed goal from the posterior and head straight to it. If the
    guess turns out wrong (ruled out by the belief), draw a new one. A fresh guess
    is also drawn at the start of each episode. Once the true goal is known,
    exploit it.

    Because the guess is random, it often commits to a FAR candidate when a near
    one was equally likely -- so it takes detours and revisits, making it less
    step-efficient than systematic search.
    """

    name = "Posterior Sampling"

    def __init__(self) -> None:
        self._guess: Cell | None = None
        self._grid_size = 5

    def reset_task(self, env: GridWorldTask) -> None:
        self._guess = None
        self._grid_size = env.grid_size

    def reset_episode(self) -> None:
        self._guess = None  # draw a new sample for the new episode

    def act(self, state: Cell, belief: GoalBelief, rng: Random) -> int:
        if belief.known_goal is not None:
            return action_toward(state, belief.known_goal, self._grid_size, rng)

        candidates = belief.candidate_cells()
        if not candidates:
            return action_toward(state, state, self._grid_size, rng)
        if self._guess is None or self._guess not in candidates:
            self._guess = candidates[rng.randrange(len(candidates))]
        return action_toward(state, self._guess, self._grid_size, rng)

"""它每次不是照一條固定路徑掃，而是：
python



candidates = belief.candidate_cells()
if self._guess is None or self._guess not in candidates:
    self._guess = candidates[rng.randrange(len(candidates))]
return action_toward(state, self._guess, ...)

意思是：從目前 posterior 裡隨機抽一個可能的 goal，先假裝它就是真 goal，然後一路往那格走。
如果抽到的 guess 錯了，踩上去後 reward 不是 +1，belief 會排除它；下一次再抽新的 guess。episode reset 時也會清掉 _guess，下一個 episode 重新抽。
它為什麼會繞遠路？因為 posterior 一開始對所有 allowed goals 是均勻的，遠近都一樣可能。從 (4,0) 出發時，Thompson sampling 可能第一抽就抽到 (0,4)。那它會一路往很遠的右上角走，只為了驗證一個候選點。如果那不是 goal，它才重新抽。下一次又可能抽到另一個很遠的點。
相比之下，Bayes-like Search 會從 (2,0) 開始，然後 (2,1), (2,2) 一路掃過去。它的資訊取得比較「路徑連續」，每一步通常都靠近下一個有用候選；Posterior Sampling 則可能在候選集合裡跳來跳去，導致 travel cost 很高。
最短版差異：
text



Bayes-like Search:
  選下一個空間上連續、未排除的候選點
  -> 探索路徑穩定、少繞路

Posterior Sampling:
  隨機抽一個未排除候選點並 committed 走過去
  -> 抽到遠點就長途跋涉，錯了再重抽

兩者最後都用同一個 helper：`action_toward` (line 13)。所以真正差別不是「怎麼走到目標」，而是「探索時選哪個候選 goal 當目標」。Bayes-like 的代表性就在這裡：它把 Bayesian belief 的候選集合變成一條有效率的 search path。"""