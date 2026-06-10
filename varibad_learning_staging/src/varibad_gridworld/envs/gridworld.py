from __future__ import annotations

from random import Random

Cell = tuple[int, int]

# 5 actions (spec section 3.4). (d_row, d_col). row 0 = top, so "up" decreases row.
ACTIONS: tuple[Cell, ...] = (
    (-1, 0),  # 0 up
    (0, 1),   # 1 right
    (1, 0),   # 2 down
    (0, -1),  # 3 left
    (0, 0),   # 4 stay
)
ACTION_NAMES: tuple[str, ...] = ("up", "right", "down", "left", "stay")
STAY = 4
N_ACTIONS = len(ACTIONS)


def default_allowed_goals(grid_size: int = 5) -> list[Cell]:
    """Top three rows (spec section 3.2): 15 cells for a 5x5 grid."""

    return [(r, c) for r in range(3) for c in range(grid_size)]


class GridWorldTask:
    """A 5x5 GridWorld BAMDP task (spec sections 3-4).

    One *task* = one hidden goal, fixed for the whole BAMDP rollout. The agent
    only learns where the goal is by stepping on it. Within a rollout the
    position resets every H steps (``reset_episode``) but the goal does NOT change
    -- the agent is meant to use earlier episodes to find the goal and later ones
    to exploit it.

    API mirrors spec section 4:
        reset_task / reset_episode / step / get_state_onehot
    """

    def __init__(
        self,
        grid_size: int = 5,
        goal: Cell | None = None,
        allowed_goals: list[Cell] | None = None,
        start_state: Cell = (4, 0),
        episode_horizon: int = 15,
        reward_goal: float = 1.0,
        reward_non_goal: float = -0.1,
    ) -> None:
        self.grid_size = grid_size
        self.allowed_goals = allowed_goals if allowed_goals is not None else default_allowed_goals(grid_size)
        self.start_state = start_state
        self.H = episode_horizon
        self.reward_goal = reward_goal
        self.reward_non_goal = reward_non_goal

        self.goal = goal
        self.state = start_state
        self.step_count = 0  # global step within the BAMDP rollout

    # --- resets ----------------------------------------------------------- #

    def reset_task(self, rng: Random) -> Cell:
        """Sample a NEW goal and reset everything. Use between BAMDP rollouts."""

        self.goal = self.allowed_goals[rng.randrange(len(self.allowed_goals))]
        self.state = self.start_state
        self.step_count = 0
        return self.state

    def reset_episode(self) -> Cell:
        """Reset POSITION only -- the goal is unchanged (spec section 17.1).

        This is the single most important invariant of the BAMDP: episode reset
        is NOT task reset.
        """

        self.state = self.start_state
        return self.state

    # --- dynamics --------------------------------------------------------- #

    def transition(self, state: Cell, action: int) -> Cell:
        d_row, d_col = ACTIONS[action]
        candidate = (state[0] + d_row, state[1] + d_col)
        if 0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size:
            return candidate
        return state  # walked into a wall -> stay put

    def step(self, action: int) -> tuple[Cell, float, bool, dict]:
        """Apply an action.

        Returns ``(next_state, reward, episode_done, info)``. Reward is based on
        the NEXT state (spec section 3.5). The episode ends every H steps and the
        position is auto-reset; the goal is never changed here.
        """

        if action not in range(N_ACTIONS):
            raise ValueError(f"action must be in 0..{N_ACTIONS - 1}")

        next_state = self.transition(self.state, action)
        on_goal = next_state == self.goal
        reward = self.reward_goal if on_goal else self.reward_non_goal

        self.state = next_state
        self.step_count += 1
        episode_done = (self.step_count % self.H == 0)

        info = {
            "goal": self.goal,           # for reward calc / eval / viz ONLY
            "state": self.state,
            "on_goal": on_goal,
            "step_in_episode": self.step_count % self.H,
            "global_step": self.step_count,
        }

        if episode_done:
            self.reset_episode()

        return next_state, reward, episode_done, info

    # --- observation ------------------------------------------------------ #

    def state_id(self, state: Cell | None = None) -> int:
        s = state if state is not None else self.state
        return s[0] * self.grid_size + s[1]

    def get_state_onehot(self, state: Cell | None = None) -> list[float]:
        """One-hot state vector of length grid_size**2 (spec section 3.3)."""

        n = self.grid_size * self.grid_size
        vec = [0.0] * n
        vec[self.state_id(state)] = 1.0
        return vec
