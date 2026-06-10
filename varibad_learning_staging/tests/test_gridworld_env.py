from random import Random
from unittest import TestCase

from varibad_gridworld.configs.gridworld import ENV_CONFIG
from varibad_gridworld.envs.gridworld import STAY, GridWorldTask, default_allowed_goals


def make_env(goal=None):
    env = GridWorldTask(
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
    )
    if goal is not None:
        env.goal = goal
        env.state = env.start_state
        env.step_count = 0
    return env


class EnvSpecTest(TestCase):
    def test_start_is_bottom_left(self) -> None:
        self.assertEqual(ENV_CONFIG["start_state"], (4, 0))

    def test_allowed_goals_are_top_three_rows(self) -> None:
        goals = default_allowed_goals(5)
        self.assertEqual(len(goals), 15)
        self.assertTrue(all(r in (0, 1, 2) for r, _ in goals))
        self.assertNotIn((4, 0), goals)  # start is never a goal

    def test_reward_based_on_next_state(self) -> None:
        env = make_env(goal=(2, 0))
        # start (4,0) -> up -> (3,0): not goal, -0.1
        _, reward, _, info = env.step(0)
        self.assertAlmostEqual(reward, -0.1)
        self.assertFalse(info["on_goal"])
        # (3,0) -> up -> (2,0): goal, +1
        _, reward, _, info = env.step(0)
        self.assertAlmostEqual(reward, 1.0)
        self.assertTrue(info["on_goal"])

    def test_goal_does_not_terminate(self) -> None:
        env = make_env(goal=(2, 0))
        env.step(0)  # (3,0)
        env.step(0)  # (2,0) goal
        # staying on goal keeps earning +1, episode not over (step 3 of 15)
        _, reward, episode_done, info = env.step(STAY)
        self.assertAlmostEqual(reward, 1.0)
        self.assertFalse(episode_done)

    def test_wall_keeps_in_place(self) -> None:
        env = make_env(goal=(0, 0))
        # from (4,0), down walks into bottom wall
        next_state, _, _, _ = env.step(2)
        self.assertEqual(next_state, (4, 0))

    def test_episode_done_every_H_steps(self) -> None:
        env = make_env(goal=(0, 0))
        dones = [env.step(STAY)[2] for _ in range(15)]
        self.assertEqual(dones[:14], [False] * 14)
        self.assertTrue(dones[14])  # done on the 15th step

    def test_episode_reset_is_not_task_reset(self) -> None:
        # spec 17.1: position resets every H steps, goal must NOT change.
        env = make_env(goal=(1, 2))
        goal_before = env.goal
        for _ in range(15):
            env.step(STAY)  # triggers one episode_done + auto reset
        self.assertEqual(env.goal, goal_before)
        self.assertEqual(env.state, env.start_state)  # position reset

    def test_reset_task_changes_goal_distribution(self) -> None:
        env = make_env()
        rng = Random(0)
        goals = set()
        for _ in range(100):
            env.reset_task(rng)
            goals.add(env.goal)
        self.assertTrue(goals.issubset(set(ENV_CONFIG["allowed_goals"])))
        self.assertGreater(len(goals), 1)  # actually varies

    def test_state_onehot(self) -> None:
        env = make_env(goal=(0, 0))
        vec = env.get_state_onehot()
        self.assertEqual(len(vec), 25)
        self.assertEqual(sum(vec), 1.0)
        # start (4,0) -> id = 4*5+0 = 20
        self.assertEqual(vec.index(1.0), 20)
