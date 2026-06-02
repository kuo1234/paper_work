from random import Random
from unittest import TestCase

from varibad_toy.grid_belief import GridBeliefEncoder
from varibad_toy.gridworld import HiddenGoalGrid


class GridWorldTest(TestCase):
    def test_wall_keeps_agent_in_place(self) -> None:
        env = HiddenGoalGrid(size=3, goal_cell=(2, 2))
        # action 0 = up; from (0,0) it walks into the top wall.
        next_state, reward, done = env.step((0, 0), 0)
        self.assertEqual(next_state, (0, 0))
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_stepping_on_goal_ends_episode(self) -> None:
        env = HiddenGoalGrid(size=3, goal_cell=(0, 1))
        next_state, reward, done = env.step((0, 0), 3)  # right
        self.assertEqual(next_state, (0, 1))
        self.assertEqual(reward, 1)
        self.assertTrue(done)

    def test_sample_task_excludes_start(self) -> None:
        rng = Random(0)
        for _ in range(50):
            env = HiddenGoalGrid.sample_task(rng, size=4, start_cell=(0, 0))
            self.assertNotEqual(env.goal_cell, (0, 0))


class GridBeliefTest(TestCase):
    def test_start_cell_ruled_out_at_init(self) -> None:
        belief = GridBeliefEncoder(size=3, start_cell=(0, 0))
        self.assertNotIn((0, 0), belief.candidate_cells())
        self.assertEqual(len(belief.candidate_cells()), 8)  # 9 cells - start

    def test_ruling_out_cells_lowers_entropy(self) -> None:
        belief = GridBeliefEncoder(size=3, start_cell=(0, 0))
        before = belief.entropy_bits()
        belief.update((0, 1), was_goal=False)
        after = belief.entropy_bits()
        self.assertLess(after, before)

    def test_finding_goal_collapses_belief(self) -> None:
        belief = GridBeliefEncoder(size=3, start_cell=(0, 0))
        belief.update((1, 1), was_goal=True)
        self.assertEqual(belief.candidate_cells(), [(1, 1)])
        self.assertEqual(belief.entropy_bits(), 0.0)
        self.assertEqual(belief.probs(), {(1, 1): 1.0})

    def test_uniform_posterior_sums_to_one(self) -> None:
        belief = GridBeliefEncoder(size=4, start_cell=(0, 0))
        total = sum(belief.probs().values())
        self.assertAlmostEqual(total, 1.0)

    def test_latent_has_fixed_length(self) -> None:
        belief = GridBeliefEncoder(size=5, start_cell=(0, 0))
        latent = belief.latent()
        self.assertEqual(len(latent), 5 * 5 + 1)  # map + entropy
