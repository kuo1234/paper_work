from random import Random
from unittest import TestCase

from varibad_gridworld.baselines.bayes_like_search import BayesLikeSearchPolicy
from varibad_gridworld.baselines.oracle_policy import OraclePolicy
from varibad_gridworld.baselines.posterior_sampling import PosteriorSamplingPolicy
from varibad_gridworld.baselines.random_policy import RandomPolicy
from varibad_gridworld.configs.gridworld import ENV_CONFIG
from varibad_gridworld.envs.gridworld import GridWorldTask
from varibad_gridworld.utils.belief import GoalBelief
from varibad_gridworld.utils.metrics import aggregate
from varibad_gridworld.utils.rollout import run_bamdp_rollout


def make_env():
    return GridWorldTask(
        allowed_goals=ENV_CONFIG["allowed_goals"],
        start_state=ENV_CONFIG["start_state"],
        episode_horizon=ENV_CONFIG["episode_horizon"],
    )


def run_many(policy_cls, n_tasks, N, seed):
    results = []
    act_rng = Random(seed + 17)
    for k in range(n_tasks):
        task_rng = Random(seed * 100003 + k)
        results.append(run_bamdp_rollout(make_env(), policy_cls(), N=N, rng=act_rng, task_rng=task_rng))
    return aggregate(results)


class BeliefTest(TestCase):
    def test_candidates_restricted_to_allowed_goals(self) -> None:
        belief = GoalBelief(ENV_CONFIG["allowed_goals"])
        self.assertEqual(len(belief.candidate_cells()), 15)
        # a bottom-row cell is never a candidate
        self.assertNotIn((4, 0), belief.candidate_cells())

    def test_ruling_out_only_allowed_cells(self) -> None:
        belief = GoalBelief(ENV_CONFIG["allowed_goals"])
        belief.update((4, 1), was_goal=False)  # not an allowed goal -> no effect
        self.assertEqual(len(belief.candidate_cells()), 15)
        belief.update((0, 0), was_goal=False)  # allowed -> ruled out
        self.assertEqual(len(belief.candidate_cells()), 14)

    def test_found_goal_collapses(self) -> None:
        belief = GoalBelief(ENV_CONFIG["allowed_goals"])
        belief.update((1, 2), was_goal=True)
        self.assertEqual(belief.candidate_cells(), [(1, 2)])
        self.assertEqual(belief.entropy_bits(), 0.0)


class RolloutTest(TestCase):
    def test_horizon_bookkeeping(self) -> None:
        result = run_bamdp_rollout(make_env(), BayesLikeSearchPolicy(), N=4,
                                   rng=Random(0), task_rng=Random(0))
        self.assertEqual(result.H_plus, 60)
        self.assertEqual(len(result.steps), 60)

    def test_belief_persists_across_episodes(self) -> None:
        result = run_bamdp_rollout(make_env(), BayesLikeSearchPolicy(), N=4,
                                   rng=Random(0), task_rng=Random(3))
        returns = result.episode_returns()
        self.assertLess(returns[0], returns[-1])  # later episodes exploit

    def test_policy_does_not_see_true_goal(self) -> None:
        # Non-oracle policies must rely only on belief, never on env.goal.
        for cls in (RandomPolicy, PosteriorSamplingPolicy, BayesLikeSearchPolicy):
            policy = cls()
            self.assertFalse(hasattr(policy, "_goal"),
                             f"{cls.__name__} appears to store the true goal")


class BaselineOrderingTest(TestCase):
    def test_expected_ordering(self) -> None:
        n, N, seed = 120, 4, 7
        oracle = run_many(OraclePolicy, n, N, seed)
        bayes = run_many(BayesLikeSearchPolicy, n, N, seed)
        posterior = run_many(PosteriorSamplingPolicy, n, N, seed)
        random = run_many(RandomPolicy, n, N, seed)

        self.assertGreaterEqual(oracle.avg_return, bayes.avg_return)
        self.assertGreater(bayes.avg_return, posterior.avg_return)
        self.assertGreater(posterior.avg_return, random.avg_return)

    def test_bayes_search_fewer_redundant_than_posterior(self) -> None:
        # spec 12.3: systematic search wastes far fewer steps than posterior sampling.
        n, N, seed = 120, 4, 7
        bayes = run_many(BayesLikeSearchPolicy, n, N, seed)
        posterior = run_many(PosteriorSamplingPolicy, n, N, seed)
        self.assertLess(bayes.avg_redundant_visits, posterior.avg_redundant_visits)

    def test_oracle_high_success(self) -> None:
        oracle = run_many(OraclePolicy, 120, 4, 7)
        self.assertGreater(oracle.success_rate_rollout, 0.99)
