from unittest import TestCase

from varibad_gridworld.utils.belief import GoalBelief
from varibad_gridworld.utils.posterior_viz import entropy_sparkline, heatmap


class PosteriorVizTest(TestCase):
    def test_entropy_sparkline_length_and_ascii(self) -> None:
        line = entropy_sparkline([4.0, 2.0, 1.0, 0.0], max_bits=4.0)
        self.assertEqual(len(line), 4)
        # all characters must be plain ASCII (console-safe)
        self.assertTrue(all(ord(ch) < 128 for ch in line))

    def test_entropy_sparkline_high_to_low(self) -> None:
        levels = "_.,-:=*#"
        line = entropy_sparkline([4.0, 0.0], max_bits=4.0)
        # first char should be the highest level, last the lowest
        self.assertEqual(line[0], levels[-1])
        self.assertEqual(line[-1], levels[0])

    def test_heatmap_dimensions(self) -> None:
        belief = GoalBelief([(r, c) for r in range(3) for c in range(5)])
        hm = heatmap(belief, grid_size=5)
        lines = hm.splitlines()
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(len(line.split()), 5)

    def test_heatmap_marks_collapsed_goal(self) -> None:
        belief = GoalBelief([(r, c) for r in range(3) for c in range(5)])
        belief.update((1, 2), was_goal=True)
        hm = heatmap(belief, grid_size=5, goal=(1, 2))
        self.assertIn("G", hm)
