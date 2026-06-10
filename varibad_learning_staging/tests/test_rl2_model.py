from unittest import TestCase

import torch

from varibad_gridworld.models.rl2 import (
    ACTION_DIM,
    HIDDEN_DIM,
    INPUT_DIM,
    STATE_DIM,
    RL2Policy,
    build_input,
)


class RL2ModelTest(TestCase):
    def test_input_dim_is_32(self) -> None:
        self.assertEqual(INPUT_DIM, STATE_DIM + ACTION_DIM + 1 + 1)
        self.assertEqual(INPUT_DIM, 32)

    def test_build_input_shape(self) -> None:
        x = build_input(
            torch.zeros(1, STATE_DIM),
            torch.zeros(1, ACTION_DIM),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
        self.assertEqual(tuple(x.shape), (1, INPUT_DIM))

    def test_step_shapes(self) -> None:
        model = RL2Policy()
        hidden = model.initial_hidden(batch_size=1)
        x_t = torch.zeros(1, INPUT_DIM)
        state = torch.zeros(1, STATE_DIM)
        logits, value, new_hidden = model.step(x_t, state, hidden)
        self.assertEqual(tuple(logits.shape), (1, ACTION_DIM))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(new_hidden.shape), (1, 1, HIDDEN_DIM))

    def test_hidden_state_actually_changes(self) -> None:
        # A non-zero input should move the hidden state (memory is alive).
        model = RL2Policy()
        hidden = model.initial_hidden(batch_size=1)
        x_t = torch.ones(1, INPUT_DIM)
        state = torch.zeros(1, STATE_DIM)
        _, _, new_hidden = model.step(x_t, state, hidden)
        self.assertFalse(torch.allclose(hidden, new_hidden))

    def test_forward_sequence_shapes(self) -> None:
        model = RL2Policy()
        T, B = 7, 3
        x_seq = torch.zeros(T, B, INPUT_DIM)
        state_seq = torch.zeros(T, B, STATE_DIM)
        logits, values, hidden = model(x_seq, state_seq)
        self.assertEqual(tuple(logits.shape), (T, B, ACTION_DIM))
        self.assertEqual(tuple(values.shape), (T, B, 1))
        self.assertEqual(tuple(hidden.shape), (1, B, HIDDEN_DIM))
