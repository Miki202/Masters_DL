import unittest

import numpy as np

from dl_lib.perceptron.square_net import SquareNet


class TestSquareNetForward(unittest.TestCase):
    def test_when_called_then_returns_a_scalar_float(self):
        # Arrange
        model = SquareNet(n_hidden=4, low=-1, high=1, seed=42)

        # Act
        actual = model.forward(np.array([2.0]))

        # Assert
        self.assertIsInstance(actual, float)

    def test_when_seeded_then_params_are_reproducible(self):
        # Arrange & Act
        first = SquareNet(n_hidden=4, low=-1, high=1, seed=42)
        second = SquareNet(n_hidden=4, low=-1, high=1, seed=42)

        # Assert
        np.testing.assert_array_equal(first.params(), second.params())


class TestSquareNetForwardBatch(unittest.TestCase):
    def test_when_called_then_matches_per_sample_forward(self):
        # Arrange
        model = SquareNet(n_hidden=4, low=-1, high=1, seed=42)
        x_values = np.array([-2.0, 0.0, 3.0])

        # Act
        batched = model.forward_batch(x_values)
        individual = np.array(
            [model.forward(np.array([x])) for x in x_values]
        )

        # Assert
        np.testing.assert_allclose(batched, individual)


class TestSquareNetParams(unittest.TestCase):
    def test_when_set_params_then_forward_uses_the_new_values(self):
        # Arrange
        model = SquareNet(n_hidden=4, low=-1, high=1, seed=42)
        new_params = np.zeros(4 * 3 + 1)

        # Act
        model.set_params(new_params)
        actual = model.forward(np.array([5.0]))

        # Assert: all-zero params -> sigmoid(0)=0.5 per hidden unit, zero output
        # weights and bias, so the network must output exactly 0.
        self.assertAlmostEqual(actual, 0.0)
