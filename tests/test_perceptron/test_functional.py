import unittest

import numpy as np

from dl_lib.perceptron import functional


class TestInitializeWeights(unittest.TestCase):
    def test_when_called_then_returns_n_values(self):
        # Arrange
        n = 3

        # Act
        actual = functional.initialize_weights(0, 1, n, seed=42)

        # Assert
        self.assertEqual(len(actual), n)

    def test_when_called_then_values_are_within_bounds(self):
        # Arrange
        low, high = -5.0, 5.0

        # Act
        actual = functional.initialize_weights(low, high, 100, seed=42)

        # Assert
        self.assertTrue(np.all(actual >= low))
        self.assertTrue(np.all(actual < high))

    def test_when_seeded_then_draws_are_reproducible(self):
        # Arrange & Act
        first = functional.initialize_weights(0, 10, 2, seed=42)
        second = functional.initialize_weights(0, 10, 2, seed=42)

        # Assert
        np.testing.assert_array_equal(first, second)
