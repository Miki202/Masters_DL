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


class TestNumericalGradient(unittest.TestCase):
    def test_when_called_on_quadratic_then_matches_analytical_derivative(self):
        # Arrange: loss(w) = w^2, whose derivative is 2w
        def loss_fn(params: np.ndarray) -> float:
            return float(params[0] ** 2)

        params = np.array([3.0])
        expected = 6.0

        # Act
        actual = functional.numerical_gradient(loss_fn, params)[0]

        # Assert
        self.assertAlmostEqual(actual, expected, places=3)

    def test_when_called_then_does_not_mutate_params(self):
        # Arrange
        def loss_fn(params: np.ndarray) -> float:
            return float(np.sum(params**2))

        params = np.array([1.0, 2.0])
        original = params.copy()

        # Act
        functional.numerical_gradient(loss_fn, params)

        # Assert
        np.testing.assert_array_equal(params, original)


class TestGradientDescentStep(unittest.TestCase):
    def test_when_called_then_moves_against_the_gradient(self):
        # Arrange
        params = np.array([5.0])
        gradient = np.array([2.0])
        learning_rate = 0.1
        expected = np.array([4.8])

        # Act
        actual = functional.gradient_descent_step(params, gradient, learning_rate)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_learning_rate_is_zero_then_params_are_unchanged(self):
        # Arrange
        params = np.array([5.0, -3.0])
        gradient = np.array([2.0, 7.0])

        # Act
        actual = functional.gradient_descent_step(params, gradient, 0.0)

        # Assert
        np.testing.assert_array_equal(actual, params)
