import unittest

import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.perceptron.perceptron import Perceptron


class TestPerceptronForward(unittest.TestCase):
    def test_when_no_bias_or_activation_then_returns_weighted_sum(self):
        # Arrange
        model = Perceptron(n_inputs=2, low=-1, high=1, seed=42)
        model.weights = np.array([2.0, 3.0])
        expected = 2.0 * 1.0 + 3.0 * 2.0

        # Act
        actual = model.forward(np.array([1.0, 2.0]))

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_using_bias_then_adds_bias_to_weighted_sum(self):
        # Arrange
        model = Perceptron(n_inputs=2, low=-1, high=1, seed=42, use_bias=True)
        model.weights = np.array([1.0, 1.0])
        model.bias = 5.0
        expected = 1.0 + 1.0 + 5.0

        # Act
        actual = model.forward(np.array([1.0, 1.0]))

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_activation_given_then_applies_it_to_the_logit(self):
        # Arrange
        model = Perceptron(
            n_inputs=2,
            low=-1,
            high=1,
            seed=42,
            use_bias=True,
            activation=activations_functional.sigmoid,
        )
        model.weights = np.array([0.0, 0.0])
        model.bias = 0.0

        # Act
        actual = model.forward(np.array([1.0, 1.0]))

        # Assert
        self.assertAlmostEqual(actual, 0.5)

    def test_when_use_bias_is_false_then_bias_is_none(self):
        # Arrange & Act
        model = Perceptron(n_inputs=2, low=-1, high=1, seed=42)

        # Assert
        self.assertIsNone(model.bias)

    def test_when_seeded_then_weights_are_reproducible(self):
        # Arrange & Act
        first = Perceptron(n_inputs=2, low=-1, high=1, seed=42)
        second = Perceptron(n_inputs=2, low=-1, high=1, seed=42)

        # Assert
        np.testing.assert_array_equal(first.weights, second.weights)
