import unittest

import numpy as np

from dl_lib.perceptron.xor import Xor, forward


class TestXorForward(unittest.TestCase):
    def test_when_called_then_returns_a_value_in_the_unit_interval(self):
        # Arrange
        model = Xor(low=-1, high=1, seed=42)

        # Act
        actual = model.forward(np.array([1.0, 0.0]))

        # Assert
        self.assertGreater(actual, 0.0)
        self.assertLess(actual, 1.0)

    def test_when_seeded_then_params_are_reproducible(self):
        # Arrange & Act
        first = Xor(low=-1, high=1, seed=42)
        second = Xor(low=-1, high=1, seed=42)

        # Assert
        np.testing.assert_array_equal(first.params(), second.params())


class TestModuleLevelForward(unittest.TestCase):
    def test_when_called_then_matches_model_forward_with_packed_inputs(self):
        # Arrange
        model = Xor(low=-1, high=1, seed=42)
        expected = model.forward(np.array([1.0, 0.0]))

        # Act
        actual = forward(model, 1.0, 0.0)

        # Assert
        self.assertAlmostEqual(actual, expected)


class TestXorParams(unittest.TestCase):
    def test_when_set_params_then_forward_uses_the_new_values(self):
        # Arrange
        model = Xor(low=-1, high=1, seed=42)
        new_params = np.zeros(9)

        # Act
        model.set_params(new_params)
        actual = model.forward(np.array([1.0, 1.0]))

        # Assert
        self.assertAlmostEqual(actual, 0.5)
