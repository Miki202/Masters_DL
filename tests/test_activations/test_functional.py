import unittest

import numpy as np
import torch

from dl_lib.activations import functional


class TestSigmoid(unittest.TestCase):
    def test_when_called_then_matches_hand_calculation(self):
        # Arrange
        x = np.array([-10.0, 0.0, 10.0])
        expected = np.array([0.0000454, 0.5, 0.9999546])

        # Act
        actual = functional.sigmoid(x)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_when_called_then_matches_torch_sigmoid(self):
        # Arrange
        x = np.array([-10.0, 0.0, 10.0])
        expected = torch.sigmoid(torch.from_numpy(x)).numpy()

        # Act
        actual = functional.sigmoid(x)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_when_called_then_output_is_within_open_unit_interval(self):
        # Arrange
        x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

        # Act
        actual = functional.sigmoid(x)

        # Assert
        self.assertTrue(np.all(actual > 0.0))
        self.assertTrue(np.all(actual < 1.0))


class TestRelu(unittest.TestCase):
    def test_when_called_then_matches_hand_calculation(self):
        # Arrange
        x = np.array([-5.0, 0.0, 5.0])
        expected = np.array([0.0, 0.0, 5.0])

        # Act
        actual = functional.relu(x)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_called_then_matches_torch_relu(self):
        # Arrange
        x = np.array([-5.0, 0.0, 5.0])
        expected = torch.relu(torch.from_numpy(x)).numpy()

        # Act
        actual = functional.relu(x)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_called_then_negative_values_are_zeroed(self):
        # Arrange
        x = np.array([-100.0, -1.0, -0.001])

        # Act
        actual = functional.relu(x)

        # Assert
        np.testing.assert_allclose(actual, np.zeros_like(x))


class TestTanh(unittest.TestCase):
    def test_when_called_then_matches_hand_calculation(self):
        # Arrange
        x = np.array([-5.0, 0.0, 5.0])
        expected = np.array([-0.9999092, 0.0, 0.9999092])

        # Act
        actual = functional.tanh(x)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_when_called_then_matches_torch_tanh(self):
        # Arrange
        x = np.array([-5.0, 0.0, 5.0])
        expected = torch.tanh(torch.from_numpy(x)).numpy()

        # Act
        actual = functional.tanh(x)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_when_called_then_output_is_within_open_interval(self):
        # Arrange
        x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

        # Act
        actual = functional.tanh(x)

        # Assert
        self.assertTrue(np.all(actual > -1.0))
        self.assertTrue(np.all(actual < 1.0))


class TestSoftmax(unittest.TestCase):
    def test_when_called_then_probabilities_sum_to_one(self):
        # Arrange
        scores = np.array([1.0, 2.0, 3.0])

        # Act
        actual = functional.softmax(scores)

        # Assert
        self.assertAlmostEqual(float(np.sum(actual)), 1.0, places=6)

    def test_when_called_then_matches_torch_softmax(self):
        # Arrange
        scores = np.array([1.0, 2.0, 3.0])
        expected = (
            torch.softmax(torch.from_numpy(scores), dim=-1).numpy()
        )

        # Act
        actual = functional.softmax(scores)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_when_scores_are_large_then_stable_version_does_not_overflow(self):
        # Arrange
        small_scores = np.array([1.0, 2.0, 3.0])
        large_scores = np.array([1000.0, 1001.0, 1002.0])

        # Act
        actual_small = functional.softmax(small_scores)
        actual_large = functional.softmax(large_scores)

        # Assert
        self.assertFalse(np.any(np.isnan(actual_large)))
        np.testing.assert_allclose(actual_large, actual_small, atol=1e-6)
