import unittest

from dl_lib.perceptron import datasets


class TestCreateDataset(unittest.TestCase):
    def test_when_called_then_returns_doubling_pairs_from_zero(self):
        # Arrange
        n = 4
        expected = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]

        # Act
        actual = datasets.create_dataset(n)

        # Assert
        self.assertEqual(actual, expected)

    def test_when_called_then_returns_n_pairs(self):
        # Arrange
        n = 6

        # Act
        actual = datasets.create_dataset(n)

        # Assert
        self.assertEqual(len(actual), n)
