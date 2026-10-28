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


class TestAndDataset(unittest.TestCase):
    def test_when_called_then_matches_the_and_truth_table(self):
        # Arrange
        expected = [
            ((0.0, 0.0), 0.0),
            ((0.0, 1.0), 0.0),
            ((1.0, 0.0), 0.0),
            ((1.0, 1.0), 1.0),
        ]

        # Act
        actual = datasets.and_dataset()

        # Assert
        self.assertEqual(actual, expected)


class TestOrDataset(unittest.TestCase):
    def test_when_called_then_matches_the_or_truth_table(self):
        # Arrange
        expected = [
            ((0.0, 0.0), 0.0),
            ((0.0, 1.0), 1.0),
            ((1.0, 0.0), 1.0),
            ((1.0, 1.0), 1.0),
        ]

        # Act
        actual = datasets.or_dataset()

        # Assert
        self.assertEqual(actual, expected)


class TestNandDataset(unittest.TestCase):
    def test_when_called_then_matches_the_nand_truth_table(self):
        # Arrange
        expected = [
            ((0.0, 0.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((1.0, 0.0), 1.0),
            ((1.0, 1.0), 0.0),
        ]

        # Act
        actual = datasets.nand_dataset()

        # Assert
        self.assertEqual(actual, expected)
