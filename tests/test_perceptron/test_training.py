import unittest

from dl_lib.activations import functional as activations_functional
from dl_lib.perceptron import datasets, training
from dl_lib.perceptron.perceptron import Perceptron


class TestTrain(unittest.TestCase):
    def test_when_called_then_loss_decreases(self):
        # Arrange
        model = Perceptron(n_inputs=2, low=-1, high=1, seed=42)
        dataset = datasets.and_dataset()

        # Act
        history = training.train(
            model, dataset, learning_rate=0.1, epochs=200, record_every=50
        )

        # Assert
        losses = [loss for _, loss in history]
        self.assertLessEqual(losses[-1], losses[0])

    def test_when_called_then_returns_one_entry_per_recorded_epoch(self):
        # Arrange
        model = Perceptron(n_inputs=2, low=-1, high=1, seed=42)
        dataset = datasets.and_dataset()

        # Act
        history = training.train(
            model, dataset, learning_rate=0.1, epochs=100, record_every=25
        )

        # Assert
        self.assertEqual([epoch for epoch, _ in history], [25, 50, 75, 100])

    def test_when_loss_is_bce_then_loss_decreases(self):
        # Arrange
        model = Perceptron(
            n_inputs=2,
            low=-1,
            high=1,
            seed=42,
            use_bias=True,
            activation=activations_functional.sigmoid,
        )
        dataset = datasets.nand_dataset()

        # Act
        history = training.train(
            model, dataset, learning_rate=0.1, epochs=200, record_every=50, loss="bce"
        )

        # Assert
        losses = [loss for _, loss in history]
        self.assertLessEqual(losses[-1], losses[0])
