import matplotlib.pyplot as plt
import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.perceptron import datasets, training
from dl_lib.perceptron.perceptron import Perceptron

LEARNING_RATE = 0.1
EPOCHS = 100_000


def describe(name: str, model: Perceptron, dataset: list[tuple], final_loss: float) -> None:
    print(f"{name} - final loss: {final_loss:.6f}")
    for x, y in dataset:
        prediction = model.forward(np.array(x))
        print(f"  AND({int(x[0])}, {int(x[1])}) = {int(y)} -> predicted {prediction:.6f}")


def plot(without_sigmoid_history: list[tuple], with_sigmoid_history: list[tuple]) -> None:
    epochs_a, losses_a = zip(*without_sigmoid_history)
    epochs_b, losses_b = zip(*with_sigmoid_history)

    plt.plot(epochs_a, losses_a, label="without sigmoid")
    plt.plot(epochs_b, losses_b, label="with sigmoid")
    plt.title("AND gate: loss with and without sigmoid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    dataset = datasets.and_dataset()

    model_no_activation = Perceptron(n_inputs=2, low=-1, high=1, seed=42, use_bias=True)
    history_no_activation = training.train(
        model_no_activation, dataset, LEARNING_RATE, EPOCHS, record_every=1
    )
    describe("Without sigmoid", model_no_activation, dataset, history_no_activation[-1][1])

    print()

    model_sigmoid = Perceptron(
        n_inputs=2,
        low=-1,
        high=1,
        seed=42,
        use_bias=True,
        activation=activations_functional.sigmoid,
    )
    history_sigmoid = training.train(
        model_sigmoid, dataset, LEARNING_RATE, EPOCHS, record_every=1
    )
    describe("With sigmoid   ", model_sigmoid, dataset, history_sigmoid[-1][1])

    plot(history_no_activation, history_sigmoid)

    return 0
