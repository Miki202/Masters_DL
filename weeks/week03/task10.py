import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.perceptron import datasets, training
from dl_lib.perceptron.perceptron import Perceptron

LEARNING_RATE = 0.1
EPOCHS = 100_000
RECORD_EVERY = 10_000


def main() -> int:
    dataset = datasets.nand_dataset()
    model = Perceptron(
        n_inputs=2,
        low=-1,
        high=1,
        seed=42,
        use_bias=True,
        activation=activations_functional.sigmoid,
    )

    history = training.train(
        model, dataset, LEARNING_RATE, EPOCHS, RECORD_EVERY, loss="bce"
    )
    for epoch, loss in history:
        print(f"  epoch {epoch:6d}: loss={loss:.6f}")

    print(f"NAND gate - final weights: {model.weights}, bias: {model.bias:.6f}")
    print("NAND gate - predictions:")
    for x, y in dataset:
        prediction = model.forward(np.array(x))
        print(f"  NAND({int(x[0])}, {int(x[1])}) = {int(y)} -> predicted {prediction:.6f}")

    return 0
