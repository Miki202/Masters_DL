import numpy as np

from dl_lib.perceptron import datasets, training
from dl_lib.perceptron.perceptron import Perceptron

LEARNING_RATE = 0.1
EPOCHS = 100_000
RECORD_EVERY = 10_000


def run_gate(name: str, dataset: list[tuple]) -> None:
    model = Perceptron(n_inputs=2, low=-1, high=1, seed=42, use_bias=True)
    print(f"{name} gate - initial weights: {model.weights}, bias: {model.bias:.6f}")

    history = training.train(model, dataset, LEARNING_RATE, EPOCHS, RECORD_EVERY)
    for epoch, loss in history:
        print(f"  epoch {epoch:6d}: loss={loss:.6f}")

    print(f"{name} gate - final weights: {model.weights}, bias: {model.bias:.6f}")
    print(f"{name} gate - predictions:")
    for x, y in dataset:
        prediction = model.forward(np.array(x))
        print(f"  {name}({int(x[0])}, {int(x[1])}) = {int(y)} -> predicted {prediction:.6f}")


def main() -> int:
    run_gate("AND", datasets.and_dataset())
    print()
    run_gate("OR", datasets.or_dataset())

    return 0
