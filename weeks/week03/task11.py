import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, functional
from dl_lib.perceptron.xor import Xor, forward

LEARNING_RATE = 0.5
EPOCHS = 10_000
RECORD_EVERY = 2_000


def bce_loss(model: Xor, params: np.ndarray, dataset: list[tuple]) -> float:
    current_params = model.params()
    model.set_params(params)

    predictions = np.array([model.forward(np.array(x)) for x, _ in dataset])
    targets = np.array([y for _, y in dataset])
    loss = losses_functional.binary_cross_entropy(targets, predictions)

    model.set_params(current_params)
    return loss


def train(model: Xor, dataset: list[tuple]) -> None:
    for epoch in range(1, EPOCHS + 1):
        params = model.params()
        gradient = functional.numerical_gradient(
            lambda p: bce_loss(model, p, dataset), params
        )
        new_params = functional.gradient_descent_step(params, gradient, LEARNING_RATE)
        model.set_params(new_params)

        if epoch % RECORD_EVERY == 0:
            print(f"  epoch {epoch:6d}: loss={bce_loss(model, model.params(), dataset):.6f}")


def describe(model: Xor, dataset: list[tuple]) -> None:
    for x, _ in dataset:
        prediction = forward(model, x[0], x[1])
        print(f"  XOR({int(x[0])}, {int(x[1])}) = {prediction:.6f}")


def main() -> int:
    dataset = datasets.xor_dataset()

    print("First attempt - seed=42:")
    model = Xor(low=-1, high=1, seed=42)
    train(model, dataset)
    describe(model, dataset)

    print("\nSecond attempt - seed=1 (a different random start):")
    model = Xor(low=-1, high=1, seed=1)
    train(model, dataset)
    describe(model, dataset)

    return 0
