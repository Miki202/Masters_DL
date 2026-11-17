import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, functional

NUDGES = [0.002, 0.001, -0.001, -0.002]


def mse_for_weight(dataset: list[tuple[float, float]], weight: float) -> float:
    x_values = np.array([x for x, _ in dataset])
    y_true = np.array([y for _, y in dataset])
    y_pred = weight * x_values
    return losses_functional.mse(y_true, y_pred)


def describe(dataset: list[tuple[float, float]], weight: float) -> None:
    print(
        f"Dataset: {dataset}\n"
        + f"w: {weight}\n"
        + f"MSE: {mse_for_weight(dataset, weight)}\n"
    )
    print("Perturbing w:")
    for nudge in NUDGES:
        new_weight = weight + nudge
        new_mse = mse_for_weight(dataset, new_weight)
        sign = "+" if nudge >= 0 else ""
        print(f"  w + ({sign}{nudge}) -> w={new_weight:.6f}, MSE={new_mse:.6f}")


def main() -> int:
    dataset = datasets.create_dataset(6)
    weight = functional.initialize_weights(0, 10, 1, 42)[0]
    describe(dataset, weight)
    return 0
