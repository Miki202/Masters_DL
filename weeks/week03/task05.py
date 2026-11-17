import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, functional

LEARNING_RATE = 0.001
EPOCHS = 500


def train(dataset: list[tuple[float, float]], w: np.ndarray) -> np.ndarray:
    x_values = np.array([x for x, _ in dataset])
    y_true = np.array([y for _, y in dataset])

    def loss_fn(params: np.ndarray) -> float:
        return losses_functional.mse(y_true, params[0] * x_values)

    for _ in range(EPOCHS):
        gradient = functional.numerical_gradient(loss_fn, w)
        w = functional.gradient_descent_step(w, gradient, LEARNING_RATE)

    return w, loss_fn(w)


def main() -> int:
    dataset = datasets.create_dataset(6)

    w_seeded = functional.initialize_weights(0, 10, 1, 42)
    print(f"Seed 42, initial w={w_seeded[0]:.6f}")
    final_w, final_loss = train(dataset, w_seeded.copy())
    print(f"After {EPOCHS} epochs: w={final_w[0]:.6f}, loss={final_loss:.6f} (target: w=2.0)")

    print("\nTraining again without a seed:")
    w_unseeded = functional.initialize_weights(0, 10, 1)
    print(f"No seed, initial w={w_unseeded[0]:.6f}")
    final_w, final_loss = train(dataset, w_unseeded.copy())
    print(f"After {EPOCHS} epochs: w={final_w[0]:.6f}, loss={final_loss:.6f} (target: w=2.0)")

    return 0
