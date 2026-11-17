import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, functional


def main() -> int:
    dataset = datasets.create_dataset(6)
    x_values = np.array([x for x, _ in dataset])
    y_true = np.array([y for _, y in dataset])

    w = functional.initialize_weights(0, 10, 1, 42)

    def loss_fn(params: np.ndarray) -> float:
        return losses_functional.mse(y_true, params[0] * x_values)

    loss = loss_fn(w)
    numerical_gradient = functional.numerical_gradient(loss_fn, w)[0]
    analytical_gradient, _ = functional.analytical_gradient(dataset, w)

    print(f"Loss: {loss:.6f}")
    print(f"Numerical gradient:  {numerical_gradient:.6f}")
    print(f"Analytical gradient: {analytical_gradient[0]:.6f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient[0]):.8f}")

    return 0
