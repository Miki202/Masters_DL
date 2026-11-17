import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, functional

LEARNING_RATE = 0.001
EPOCHS = 10


def mse_for_weight(
    x_values: np.ndarray, y_true: np.ndarray, params: np.ndarray
) -> float:
    y_pred = params[0] * x_values
    return losses_functional.mse(y_true, y_pred)


def main() -> int:
    dataset = datasets.create_dataset(6)
    x_values = np.array([x for x, _ in dataset])
    y_true = np.array([y for _, y in dataset])

    w = functional.initialize_weights(0, 10, 1, 42)

    def loss_fn(params: np.ndarray) -> float:
        return mse_for_weight(x_values, y_true, params)

    loss_before = loss_fn(w)
    gradient = functional.numerical_gradient(loss_fn, w)

    w_no_learning_rate = functional.gradient_descent_step(w, gradient, 1.0)
    w_scaled = functional.gradient_descent_step(w, gradient, LEARNING_RATE)

    print(f"Loss before update: {loss_before:.6f}")
    print(f"Numerical gradient: {gradient[0]:.6f}")
    print(
        f"w after subtracting the raw gradient: w={w_no_learning_rate[0]:.6f}, "
        + f"loss={loss_fn(w_no_learning_rate):.6f}"
    )
    print(
        f"w after one step with learning_rate={LEARNING_RATE}: "
        + f"w={w_scaled[0]:.6f}, loss={loss_fn(w_scaled):.6f}"
    )

    print(f"\nLoss before update: {loss_before:.6f}")
    for epoch in range(1, EPOCHS + 1):
        gradient = functional.numerical_gradient(loss_fn, w)
        w = functional.gradient_descent_step(w, gradient, LEARNING_RATE)
        print(f"epoch {epoch:2d}: w={w[0]:.6f}, loss={loss_fn(w):.6f}")

    return 0
