from collections.abc import Callable

import numpy as np


def initialize_weights(
    low_bound: float, high_bound: float, n: int, seed: int | None = None
) -> np.ndarray:
    range_to_draw = np.random.default_rng(seed)
    return range_to_draw.uniform(low_bound, high_bound, n)


def numerical_gradient(
    loss_fn: Callable[[np.ndarray], float], params: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    gradient = np.zeros_like(params, dtype=float)

    for i in range(len(params)):
        bumped_up = params.copy()
        bumped_up[i] += eps

        bumped_down = params.copy()
        bumped_down[i] -= eps

        gradient[i] = (loss_fn(bumped_up) - loss_fn(bumped_down)) / (2 * eps)

    return gradient


def gradient_descent_step(
    params: np.ndarray, gradient: np.ndarray, learning_rate: float
) -> np.ndarray:
    return params - learning_rate * gradient


def analytical_gradient(
    dataset: list[tuple],
    weights: np.ndarray,
    bias: float | None = None,
    activation: Callable[[np.ndarray], np.ndarray] | None = None,
    loss: str = "mse",
) -> tuple[np.ndarray, float]:
    x = np.array([np.atleast_1d(sample[0]) for sample in dataset], dtype=float)
    y = np.array([sample[1] for sample in dataset], dtype=float)
    n = len(y)

    logits = x @ weights
    if bias is not None:
        logits = logits + bias

    predictions = activation(logits) if activation is not None else logits
    error = predictions - y

    coefficient = 2 / n if loss == "mse" else 1 / n

    weight_gradient = coefficient * (x.T @ error)
    bias_gradient = coefficient * float(np.sum(error))

    return weight_gradient, bias_gradient
