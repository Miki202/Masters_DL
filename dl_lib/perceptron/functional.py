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
