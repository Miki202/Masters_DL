from collections.abc import Callable

import numpy as np

from dl_lib.perceptron.functional import initialize_weights


class Perceptron:
    def __init__(
        self,
        n_inputs: int,
        low: float,
        high: float,
        seed: int | None = None,
        use_bias: bool = False,
        activation: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.use_bias = use_bias
        self.activation = activation

        n_draws = n_inputs + (1 if use_bias else 0)
        drawn = initialize_weights(low, high, n_draws, seed)

        self.weights = drawn[:n_inputs]
        self.bias = drawn[-1] if use_bias else None

    def forward(self, x: np.ndarray) -> float:
        z = float(np.dot(x, self.weights))

        if self.use_bias:
            z += self.bias

        if self.activation is not None:
            z = float(self.activation(np.array([z]))[0])

        return z
