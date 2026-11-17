import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import initialize_weights


class SquareNet:
    def __init__(
        self, n_hidden: int, low: float, high: float, seed: int | None = None
    ) -> None:
        self.n_hidden = n_hidden

        n_draws = n_hidden * 3 + 1
        drawn = initialize_weights(low, high, n_draws, seed)

        self.hidden_weights = drawn[0:n_hidden]
        self.hidden_biases = drawn[n_hidden : 2 * n_hidden]
        self.output_weights = drawn[2 * n_hidden : 3 * n_hidden]
        self.output_bias = float(drawn[3 * n_hidden])

    def forward(self, x: np.ndarray) -> float:
        hidden_logits = self.hidden_weights * x[0] + self.hidden_biases
        hidden_activations = sigmoid(hidden_logits)
        output = float(hidden_activations @ self.output_weights + self.output_bias)
        return output

    def forward_batch(self, x_values: np.ndarray) -> np.ndarray:
        hidden_logits = x_values[:, None] * self.hidden_weights[None, :] + self.hidden_biases[None, :]
        hidden_activations = sigmoid(hidden_logits)
        return hidden_activations @ self.output_weights + self.output_bias

    def params(self) -> np.ndarray:
        return np.concatenate(
            [self.hidden_weights, self.hidden_biases, self.output_weights, [self.output_bias]]
        )

    def set_params(self, params: np.ndarray) -> None:
        n = self.n_hidden
        self.hidden_weights = params[0:n]
        self.hidden_biases = params[n : 2 * n]
        self.output_weights = params[2 * n : 3 * n]
        self.output_bias = float(params[3 * n])
