import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import initialize_weights


class Xor:
    def __init__(self, low: float, high: float, seed: int | None = None) -> None:
        drawn = initialize_weights(low, high, 9, seed)

        self.hidden_weights = drawn[0:4].reshape(2, 2)
        self.hidden_biases = drawn[4:6]
        self.output_weights = drawn[6:8]
        self.output_bias = drawn[8]

    def forward(self, x: np.ndarray) -> float:
        hidden_logits = self.hidden_weights @ x + self.hidden_biases
        hidden_activations = sigmoid(hidden_logits)

        output_logit = float(hidden_activations @ self.output_weights + self.output_bias)
        return float(sigmoid(np.array([output_logit]))[0])

    def params(self) -> np.ndarray:
        return np.concatenate(
            [
                self.hidden_weights.flatten(),
                self.hidden_biases,
                self.output_weights,
                [self.output_bias],
            ]
        )

    def set_params(self, params: np.ndarray) -> None:
        self.hidden_weights = params[0:4].reshape(2, 2)
        self.hidden_biases = params[4:6]
        self.output_weights = params[6:8]
        self.output_bias = float(params[8])


def forward(model: Xor, x1: float, x2: float) -> float:
    return model.forward(np.array([x1, x2]))
