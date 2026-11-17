import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import datasets, functional
from dl_lib.perceptron.perceptron import Perceptron


def main() -> int:
    dataset = datasets.and_dataset()
    model = Perceptron(
        n_inputs=2,
        low=-1,
        high=1,
        seed=42,
        use_bias=True,
        activation=activations_functional.sigmoid,
    )
    print(
        f"AND gate (sigmoid) - weights: {model.weights}, bias: {model.bias:.6f}"
    )

    def bce_loss(params: np.ndarray) -> float:
        weights, bias = params[:-1], params[-1]
        predictions = np.array(
            [activations_functional.sigmoid(np.array([x @ weights + bias]))[0] for x, _ in dataset]
        )
        targets = np.array([y for _, y in dataset])
        return losses_functional.binary_cross_entropy(targets, predictions)

    params = np.concatenate([model.weights, [model.bias]])
    numerical = functional.numerical_gradient(bce_loss, params)
    numerical_weight_gradient, numerical_bias_gradient = numerical[:-1], numerical[-1]

    analytical_weight_gradient, analytical_bias_gradient = functional.analytical_gradient(
        dataset,
        model.weights,
        bias=model.bias,
        activation=activations_functional.sigmoid,
        loss="bce",
    )

    print(f"Numerical weight gradient:  {numerical_weight_gradient}")
    print(f"Analytical weight gradient: {analytical_weight_gradient}")
    print(
        "Weight gradient difference: "
        + f"{np.abs(numerical_weight_gradient - analytical_weight_gradient)}"
    )
    print(f"Numerical bias gradient:  {numerical_bias_gradient:.6f}")
    print(f"Analytical bias gradient: {analytical_bias_gradient:.6f}")
    print(
        "Bias gradient difference: "
        + f"{abs(numerical_bias_gradient - analytical_bias_gradient):.8f}"
    )

    return 0
