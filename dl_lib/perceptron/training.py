import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron.functional import gradient_descent_step, numerical_gradient
from dl_lib.perceptron.perceptron import Perceptron


def _params_from_model(model: Perceptron) -> np.ndarray:
    if model.use_bias:
        return np.concatenate([model.weights, [model.bias]])
    return model.weights.copy()


def _apply_params(model: Perceptron, params: np.ndarray) -> None:
    n_weights = len(model.weights)
    model.weights = params[:n_weights]
    if model.use_bias:
        model.bias = float(params[-1])


def _loss(model: Perceptron, dataset: list[tuple], loss: str) -> float:
    predictions = np.array(
        [model.forward(np.atleast_1d(x)) for x, _ in dataset]
    )
    targets = np.array([y for _, y in dataset])
    if loss == "bce":
        return losses_functional.binary_cross_entropy(targets, predictions)
    return losses_functional.mse(targets, predictions)


def train(
    model: Perceptron,
    dataset: list[tuple],
    learning_rate: float,
    epochs: int,
    record_every: int,
    loss: str = "mse",
) -> list[tuple[int, float]]:
    def loss_fn(params: np.ndarray) -> float:
        current_params = _params_from_model(model)
        _apply_params(model, params)
        current_loss = _loss(model, dataset, loss)
        _apply_params(model, current_params)
        return current_loss

    history = []
    for epoch in range(1, epochs + 1):
        params = _params_from_model(model)
        gradient = numerical_gradient(loss_fn, params)
        new_params = gradient_descent_step(params, gradient, learning_rate)
        _apply_params(model, new_params)

        if epoch % record_every == 0 or epoch == epochs:
            history.append((epoch, _loss(model, dataset, loss)))

    return history
