import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import functional
from dl_lib.perceptron.square_net import SquareNet

N_HIDDEN = 12
LEARNING_RATE = 0.01
EPOCHS = 80_000
RECORD_EVERY = 10_000


def mse_loss(model: SquareNet, params: np.ndarray, x_values: np.ndarray, y_true: np.ndarray) -> float:
    current_params = model.params()
    model.set_params(params)
    predictions = model.forward_batch(x_values)
    loss = losses_functional.mse(y_true, predictions)
    model.set_params(current_params)
    return loss


def train(model: SquareNet, x_values: np.ndarray, y_true: np.ndarray) -> None:
    print("Training:")
    for epoch in range(1, EPOCHS + 1):
        params = model.params()
        gradient = functional.numerical_gradient(
            lambda p: mse_loss(model, p, x_values, y_true), params
        )
        new_params = functional.gradient_descent_step(params, gradient, LEARNING_RATE)
        model.set_params(new_params)

        if epoch % RECORD_EVERY == 0:
            loss = mse_loss(model, model.params(), x_values, y_true)
            print(f"  epoch {epoch:6d}: loss={loss:.6f}")


def main() -> int:
    x_values = np.arange(-8, 9, dtype=float)
    y_true = x_values**2

    model = SquareNet(n_hidden=N_HIDDEN, low=-1, high=1, seed=42)
    train(model, x_values, y_true)

    print("\nPredictions:")
    for x in (-8, -7, -2, 0, 2, 7, 8):
        prediction = model.forward(np.array([float(x)]))
        print(f"  model({x}) = {prediction:.3f} (x^2 = {x**2})")

    return 0
