import matplotlib.pyplot as plt
import numpy as np

from dl_lib.losses import functional

FIXED_ERRORS = np.array([1.0, 0.5, -0.5])


def errors_to_loss(growing_error: float) -> tuple[float, float, float]:
    errors = np.concatenate([FIXED_ERRORS, [growing_error]])
    y_true = np.zeros_like(errors)
    return (
        functional.mae(y_true, errors),
        functional.mse(y_true, errors),
        functional.rmse(y_true, errors),
    )


def describe() -> None:
    for sample in (1.0, 5.0, 10.0):
        mean_absolute_error, mean_squared_error, root_mean_squared_error = (
            errors_to_loss(sample)
        )
        print(
            f"Growing error={sample}: MAE={mean_absolute_error:.4f}, "
            + f"MSE={mean_squared_error:.4f}, RMSE={root_mean_squared_error:.4f}"
        )


def plot(growing_errors: np.ndarray) -> None:
    losses = np.array([errors_to_loss(error) for error in growing_errors])

    plt.plot(growing_errors, losses[:, 0], label="MAE")
    plt.plot(growing_errors, losses[:, 1], label="MSE")
    plt.plot(growing_errors, losses[:, 2], label="RMSE")
    plt.title("MAE, MSE, and RMSE as one prediction error grows")
    plt.xlabel(
        "the growing error, with three other errors "
        + f"({FIXED_ERRORS[0]:g}, {FIXED_ERRORS[1]:g}, {FIXED_ERRORS[2]:g}) held fixed"
    )
    plt.ylabel("loss value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    growing_errors = np.linspace(0, 10, 101)

    describe()
    plot(growing_errors=growing_errors)

    return 0
