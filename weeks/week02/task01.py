import numpy as np


def describe(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    mean_error = errors.mean()
    mean_absolute_error = absolute_errors.mean()
    mean_squared_error = squared_errors.mean()
    root_mean_squared_error = np.sqrt(mean_squared_error)

    print(
        f"y_true: {y_true}\n y_pred: {y_pred}\n errors: {errors}\n absolute errors: {absolute_errors}\n"
        + f"squared errors: {squared_errors}\n mean error: {mean_error:.4f}\n MAE (mean absolute error): {mean_absolute_error:.4f}\n"
        + f"MSE (mean squared error): {mean_squared_error:.4f}\n RMSE (root mean squared error): {root_mean_squared_error:.4f}"
    )


def main() -> int:
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    describe(y_true=y_true, y_pred=y_pred)

    return 0
