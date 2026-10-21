import numpy as np
import torch

from dl_lib.losses import functional


def describe(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_outlier: np.ndarray
) -> None:
    mean_squared_error = functional.mse(y_true, y_pred)
    torch_mean_squared_error = torch.nn.functional.mse_loss(
        torch.from_numpy(y_pred), torch.from_numpy(y_true)
    ).item()

    mean_absolute_error_outlier = functional.mae(y_true, y_pred_outlier)
    mean_squared_error_outlier = functional.mse(y_true, y_pred_outlier)

    print(
        f"y_true: {y_true}\n"
        + f"y_pred: {y_pred}\n"
        + f"MSE: {mean_squared_error:.4f}\n"
        + f"MSE (torch.nn.functional.mse_loss): {torch_mean_squared_error:.4f}\n\n"
        + f"y_pred with one extreme outlier: {y_pred_outlier}\n"
        + f"MAE with outlier: {mean_absolute_error_outlier:.4f}\n"
        + f"MSE with outlier: {mean_squared_error_outlier:.4f}"
    )


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])
    y_pred_outlier = y_pred.copy()
    y_pred_outlier[-1] = 100.0

    describe(y_true=y_true, y_pred=y_pred, y_pred_outlier=y_pred_outlier)

    return 0
