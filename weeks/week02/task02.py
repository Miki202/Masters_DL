import numpy as np
import torch

from dl_lib.losses import functional


def describe(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mean_absolute_error = functional.mae(y_true, y_pred)
    torch_mean_absolute_error = torch.nn.functional.l1_loss(
        torch.from_numpy(y_pred), torch.from_numpy(y_true)
    ).item()

    print(
        f"y_true: {y_true}\n"
        + f"y_pred: {y_pred}\n"
        + f"MAE: {mean_absolute_error:.4f}\n"
        + f"MAE (torch.nn.functional.l1_loss): {torch_mean_absolute_error:.4f}"
    )


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])

    describe(y_true=y_true, y_pred=y_pred)

    return 0
