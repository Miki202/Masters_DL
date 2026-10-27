import numpy as np

from dl_lib.metrics import functional


def describe(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = functional.confusion_matrix(y_true, y_pred)
    print(
        f"y_true: {y_true}\n"
        + f"y_pred: {y_pred}\n\n"
        + f"True positives:  {matrix['tp']}\n"
        + f"True negatives:  {matrix['tn']}\n"
        + f"False positives: {matrix['fp']}\n"
        + f"False negatives: {matrix['fn']}"
    )


def main() -> int:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    describe(y_true=y_true, y_pred=y_pred)

    return 0
