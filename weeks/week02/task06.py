import numpy as np

from dl_lib.metrics import functional


def describe(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = functional.confusion_matrix(y_true, y_pred)

    print(f"y_true: {y_true}\ny_pred: {y_pred}\n")
    print(f"True positives:  {matrix['tp']}")
    print(f"True negatives:  {matrix['tn']}")
    print(f"False positives: {matrix['fp']}")
    print(f"False negatives: {matrix['fn']}")


def main() -> int:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    describe(y_true=y_true, y_pred=y_pred)

    return 0
