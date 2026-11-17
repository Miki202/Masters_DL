import numpy as np

from dl_lib.metrics import functional


def describe(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("Dataset: 990 negative, 10 positive")
    print("Model: always predicts negative, never looks at the input\n")
    print(f"accuracy:  {functional.accuracy(y_true, y_pred):.4f}")
    print(f"precision: {functional.precision(y_true, y_pred):.4f}")
    print(f"recall:    {functional.recall(y_true, y_pred):.4f}")
    print(f"f1:        {functional.f1_score(y_true, y_pred):.4f}")


def main() -> int:
    y_true = np.array([0] * 990 + [1] * 10)
    y_pred = np.zeros_like(y_true)

    describe(y_true=y_true, y_pred=y_pred)

    return 0
