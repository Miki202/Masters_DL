import numpy as np

from dl_lib.metrics import functional


def describe(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(
        f"{name}:\n"
        + f"  accuracy:  {functional.accuracy(y_true, y_pred):.4f}\n"
        + f"  precision: {functional.precision(y_true, y_pred):.4f}\n"
        + f"  recall:    {functional.recall(y_true, y_pred):.4f}\n"
        + f"  f1:        {functional.f1_score(y_true, y_pred):.4f}"
    )


def main() -> int:
    rng = np.random.default_rng(42)

    y_true_balanced = np.array([1, 0, 1, 0, 1])
    y_pred_balanced = np.array([1, 0, 0, 1, 1])

    y_true_imbalanced = np.array([0] * 990 + [1] * 10)
    y_pred_imbalanced = y_true_imbalanced.copy()
    positive_indices = np.flatnonzero(y_true_imbalanced)
    missed_positives = rng.choice(positive_indices, size=3, replace=False)
    y_pred_imbalanced[missed_positives] = 0

    describe(name="Balanced dataset", y_true=y_true_balanced, y_pred=y_pred_balanced)
    print()
    describe(
        name="Imbalanced dataset (990 negative, 10 positive)",
        y_true=y_true_imbalanced,
        y_pred=y_pred_imbalanced,
    )

    return 0
