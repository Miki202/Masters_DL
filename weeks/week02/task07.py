import numpy as np

from dl_lib.metrics import functional


def describe(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"{name}:")
    print(f"  accuracy:  {functional.accuracy(y_true, y_pred):.4f}")
    print(f"  precision: {functional.precision(y_true, y_pred):.4f}")
    print(f"  recall:    {functional.recall(y_true, y_pred):.4f}")
    print(f"  f1:        {functional.f1_score(y_true, y_pred):.4f}")


def build_balanced_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y_true = rng.integers(0, 2, size=5)
    y_pred = rng.integers(0, 2, size=5)
    return y_true, y_pred


def build_imbalanced_dataset(
    rng: np.random.Generator, missed_positives: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0] * 990 + [1] * 10)
    rng.shuffle(y_true)

    y_pred = y_true.copy()
    positive_indices = np.flatnonzero(y_true == 1)
    missed_indices = rng.choice(
        positive_indices, size=missed_positives, replace=False
    )
    y_pred[missed_indices] = 0

    return y_true, y_pred


def main() -> int:
    balanced_y_true, balanced_y_pred = build_balanced_dataset(
        np.random.default_rng(seed=14)
    )
    describe("Balanced dataset", balanced_y_true, balanced_y_pred)

    print()

    imbalanced_y_true, imbalanced_y_pred = build_imbalanced_dataset(
        np.random.default_rng(seed=0)
    )
    describe(
        "Imbalanced dataset (990 negative, 10 positive)",
        imbalanced_y_true,
        imbalanced_y_pred,
    )

    return 0
