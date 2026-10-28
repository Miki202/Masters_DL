import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics import functional


def build_dataset(
    rng: np.random.Generator, n_per_class: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    negative_probabilities = rng.beta(a=2, b=5, size=n_per_class)
    positive_probabilities = rng.beta(a=5, b=2, size=n_per_class)

    y_true = np.array([0] * n_per_class + [1] * n_per_class)
    probabilities = np.concatenate([negative_probabilities, positive_probabilities])

    return y_true, probabilities


def sweep_thresholds(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    precisions = np.empty_like(thresholds)
    recalls = np.empty_like(thresholds)

    for i, threshold in enumerate(thresholds):
        y_pred = functional.apply_threshold(probabilities, threshold)
        precisions[i] = functional.precision(y_true, y_pred)
        recalls[i] = functional.recall(y_true, y_pred)

    return precisions, recalls


def describe(
    thresholds: np.ndarray, precisions: np.ndarray, recalls: np.ndarray
) -> None:
    for sample in (0.2, 0.5, 0.8):
        index = np.argmin(np.abs(thresholds - sample))
        print(
            f"threshold={thresholds[index]:.1f}: "
            + f"precision={precisions[index]:.4f}, recall={recalls[index]:.4f}"
        )


def plot(
    thresholds: np.ndarray, precisions: np.ndarray, recalls: np.ndarray
) -> None:
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.title("Precision and recall as the threshold changes")
    plt.xlabel("threshold")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    rng = np.random.default_rng(seed=0)
    y_true, probabilities = build_dataset(rng)

    thresholds = np.linspace(0, 1, 101)
    precisions, recalls = sweep_thresholds(y_true, probabilities, thresholds)

    describe(thresholds=thresholds, precisions=precisions, recalls=recalls)
    plot(thresholds=thresholds, precisions=precisions, recalls=recalls)

    return 0
