import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics import functional


def describe(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    precisions = []
    recalls = []
    for threshold in thresholds:
        predictions = functional.apply_threshold(probabilities, threshold)
        precisions.append(functional.precision(y_true, predictions))
        recalls.append(functional.recall(y_true, predictions))
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    for sample in (0.2, 0.5, 0.8):
        index = np.argmin(np.abs(thresholds - sample))
        print(
            f"threshold={thresholds[index]:.1f}: "
            + f"precision={precisions[index]:.4f}, recall={recalls[index]:.4f}"
        )

    return precisions, recalls


def plot(thresholds: np.ndarray, precisions: np.ndarray, recalls: np.ndarray) -> None:
    plt.plot(thresholds, precisions, marker="o", label="precision")
    plt.plot(thresholds, recalls, marker="o", label="recall")
    plt.title("Precision and recall as the classification threshold rises")
    plt.xlabel("threshold")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    rng = np.random.default_rng(42)

    positive_probabilities = rng.uniform(0.2, 0.75, size=20)
    negative_probabilities = rng.uniform(0.0, 0.75, size=80)

    y_true = np.concatenate([np.ones(20, dtype=int), np.zeros(80, dtype=int)])
    probabilities = np.concatenate([positive_probabilities, negative_probabilities])

    thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    precisions, recalls = describe(
        y_true=y_true, probabilities=probabilities, thresholds=thresholds
    )
    plot(thresholds=thresholds, precisions=precisions, recalls=recalls)

    return 0
