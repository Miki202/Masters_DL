import numpy as np

from dl_lib.losses import functional


def describe_binary(y_true: np.ndarray, p: np.ndarray) -> None:
    loss = functional.binary_cross_entropy(y_true, p)
    confidently_right = functional.binary_cross_entropy(
        np.array([1.0]), np.array([0.999])
    )
    confidently_wrong = functional.binary_cross_entropy(
        np.array([1.0]), np.array([0.001])
    )

    print(
        f"Binary cross-entropy on y_true={y_true}, p={p}: {loss:.4f}\n\n"
        + f"y_true=1, p=0.999 (confidently right): loss={confidently_right:.4f}\n"
        + f"y_true=1, p=0.001 (confidently wrong): loss={confidently_wrong:.4f}"
    )


def describe_categorical(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    loss = functional.categorical_cross_entropy(y_true, probabilities)
    confidently_right = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.999, 0.0005, 0.0005]])
    )
    confidently_wrong = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.001, 0.4995, 0.4995]])
    )

    print(
        f"Multiclass cross-entropy on y_true={y_true}: {loss:.4f}\n\n"
        + f"y_true=0, p_true=0.999 (confidently right): loss={confidently_right:.4f}\n"
        + f"y_true=0, p_true=0.001 (confidently wrong): loss={confidently_wrong:.4f}"
    )


def main() -> int:
    y_true = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.2, 0.6])
    describe_binary(y_true=y_true, p=p)
    print()

    y_true_multiclass = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    describe_categorical(y_true=y_true_multiclass, probabilities=probabilities)

    return 0
