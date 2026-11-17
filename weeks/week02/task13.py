import numpy as np

from dl_lib.losses import functional


def describe_binary() -> None:
    y_true = np.array([1.0, 0.0])
    p = np.array([0.9, 0.1])
    loss = functional.binary_cross_entropy(y_true, p)
    print(f"Binary cross entropy for y_true={y_true}, p={p}: {loss:.4f}\n")

    confidently_right = functional.binary_cross_entropy(
        np.array([1.0]), np.array([0.999])
    )
    confidently_wrong = functional.binary_cross_entropy(
        np.array([1.0]), np.array([0.001])
    )
    print(
        f"Confidently right (p=0.999): {confidently_right:.4f}\n"
        + f"Confidently wrong (p=0.001): {confidently_wrong:.4f}\n"
    )


def describe_categorical() -> None:
    y_true = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    loss = functional.categorical_cross_entropy(y_true, probabilities)
    print(
        f"Categorical cross entropy for y_true={y_true}:\n{probabilities}\n"
        + f"loss: {loss:.4f}\n"
    )

    confidently_right = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.999, 0.0005, 0.0005]])
    )
    confidently_wrong = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.001, 0.4995, 0.4995]])
    )
    print(
        f"Confidently right (p=0.999 on true class): {confidently_right:.4f}\n"
        + f"Confidently wrong (p=0.001 on true class): {confidently_wrong:.4f}"
    )


def main() -> int:
    describe_binary()
    describe_categorical()

    return 0
