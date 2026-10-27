import numpy as np
import torch

from dl_lib.activations import functional


def naive_softmax(x: np.ndarray) -> np.ndarray:
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def describe(scores: np.ndarray, large_scores: np.ndarray) -> None:
    probabilities = functional.softmax(scores)
    naive_probabilities = naive_softmax(large_scores)
    stable_probabilities = functional.softmax(large_scores)
    torch_probabilities = torch.softmax(torch.from_numpy(scores), dim=-1).numpy()

    print(
        f"scores: {scores}\n"
        + f"softmax: {probabilities}\n"
        + f"sum of probabilities: {probabilities.sum():.4f}\n"
        + f"torch.softmax: {torch_probabilities}\n\n"
        + f"Large scores: {large_scores}\n"
        + f"Naive softmax (unstable): {naive_probabilities}\n"
        + f"Stable softmax: {stable_probabilities}"
    )


def main() -> int:
    scores = np.array([1.0, 2.0, 3.0])
    large_scores = np.array([1000.0, 1001.0, 1002.0])

    describe(scores=scores, large_scores=large_scores)

    return 0
