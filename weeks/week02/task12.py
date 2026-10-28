import numpy as np
import torch

from dl_lib.activations import functional


def naive_softmax(x: np.ndarray) -> np.ndarray:
    exponentials = np.exp(x)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def describe(scores: np.ndarray, large_scores: np.ndarray) -> None:
    probabilities = functional.softmax(scores)
    print(f"scores: {scores}")
    print(f"softmax: {probabilities}")
    print(f"sum of probabilities: {probabilities.sum():.4f}\n")

    print(f"Large scores: {large_scores}")
    print(f"Naive softmax (unstable): {naive_softmax(large_scores)}")
    print(f"Stable softmax: {functional.softmax(large_scores)}")

    torch_probabilities = torch.softmax(torch.from_numpy(scores), dim=-1).numpy()
    print(f"\ntorch.softmax: {torch_probabilities}")


def main() -> int:
    scores = np.array([1.0, 2.0, 3.0])
    large_scores = np.array([1000.0, 1001.0, 1002.0])

    describe(scores=scores, large_scores=large_scores)

    return 0
