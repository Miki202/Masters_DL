import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def describe(full_range: np.ndarray, sample_points: np.ndarray) -> None:
    sigmoid_range = functional.sigmoid(full_range)
    relu_range = functional.relu(full_range)
    tanh_range = functional.tanh(full_range)

    print(f"sigmoid range: [{sigmoid_range.min():.4f}, {sigmoid_range.max():.4f}]")
    print(f"relu range:    [{relu_range.min():.4f}, {relu_range.max():.4f}]")
    print(f"tanh range:    [{tanh_range.min():.4f}, {tanh_range.max():.4f}]")
    print()

    for x in sample_points:
        sigmoid_value = functional.sigmoid(np.array([x]))[0]
        relu_value = functional.relu(np.array([x]))[0]
        tanh_value = functional.tanh(np.array([x]))[0]
        print(
            f"At x={x:g}: sigmoid={sigmoid_value:.4f}, "
            + f"relu={relu_value:.4f}, tanh={tanh_value:.4f}"
        )

    torch_x = torch.from_numpy(sample_points)
    print(f"\ntorch.relu: {torch.relu(torch_x).numpy()}")
    print(f"torch.tanh: {torch.tanh(torch_x).numpy()}")


def plot(full_range: np.ndarray) -> None:
    plt.plot(full_range, functional.sigmoid(full_range), label="sigmoid")
    plt.plot(full_range, functional.relu(full_range), label="relu")
    plt.plot(full_range, functional.tanh(full_range), label="tanh")
    plt.title("Comparing sigmoid, relu, and tanh")
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    full_range = np.linspace(-10, 10, 1001)
    sample_points = np.array([-5.0, 0.0, 5.0])

    describe(full_range=full_range, sample_points=sample_points)
    plot(full_range=full_range)

    return 0
