import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def describe(x_range: np.ndarray) -> None:
    sigmoid_range = functional.sigmoid(x_range)
    relu_range = functional.relu(x_range)
    tanh_range = functional.tanh(x_range)

    sample_points = np.array([-5.0, 0.0, 5.0])
    sigmoid_samples = functional.sigmoid(sample_points)
    relu_samples = functional.relu(sample_points)
    tanh_samples = functional.tanh(sample_points)

    torch_relu_samples = torch.relu(torch.from_numpy(sample_points)).numpy()
    torch_tanh_samples = torch.tanh(torch.from_numpy(sample_points)).numpy()

    print(
        f"sigmoid range: [{sigmoid_range.min():.4f}, {sigmoid_range.max():.4f}]\n"
        + f"relu range:    [{relu_range.min():.4f}, {relu_range.max():.4f}]\n"
        + f"tanh range:    [{tanh_range.min():.4f}, {tanh_range.max():.4f}]\n\n"
        + f"At x=-5: sigmoid={sigmoid_samples[0]:.4f}, relu={relu_samples[0]:.4f}, "
        + f"tanh={tanh_samples[0]:.4f}\n"
        + f"At x=0:  sigmoid={sigmoid_samples[1]:.4f}, relu={relu_samples[1]:.4f}, "
        + f"tanh={tanh_samples[1]:.4f}\n"
        + f"At x=5:  sigmoid={sigmoid_samples[2]:.4f}, relu={relu_samples[2]:.4f}, "
        + f"tanh={tanh_samples[2]:.4f}\n\n"
        + f"torch.relu at [-5, 0, 5]: {torch_relu_samples}\n"
        + f"torch.tanh at [-5, 0, 5]: {torch_tanh_samples}"
    )


def plot(
    x_range: np.ndarray,
    sigmoid_range: np.ndarray,
    relu_range: np.ndarray,
    tanh_range: np.ndarray,
) -> None:
    plt.plot(x_range, sigmoid_range, label="sigmoid")
    plt.plot(x_range, relu_range, label="relu")
    plt.plot(x_range, tanh_range, label="tanh")
    plt.title("sigmoid, relu, and tanh")
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    x_range = np.linspace(-10, 10, 200)

    describe(x_range=x_range)
    plot(
        x_range=x_range,
        sigmoid_range=functional.sigmoid(x_range),
        relu_range=functional.relu(x_range),
        tanh_range=functional.tanh(x_range),
    )

    return 0
