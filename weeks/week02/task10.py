import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def describe(x_range: np.ndarray) -> None:
    sigmoid_range = functional.sigmoid(x_range)
    torch_sigmoid = torch.sigmoid(torch.from_numpy(x_range)).numpy()

    print(
        f"sigmoid(-10) = {functional.sigmoid(np.array([-10.0]))[0]:.6f}\n"
        + f"sigmoid(0)   = {functional.sigmoid(np.array([0.0]))[0]:.6f}\n"
        + f"sigmoid(10)  = {functional.sigmoid(np.array([10.0]))[0]:.6f}\n"
        + f"Output range: [{sigmoid_range.min():.6f}, {sigmoid_range.max():.6f}]\n"
        + f"torch.sigmoid range: [{torch_sigmoid.min():.6f}, {torch_sigmoid.max():.6f}]"
    )


def plot(x_range: np.ndarray, sigmoid_range: np.ndarray) -> None:
    plt.plot(x_range, sigmoid_range)
    plt.title("sigmoid(x)")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    x_range = np.linspace(-10, 10, 200)

    describe(x_range=x_range)
    plot(x_range=x_range, sigmoid_range=functional.sigmoid(x_range))

    return 0
