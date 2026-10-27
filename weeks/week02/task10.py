import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def describe(x: np.ndarray) -> None:
    for value in x:
        print(f"sigmoid({value:g}) = {functional.sigmoid(np.array([value]))[0]:.6f}")

    full_range = np.linspace(-10, 10, 1001)
    outputs = functional.sigmoid(full_range)
    print(f"Output range: [{outputs.min():.6f}, {outputs.max():.6f}]")

    torch_outputs = torch.sigmoid(torch.from_numpy(x)).numpy()
    print(f"torch.sigmoid: {torch_outputs}")


def plot(full_range: np.ndarray) -> None:
    plt.plot(full_range, functional.sigmoid(full_range))
    plt.title("Sigmoid activation")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> int:
    x = np.array([-10.0, 0.0, 10.0])

    describe(x)
    plot(np.linspace(-10, 10, 1001))

    return 0
