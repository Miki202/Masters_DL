import matplotlib.pyplot as plt
import numpy as np


def describe(
    x_array: np.ndarray,
    y_array: np.ndarray,
    slope: float,
    intercept: float,
) -> None:
    print(
        f"Generated {x_array.size} points for y = {slope}*x + {intercept} + noise\n"
        + f"x range: [{x_array.min():.2f}, {x_array.max():.2f}]\n"
        + f"y range: [{y_array.min():.2f}, {y_array.max():.2f}]"
    )


def plot(x_array: np.ndarray, y_array: np.ndarray) -> None:
    plt.scatter(x_array, y_array)
    plt.title("Synthetic linear data with noise")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def main() -> int:
    slope = 3.0
    intercept = 1.0

    x_array = np.linspace(0, 10, 50)
    noise = np.random.default_rng(42).standard_normal(x_array.size)
    y_array = slope * x_array + intercept + noise

    describe(
        x_array=x_array,
        y_array=y_array,
        slope=slope,
        intercept=intercept,
    )
    plot(x_array=x_array, y_array=y_array)

    return 0
