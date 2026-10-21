import matplotlib.pyplot as plt
import numpy as np


def describe(sin_array: np.ndarray, samples: np.ndarray) -> None:
    print(
        f"sin(x) range: [{sin_array.min():.4f}, {sin_array.max():.4f}]\n"
        + f"Histogram sample count: {samples.size}"
    )


def plot(
    x_array: np.ndarray,
    sin_array: np.ndarray,
    cos_array: np.ndarray,
    samples: np.ndarray,
) -> None:
    figure, (line_axes, histogram_axes) = plt.subplots(1, 2)

    line_axes.plot(x_array, sin_array, label="sin(x)")
    line_axes.plot(x_array, cos_array, label="cos(x)")
    line_axes.set_title("Line plot")
    line_axes.set_xlabel("x")
    line_axes.set_ylabel("y")
    line_axes.legend()
    line_axes.grid(True)

    histogram_axes.hist(samples, bins=30)
    histogram_axes.set_title("Histogram")
    histogram_axes.set_xlabel("value")
    histogram_axes.set_ylabel("count")
    histogram_axes.grid(True)

    figure.tight_layout()
    plt.show()


def main() -> int:
    x_array = np.linspace(0, 2 * np.pi, 200)
    sin_array = np.sin(x_array)
    cos_array = np.cos(x_array)
    samples = np.random.default_rng(42).standard_normal(1_000)

    describe(sin_array=sin_array, samples=samples)
    plot(
        x_array=x_array,
        sin_array=sin_array,
        cos_array=cos_array,
        samples=samples,
    )

    return 0
