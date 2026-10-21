import matplotlib.pyplot as plt
import numpy as np


def describe(name: str, samples: np.ndarray) -> None:
    print(
        f"{name}: mean={samples.mean():.2f}, std={samples.std():.2f}, "
        + f"min={samples.min():.2f}, max={samples.max():.2f}"
    )


def plot(name: str, samples: np.ndarray, axes: plt.Axes) -> None:
    axes.hist(samples, bins=40)
    axes.set_title(name)
    axes.set_xlabel("value")
    axes.set_ylabel("count")
    axes.grid(True)


def main() -> int:
    distributions = {
        "Uniform": np.random.default_rng(42).uniform(size=10_000),
        "Normal": np.random.default_rng(42).standard_normal(10_000),
        "Exponential": np.random.default_rng(42).exponential(size=10_000),
    }

    figure, all_axes = plt.subplots(1, 3)
    for (name, samples), axes in zip(distributions.items(), all_axes):
        describe(name=name, samples=samples)
        plot(name=name, samples=samples, axes=axes)

    figure.tight_layout()
    plt.show()

    return 0
