import numpy as np

from dl_lib.preprocessing import functional


def describe(
    array: np.ndarray,
    normalized: np.ndarray,
    standardized: np.ndarray,
) -> None:
    print(
        f"Original:     {array}\n"
        + f"Min-max:      {normalized}\n"
        + f"Standardized: {standardized}\n\n"
        + f"Min-max range: [{normalized.min():.4f}, {normalized.max():.4f}]\n"
        + f"Standardized mean/std: {standardized.mean():.4f} / {standardized.std():.4f}"
    )


def main() -> int:
    array = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

    describe(
        array=array,
        normalized=functional.min_max_normalize(array, range_a=0.0, range_b=1.0),
        standardized=functional.standardize(array),
    )

    return 0
