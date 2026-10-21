import numpy as np


def describe(label: str, array: np.ndarray) -> None:
    print(
        f"{label}: shape={array.shape}, dtype={array.dtype}, "
        f"ndim={array.ndim}, size={array.size}"
    )
    print(array)


def main() -> int:
    vector = np.array([1, 2, 3, 4, 5])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    volume = np.arange(24).reshape(2, 3, 4)

    describe("1D array (vector)", vector)
    print()
    describe("2D array (matrix)", matrix)
    print()
    describe("3D array (volume)", volume)

    return 0
