import numpy as np


def describe(first_array: np.ndarray, second_array: np.ndarray) -> None:
    print(
        f"A:\n{first_array}\n\n"
        + f"B:\n{second_array}\n\n"
        + f"Elementwise:\n{first_array * second_array}\n\n"
        + f"Matrix multiplication:\n{first_array @ second_array}\n\n"
        + f"The transpose of A is:\n{first_array.T}"
    )


def main() -> int:
    first_array = np.array([[1, 2], [3, 4]])
    second_array = np.array([[5, 6], [7, 8]])
    describe(first_array=first_array, second_array=second_array)

    return 0
