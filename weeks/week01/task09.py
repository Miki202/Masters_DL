import numpy as np


def describe(
    array: np.ndarray,
    scalar: int,
    row: np.ndarray,
    column: np.ndarray,
    incompatible: np.ndarray,
) -> None:
    print(
        f"Matrix + scalar ({scalar}): {array + scalar} \n"
        + f"Matrix {array.shape} + row vector {row.shape}: {array + row} \n"
        + f"Matrix {array.shape} + column vector {column.shape}: {array + column} \n"
    )
    print(f"Matrix {array.shape} + incompatible {incompatible.shape} raises:")
    try:
        print(array + incompatible)
    except ValueError as error:
        print(error)


def main() -> int:
    two_d_array = np.array([[1, 2, 3], [4, 5, 6]])

    describe(
        array=two_d_array,
        scalar=10,
        row=np.array([10, 20, 30]),
        column=np.array([[100], [200]]),
        incompatible=np.array([1, 2]),
    )

    return 0
