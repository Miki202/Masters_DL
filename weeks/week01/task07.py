import numpy as np


def describe(
    array: np.ndarray,
    row: int,
    column: int,
    rows: slice,
    columns: slice,
    filter_value: int,
) -> None:
    mask = array > filter_value
    print(
        f"Matrix: {array} \n {row} row: {array[row - 1]} \n {column} column: {array[:, column - 1]} \n"
        + f"Submatrix: {array[rows, columns]} \n Boolean mask for more than {filter_value}: {mask} \n"
        + f"Values satisfying the mask: {array[mask]}"
    )


def main() -> int:
    two_d_array = np.arange(1, 13).reshape(3, 4)
    describe(
        array=two_d_array,
        row=2,
        column=3,
        rows=slice(0, 2),
        columns=slice(1, 3),
        filter_value=6,
    )
    return 0
