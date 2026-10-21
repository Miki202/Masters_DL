import numpy as np


def min_max_normalize(
    array: np.ndarray, range_a: float = 0.0, range_b: float = 1.0
) -> np.ndarray:
    minimum = array.min()
    maximum = array.max()
    if minimum == maximum:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum) * (range_b - range_a) + range_a


def standardize(array: np.ndarray) -> np.ndarray:
    mean = array.mean()
    standard_deviation = array.std()
    if standard_deviation == 0:
        return np.zeros_like(array)
    return (array - mean) / standard_deviation
