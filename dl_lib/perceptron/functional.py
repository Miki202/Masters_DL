import numpy as np


def initialize_weights(low_bound:float, high_bound: float, n: int, seed: int | None = None) -> np.ndarray:
    range_to_draw = np.random.default_rng(seed)
    return range_to_draw.uniform(low_bound, high_bound, n)
