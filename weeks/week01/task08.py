import time

import numpy as np


def main() -> int:
    one_d_array = np.random.rand(1_000_000)
    start_time_loop = time.perf_counter()
    loop_total = 0.0
    for value in one_d_array:
        loop_total += float(value)
    loop_seconds = time.perf_counter() - start_time_loop

    start_time_vectorized = time.perf_counter()
    vectorized_total = float(one_d_array.sum())
    vectorized_seconds = time.perf_counter() - start_time_vectorized

    print(f"Array length: {one_d_array.size:,}")
    print(
        f"Loop sum: {loop_total:.4f} ({loop_seconds:.4f}s) \n"
        + f"Vectorized sum: {vectorized_total:.4f} ({vectorized_seconds:.4f}s) \n"
        + f"Vectorized speedup: {loop_seconds / vectorized_seconds:.0f}x \n"
        + f"Vectorized mean: {one_d_array.mean():.4f} \n"
        + f"Vectorized std: {one_d_array.std():.4f} \n"
        + f"Vectorized min/max: {one_d_array.min():.4f} / {one_d_array.max():.4f}"
    )
    return 0
