import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))