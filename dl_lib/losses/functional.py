import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def binary_cross_entropy(
    y_true: np.ndarray, p: np.ndarray, epsilon: float = 1e-12
) -> float:
    clipped_p = np.clip(p, epsilon, 1 - epsilon)
    losses = -(y_true * np.log(clipped_p) + (1 - y_true) * np.log(1 - clipped_p))
    return float(np.mean(losses))


def categorical_cross_entropy(
    y_true: np.ndarray, probabilities: np.ndarray, epsilon: float = 1e-12
) -> float:
    clipped_probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    true_class_probabilities = clipped_probabilities[
        np.arange(len(y_true)), y_true
    ]
    return float(-np.mean(np.log(true_class_probabilities)))
