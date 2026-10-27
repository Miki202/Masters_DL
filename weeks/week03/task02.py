from dl_lib.perceptron import datasets, functional
from dl_lib.losses import functional as losses_functional

NUDGES = [0.002, 0.001, -0.001, -0.002]

def mse_for_weight(dataset: list[tuple[float, float]], weight: float) -> float:
    x_values = [x for x, _ in dataset]
    y_true = [y for _, y in dataset]
    y_pred = [weight*x for x in x_values]
    return losses_functional.mse(y_true, y_pred)

def describe(dataset: list[tuple[float, float]], weight: float) -> None:
    print(f"Dataset: {dataset}\n"
          + f"w: {weight}\n"
          + f"MSE: {mse_for_weight(dataset, weight)}")
    print("Perturbing w:")
    for new in NUDGES:
        new_weight = weight + new
        new_mse = mse_for_weight(dataset, new_weight)
        sign = "+" if new >= 0 else ""
        print(f"w + ({sign}{new}) -> w={new_weight}, MSE={new_mse}")

def main() -> int:
    datset = datasets.create_dataset(6)
    weight = functional.initialize_weights(0, 10, 1, 42)[0]
    describe(datset, weight)
    return 0