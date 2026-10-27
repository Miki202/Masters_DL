from dl_lib.perceptron import datasets, functional

def describe(dataset: list[tuple[float, float]], first_weight: float, second_weight: float) -> None:
    print(f"Dataset: {dataset}\n"
          + f"first_weight: {first_weight}\n"
          + f"second_weight: {second_weight}")


def main() -> int:
    dataset = datasets.create_dataset(4)
    first_weight, second_weight = functional.initialize_weights(-1, 1, 2, 42)
    describe(dataset, first_weight, second_weight)
    return 0
