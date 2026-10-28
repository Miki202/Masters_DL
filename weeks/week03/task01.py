from dl_lib.perceptron import datasets, functional


def describe(
    dataset: list[tuple[float, float]], first_weight: float, second_weight: float
) -> None:
    print(
        f"Dataset: {dataset}\n"
        + f"initialize_weights(0, 100): {first_weight}\n"
        + f"initialize_weights(0, 10):  {second_weight}"
    )


def main() -> int:
    dataset = datasets.create_dataset(4)
    first_weight = functional.initialize_weights(0, 100, 1, 42)[0]
    second_weight = functional.initialize_weights(0, 10, 1, 42)[0]
    describe(dataset, first_weight, second_weight)
    return 0
