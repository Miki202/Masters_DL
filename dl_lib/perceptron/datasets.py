def create_dataset(n: int) -> list[tuple[float, float]]:
    return [(float(x), float(2 * x)) for x in range(n)]


def and_dataset() -> list[tuple[tuple[float, float], float]]:
    return [
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 0.0),
        ((1.0, 0.0), 0.0),
        ((1.0, 1.0), 1.0),
    ]


def or_dataset() -> list[tuple[tuple[float, float], float]]:
    return [
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 1.0),
    ]
