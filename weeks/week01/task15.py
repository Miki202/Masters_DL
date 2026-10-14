import numpy as np
import torch


def describe(
    array: np.ndarray,
    tensor: torch.Tensor,
    device: torch.device,
    moved_tensor: torch.Tensor,
) -> None:
    print(
        f"NumPy array:{array}\n"
        + f"dtype={array.dtype}, shape={array.shape}\n\n"
        + f"PyTorch tensor:\n{tensor}\n"
        + f"dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}\n\n"
        + f"Selected device: {device}\n"
        + f"Tensor moved to device: {moved_tensor.device}, dtype={moved_tensor.dtype}"
    )


def main() -> int:
    array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor = torch.from_numpy(array)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    describe(
        array=array,
        tensor=tensor,
        device=device,
        moved_tensor=tensor.to(device),
    )

    return 0
