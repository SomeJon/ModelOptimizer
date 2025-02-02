import os
import torch
from torch.utils.data import DataLoader


def get_optimal_num_workers():
    """
    Determines a dynamic number of workers based on the available CPU cores.
    Returns at least 2 and at most 8 workers, using roughly half the CPU cores.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 2  # Fallback default
    return min(8, max(2, cpu_count // 2))


def get_data_loader(dataset, batch_size, shuffle, use_cuda):
    """
    Creates a DataLoader with dynamic settings.

    If CUDA is available and the device is an RTX 3090, it uses a dynamic number of workers
    based on the CPU count. Otherwise, it sets num_workers to 0.

    Parameters:
    - dataset (Dataset): The dataset to load.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): Whether to shuffle the data.
    - use_cuda (bool): Whether CUDA (GPU) is available.

    Returns:
    - DataLoader: Configured DataLoader.
    """
    if use_cuda:
        pin_memory = True
        device_name = torch.cuda.get_device_name(0)
        # Check if the device name contains 'rtx 3090' (case-insensitive)
        if "rtx 3090" in device_name.lower():
            num_workers = get_optimal_num_workers()
        else:
            num_workers = 0
    else:
        pin_memory = False
        num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
