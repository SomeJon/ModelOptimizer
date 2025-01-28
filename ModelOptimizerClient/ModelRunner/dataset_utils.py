import os
from torch.utils.data import DataLoader


def get_split_sizes(dataset_length, train_ratio=0.8):
    """
    Calculates the sizes for training and validation splits.

    Parameters:
    - dataset_length (int): Total number of samples in the dataset.
    - train_ratio (float): Proportion of data to be used for training.

    Returns:
    - train_size (int): Number of samples for training.
    - valid_size (int): Number of samples for validation.
    """
    train_size = int(train_ratio * dataset_length)
    valid_size = dataset_length - train_size
    return train_size, valid_size


def get_optimal_num_workers():
    """
    Determines the optimal number of workers based on CPU cores.

    Returns:
    - int: Number of workers.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 2  # Default value
    else:
        return min(8, max(2, cpu_count // 2))  # At least 2, at most 8


def get_data_loader(dataset, batch_size, shuffle, use_cuda):
    """
    Creates a DataLoader with dynamic settings based on hardware.

    Parameters:
    - dataset (Dataset): The dataset to load.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): Whether to shuffle the data.
    - use_cuda (bool): Whether CUDA (GPU) is available.

    Returns:
    - DataLoader: Configured DataLoader.
    """
    # Determine optimal num_workers

    # Dynamically set num_workers and pin_memory
    if use_cuda:
        pin_memory = True
    else:
        pin_memory = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
