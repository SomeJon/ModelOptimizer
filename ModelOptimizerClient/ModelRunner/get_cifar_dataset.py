from torchvision import datasets, transforms


def get_cifar10_datasets(normalization='None', train_ratio=0.8):
    """
    Downloads and returns the CIFAR-10 training, validation, and testing datasets with specified transformations.

    Parameters:
    - normalization (str): The type of normalization to apply ('StandardScaler', 'MinMaxScaler', 'Normalizer', or 'None').
    - train_ratio (float): Proportion of the training dataset to be used for training (default is 0.8).

    Returns:
    - train_dataset (Subset): Training subset.
    - valid_dataset (Subset): Validation subset.
    - test_dataset (Dataset): Testing dataset.
    """

    # Define mean and std for CIFAR-10 for normalization
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # Define transformations
    transform_list = [
        transforms.ToTensor(),
    ]

    if normalization == 'StandardScaler':
        transform_list.append(transforms.Normalize(cifar10_mean, cifar10_std))
    elif normalization == 'MinMaxScaler':
        # ToTensor already scales inputs to [0, 1]
        pass  # No additional transform needed
    elif normalization == 'Normalizer':
        # Normalizer scales input vectors individually to unit norm
        transform_list.append(transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]))
    elif normalization == 'None':
        pass  # No normalization
    else:
        raise ValueError(f"Unsupported normalization type: {normalization}")

    transform = transforms.Compose(transform_list)

    # Download full training dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Download testing dataset
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset
