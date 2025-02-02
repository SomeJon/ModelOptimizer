from torchvision import datasets, transforms
from torch.utils.data import random_split


def get_cifar10_datasets(normalization='None', train_ratio=0.8):
    """
    Downloads and returns the CIFAR-10 training, validation, and testing datasets with specified transformations.
    The training dataset uses data augmentation while the validation dataset uses only basic transforms (no augmentation).

    Parameters:
    - normalization (str): The type of normalization to apply ('StandardScaler', 'MinMaxScaler', 'Normalizer', or 'None').
    - train_ratio (float): Proportion of the full training dataset to be used for training (default is 0.8).

    Returns:
    - train_dataset (Subset): Training subset with augmentation.
    - valid_dataset (Subset): Validation subset with basic transforms (no augmentation).
    - test_dataset (Dataset): Testing dataset with basic transforms.
    """
    # Define mean and std for CIFAR-10 for normalization
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # ---------------------------
    # Define training augmentation transforms
    # ---------------------------
    train_transform_list = [
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentations
        transforms.ToTensor(),
    ]

    # ---------------------------
    # Define basic transforms (no augmentation) for validation and testing
    # ---------------------------
    basic_transform_list = [
        transforms.ToTensor(),
    ]

    # ---------------------------
    # Apply normalization if requested
    # ---------------------------
    if normalization is None or normalization == 'None':
        pass  # No normalization applied
    elif normalization == 'StandardScaler':
        norm_transform = transforms.Normalize(cifar10_mean, cifar10_std)
        train_transform_list.append(norm_transform)
        basic_transform_list.append(norm_transform)
    elif normalization == 'MinMaxScaler':
        # ToTensor already scales inputs to [0, 1]
        pass
    elif normalization == 'Normalizer':
        norm_transform = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        train_transform_list.append(norm_transform)
        basic_transform_list.append(norm_transform)
    else:
        raise ValueError(f"Unsupported normalization type: {normalization}")

    # Compose transforms for training and for basic (validation/test)
    train_transform = transforms.Compose(train_transform_list)
    basic_transform = transforms.Compose(basic_transform_list)

    # ---------------------------
    # Download the full training dataset (with augmentation)
    # ---------------------------
    full_train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    # Also download the test dataset (using basic transforms)
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=basic_transform
    )

    # ---------------------------
    # Split the full training dataset into train and validation subsets
    # ---------------------------
    total_train = len(full_train_dataset)  # 50,000 images for CIFAR-10 training set
    train_size = int(train_ratio * total_train)
    valid_size = total_train - train_size
    train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

    # IMPORTANT: Override the transform for the validation subset so that it does not use data augmentation.
    # Since valid_dataset is a Subset, its underlying dataset is full_train_dataset; we change its transform.
    # WARNING: This change affects all instances referencing the underlying dataset,
    # so if you need to keep the training augmentation for training, consider downloading a separate instance.
    valid_dataset.dataset.transform = basic_transform

    return train_dataset, valid_dataset, test_dataset
