import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_cifar10_datasets(data_dir='./data'):
    """
    Downloads and returns the CIFAR-10 training and testing datasets with standard transformations.

    Parameters:
    - data_dir (str): Directory where the CIFAR-10 data will be stored/downloaded.

    Returns:
    - train_dataset (torch.utils.data.Dataset): Training dataset for CIFAR-10.
    - test_dataset (torch.utils.data.Dataset): Testing dataset for CIFAR-10.
    """

    # Define the standard transformation: convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 training set mean
            std=[0.2023, 0.1994, 0.2010]  # CIFAR-10 training set std
        )
    ])

    # Initialize the training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Initialize the testing dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset
