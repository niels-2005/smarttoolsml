import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageDataset(Dataset):
    """Custom Dataset for loading images from a directory.

    Args:
        data_dir (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        data (ImageFolder): Instance of ImageFolder with the provided directory and transform.
        filepaths (list): List of file paths for each image in the dataset.

    Methods:
        __len__(): Returns the total number of samples.
        __getitem__(idx): Returns the sample at the given index.
        classes: Property to get the class names.
        get_filepaths(): Returns the list of file paths for each image.
    """

    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.filepaths = [s[0] for s in self.data.samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

    def get_filepaths(self):
        return self.filepaths


def get_datasets(train_folder, val_folder, test_folder):
    """Creates datasets for training, validation, and testing.

    Args:
        train_folder (str): Path to the training data directory.
        val_folder (str): Path to the validation data directory.
        test_folder (str): Path to the testing data directory.

    Returns:
        tuple: A tuple containing the training, validation, and testing datasets.

    Example usage:
        train_dataset, val_dataset, test_dataset = get_datasets(
            train_folder="path/to/train",
            val_folder="path/to/val",
            test_folder="path/to/test"
        )
    """
    # Train Augmentations if needed
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Datasets
    train_dataset = ImageDataset(data_dir=train_folder, transform=train_transform)
    val_dataset = ImageDataset(data_dir=val_folder, transform=test_transform)
    test_dataset = ImageDataset(data_dir=test_folder, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size):
    """Creates dataloaders for training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Testing dataset.
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple: A tuple containing the training, validation, and testing dataloaders.

    Example usage:
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(
            train_dataset, val_dataset, test_dataset, batch_size=32
        )
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_target_to_class(train_folder: str):
    """_summary_

    Args:
        train_folder (str): _description_

    Returns:
        _type_: _description_
    """
    target_to_class = {v: k for k, v in ImageFolder(train_folder).class_to_idx.items()}
    return target_to_class
