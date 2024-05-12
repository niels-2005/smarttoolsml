import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class ImageDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_

    Example usage:
        dataset = PlayingCardDataset(data_dir=CFG.train_folder)
    """

    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def get_dataloader(folder: str, batch_size: int, shuffle: bool = True, transform=None):
    """_summary_

    Args:
        folder (str): _description_
        batch_size (int): _description_
        shuffle (bool, optional): _description_. Defaults to True.
        transform (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_

    Example usage:
        folder = "./train"

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        batch_size = 32

        get_dataloader(folder=folder, transform=transform, batch_size=batch_size, shuffle=True)
    """
    dataset = ImageDataset(folder, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_target_to_class(train_folder: str):
    """_summary_

    Args:
        train_folder (str): _description_

    Returns:
        _type_: _description_
    """
    target_to_class = {v: k for k, v in ImageFolder(train_folder).class_to_idx.items()}
    return target_to_class
