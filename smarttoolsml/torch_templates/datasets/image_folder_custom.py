import os
import pathlib
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderCustom(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_

    Example usage:
        # Augment train data
        train_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        # Don't augment test data, only reshape
        test_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)
        test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                            transform=test_transforms)
    """

    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


def find_classes(directory: str) -> tuple[list[str], dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def display_random_images(
    dataset: torch.utils.data.dataset.Dataset,
    classes: list[str] = None,
    n: int = 10,
    display_shape: bool = True,
    seed: int = None,
):
    """_summary_

    Args:
        dataset (torch.utils.data.dataset.Dataset): _description_
        classes (list[str], optional): _description_. Defaults to None.
        n (int, optional): _description_. Defaults to 10.
        display_shape (bool, optional): _description_. Defaults to True.
        seed (int, optional): _description_. Defaults to None.

    Example usage:
        data = ImageFolderCustom()
        classes = data.classes
        display_random_images(data, n=5, classes=classes, seed=42)
    """
    if n > 10:
        n = 10
        display_shape = False
        print(
            f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display."
        )

    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16, 8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        targ_image_adjust = targ_image.permute(1, 2, 0)

        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
