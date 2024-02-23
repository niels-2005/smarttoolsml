import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_images_from_folder(
    folder: str,
    n_images: int = 16,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 10,
) -> None:
    """
    Displays a specified number of images from a given folder in a grid layout, annotating each image with its class name and dimensions.

    Args:
        folder (str): The path to the directory containing the images. The directory can contain subdirectories representing different classes.
        n_images (int, optional): The total number of images to be displayed. Defaults to 16.
        images_per_row (int, optional): The number of images displayed per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure used to display the images. The height is automatically adjusted based on the number of rows. Defaults to 20.
        fontsize (int, optional): The font size used for the annotations on each subplot. Defaults to 10.

    Returns:
        None: This function does not return any value. It directly plots the images using matplotlib.

    Example usage:
        plot_images_from_folder(
            folder='/test',
            n_images=16,
            images_per_row=4,
            figsize_width=20,
            fontsize=10
        )

    Note:
        - The function assumes that images are stored directly in the specified folder or within its subdirectories.
        - The `folder` argument should point to a directory structure compatible with the expected image sources.
        - Images are selected randomly from the directory, so displayed images will vary with each function call.
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    ax = ax.reshape(-1)

    for i in range(n_images):
        img, class_name = get_random_image_and_class(folder=folder)
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(f"{class_name}, {img.shape}", fontsize=fontsize)

    for j in range(i + 1, n_rows * n_cols):
        ax[j].axis("off")


def get_random_image_and_class(folder: str) -> tuple[np.ndarray, str]:
    """
    Selects a random image from a specified folder and its class name.

    Args:
        folder (str): Path to the folder containing class subfolders with images.

    Returns:
        tuple[np.ndarray, str]: A tuple containing the normalized image as a NumPy array and the class name (subfolder name).

    Example usage:
        img, class_name = get_random_image_and_class("./data/train")
    """
    random_target_folder = random.choice(os.listdir(folder))
    target_path_folder = os.path.join(folder, random_target_folder)
    random_target_image = random.choice(os.listdir(target_path_folder))
    target_path_file = os.path.join(target_path_folder, random_target_image)
    img = mpimg.imread(target_path_file) / 255.0

    return img, random_target_folder


def plot_images_from_dataset(
    files: tf.data.Dataset,
    class_names: list,
    n_images: int = 16,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 10,
) -> None:
    """
    Displays a specified number of images from a TensorFlow dataset in a grid layout, annotating each image with its class name.

    Args:
        files (tf.data.Dataset): The TensorFlow dataset containing tuples of images and labels.
        class_names (list): A list of class names corresponding to the labels in the dataset.
        n_images (int, optional): The total number of images to be displayed. Defaults to 16.
        images_per_row (int, optional): The number of images displayed per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure used to display the images. The height is automatically adjusted based on the number of rows. Defaults to 20.
        fontsize (int, optional): The font size used for the annotations on each subplot. Defaults to 10.

    Returns:
        None: This function does not return any value. It directly plots the images using matplotlib.

    Example usage:
        # Assuming `dataset` is a TensorFlow dataset and `class_names` is a list of class names corresponding to dataset labels.
        plot_images_from_dataset(
            files=train_files,
            class_names=['cat', 'dog'],
            n_images=16,
            images_per_row=4,
            figsize_width=20,
            fontsize=10
        )

    Note:
        - This function is designed to work with TensorFlow datasets that return images and labels in separate tensors.
        - Ensure the provided `files` dataset is batched appropriately, as this function takes only the first batch for plotting.
        - The actual number of plotted images will be the minimum of `n_images` and the batch size of `files`.
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    ax = ax.reshape(-1)

    for images, labels in files.take(1):
        labels = labels.numpy().astype(int).reshape(-1)

        for i in range(n_images):
            ax[i].imshow(images[i].numpy().astype("uint8"))
            ax[i].axis("off")
            ax[i].set_title(
                f"{class_names[labels[i]]}, {images[i].shape}", fontsize=fontsize
            )
