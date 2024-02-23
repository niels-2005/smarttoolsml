import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_images_from_folder(
    folder: str,
    figsize: tuple[int, int] = (10, 10),
    num_images: int = 16,
    num_subplot: int = 4,
) -> None:
    """
    Displays a selection of images from a given folder in a figure with subplots.

    This function loads random images from the specified folder and displays them in a grid of subplots.
    Each subplot shows an image along with a title that includes the class name and the image dimensions.
    The function is designed so that the number of images (`num_images`) should be a square of the number of
    subplots per row and column (`num_subplot`), e.g., 16 images in a 4x4 grid or 9 images in a 3x3 grid.

    Args:
        folder (str): The path to the folder from which the images will be loaded.
        figsize (tuple[int, int], optional): The size of the figure containing the subplots.
                                              Defaults to (10, 10).
        num_images (int, optional): The total number of images to display. Default is 16.
                                    It's important that `num_images` is a perfect square of `num_subplot`
                                    to ensure an even distribution of images across the subplots.
        num_subplot (int, optional): The number of subplots per row and column. Default is 4.
                                     This value determines the layout of the subplots in the figure.

    Returns:
        None: The function does not return anything but directly displays the generated figure with the images.

    Example:
        plot_images_from_folder(folder='path/to/image/folder', figsize=(12, 12), num_images=9, num_subplot=3)

    Important:
        Ensure that `num_images` is a perfect square of `num_subplot` (e.g., 16 images for 4 subplots),
        to guarantee correct display. Otherwise, displaying the subplots may not function as expected.
    """
    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(num_subplot, num_subplot, i + 1)
        img, class_name = get_random_image_and_class(folder=folder)
        plt.imshow(img)
        plt.title(f"{class_name}, {img.shape}")
        plt.axis(False)


def get_random_image_and_class(folder: str) -> tuple[np.ndarray, str]:
    """
    Selects a random image from a specified folder and its class name.

    This function navigates through a given folder, randomly selects a subfolder (representing a class),
    and then randomly selects an image file from this subfolder. The image is loaded, normalized, and returned
    along with the name of its class (subfolder name).

    Args:
        folder (str): Path to the folder containing class subfolders with images.

    Returns:
        Tuple[np.ndarray, str]: A tuple containing the normalized image as a NumPy array and the class name (subfolder name).

    Example usage:
        img, class_name = get_random_image_and_class("./data/train")
    """
    # Get a random class folder
    random_target_folder = random.choice(os.listdir(folder))
    # Construct the full path to the class folder
    target_path_folder = os.path.join(folder, random_target_folder)
    # Get a random image name from the class folder
    random_target_image = random.choice(os.listdir(target_path_folder))
    # Construct the full path to the image file
    target_path_file = os.path.join(target_path_folder, random_target_image)
    # Read the image file and normalize it
    img = mpimg.imread(target_path_file) / 255.0

    return img, random_target_folder


def plot_images_from_dataset(
    files: tf.data.Dataset,
    class_names: list,
    num_images: int = 16,
    num_subplot: int = 4,
    figsize: tuple[int, int] = (12, 12),
) -> None:
    """
    Plots a selection of images from a TensorFlow dataset in a grid layout.

    This function takes the first batch of images from the provided dataset and plots a specified
    number of images in a grid format defined by `num_subplot` by `num_subplot`. Each image is displayed
    with its corresponding class name and image dimensions as the title. The labels are expected to
    be in integer format, which are used to index into the provided list of class names to retrieve
    the appropriate label for each image.

    Args:
        files (tf.data.Dataset): The TensorFlow dataset containing tuples of images and labels.
        class_names (list): A list of class names corresponding to the labels in the dataset.
        num_images (int, optional): The total number of images to display. Default is 16.
                                    It's important that `num_images` is a perfect square of `num_subplot`
                                    to ensure an even distribution of images across the subplots.
        num_subplot (int, optional): The number of subplots per row and column. Default is 4.
                                     This value determines the layout of the subplots in the figure.
        figsize (tuple[int, int], optional): The size of the figure to display the images. Defaults to (12, 12).

    Returns:
        None: This function does not return any value. It plots the images directly using matplotlib.

    Example usage:
        # Assuming `dataset` is your TensorFlow dataset and `class_names` is your list of class names.

        plot_images_from_dataset(files=dataset,
                                 class_names=class_names,
                                 num_images=9,
                                 num_subplot=3,
                                 figsize=(10, 10))

    Note:
        - The function automatically converts the labels to integers and reshapes them to a flat array
          to ensure proper indexing. Images are converted to 'uint8' format for proper display.
        - It is assumed that the dataset returns images in a format compatible with matplotlib's `imshow` method.
        - Ensure `num_images` does not exceed the actual number of images in the batch provided by `files.take(1)`.
    """

    plt.figure(figsize=figsize)

    for images, labels in files.take(1):
        labels = labels.numpy().astype(int).reshape(-1)

        for i in range(num_images):
            plt.subplot(num_subplot, num_subplot, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"{class_names[labels[i]]}, {images[i].shape}")
            plt.axis("off")
