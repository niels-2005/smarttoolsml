import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
