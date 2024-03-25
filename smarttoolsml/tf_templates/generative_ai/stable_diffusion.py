import tensorflow as tf
import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, figsize: tuple[int, int] = (20, 20)):
    """
    Displays a list of images in a single row with matplotlib.

    Args:
        images (list): A list of images to display. Each image in the list should be in a format compatible with matplotlib's imshow function.
        figsize (tuple[int, int], optional): The size of the figure to display the images. Defaults to (20, 20).

    """
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


def generate_images(
    prompt: str,
    img_width: int = 512,
    img_height: int = 512,
    n_images: int = 1,
    figsize: tuple[int, int] = (20, 20),
) -> np.ndarray:
    """
    Generates images based on a text prompt using the Stable Diffusion model from keras_cv.

    Args:
        prompt (str): The text prompt to generate images for.
        img_width (int, optional): The width of the generated images. Defaults to 512.
        img_height (int, optional): The height of the generated images. Defaults to 512.
        n_images (int, optional): The number of images to generate. Defaults to 1.
        figsize (tuple[int, int], optional): The figure size for displaying the generated images using the plot_images function. Defaults to (20, 20).

    Returns:
        list: A list of generated images, each as a numpy array.

    Example usage:
        images = generate_images(
            prompt="A futuristic city skyline at sunset",
            img_width=512,
            img_height=512,
            n_images=1,
            figsize=(10, 10)
        )
    """
    model = keras_cv.models.StableDiffusion(img_width=img_width, img_height=img_height)
    images = model.text_to_image(prompt, batch_size=n_images)
    plot_images(images=images, figsize=figsize)
    return images
