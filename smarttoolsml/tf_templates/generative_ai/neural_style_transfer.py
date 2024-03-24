import os

import tensorflow as tf

# Load compressed models from tensorflow_hub
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow_hub as hub
from PIL import Image


def load_img(path_to_img: str):
    """
    Loads an image from a specified path and scales it proportionally so that its longest side is at most 512 pixels.

    Args:
        path_to_img (str): The file path to the image to be loaded.

    Returns:
        Tensor: A tensor representing the loaded and scaled image, with an added batch dimension.
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def imshow(ax, image, title=None, display_size=(112, 224)):
    """
    Displays an image on a given axis object, scaling the image to a specified size.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The axis object on which the image will be displayed.
        image (Tensor): The image to display.
        title (str, optional): The title of the image. Defaults to None.
        display_size (tuple[int, int], optional): The size to which the image is scaled, as a tuple (width, height). Defaults to (112, 224).
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    image = tf.image.resize(image, display_size)

    ax.imshow(image)
    ax.axis("off")

    if title:
        ax.set_title(title)


def tensor_to_image(tensor) -> PIL.Image:
    """
    Converts an image tensor into a PIL.Image object.

    Args:
        tensor (Tensor): The tensor representing the image.

    Returns:
        PIL.Image: The converted image as a PIL.Image object.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)


def transfer_style(
    content_path: str,
    style_path: str,
    show_images_to_transfer: bool = True,
    hub_model_path: str = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2",
    figsize: tuple[int, int] = (12, 12),
    display_size: tuple[int, int] = (112, 224),
    should_save_image: bool = False,
    save_path: str = "stylized_image.png",
    save_image_size: tuple[int, int] = (512, 256),
) -> PIL.Image:
    """
    Applies style transfer to a content image using a style image and optionally displays and saves the result image.

    Args:
        content_path (str): Path to the content image.
        style_path (str): Path to the style image.
        show_images_to_transfer (bool, optional): If True, displays the content and style image before style transfer. Defaults to True.
        hub_model_path (str, optional): The URL path to the TensorFlow Hub model used for the style transfer. Defaults to Magenta's model.
        figsize (tuple[int, int], optional): The figure size for displaying the images. Defaults to (12, 12).
        display_size (tuple[int, int], optional): The size to which images are scaled for display. Defaults to (112, 224).
        should_save_image (bool, optional): If True, saves the transferred image to the specified path. Defaults to False.
        save_path (str, optional): The file path where the transferred image should be saved. Defaults to "stylized_image.png".
        save_image_size (tuple[int, int], optional): The size to which the transferred image should be resized before saving. Defaults to (512, 256).

    Returns:
        PIL.Image: The image after style transfer as a PIL.Image object.

    Example usage:
        transferred_image = transfer_style(
            content_path="path/to/content.jpg",
            style_path="path/to/style.jpg",
            show_images_to_transfer=True,
            should_save_image=True,
            save_path="path/to/save/stylized_image.png"
        )
    """
    content_image = load_img(path_to_img=content_path)
    style_image = load_img(path_to_img=style_path)

    if show_images_to_transfer:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        imshow(
            ax[0], image=content_image, title="Content Image", display_size=display_size
        )
        imshow(ax[1], image=style_image, title="Style Image", display_size=display_size)
        plt.show()

    hub_model = hub.load(hub_model_path)
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    image = tensor_to_image(tensor=stylized_image)
    plt.imshow(image)
    plt.title("Style Transferred Image")
    plt.axis("off")
    plt.show()

    if should_save_image:
        resized_image = image.resize(save_image_size, Image.LANCZOS)
        resized_image.save(save_path)

    return image
