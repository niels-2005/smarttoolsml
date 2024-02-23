from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def apply_and_plot_augmentation_test(
    image: np.ndarray,
    img_augmentation_layers: list,
    n_images: int = 10,
    images_per_row: int = 4,
    figsize_width: int = 20,
) -> None:
    """
    Applies a series of image augmentation layers to a single image and displays the augmented images in a grid layout.

    Args:
        image (np.ndarray): The original image to augment, expected in the format (height, width, channels).
        img_augmentation_layers (list): A list of TensorFlow/Keras image augmentation layers to apply to the image.
        n_images (int, optional): The total number of augmented images to display. Defaults to 10.
        images_per_row (int, optional): The number of images to display per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure used to display the images. The height is automatically adjusted based on the number of rows. Defaults to 20.

    Returns:
        None: This function does not return any value. It directly displays the images using matplotlib.

    Example usage:
        from smarttoolsml.tf_templates.image_classification.img_plotting import get_random_image_and_class

        TEST_DIR = './test'

        img, class_name = get_random_image_and_class(folder=TEST_DIR)

        img_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(height_factor=(0.05), width_factor=(0.05)),
        ]

        # Apply and plot the augmentations
        apply_and_plot_augmentation_test(
            image=img,
            img_augmentation_layers=img_augmentation_layers,
            n_images=10,
            images_per_row=4,
            figsize_width=20
        )

    Note:
        - Ensure that the input image and augmentation layers are compatible with TensorFlow/Keras.
        - The function automatically handles the creation and layout of the subplot grid based on the specified number of images and images per row.
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    ax = ax.reshape(-1)

    for i in range(n_images):
        img_tensor = tf.expand_dims(image, axis=0)  # add dimension

        for layer in img_augmentation_layers:
            img_tensor = layer(img_tensor, training=True)

        aug_img = img_tensor.numpy().squeeze()  # remove dimension

        ax[i].imshow(aug_img)
        ax[i].axis("off")

    for j in range(i + 1, n_cols * n_rows):
        ax[j].axis("off")

    plt.tight_layout()
    plt.show()
