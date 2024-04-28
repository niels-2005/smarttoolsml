import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def preprocess_image(image_path: str) -> tf.Tensor:
    """
    Loads an image from a given path and preprocesses it to make it ready for model input. This includes decoding the
    image, removing the alpha channel if present, resizing the image to a multiple of 4, and normalizing pixel values.

    Args:
        image_path (str): The file path to the image to be processed.

    Returns:
        tf.Tensor: A 4D tensor of the preprocessed image with shape (1, height, width, channels), ready for the model.
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def save_hr_image(image: tf.Tensor, filename: str) -> None:
    """
    Saves a high-resolution image tensor to a file. If the input is not a PIL Image, it converts the tensor to a
    uint8 image and clips values to ensure they are within the valid range for image data. The saved image will
    have a '.jpg' extension.

    Args:
        image (tf.Tensor): A 3D image tensor of shape (height, width, channels).
        filename (str): The base filename (without extension) where the image will be saved.

    Returns:
        None: This function does not return a value but saves the image to the filesystem.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


def plot_image(image: tf.Tensor, title: str = "hr_image") -> None:
    """
    Displays an image from a tensor. It ensures that the pixel values are clipped to the valid range for images
    (0, 255) before displaying. This function is intended for use within notebooks or Python environments capable
    of rendering plots.

    Args:
        image (tf.Tensor): A 3D image tensor of shape (height, width, channels).
        title (str, optional): A title for the image plot. Defaults to an empty string.

    Returns:
        None: This function does not return a value but displays the image using matplotlib.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def upscale_image(
    image_path: str,
    display_original_img: bool = True,
    model_hub_path: str = "https://tfhub.dev/captain-pool/esrgan-tf2/1",
    save_image: bool = False,
    save_img_filename: str = "hr_image",
) -> tf.Tensor:
    """
    Performs image super-resolution on a given image using a pre-trained ESRGAN model from TensorFlow Hub. Optionally,
    displays the original and super-resolved images, and can save the high-resolution image to disk.

    Args:
        image_path (str): The file path to the image to be upscaled.
        display_original_img (bool, optional): If True, the original image will be displayed. Defaults to True.
        model_hub_path (str, optional): The URL to the TensorFlow Hub model to be used for super-resolution.
            Defaults to "https://tfhub.dev/captain-pool/esrgan-tf2/1".
        save_image (bool, optional): If True, the super-resolved image will be saved to disk. Defaults to False.
        save_img_filename (str, optional): The base filename (without extension) for the saved high-resolution image.
            Defaults to "hr_image".

    Returns:
        tf.Tensor: The super-resolved image as a 4D tensor.

    Example usage:
        image = upscale_image(
            image_path="path/to/your/image.png",
            display_original_img=True,
            save_image=True,
            save_img_filename="path/to/save/super_resolved_image"
        )
    """
    model = hub.load(model_hub_path)
    img = preprocess_image(image_path)

    if display_original_img:
        plot_image(tf.squeeze(img), title="Original Image")

    hr_image = model(img)
    hr_image = tf.squeeze(hr_image)

    plot_image(tf.squeeze(hr_image), title="Image Super Resolution")

    if save_image:
        save_hr_image(image=hr_image, filename=save_img_filename)

    return hr_image
