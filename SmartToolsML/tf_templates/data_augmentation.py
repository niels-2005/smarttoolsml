from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def apply_and_plot_augmentation(
    image: np.ndarray,
    img_augmentation_layers: list,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Applies a list of augmentation layers to an image and plots nine results.

    Args:
        image (np.ndarray): The image to augment, expected shape (height, width, channels).
        img_augmentation_layers (list): A list of Keras augmentation layers to apply.

    Example usage:
        apply_and_plot_augmentation(image, img_augmentation_layers)

        img_augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            layers.RandomContrast(factor=0.1),
            layers.RandomBrightness(factor=0.1),
            layers.Rescaling(1.0 / 255)
        ]
    """
    plt.figure(figsize=figsize)
    for i in range(9):
        # Convert the image into a batch of one to pass through the Keras layers
        img_tensor = tf.expand_dims(image, axis=0)
        # Apply each augmentation layer in sequence
        for layer in img_augmentation_layers:
            img_tensor = layer(img_tensor, training=True)
        # Convert the tensor back to a numpy array for plotting
        aug_img = img_tensor.numpy()
        plt.subplot(3, 3, i + 1)
        plt.imshow(aug_img[0].astype("uint8"))  #  in uint8 format
        plt.axis("off")
    plt.show()


def img_augmentation(images: tf.Tensor, img_augmentation_layers: list) -> tf.Tensor:
    """
    Applies a series of augmentation layers to the input images.

    Args:
        images (tf.Tensor): A batch of images to augment.
        img_augmentation_layers (list): A list of Keras image augmentation layers to apply.

    Returns:
        tf.Tensor: The augmented images.
    """
    for layer in img_augmentation_layers:
        images = layer(images, training=True)  # augmentation is applied
    return images


def input_preprocess_train(
    image: tf.Tensor, label: tf.Tensor, num_classes: int, img_augmentation_layers: list
) -> tuple:
    """
    Applies image augmentation and one-hot encodes the labels for training images.

    Args:
        image (tf.Tensor): A single image to preprocess.
        label (tf.Tensor): The label of the image.
        num_classes (int): The total number of classes.
        img_augmentation_layers (list): A list of Keras image augmentation layers to apply.

    Returns:
        tuple: A tuple of the augmented image and its one-hot encoded label.
    """
    image = img_augmentation(image, img_augmentation_layers)
    label = tf.one_hot(label, num_classes)
    return image, label


def input_preprocess_test(
    image: tf.Tensor, label: tf.Tensor, num_classes: int
) -> tuple:
    """
    One-hot encodes the labels for test images.

    Args:
        image (tf.Tensor): A single image to preprocess.
        label (tf.Tensor): The label of the image.
        num_classes (int): The total number of classes.

    Returns:
        tuple: A tuple of the image and its one-hot encoded label.
    """
    label = tf.one_hot(label, num_classes)
    return image, label


def preprocess_data_augmentation(
    train_files: tf.data.Dataset,
    valid_files: tf.data.Dataset,
    test_files: tf.data.Dataset,
    num_classes: int,
    batch_size: int,
    img_augmentation_layers: list,
    include_test_files: bool = True,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    drop_remainder: bool = True,
) -> tuple:
    """
    Prepares and augments training, validation, and optional test datasets for model training.

    This function applies image augmentations to the training dataset and preprocesses all datasets by batching, prefetching,
    and optionally dropping the remainder of batches. It also one-hot encodes the labels of images in all datasets.

    Args:
        train_files (tf.data.Dataset): The dataset containing training images and labels.
        valid_files (tf.data.Dataset): The dataset containing validation images and labels.
        test_files (tf.data.Dataset): The dataset containing test images and labels.
        num_classes (int): The total number of classes in the dataset.
        batch_size (int): The size of the batches of data to use during training.
        img_augmentation_layers (list): A list of TensorFlow/Keras image augmentation layers to apply to the training images.
        include_test_files (bool, optional): Flag indicating whether to include and preprocess the test dataset. Defaults to True.
        prefetch_buffer (int, optional): The number of batches to prefetch. Can improve performance. Defaults to tf.data.AUTOTUNE.
        num_parallel_calls (int, optional): The number of parallel processing threads. Defaults to tf.data.AUTOTUNE.
        drop_remainder (bool, optional): Whether to drop the last batch if it's smaller than the specified batch size. Defaults to True.

    Returns:
        tuple: A tuple containing the processed training, validation, and (if included) test datasets, in that order. Each dataset is batched,
        prefetched, and has its labels one-hot encoded.

    Example usage:
        preprocess_data_augmentation(train_files,
                                     valid_files,
                                     test_files,
                                     num_classes=10,
                                     batch_size=32,
                                     img_augmentation_layers=img_augmentation_layers,
                                     include_test_files=True,
                                     prefetch_buffer=tf.data.AUTOTUNE,
                                     num_parallel_calls=tf.data.AUTOTUNE,
                                     drop_remainder=True)

        img_augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            layers.RandomContrast(factor=0.1),
            layers.RandomBrightness(factor=0.1),
            layers.Rescaling(1.0 / 255)
        ]
    """
    preprocess_train_with_aug = partial(
        input_preprocess_train,
        num_classes=num_classes,
        img_augmentation_layers=img_augmentation_layers,
    )

    # augmentate the train_files
    train_files = train_files.map(
        lambda image, label: preprocess_train_with_aug(image, label),
        num_parallel_calls=num_parallel_calls,
    )
    train_files = train_files.batch(
        batch_size=batch_size, drop_remainder=drop_remainder
    )
    train_files = train_files.prefetch(prefetch_buffer)

    # dont need to be augmentated
    preprocess_test = partial(input_preprocess_test, num_classes=num_classes)
    valid_files = valid_files.map(
        lambda image, label: preprocess_test(image, label),
        num_parallel_calls=num_parallel_calls,
    )
    valid_files = valid_files.batch(
        batch_size=batch_size, drop_remainder=drop_remainder
    )

    # parameter include_test_files = True
    if include_test_files:
        test_files = test_files.map(
            lambda image, label: preprocess_test(image, label),
            num_parallel_calls=num_parallel_calls,
        )
        test_files = test_files.batch(
            batch_size=batch_size, drop_remainder=drop_remainder
        )
        return train_files, valid_files, test_files
    else:
        return train_files, valid_files
