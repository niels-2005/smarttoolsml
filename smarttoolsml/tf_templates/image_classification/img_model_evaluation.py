import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model


def plot_and_predict_img_from_folder(
    model: Model,
    folder: str,
    class_names: list,
    img_shape: int = 224,
    preprocess_fn=None,
    is_categorical: bool = False,
    color_channels: int = 3,
    n_images: int = 20,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 10,
) -> None:
    """
    Displays a specified number of images from given folders in a grid layout, along with their actual and predicted class labels, using a trained model
    for predictions.

    Args:
        model (Model): The trained model used for making predictions.
        folder (str): The path to the directory containing class subfolders with images. Each subfolder represents a class.
        class_names (list): A list of class names corresponding to the subfolders in the directory.
        img_shape (int, optional): The target size to which the images are resized before prediction. Defaults to 224.
        preprocess_fn (callable, optional): The preprocessing function applied to images before prediction. If None, images are scaled to [0, 1]. Defaults to None.
        is_categorical (bool, optional): Specifies whether the classification task is categorical (True) or binary (False). Defaults to False.
        color_channels (int, optional): The number of color channels in the images. Defaults to 3 for RGB images.
        n_images (int, optional): The total number of images to display. Defaults to 20.
        images_per_row (int, optional): The number of images displayed per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure used to display the images. Defaults to 20.
        fontsize (int, optional): The font size used for the image titles. Defaults to 10.

    Example usage:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.resnet50 import preprocess_input

        model = load_model('path/to/your/model.h5')
        folder = '/test' (important that /test, not /test/)
        class_names = ['cat', 'dog']

        def preprocess_fn(image):
            return preprocess_input(image)

        plot_and_predict_img_from_folder(model=model,
                                         folder=folder,
                                         class_names=class_names,
                                         img_shape=224,
                                         preprocess_fn=custom_preprocess_fn,
                                         is_categorical=True,
                                         color_channels=3,
                                         n_images=20,
                                         images_per_row=4,
                                         figsize_width=20,
                                         fontsize=10)

    Note:
        - The function randomly selects images from the specified folder, so the displayed images will vary with each call.
        - Ensure the `folder` argument points to a directory structure compatible with the expected class subfolders.
        - The preprocessing function should be compatible with the model's expected input format.
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    ax = ax.reshape(-1)  # needed for single subplots

    for i in range(n_images):

        class_name = random.choice(class_names)
        filenames = os.listdir(os.path.join(folder, class_name))
        filename = random.choice(filenames)
        filepath = os.path.join(folder, class_name, filename)

        img = tf.io.read_file(filepath)
        img = tf.io.decode_image(img, channels=color_channels)
        img = tf.image.resize(img, [img_shape, img_shape])

        if preprocess_fn:
            img_preprocessed = preprocess_fn(img)
            pred_probs = model.predict(
                tf.expand_dims(img_preprocessed, axis=0)
            )  # model needs shape [None, 224, 224, 3]
            img_to_show = img_preprocessed.numpy()
            img_to_show = (img_to_show - img_to_show.min()) / (
                img_to_show.max() - img_to_show.min()
            )  # get preprocessed image back to [0, 1] for plotting
        else:
            pred_probs = model.predict(tf.expand.dims(img, axis=0))
            img_to_show = img.numpy() / 255.0

        if is_categorical:
            pred_class = class_names[pred_probs.argmax()]
        else:
            pred_prob = pred_probs.reshape(-1)[0]
            pred_class = class_names[int(pred_prob > 0.5)]

        ax[i].imshow(img_to_show)
        ax[i].axis("off")
        title_color = "g" if class_name == pred_class else "r"
        ax[i].set_title(
            f"Actual: {class_name}, Pred: {pred_class}, Prob: {pred_probs.max():.2f}",
            color=title_color,
            fontsize=fontsize,
        )

    # ignore empty subplots
    for j in range(i + 1, n_rows * n_cols):
        ax[j].axis("off")

    plt.tight_layout()
    plt.show()


def get_filepaths(files: tf.data.Dataset, path: str) -> np.ndarray:
    """
    Retrieves file paths from a specified directory and its subdirectories matching a given pattern using TensorFlow's dataset utilities.

    Args:
        files (tf.data.Dataset): A TensorFlow Dataset object, used here as a namespace to access the `list_files` method.
        path (str): The path to the directory containing the files of interest, including a pattern to match files. For example, './test/*/*.jpg' matches
                    all JPG files in all subdirectories of 'test'.

    Returns:
        np.ndarray: An array of file paths matching the specified pattern in the given directory, converted from TensorFlow string tensors to Python strings for compatibility with non-TensorFlow processing.

    Example usage:
        path = './test/*/*.jpg'

        # Getting file paths
        filepaths = get_filepaths(test_files, path)

    Note:
        - Ensure the `path` argument correctly specifies the pattern to match the desired files within the directory structure.
    """
    filepaths = []

    for filepath in files.list_files(path, shuffle=False):
        decoded_filepath = filepath.numpy().decode("utf-8")
        filepaths.append(decoded_filepath)

    return np.array(filepaths)


def get_predictions_as_df(
    filepaths: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_probs: np.ndarray,
    class_names: np.ndarray,
    is_binary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates two DataFrames: one containing all predictions and another highlighting the wrong predictions made by a classification model.

    This function creates a comprehensive DataFrame of model predictions for further analysis and a subset DataFrame that filters out incorrect predictions based on the comparison between predicted and true labels. It supports both binary and multi-class classification tasks. For binary classification, it reshapes the predictions and true labels to ensure they are one-dimensional. The resulting DataFrames include columns for image paths, true labels, predicted labels, prediction confidence, and class names for both true and predicted labels.

    Args:
        filepaths (np.ndarray): An array of file paths corresponding to the images or data points evaluated by the model.
        y_pred (np.ndarray): An array of predicted labels by the model.
        y_true (np.ndarray): An array of true labels.
        pred_probs (np.ndarray): An array of prediction probabilities outputted by the model.
        class_names (np.ndarray): An array of class names corresponding to the labels.
        is_binary (bool, optional): Indicates if the classification task is binary. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. A DataFrame (`pred_df`) with all predictions, including image path (`img_path`), true label (`y_true`),
               predicted label (`y_pred`), prediction confidence (`pred_conf`), true class name (`y_true_classname`),
               and predicted class name (`y_pred_classname`).
            2. A DataFrame (`wrong_predictions`) of the same structure as `pred_df` but filtered to only include instances
               where the model's prediction was incorrect, sorted by prediction confidence in descending order.

    Example usage:
        all_predictions_df, wrong_predictions_df = get_wrong_predictions_as_df(
            filepaths=filepaths,
            y_pred=model_predictions,
            y_true=true_labels,
            pred_probs=prediction_probabilities,
            class_names=np.array(["class0", "class1"]),
            is_binary=True
        )

    Note:
        - For binary classification, ensure `y_pred`, `y_true`, and `class_names` are correctly formatted to represent binary outcomes.
        - The `pred_probs` should be the raw output probabilities from the model, with the maximum value per prediction used to determine confidence.
    """
    if is_binary:
        y_pred = y_pred.reshape(-1).astype(int)
        y_true = y_true.reshape(-1).astype(int)

    pred_df = pd.DataFrame(
        {
            "img_path": filepaths,
            "y_true": y_true,
            "y_pred": y_pred,
            "pred_conf": pred_probs.max(axis=1),
            "y_true_classname": [class_names[i] for i in y_true],
            "y_pred_classname": [class_names[i] for i in y_pred],
        }
    )

    pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
    wrong_predictions = pred_df[pred_df["pred_correct"] == False].sort_values(
        "pred_conf", ascending=False
    )

    return pred_df, wrong_predictions


def plot_wrong_predictions(
    wrong_predictions: pd.DataFrame,
    n_images: int = 20,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 12,
):
    """
    Plots a specified number of wrong predictions from a DataFrame, displaying each image with its true and predicted class names, and image path below.

    Args:
        wrong_predictions (pd.DataFrame): A DataFrame containing the wrong predictions. It must include columns 'img_path', 'y_true_classname', and 'y_pred_classname'.
        n_images (int, optional): The total number of wrong prediction images to display. Defaults to 20.
        images_per_row (int, optional): The number of images to display per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure to display the images. The height is automatically calculated based on the number of rows. Defaults to 20.
        fontsize (int, optional): The font size of the title for each subplot, which includes the true and predicted class names, and the image path. Defaults to 12.

    Example usage:
        # Assuming `wrong_predictions` is your DataFrame containing the columns 'img_path', 'y_true_classname', and 'y_pred_classname'.
        plot_wrong_predictions(wrong_predictions=wrong_predictions, n_images=20, images_per_row=4, figsize_width=20, fontsize=12)

    Note:
        - The images are assumed to be stored at the paths specified in the 'img_path' column of the `wrong_predictions` DataFrame.
        - The function automatically adjusts the figure's height based on the number of rows required to display `n_images` images, with `images_per_row` images per row.
        - Any excess subplot axes not used for displaying images are hidden to maintain a clean figure layout.
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    axes = axes.reshape(-1)

    for i, row in enumerate(wrong_predictions.iloc[:n_images].itertuples()):
        img_path = row.img_path
        true_classname = row.y_true_classname
        pred_classname = row.y_pred_classname
        pred_conf = row.pred_conf

        img = mpimg.imread(img_path)
        axes[i].imshow(img / 255.0)

        axes[i].set_title(
            f"True: {true_classname}, Pred: {pred_classname}\n prob: {pred_conf:.4f}\n img_path: {img_path}",
            fontsize=fontsize,
        )

        axes[i].axis("off")

    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def pred_and_plot_random_image(
    filepath: str,
    model: Model,
    class_names: list,
    preprocess_fn=None,
    is_categorical: bool = False,
    color_channels: int = 3,
    img_shape: int = 224,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """
    Loads an image from a specified filepath, optional processes it, predicts its class using a trained model, and plots both the original and processed images
    side by side.

    Args:
        filepath (str): The path to the image file to be loaded and predicted.
        model (Model): The trained TensorFlow/Keras model used for making predictions.
        class_names (list): A list of class names that correspond to the model's output classes.
        preprocess_fn (callable, optional): A function to preprocess the image before making a prediction. If None, the image is only resized and normalized. Defaults to None.
        is_categorical (bool, optional): Specifies whether the model's prediction task is categorical (True) or binary (False). Defaults to False.
        color_channels (int, optional): The number of color channels in the image. Defaults to 3 (for RGB images).
        img_shape (int, optional): The size to which the image is resized before prediction. Defaults to 224.
        figsize (tuple[int, int], optional): The size of the figure in which the images are plotted. Defaults to (10, 5).

    Returns:
        None: This function does not return any value. It directly plots the images using matplotlib.

    Example usage:
        from tensorflow.keras.models import load_model

        model = load_model('/path/to/your/model.h5')
        class_names = ['cat', 'dog']
        filepath = '/path/to/your/image.jpg'

        def custom_preprocess_fn(image):
            # Custom preprocessing steps
            return processed_image

        pred_and_plot_random_image(
            filepath=filepath,
            model=model,
            class_names=class_names,
            preprocess_fn=custom_preprocess_fn,
            is_categorical=False,
            color_channels=3,
            img_shape=224,
            figsize=(10, 5)
        )

    Note:
        - The function assumes the provided model expects input images of shape [None, Height, Width, Color Channels].
        - If a preprocess function is provided, it should handle the resizing and normalization of the image as needed by the model.
    """

    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img, channels=color_channels)
    original_img = img
    img = tf.image.resize(img, [img_shape, img_shape])

    if preprocess_fn:
        img_preprocessed = preprocess_fn(img)
        pred_probs = model.predict(
            tf.expand_dims(img_preprocessed, axis=0)
        )  # model needs shape [None, Height, Width, Color Channels]
        img_to_show = img_preprocessed.numpy()
        img_to_show = (img_to_show - img_to_show.min()) / (
            img_to_show.max() - img_to_show.min()
        )  # get preprocessed image back to [0, 1] for plotting
    else:
        pred_probs = model.predict(tf.expand.dims(img, axis=0))
        img_to_show = img.numpy() / 255.0

    if is_categorical:
        pred_class = class_names[pred_probs.argmax()]
    else:
        pred_prob = pred_probs.reshape(-1)[0]
        pred_class = class_names[int(pred_prob > 0.5)]

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # original image
    ax[0].imshow(original_img.numpy() / 255.0)
    ax[0].axis("off")
    ax[0].set_title(f"Original Image\n Shape: {original_img.shape}")

    # Predicted Class (image)
    ax[1].imshow(img_to_show)
    ax[1].axis("off")
    ax[1].set_title(f"Predicted Class: {pred_class}\n Shape: {img_to_show.shape}")

    plt.tight_layout()
    plt.show()
