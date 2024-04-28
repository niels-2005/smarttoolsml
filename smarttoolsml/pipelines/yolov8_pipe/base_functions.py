import os
import random

import cv2
import matplotlib.pyplot as plt
from cfg import CFG
from PIL import Image
from ultralytics import YOLO


def load_model(preset_name: str = CFG.yolo_preset_name) -> YOLO:
    """
    Initializes and loads a YOLO model based on a given preset name.

    Args:
        preset_name (str): The name of the pre-trained model to be loaded.

    Returns:
        YOLO: An instance of the YOLO model.
    """
    model = YOLO(preset_name)
    return model


def get_best_model(post_training_path: str = CFG.post_train_files_path) -> YOLO:
    """
    Loads the best YOLO model based on the weights saved during training.

    Args:
        post_training_path (str, optional): Path to the directory containing the best model's weights. Defaults to "./runs/detect/train".

    Returns:
        YOLO: The best YOLO model.
    """
    best_model_path = os.path.join(post_training_path, "weights/best.pt")
    best_model = YOLO(best_model_path)
    return best_model


def plot_random_samples(
    path: str,
    format: str = CFG.file_format,
    n_images: int = 8,
    nrows: int = 2,
    ncols: int = 4,
    figsize: tuple[int, int] = (20, 10),
    plot_title: str = "Sample Images from Path",
    fontsize: int = 20,
):
    """
    Plots random sample images from a specified directory.

    Args:
        path (str): Directory path containing images.
        format (str, optional): Image file format to look for. Defaults to ".jpg".
        n_images (int, optional): Number of images to sample and plot. Defaults to 8.
        nrows (int, optional): Number of rows in the subplot grid. Defaults to 2.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 4.
        figsize (tuple[int, int], optional): Figure size of the plot. Defaults to (20, 10).
        plot_title (str, optional): Title of the plot. Defaults to "Sample Images from Path".
        fontsize (int, optional): Font size for the plot title. Defaults to 20.

    Returns:
        None
    """
    image_files = [file for file in os.listdir(path) if file.endswith(format)]

    selected_images = random.sample(image_files, n_images)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, img_file in zip(ax.ravel(), selected_images):
        img_path = os.path.join(path, img_file)
        image = Image.open(img_path)
        ax.imshow(image)
        ax.axis("off")

    plt.suptitle(plot_title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def predict_random_samples(
    model: YOLO,
    img_path: str,
    conf: float = CFG.conf,
    imgsz: int = CFG.imgsz,
    format: str = CFG.file_format,
    n_images: int = 16,
    nrows: int = 4,
    ncols: int = 4,
    plot_title: str = "Random Image Predictions",
    figsize: tuple[int, int] = (20, 20),
    fontsize: int = 24,
) -> None:
    """
    Predicts and plots random sample images with detected objects using the specified YOLO model.

    Args:
        model: The YOLO model to use for predictions.
        img_path (str): Path to the directory containing images for prediction.
        conf (float, optional): Confidence threshold for the predictions. Defaults to 0.5.
        imgsz (int, optional): Size to which the images are resized before prediction. Defaults to 640.
        format (str, optional): Image file format to consider for predictions. Defaults to ".jpg".
        n_images (int, optional): Number of images to sample and predict. Defaults to 9.
        nrows (int, optional): Number of rows in the subplot grid. Defaults to 3.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 3.
        plot_title (str, optional): Title for the plot of predictions. Defaults to "Random Image Predictions".
        figsize (tuple[int, int], optional): Figure size for the plot. Defaults to (20, 21).
        fontsize (int, optional): Font size for the plot title. Defaults to 24.

    Returns:
        None
    """

    image_files = [file for file in os.listdir(img_path) if file.endswith(format)]

    selected_images = random.sample(image_files, n_images)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(plot_title, fontsize=fontsize)

    for i, ax in enumerate(ax.flatten()):
        image_path = os.path.join(img_path, selected_images[i])
        results = model.predict(source=image_path, imgsz=imgsz, conf=conf)
        annotated_image = results[0].plot(line_width=1)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated_image_rgb)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
