import os

from base_functions import load_model, plot_random_samples, predict_random_samples
from cfg import CFG
from PIL import Image


def data_pipeline():
    print("1. Loading Model...")
    model = load_model()
    print("2. Checking Datasets...")
    check_img_dataset_values()
    print("3. Plot Samples from Training Set...")
    plot_random_samples(
        path=CFG.img_train_path, plot_title="8 Samples from Training Set"
    )
    print("4. Plot Samples from Validation Set...")
    plot_random_samples(
        path=CFG.img_valid_path, plot_title="8 Samples from Validation Set"
    )
    print("5. Base Model predicts on Validation Set (conf=0)...")
    predict_random_samples(model=model, img_path=CFG.img_valid_path, conf=0)
    print("6. Base Model predicts on Validation Set (conf=0.5)")
    predict_random_samples(model=model, img_path=CFG.img_valid_path)


def check_img_dataset_values(
    img_train_path: str = CFG.img_train_path,
    img_valid_path: str = CFG.img_valid_path,
    format: str = CFG.file_format,
):
    """
    Checks and prints the number of images and unique image sizes in training and validation datasets.

    Args:
        img_train_path (str): The file path to the training images.
        img_valid_path (str): The file path to the validation images.
        format (str, optional): The image file format to be considered. Defaults to ".jpg".

    Returns:
        None
    """
    num_train_images = 0
    num_valid_images = 0

    train_image_sizes = set()
    valid_image_sizes = set()

    # Check train images sizes and count
    for filename in os.listdir(img_train_path):
        if filename.endswith(format):
            num_train_images += 1
            image_path = os.path.join(img_train_path, filename)
            with Image.open(image_path) as img:
                train_image_sizes.add(img.size)

    # Check validation images sizes and count
    for filename in os.listdir(img_valid_path):
        if filename.endswith(format):
            num_valid_images += 1
            image_path = os.path.join(img_valid_path, filename)
            with Image.open(image_path) as img:
                valid_image_sizes.add(img.size)

    # Print the results
    print(f"Number of training images: {num_train_images}")
    print(f"Number of validation images: {num_valid_images}")

    # Check if all images in training set have the same size
    if len(train_image_sizes) == 1:
        print(f"All training images have the same size: {train_image_sizes.pop()}")
    else:
        print("Training images have varying sizes.")

    # Check if all images in validation set have the same size
    if len(valid_image_sizes) == 1:
        print(f"All validation images have the same size: {valid_image_sizes.pop()}")
    else:
        print("Validation images have varying sizes.")
