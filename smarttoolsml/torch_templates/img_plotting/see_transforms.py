import random

import matplotlib.pyplot as plt
from PIL import Image


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.

    Example usage:
        image_paths = get_image_paths()

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense
            transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
        ])

        plot_transformed_images(image_paths, transform=transform, n=3)
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def get_image_paths(image_path: str, format: str):
    """_summary_

    Args:
        image_path (str): _description_
        format (str): _description_

    Returns:
        _type_: _description_

    Example usage:
        data_path = Path("data/")
        image_path = data_path / "pizza_steak_sushi"
        format = "*/*/*.jpg"
        img_paths = get_image_paths(image_path=image_path, format=format)

        If you want to get a random path from list:
            random_path = random.choice(img_paths)
            img_class = random_image_path.parent.stem
    """
    img_paths_list = list(image_path.glob(format))
    return img_paths_list
