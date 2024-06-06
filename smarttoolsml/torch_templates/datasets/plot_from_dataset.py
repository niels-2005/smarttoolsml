import matplotlib.pyplot as plt


def plot_images_from_dataloader(
    dataloader,
    class_names: list,
    n_images: int = 16,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 10,
) -> None:
    """
    Displays a specified number of images from a PyTorch DataLoader in a grid layout, annotating each image with its class name.

    Args:
        dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader containing batches of images and labels.
        class_names (list): A list of class names corresponding to the labels in the dataset.
        n_images (int, optional): The total number of images to be displayed. Defaults to 16.
        images_per_row (int, optional): The number of images displayed per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure used to display the images. The height is automatically adjusted based on the number of rows. Defaults to 20.
        fontsize (int, optional): The font size used for the annotations on each subplot. Defaults to 10.
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    ax = ax.flatten()

    for images, labels in dataloader:
        images = images.numpy()
        labels = labels.numpy()
        for i in range(min(n_images, len(images))):
            ax[i].imshow(images[i].transpose((1, 2, 0)))
            ax[i].axis("off")
            ax[i].set_title(
                f"{class_names[labels[i]]}, {images[i].shape}", fontsize=fontsize
            )
        break

    for j in range(i + 1, n_rows * n_cols):
        ax[j].axis("off")
    fig.suptitle(f"Images from DataLoader")
    plt.show()
