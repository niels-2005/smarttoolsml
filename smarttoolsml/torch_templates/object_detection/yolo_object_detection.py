import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor


def plot_image_object_detection(
    image_path: str,
    threshold: float = 0.9,
    preset_name: str = "hustvl/yolos-tiny",
    fontsize: int = 12,
    linewidth: int = 2,
    linecolor: str = "r",
    alpha: float = 0.5,
    figsize: tuple[int, int] = (15, 20),
) -> None:
    """
    Plots an image with object detection annotations using a pre-trained YOLOS model.

    Args:
        image_path (str): Path to the image to be analyzed.
        threshold (float, optional): Confidence threshold for displaying detected objects. Only objects with a confidence score above this threshold will be shown. Defaults to 0.9.
        preset_name (str, optional): The name of the pre-trained model to be loaded from Hugging Face's model repository. Defaults to 'hustvl/yolos-tiny'.
        fontsize (int, optional): Font size for the labels displayed on the bounding boxes. Defaults to 12.
        linewidth (int, optional): Line width for the bounding boxes. Defaults to 2.
        linecolor (str, optional): Color for the bounding box lines. Defaults to 'r' for red.
        alpha (float, optional): Transparency for the background of the label text. Defaults to 0.5.
        figsize (tuple[int, int], optional): Size of the figure that displays the images. Defaults to (15, 20).

    Returns:
        None. The function directly displays the images using matplotlib.

    Example:
        plot_image_object_detection('path/to/image.jpg',
                                    threshold=0.9,
                                    preset_name='hustvl/yolos-tiny',
                                    fontsize=12,
                                    linewidth=2,
                                    linecolor='r',
                                    alpha=0.5,
                                    figsize=(15, 20))
    """

    image = Image.open(image_path)

    model = YolosForObjectDetection.from_pretrained(preset_name)
    image_processor = YolosImageProcessor.from_pretrained(preset_name)

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Original Image")

    ax[1].imshow(image)
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        # draw Bounding Boxes
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=linewidth,
            edgecolor=linecolor,
            facecolor="none",
        )
        ax[1].add_patch(rect)

        # add Text to Bounding Boxes
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        ax[1].text(
            box[0],
            box[1] - 5,
            label_text,
            color="white",
            fontsize=fontsize,
            bbox=dict(facecolor=linecolor, alpha=alpha),
        )

    ax[1].axis("off")
    ax[1].set_title("Image with Object Detection")

    plt.show()
