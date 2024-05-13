import random

import matplotlib.pyplot as plt
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_random_samples(dataloader, k: int = 10):
    samples = []
    labels = []
    for sample, label in random.sample(list(dataloader), k=k):
        samples.append(sample)
        labels.append(label)
    return samples, labels


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)


def plot_prediction_dataloader(
    model,
    dataloader,
    classes: list,
    is_categorical: bool = True,
    n_rows: int = 3,
    n_cols: int = 3,
    figsize: tuple[int, int] = (9, 9),
):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        classes (list): _description_
        is_categorical (bool, optional): _description_. Defaults to True.
        n_rows (int, optional): _description_. Defaults to 3.
        n_cols (int, optional): _description_. Defaults to 3.
        figsize (tuple[int, int], optional): _description_. Defaults to (9, 9).

    Example usage:
        model = Model()
        dataloader = get_dataloader()
        classes = ["Class 1", "Class 2"]
        is_categorical = True (Multiclass Classification)

        plot_prediction_dataloader(model=model, dataloader=dataloader, classes=classes, is_categorical=is_categorical)
    """
    samples, labels = get_random_samples(dataloader=dataloader)
    pred_probs = make_predictions(model=model, data=samples)
    if is_categorical:
        pred_classes = pred_probs.argmax(dim=1)
    else:
        pred_classes = torch.sigmoid(pred_probs) >= 0.5

    plt.figure(figsize=figsize)
    for i, sample in enumerate(samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(sample.squeeze())
        pred_label = classes[pred_classes[i]]
        true_label = classes[labels[i]]
        plot_title = f"Pred: {pred_label} | True: {true_label}"
        if pred_label == true_label:
            plt.title(plot_title, c="g")
        else:
            plt.title(plot_title, c="r")
        plt.axis("off")
    plt.show()
