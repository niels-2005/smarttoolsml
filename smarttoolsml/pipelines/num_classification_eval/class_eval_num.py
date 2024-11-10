import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


def classification_evaluation_pipeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    metrics_average: str = "weighted",
    save_folder: str = "save_folder",
) -> None:
    """
    Evaluates the classification model by generating a comprehensive report including classification
    metrics and a confusion matrix. Optionally, it can also return a DataFrame containing incorrect predictions if specified.

    Args:
        X_test (pd.Series): The input text data used for testing the model.
        y_true (np.ndarray): True labels of the test data.
        y_pred (np.ndarray): Predicted labels as returned by the classifier.
        classes (list): List of class names for more interpretable visualizations.
        get_wrong_preds (bool, optional): Flag to determine if the function should return a DataFrame with wrong predictions. Defaults to False.

    Returns:
        None or (pd.DataFrame, pd.DataFrame): If get_wrong_preds is True, returns a tuple of two DataFrames:
            1. DataFrame of the test data, predictions, and true labels.
            2. DataFrame of incorrect predictions only.

    Example usage:
        y_pred = model.predict(X_test)
        classes = ["Class 0", "Class 1"]
        save_folder = "model_1"
        metrics_average = "weighted
        classification_evaluation_pipeline(y_true=y_test, y_pred=y_pred, classes=classes, save_folder=save_folder, metrics_average=metrics_average)
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Printing Classification Report")
    print(classification_report(y_pred=y_pred, y_true=y_true, target_names=classes))

    print("Plotting Classification Report with Support")
    report = classification_report(
        y_pred=y_pred, y_true=y_true, output_dict=True, target_names=classes
    )
    plot_classification_report_with_support(report=report, save_folder=save_folder)
    # save report as csv
    report_df = pd.DataFrame(report)
    report_df.to_csv(f"{save_folder}/classification_report.csv")

    print("Plotting Confusion Matrix")
    make_confusion_matrix(
        y_true=y_true, y_pred=y_pred, classes=classes, save_folder=save_folder
    )

    # save model metrics as csv
    print("Calculating Accuracy, F1-Score, Recall, Precision")
    df_metrics = calculate_metrics(
        y_pred=y_pred, y_true=y_true, average=metrics_average
    )
    df_metrics.to_csv(f"{save_folder}/model_metrics_{metrics_average}.csv")


def plot_classification_report_with_support(report: dict, save_folder: str):
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    metrics = ["precision", "recall", "f1-score", "support"]
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.matshow(data, cmap="coolwarm")
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    # Adding the text
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.title("Classification Report with Support")
    plt.savefig(f"{save_folder}/classification_report.png")
    plt.show()


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_folder: str,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
) -> None:
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels, with options to normalize
    and save the figure.

    Args:
      y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
      y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
      classes (np.ndarray): Array of class labels (e.g., string form). If `None`, integer labels are used.
      figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
      text_size (int): Size of output figure text (default=15).
      norm (bool): If True, normalize the values in the confusion matrix (default=False).
      savefig (bool): If True, save the confusion matrix plot to the current working directory (default=False).

    Returns:
        None: This function does not return a value but displays a Confusion Matrix. Optionally, it saves the plot.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10,
                            norm=True,
                            savefig=True)
    """
    # Create the confusion matrix
    cm = (
        confusion_matrix(y_true, y_pred, normalize="true")
        if norm
        else confusion_matrix(y_true, y_pred)
    )

    # Plot the figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    # Set class labels
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(len(cm))

    # Set the labels and titles
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Annotate the cells with the appropriate values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )

    plt.tight_layout()
    # Save the figure if requested
    plt.savefig(f"{save_folder}/confusion_matrix.png")
    plt.show()


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
):
    # Berechnung der Metriken
    acc_score = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average=average) * 100
    precision = precision_score(y_true, y_pred, average=average) * 100
    recall = recall_score(y_true, y_pred, average=average) * 100

    df_dict = {
        f"accuracy": [round(acc_score, 2)],
        f"f1-score_{average}": [round(f1, 2)],
        f"precision_{average}": [round(precision, 2)],
        f"recall_{average}": [round(recall, 2)],
    }

    df_metrics = pd.DataFrame(df_dict)
    return df_metrics
