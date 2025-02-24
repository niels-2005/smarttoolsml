import itertools
import os
import random
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import Model


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
    savefig: bool = False,
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
    if savefig:
        plt.savefig("confusion_matrix.png")
    plt.show()


def show_model_prediction_time(
    model: Model, samples: np.ndarray, figsize: tuple[int, int] = (10, 10)
) -> tuple[float, float]:
    """
    Times how long a model takes to make predictions on samples.

    Args:
        model: A trained model, capable of making predictions.
        samples: A batch of samples to predict on. Expected to be in the correct format for the model.
        figsize (tuple[int, int]): Size of output figure (default=(10, 10)).

    Returns:
        total_time (float): Total elapsed time for the model to make predictions on samples, in seconds.
        time_per_pred (float): Average time in seconds per single sample prediction.

    Example usage:
        total_time, time_per_pred = show_model_prediction_time(model, samples)
    """
    start_time = time.perf_counter()  # get start time
    model.predict(samples)  # make predictions
    end_time = time.perf_counter()  # get finish time
    total_time = end_time - start_time  # calculate how long predictions took to make
    time_per_pred = total_time / len(samples)  # find prediction time per sample

    plt.figure(figsize=figsize)
    plt.scatter(total_time, time_per_pred)
    plt.title("Time how long a model takes to make predictions on sample")
    plt.xlabel(f"Total time: {total_time:.5f} s")
    plt.ylabel(f"Time per prediction: {time_per_pred:.5f} s")
    plt.show()

    return total_time, time_per_pred


def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
    figsize: tuple[int, int] = (10, 10),
    color: str = "blue",
) -> dict:
    """
    This function uses the 'accuracy_score' and 'precision_recall_fscore_support' from scikit-learn to calculate
    the accuracy, precision, recall, and f1-score of a classification model, with metrics returned as percentages.
    It also plots these metrics in a bar chart, using Matplotlib, with customizable figure size and bar color.

    Args:
        y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
        y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
        average (str, optional): The strategy for averaging. Can be one of 'micro', 'macro', 'samples', 'weighted', or 'binary'.
                                 Defaults to 'weighted' which accounts for label imbalance by computing the average of binary metrics
                                 in which each class's score is weighted by its presence in the true data sample.
        figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
        color (str, optional): Color of the bars in the plot. Can be a single color format string, or a sequence of color
                               specifications of length equal to the number of bars. Accepts name of a color (e.g., 'blue', 'green'),
                               hex string (e.g., '#008000'), RGB tuple (e.g., (0,1,0)), or grayscale intensity (e.g., '0.5').
                               Defaults to 'blue'. Example colors: 'red', '#FFDD44', (0.1, 0.2, 0.5), '0.75'.

    Returns:
        dict[str, float]: A dictionary containing the accuracy, precision, recall, and f1-score, each as a percentage.

    Example usage:
        model_metrics = calculate_model_metrics(y_true, y_pred)
        model_metrics_custom_color = calculate_model_metrics(y_true, y_pred, color='green')
        model_metrics_rgb_color = calculate_model_metrics(y_true, y_pred, color=(0.5, 0.2, 0.8))
    """
    # Calculate model accuracy and convert to percentage
    model_accuracy = accuracy_score(y_true, y_pred)

    # Calculate model precision, recall, and f1 score using specified average method and convert to percentages
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )

    model_results = {
        "accuracy": model_accuracy * 100,
        "precision": model_precision * 100,
        "recall": model_recall * 100,
        "f1": model_f1 * 100,
    }

    plt.figure(figsize=figsize)
    plt.bar(model_results.keys(), model_results.values(), color=color)
    plt.title("Model Metrics")
    plt.xlabel("Metric Names")
    plt.ylabel("Metric Values in %")
    plt.show()

    return model_results


def calculate_classes_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> pd.DataFrame:
    """
    Calculates precision, recall, and f1-score for each class based on the true and predicted labels.

    This function uses the `classification_report` from scikit-learn to generate a report on precision,
    recall, and f1-score for each class. It then organizes this information into a pandas DataFrame
    for easier analysis and visualization.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels, same shape as y_true.
        classes (np.ndarray): Array of class labels as strings.

    Returns:
        pd.DataFrame: A DataFrame containing the class names, their corresponding f1-score, precision,
                      and recall. Each row corresponds to a class.

    Example usage:
        class_names = ['cats', 'dogs']

        calculate_classes_metrics(y_true, y_pred, class_names=class_names)
    """
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    # Delete bottom Metrics
    report.pop("accuracy", None)
    report.pop("macro avg", None)
    report.pop("weighted avg", None)

    # generate pandas dataframe
    df = pd.DataFrame.from_dict(report).transpose().reset_index()
    df.rename(columns={"index": "class_name"}, inplace=True)

    return df


def plot_metric_from_classes(
    df: pd.DataFrame,
    metric: str,
    df_class_name_column: str = "class_name",
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Plots a horizontal bar chart of given metric scores for different classes.

    This function takes a pandas DataFrame containing metrics for different classes,
    a metric name to plot, and the DataFrame column name that contains class names.
    It then plots a horizontal bar chart showing the metric scores for each class,
    sorted in ascending order. Additionally, it annotates each bar with the metric score.

    Args:
        df (pd.DataFrame): The DataFrame containing the metric scores and class names.
        metric (str): The name of the metric column in `df` to plot.
                      This metric will be displayed on the x-axis. (precision, recall, f1-score or support)
        df_class_name_column (str): The name of the column in `df` that contains the class names.
                                    These class names will be displayed on the y-axis.
        figsize (tuple[int, int]): A tuple specifying the width and height in inches of the figure to be plotted.
                                   This allows customization of the plot size for better readability and fitting into different contexts.

    Returns:
        None: This function does not return a value. It generates a plot.

    Example usage:
        plot_metric_from_classes(df, metric='f1-score', df_class_name_column='class names', figsize=(10, 10))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # sort df with ascending=True (necessary because ylabels wouldnt have the exact values)
    sorted_df = df.sort_values(by=[metric], ascending=True)

    # num_classes in range for y, x
    range_num_classes = range(len(sorted_df[df_class_name_column]))

    # create barh chart
    scores = ax.barh(range_num_classes, sorted_df[metric])
    ax.set_yticks(range_num_classes)
    ax.set_yticklabels(sorted_df[df_class_name_column])
    ax.set_xlabel(f"{metric}")
    ax.set_title(f"{metric} for Different Classes")

    # write to the right the metric score (%) for each class.
    for rect in scores:
        width = rect.get_width()
        ax.text(
            1.03 * width,
            rect.get_y() + rect.get_height() / 1.5,
            f"{width:.2f}",
            ha="center",
            va="center",
        )
