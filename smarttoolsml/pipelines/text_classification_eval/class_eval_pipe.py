import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_evaluation_pipeline(
    X_test: pd.Series,
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
        classification_evaluation_pipeline(X_test=X_test, y_true=y_test, y_pred=y_pred, y_prob=y_prob, classes=classes)

        # with getting Wrong Predictions
        df, wrong_preds = classification_evaluation_pipeline(X_test=X_test, y_true=y_test, y_pred=y_pred, classes=classes, get_wrong_preds=True)
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
    report_df = pd.DataFrame(report)
    report_df.to_csv(f"{save_folder}/classification_report.csv")

    print("Plot Confusion Matrix")
    make_confusion_matrix(
        y_true=y_true, y_pred=y_pred, classes=classes, save_folder=save_folder
    )

    print("Getting wrong Predictions.")
    df_predictions, wrong_preds = get_wrong_predictions(
        X_test=X_test, y_pred=y_pred, y_true=y_true, classes=classes
    )
    df_predictions.to_csv(f"{save_folder}/all_model_predictions.csv")
    wrong_preds.to_csv(f"{save_folder}/model_wrong_predictions.csv")

    print("Calculating Accuracy, F1-Score, Precision and Recall")
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
    plt.savefig(f"{save_folder}/classification_report.png")
    plt.show()


def get_wrong_predictions(
    X_test: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    is_binary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies and returns the correct and incorrect predictions made by a classification model.
    The function creates a DataFrame that includes the test inputs, actual and predicted labels, and class names.
    It also visualizes the distribution of correct and incorrect predictions.

    Args:
        X_test (pd.Series): The input text data that was used for testing the model, used here to trace back incorrect predictions to the original inputs.
        y_true (np.ndarray): The actual labels from the test data, representing the true classes of the inputs.
        y_pred (np.ndarray): The predicted labels produced by the classification model, used to compare against the true labels to determine prediction correctness.
        classes (list): A list of class names corresponding to the label indices, used to convert label indices into human-readable class names for easier interpretation and visualization.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. The first DataFrame includes all predictions with columns for the text, actual and predicted labels, and whether each prediction was correct.
            2. The second DataFrame is a subset of the first and includes only the rows where the predictions were incorrect.

    The function also plots a count plot showing the balance between correct and incorrect predictions across predicted class labels.
    """

    if is_binary:
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

    df_dict = {
        "text": X_test.values,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_classnames": [classes[i] for i in y_true],
        "y_pred_classnames": [classes[i] for i in y_pred],
    }

    df_pred = pd.DataFrame(df_dict).reset_index(drop=True)
    df_pred["pred_correct"] = df_pred["y_true"] == df_pred["y_pred"]

    plt.figure(figsize=(8, 4))
    sns.countplot(x="pred_correct", hue="y_pred_classnames", data=df_pred)
    plt.title("Balance between Predictions")
    plt.show()

    wrong_preds = df_pred[df_pred["pred_correct"] == False].reset_index(drop=True)
    return df_pred, wrong_preds


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
):
    acc_score = accuracy_score(y_pred=y_pred, y_true=y_true, average=average)
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average=average)
    precision = precision_score(y_pred=y_pred, y_true=y_true, average=average)
    recall = recall_score(y_pred=y_pred, y_true=y_true, average=average)

    df_dict = {
        f"accuracy_{average}": acc_score,
        f"f1-score_{average}": f1,
        f"precision_{average}": precision,
        f"recall_{average}": recall,
    }

    df_metrics = pd.DataFrame(df_dict)
    return df_metrics
