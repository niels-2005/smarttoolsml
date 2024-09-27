
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score


import itertools


def plot_classification_report_with_support(y_pred: np.ndarray, y_true: np.ndarray, classes: list, save_figure: bool = False):
    report = classification_report(
            y_pred=y_pred, y_true=y_true, output_dict=True, target_names=classes
        )
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
    if save_figure:
        plt.savefig("classification_report.png")
    plt.show()


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
    save_figure: bool = False,
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
    if save_figure:
        plt.savefig("confusion_matrix.png")
    plt.show()


def get_wrong_predictions_numeric(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list, is_binary: bool = True
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
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

    df_dict = {
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


def get_wrong_predictions_text(
    X_test: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, classes: list, is_binary: bool = True
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"):
        acc_score = accuracy_score(y_pred=y_pred, y_true=y_true, average=average)
        f1 = f1_score(y_pred=y_pred, y_true=y_true, average=average)
        precision = precision_score(y_pred=y_pred, y_true=y_true, average=average)
        recall = recall_score(y_pred=y_pred, y_true=y_true, average=average)

        df_dict = {
            f"accuracy_{average}": acc_score,
            f"f1-score_{average}": f1, 
            f"precision_{average}": precision,
            f"recall_{average}": recall
        }

        df_metrics = pd.DataFrame(df_dict)
        return df_metrics


def plot_roc_auc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_figure: bool = True):
    """
    Example usage:
        y_pred_proba = model.predict_proba(X_test)[:, 1] 
        y_true = [...]

        plot_roc_auc_curve(y_true=y_true, y_pred_proba=y_pred_proba)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba) 
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_figure:
        plt.savefig("roc_auc_curve.png")
    plt.show()
