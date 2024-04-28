import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from base_functions import get_best_model, predict_random_samples
from cfg import CFG


def evaluation_pipeline():
    print("1. Print Evaluations...")
    df = get_results_df()
    print("2. Loading Best Model...")
    model = get_best_model()
    print("3. Predicting Samples from Validation Set (conf=0)...")
    predict_random_samples(
        model=model,
        img_path=CFG.img_valid_path,
        conf=0,
        plot_title="Validation Set Predictions (conf=0)",
    )
    print("4. Predicting Samples from Validation Set (conf=0.5)")
    predict_random_samples(
        model=model,
        img_path=CFG.img_valid_path,
        plot_title="Validation Set Predictions (conf=0.5)",
    )


def get_results_df(
    post_training_path: str = CFG.post_train_files_path,
    plot_learning_curves: bool = True,
    plot_confusion_matrix: bool = True,
) -> pd.DataFrame:
    """
    Loads training results from a CSV file, optionally plots learning curves and confusion matrix, and returns the results as a DataFrame.

    Args:
        post_training_path (str, optional): Path to the directory containing training results. Defaults to "./runs/detect/train".
        plot_learning_curves (bool, optional): Whether to plot learning curves for losses and metrics. Defaults to True.
        plot_confusion_matrix (bool, optional): Whether to display the normalized confusion matrix. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the training results.
    """
    results_csv_path = os.path.join(post_training_path, "results.csv")

    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    if plot_learning_curves:
        plot_loss_learning_curve(
            df=df,
            train_loss_col="train/box_loss",
            val_loss_col="val/box_loss",
            plot_title="Box Loss Learning Curve",
        )
        plot_loss_learning_curve(
            df=df,
            train_loss_col="train/cls_loss",
            val_loss_col="val/cls_loss",
            plot_title="Classification Loss Learning Curve",
        )
        plot_loss_learning_curve(
            df=df,
            train_loss_col="train/dfl_loss",
            val_loss_col="val/dfl_loss",
            plot_title="Distribution Focal Loss Learning Curve",
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/precision(B)", plot_title="Metrics Precision (B)"
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/recall(B)", plot_title="Metrics Recall (B)"
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/mAP50(B)", plot_title="Metrics mAP50 (B)"
        )
        plot_metric_learning_curve(
            df=df, metric_col="metrics/mAP50-95(B)", plot_title="Metrics mAP50-95 (B)"
        )

    if plot_confusion_matrix:
        cm_path = os.path.join(post_training_path, "confusion_matrix_normalized.png")
        plot_norm_confusion_matrix(cm_path=cm_path)

    return df


def plot_loss_learning_curve(
    df: pd.DataFrame,
    train_loss_col: str,
    val_loss_col: str,
    plot_title: str,
    train_color: str = "#141140",
    train_linestyle: str = "-",
    valid_color: str = "orangered",
    valid_linestyle: str = "--",
    linewidth: int = 2,
) -> None:
    """
    Plots a learning curve for training and validation losses over epochs.

    Args:
        df (pd.DataFrame): DataFrame containing the loss data across epochs.
        train_loss_col (str): Column name for the training loss.
        val_loss_col (str): Column name for the validation loss.
        plot_title (str): Title of the plot.
        train_color (str, optional): Color for the training loss line. Defaults to "#141140".
        train_linestyle (str, optional): Line style for the training loss line. Defaults to "-".
        valid_color (str, optional): Color for the validation loss line. Defaults to "orangered".
        valid_linestyle (str, optional): Line style for the validation loss line. Defaults to "--".
        linewidth (int, optional): Width of the line. Defaults to 2.

    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df,
        x="epoch",
        y=train_loss_col,
        label="Train Loss",
        color=train_color,
        linestyle=train_linestyle,
        linewidth=linewidth,
    )
    sns.lineplot(
        data=df,
        x="epoch",
        y=val_loss_col,
        label="Validation Loss",
        color=valid_color,
        linestyle=valid_linestyle,
        linewidth=linewidth,
    )
    plt.title(plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_metric_learning_curve(
    df: pd.DataFrame,
    metric_col: str,
    plot_title: str,
    color: str = "#141140",
    linestyle: str = "-",
    linewidth: int = 2,
) -> None:
    """
    Plots a learning curve for a specific metric over epochs.

    Args:
        df (pd.DataFrame): DataFrame containing the metric data across epochs.
        metric_col (str): Column name for the metric to plot.
        plot_title (str): Title of the plot.
        color (str, optional): Color of the metric line. Defaults to "#141140".
        linestyle (str, optional): Line style of the metric line. Defaults to "-".
        linewidth (int, optional): Width of the line. Defaults to 2.

    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df,
        x="epoch",
        y=metric_col,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    plt.title(plot_title)
    plt.show()


def plot_norm_confusion_matrix(cm_path: str) -> None:
    """
    Displays a normalized confusion matrix from a specified file path.

    Args:
        cm_path (str): Path to the image file of the normalized confusion matrix.

    Returns:
        None
    """
    cm_img = cv2.imread(cm_path)
    cm_img = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm_img)
    plt.axis("off")
    plt.show()
