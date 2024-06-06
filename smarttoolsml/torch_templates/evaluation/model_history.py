import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def compare_models(model_0_df: pd.DataFrame, model_1_df: pd.DataFrame):
    """_summary_

    Args:
        model_0_df (pd.DataFrame): _description_
        model_1_df (pd.DataFrame): _description_

    Example usage:
        model_0_results, model_0_df, model_0_df_results = train_with_callbacks()
        model_1_results, model_1_df, model_1_df_results = train_with_callbacks()

        or load csv file.

        compare_models(model_0_df_results, model_1_df_results)
    """
    plt.figure(figsize=(15, 10))
    # Get number of epochs
    epochs = range(len(model_0_df))

    # Plot train loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
    plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot test loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
    plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot train accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
    plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
    plt.title("Train Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot test accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
    plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
