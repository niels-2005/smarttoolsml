import pickle

import matplotlib.pyplot as plt


def save_history(history: dict, filename: str) -> None:
    """Saves the training history of a model to a pickle file.

    Args:
        history (History): A Model History object containing the training history information.
        filename (str): The base name of the file in which to save the training history.
                        The function automatically appends the '.pkl' extension.

    Returns:
        None

    Example usage:
        save_history(history, filename='model_history')
    """
    history_dict = history.history
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(history_dict, f)


def load_history(filename: str) -> dict:
    """Loads the training history of a model from a pickle file.

    Args:
        filename (str): The base name of the file from which to load the training history.
                        The file is expected to have the '.pkl' extension.

    Returns:
        dict: A dictionary containing the training history.

    Example usage:
        load_history(filename='model_history')
    """
    with open(f"{filename}.pkl", "rb") as f:
        history_dict = pickle.load(f)
    return history_dict


def plot_finetune_comparison(
    before_ft_history: dict,
    ft_history: dict,
    initial_epochs: int,
    figsize: tuple[int, int],
) -> None:
    """
    Compares two model training history dictionaries to visualize the training and validation accuracy and loss
    before and after fine-tuning. This function plots the accuracy and loss for training and validation sets
    across epochs and marks the point where fine-tuning begins to highlight the improvement or degradation.

    Args:
        before_ft_history (dict): A dictionary containing the training history of the model before fine-tuning.
                                 Expected to have keys 'accuracy', 'loss', 'val_accuracy', and 'val_loss'.
        ft_history (dict): A dictionary containing the training history of the model after fine-tuning.
                            Expected to have keys 'accuracy', 'loss', 'val_accuracy', and 'val_loss'.
        initial_epochs (int): The epoch at which fine-tuning started. This is used to mark the transition
                              point in the plots.
        figsize (tuple[int, int]): A tuple specifying the width and height in inches of the figure to be plotted.
                                   This allows customization of the plot size for better readability and fitting into different contexts.

    Returns:
        None: This function does not return a value but displays a matplotlib plot.

    Example usage:
        plot_finetune_comparison(feature_ext_history, fine_tune_history, initial_epochs=10, figsize=(10, 10))
    """
    # Get original history measurements directly from the dictionaries
    acc = before_ft_history["accuracy"]
    loss = before_ft_history["loss"]

    val_acc = before_ft_history["val_accuracy"]
    val_loss = before_ft_history["val_loss"]

    # Combine original history with new history
    total_acc = acc + ft_history["accuracy"]
    total_loss = loss + ft_history["loss"]

    total_val_acc = val_acc + ft_history["val_accuracy"]
    total_val_loss = val_loss + ft_history["val_loss"]

    # Make plots
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )  # reshift plot around epochs
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )  # reshift plot around epochs
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()


def plot_training_history(history: dict, figsize: tuple[int, int]) -> None:
    """
    Plots the loss and accuracy curves for training and validation in a vertical layout.

    This function accepts a model history dictionary and plots the training and validation loss,
    as well as the accuracy over the epochs. It creates two separate plots: one for the loss and
    another for the accuracy, allowing for a clear visualization of the model's performance over time.

    Args:
        history (dict): A model history dictionary containing the history of training/validation loss and accuracy,
                        recorded at the end of each epoch. Expected keys are 'loss', 'val_loss', 'accuracy', and 'val_accuracy'.
        figsize (tuple[int, int]): A tuple specifying the width and height in inches of the figure to be plotted.
                                   This allows customization of the plot size for better readability and fitting into different contexts.

    Returns:
        None: This function does not return any value. It generates and displays matplotlib plots, visualizing the
              training and validation loss and accuracy over epochs.

    Example usage:
        plot_training_history(history, figsize=(10, 10))
    """
    loss = history["loss"]
    val_loss = history["val_loss"]

    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    epochs = range(1, len(history["loss"]) + 1)  # Start epochs at 1

    # Plotting setup for a vertical layout
    plt.figure(figsize=figsize)  # Use provided figure size

    # Plot loss
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot = loss
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot = accuracy
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()  # Adjust layout to not overlap
    plt.show()
