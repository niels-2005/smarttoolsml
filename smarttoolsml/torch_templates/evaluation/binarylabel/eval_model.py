import os

import matplotlib.pyplot as plt
import pandas as pd
import torch


def accuracy_fn(y_true, y_logits):
    """Calculates accuracy for binary classification.

    Args:
        y_true (torch.Tensor): True labels for predictions.
        y_logits (torch.Tensor): Logits predicted by the model.

    Returns:
        float: Accuracy value between y_true and y_pred, e.g., 78.45
    """
    y_pred = (torch.sigmoid(y_logits) >= 0.5).float()
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def eval_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    df: pd.DataFrame,
    model_folder,
    device,
):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on dataloader.
        dataloader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of the model.
        df (pd.DataFrame): DataFrame to store the evaluation results.
        model_folder (str): Path to the folder where the evaluation results should be saved.
        device (torch.device, optional): Target device to compute on. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        dict: Results of model making predictions on dataloader.

    Example usage:
        model = Model()
        dataloader = get_dataloader()
        loss_fn = nn.BCEWithLogitsLoss()
        df = pd.DataFrame()

        eval_model(model=model, dataloader=dataloader, loss_fn=loss_fn, df=df, model_folder="path/to/save")
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to target device
            X, y = X.to(device), y.to(device).unsqueeze(1).float()
            y_pred = model(X)
            loss += loss_fn(y_pred, y).item()
            acc += accuracy_fn(y_true=y, y_logits=y_pred)

        # Scale loss and accuracy
        loss /= len(dataloader)
        acc /= len(dataloader)

        df["test_loss"] = loss
        df["test_acc"] = acc

        csv_path = os.path.join(model_folder, "model_stats.csv")
        df.to_csv(csv_path, index=False)


def plot_loss_curves(results: dict[str, list[float]], model_folder):
    """Plots training curves of a results dictionary and saves the plot.

    Args:
        results (dict): Dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}.
        model_folder (str): Path to the folder where the plot should be saved.

    Example usage:
        results = {
            "train_loss": [0.6, 0.4, 0.3],
            "train_acc": [60, 70, 80],
            "val_loss": [0.5, 0.35, 0.25],
            "val_acc": [65, 75, 85]
        }
        plot_loss_curves(results)
    """
    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["val_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"]

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

    save_path = os.path.join(model_folder, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
