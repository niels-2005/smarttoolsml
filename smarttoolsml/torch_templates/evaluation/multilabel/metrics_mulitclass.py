import mlxtend
import torch
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_y_pred_multilabel(model, test_loader):
    """Generates predictions for a multilabel classification model on a given test dataset.

    Args:
        model (torch.nn.Module): A PyTorch model for multilabel classification.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.

    Returns:
        torch.Tensor: A tensor containing the predicted labels for the test dataset.

    Example usage:
        model = YourMultilabelModel()
        test_loader = get_dataloader()
        y_pred = get_y_pred_multilabel(model=model, test_loader=test_loader)
    """
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_loader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(
                dim=1
            )  # Note: perform softmax on the "logits" dimension, not "batch" dimension
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor


def get_y_true(dataset):
    """Extracts true labels from a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the true labels.

    Returns:
        torch.Tensor: A tensor containing the true labels.

    Example usage:
        test_dataset = ImageDataset()
        y_true = get_y_true(dataset=test_dataset)
    """
    y_true = [label for _, label in dataset]
    y_true_tensor = torch.tensor(y_true)
    return y_true_tensor


def plot_confusion_matrix(class_names: list, model, dataloader, dataset):
    """Plots a confusion matrix for a given model and dataset.

    Args:
        class_names (list): List of class names for the classification task.
        model (torch.nn.Module): A PyTorch model to generate predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        dataset (torch.utils.data.Dataset): The test dataset containing the true labels.

    Example usage:
        classes = Dataset.classes
        model = YourMultilabelModel()
        dataloader = get_dataloader()
        dataset = ImageDataset()

        plot_confusion_matrix(class_names=classes, model=model, dataloader=dataloader, dataset=dataset)
    """
    y_pred = get_y_pred_multilabel(model=model, test_loader=dataloader)
    y_true = get_y_true(dataset=dataset)

    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=y_pred, target=y_true)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=class_names,  # Turn the row and column labels into class names
        figsize=(10, 7),
    )
    return fig, ax
