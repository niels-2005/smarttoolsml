import torch 
from tqdm.notebook import tqdm
import mlxtend 
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_y_pred_multilabel(model, test_loader):
    """_summary_

    Args:
        model (_type_): _description_
        test_loader (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        model = model()
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
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor


def plot_confusion_matrix(class_names: list, y_true: torch.Tensor, y_pred: torch.Tensor):
    """_summary_

    Args:
        class_names (list): _description_
        data (_type_): _description_
        y_pred (torch.Tensor): _description_

    Example usage:
        class_names = Dataset.classes 
        y_pred = get_y_pred_multilabel()
        y_true = ...
        plot_confusion_matrix(class_names=class_names, y_pred=y_pred, y_true=y_true)

    """
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_pred,
                            target=y_true)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=class_names, # turn the row and column labels into class names
        figsize=(10, 7)
    );