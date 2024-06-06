import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


def accuracy_fn(y_true, y_logits):
    """Calculates accuracy for binary classification.

    Args:
        y_true (torch.Tensor): True labels for predictions.
        y_logits (torch.Tensor): Logits predicted by the model.

    Returns:
        float: Accuracy value between y_true and y_pred, e.g., 78.45.
    """
    y_pred = torch.sigmoid(y_logits) >= 0.5
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def get_current_lr(optimizer):
    """Retrieve the current learning rate from optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to retrieve the learning rate.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_step(model, dataloader, loss_fn, optimizer, device):
    """Performs a single training step.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        loss_fn (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: Training loss and accuracy.
    """
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).unsqueeze(1).float()

        # Forward pass
        y_logits = model(X)

        # Calculate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_logits)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def val_step(dataloader, model, loss_fn, device):
    """Performs a single validation step.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        model (torch.nn.Module): The model to validate.
        loss_fn (torch.nn.Module): The loss function.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: Validation loss and accuracy.
    """
    val_loss, val_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1).float()
            y_logits = model(X)
            val_loss += loss_fn(y_logits, y).item()
            val_acc += accuracy_fn(y, y_logits)

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    early_stopper,
    lr_scheduler,
    device,
    return_df: bool = True,
    epochs: int = 5,
):
    """Trains a model with callbacks for early stopping and learning rate scheduling.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn (torch.nn.Module): The loss function.
        early_stopper: An instance of EarlyStopping for early stopping.
        lr_scheduler: Learning rate scheduler.
        device (torch.device): The device to run the model on.
        return_df (bool, optional): Whether to return the results as a DataFrame. Defaults to True.
        epochs (int, optional): The number of epochs to train. Defaults to 5.

    Returns:
        tuple: Results dictionary and optionally a DataFrame if return_df is True.

    Example usage:
        model = YourModel()
        train_dataloader = get_dataloader(train_dataset)
        val_dataloader = get_dataloader(val_dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        early_stopper = EarlyStopping(patience=8, verbose=True, path="best_model.pt")
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4, verbose=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epochs = 10

        results, df = train_model(
            model, train_dataloader, val_dataloader, optimizer, loss_fn, early_stopper, lr_scheduler, device, epochs=epochs
        )
    """
    # 2. Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    initial_lr = get_current_lr(optimizer)

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        current_lr = get_current_lr(optimizer)

        # 4. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["lr"].append(current_lr)

        # 5. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"lr: {current_lr}"
        )

        # 6. Update Scheduler
        lr_scheduler.step(val_loss)

        # 7. Check Early Stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            early_stopped_epoch = epoch + 1
            print(f"Early Stopping at Epoch: {epoch+1}")
            break

    early_stopper.load_best_weights(model)

    if return_df:
        df_results = {
            "Model": model.__class__.__name__,
            "Best Model Path": early_stopper.path,
            "Original Epochs": epochs,
            "Early Stopped Epoch": early_stopped_epoch,
            "Optimizer": optimizer.__class__.__name__,
            "Loss Function": loss_fn.__class__.__name__,
            "initial_lr": initial_lr,
            "final_lr": min(results["lr"]),
            "train_loss": min(results["train_loss"]),
            "train_acc": max(results["train_acc"]),
            "val_loss": min(results["val_loss"]),
            "val_acc": max(results["val_acc"]),
        }
        df = pd.DataFrame([df_results])
        df_results_all = pd.DataFrame([results])
        return results, df, df_results_all
    else:
        return results


def train_model_with_tensorboard(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    early_stopper,
    lr_scheduler,
    device,
    return_df: bool = True,
    epochs: int = 5,
):
    """Trains a model with callbacks for early stopping and learning rate scheduling.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn (torch.nn.Module): The loss function.
        early_stopper: An instance of EarlyStopping for early stopping.
        lr_scheduler: Learning rate scheduler.
        device (torch.device): The device to run the model on.
        return_df (bool, optional): Whether to return the results as a DataFrame. Defaults to True.
        epochs (int, optional): The number of epochs to train. Defaults to 5.

    Returns:
        tuple: Results dictionary and optionally a DataFrame if return_df is True.

    Example usage:
        model = YourModel()
        train_dataloader = get_dataloader(train_dataset)
        val_dataloader = get_dataloader(val_dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        early_stopper = EarlyStopping(patience=8, verbose=True, path="best_model.pt")
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4, verbose=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epochs = 10

        results, df = train_model(
            model, train_dataloader, val_dataloader, optimizer, loss_fn, early_stopper, lr_scheduler, device, epochs=epochs
        )
    """
    # 2. Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # Writer for Tensorboard
    writer = SummaryWriter()

    # Initial Learning Rate for Tracking
    initial_lr = get_current_lr(optimizer)

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        # get current learning rate
        current_lr = get_current_lr(optimizer)

        # 4. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["lr"].append(current_lr)

        # 5. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"lr: {current_lr}"
        )

        # Tensorboard Tracking
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
            global_step=epoch,
        )

        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train_acc": train_acc, "val_acc": val_acc},
            global_step=epoch,
        )

        # (BATCH, COLOR_CHANNELS, WIDTH, HEIGHT)
        writer.add_graph(
            model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device)
        )

        # 6. Update Scheduler
        lr_scheduler.step(val_loss)

        # 7. Check Early Stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            early_stopped_epoch = epoch + 1
            print(f"Early Stopping at Epoch: {epoch+1}")
            break

    writer.close()

    early_stopper.load_best_weights(model)

    if return_df:
        df_results = {
            "Model": model.__class__.__name__,
            "Best Model Path": early_stopper.path,
            "Original Epochs": epochs,
            "Early Stopped Epoch": early_stopped_epoch,
            "Optimizer": optimizer.__class__.__name__,
            "Loss Function": loss_fn.__class__.__name__,
            "initial_lr": initial_lr,
            "final_lr": min(results["lr"]),
            "train_loss": min(results["train_loss"]),
            "train_acc": max(results["train_acc"]),
            "val_loss": min(results["val_loss"]),
            "val_acc": max(results["val_acc"]),
        }
        df = pd.DataFrame([df_results])
        df_results_all = pd.DataFrame([results])
        return results, df, df_results_all
    else:
        return results
