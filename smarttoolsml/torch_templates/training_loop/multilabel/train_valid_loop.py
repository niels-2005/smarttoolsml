import torch
from tqdm.notebook import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def get_current_lr(optimizer):
    """Retrieve the current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = device,
):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        loss_fn (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        device (torch.device, optional): _description_. Defaults to device.

    Returns:
        _type_: _description_

    Example usage:
        model = Model()
        dataloader = get_dataloader()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        train_loss, train_acc = train_step(model, dataloader, loss_fn, optimizer)
    """
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y, y_pred=y_pred.argmax(dim=1)
        )  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def val_step(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device = device,
):
    """_summary_

    Args:
        dataloader (torch.utils.data.DataLoader): _description_
        model (torch.nn.Module): _description_
        loss_fn (torch.nn.Module): _description_
        device (torch.device, optional): _description_. Defaults to device.

    Returns:
        _type_: _description_

    Example usage:
        dataloader = get_dataloader()
        model = Model()
        loss_fn = nn.CrossEntropyLoss()

        val_loss, val_acc = val_step(dataloader, model, loss_fn)
    """
    val_loss, val_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            val_loss += loss_fn(test_pred, y)
            val_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),  # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
        return val_loss, val_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int = 5,
):

    # 1. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 2. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn
        )

        # 3. Update results dictionary
        results["train_loss"].append(train_loss.detach().cpu().item())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss.detach().cpu().item())
        results["test_acc"].append(val_acc)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            f"test_acc: {val_acc:.4f}"
        )

    # 5. Return the filled results at the end of the epochs
    return results


def train_with_callbacks(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    early_stopper,
    lr_scheduler,
    epochs: int = 5,
):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        val_dataloader (torch.utils.data.DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss_fn (torch.nn.Module): _description_
        early_stopper (_type_): _description_
        lr_scheduler (_type_): _description_
        epochs (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_

    Example usage:
        model = Model()
        train_dataloader = get_dataloader()
        val_dataloader = get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        early_stopper = EarlyStopping(patience=8, verbose=True, path="best_model.pt")
        lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=4, verbose=True)
        epochs = 10

        train_with_callbacks(model, train_dataloader, val_dataloader, optimizer, loss_fn, early_stopper, lr_scheduler, epochs)
    """
    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        # 4. Update results dictionary
        results["train_loss"].append(train_loss.detach().cpu().item())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss.detach().cpu().item())
        results["test_acc"].append(val_acc)

        current_lr = get_current_lr(optimizer)

        # 5. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            f"test_acc: {val_acc:.4f} | "
            f"lr: {current_lr}"
        )

        # 6. Update Scheduler
        lr_scheduler.step(val_loss)

        # 7. Check Early Stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early Stopping at Epoch: {epoch+1}")
            break

    # 6. Return the filled results at the end of the epochs
    return results
