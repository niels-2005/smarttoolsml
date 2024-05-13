import torch
from tqdm.notebook import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_logits):
    """Calculates accuracy for binary classification."""
    y_pred = (
        torch.sigmoid(y_logits) >= 0.5
    ) 
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def get_current_lr(optimizer):
    """Retrieve the current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_step(model, dataloader, loss_fn, optimizer, device):
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


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int = 5,
):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        val_dataloader (torch.utils.data.DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss_fn (torch.nn.Module): _description_
        epochs (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_

    Example usage:
        model = Model()
        train_dataloader = get_dataloader()
        val_dataloader = get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        epochs = 10
        
        results = train(model, train_dataloder, val_dataloader, optimizer, loss_fn, epochs)
    """
    # 1. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 2. Loop through training and testing steps for a number of epochs
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

        # 3. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss)
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
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss)
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
