import torch


def accuracy_fn(y_true, y_logits):
    """Calculates accuracy for binary classification."""
    y_pred = (
        torch.sigmoid(y_logits) >= 0.5
    )  # Umwandeln von Logits zu binären Klassenlabels
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(model, dataloader, loss_fn, optimizer, device):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        model = Model()
        dataloader = get_dataloader()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_loss, train_acc = train_step(model, dataloader, loss_fn, optimizer)
    """
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)

        # Calculate loss
        loss = loss_fn(y_logits, y.float())
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
    """_summary_

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        loss_fn (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        dataloader = get_dataloader()
        model = Model()
        loss_fn = nn.BCEWithLogitsLoss()

        val_loss, val_acc = val_step(dataloader, model, loss_fn)
    """
    val_loss, val_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            val_loss += loss_fn(
                y_logits, y.float()
            ).item()  # y.float() für BCEWithLogitsLoss
            val_acc += accuracy_fn(y, y_logits)

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc
