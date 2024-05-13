import torch
from tqdm.notebook import tqdm
from train_val_step import *

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn
        )

        # 4. Update results dictionary
        results["train_loss"].append(train_loss.detach().cpu().item())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss.detach().cpu().item())
        results["test_acc"].append(val_acc)

        # 5. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            f"test_acc: {val_acc:.4f}"
        )

        # 6. Update Scheduler
        lr_scheduler.step(val_loss)

        # 7. Check Early Stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early Stopping at Epoch: {epoch+1}")

    # 6. Return the filled results at the end of the epochs
    return results
