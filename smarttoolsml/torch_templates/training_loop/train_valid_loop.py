import torch 
from torch import nn 
from torch import optim 
from tqdm.notebook import tqdm 


def training_valid_loop(model, train_loader, val_loader):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_

    Example usage:
        model = SimpleCardClassifer(num_classes=53)

        # smarttoolsml.torch_templates.datasets
        train_loader = get_dataloader()
        val_loader = get_dataloader()
        
        training_valid_loop(model=model, train_loader=train_loader, val_loader=val_loader)
    """
    num_epochs = 5
    train_losses, val_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")