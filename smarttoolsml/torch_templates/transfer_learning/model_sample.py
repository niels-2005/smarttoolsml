import torch
import torchvision
from torch import nn


def get_model(
    classes: list,
    freeze_layers: bool = True,
    seed: int = 42,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    if freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    output_shape = len(classes)

    # in_features = 1280 (because efficientnetb0)
    model.classifer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=1280, out_features=output_shape, bias=True).to(device),
    )

    return model
