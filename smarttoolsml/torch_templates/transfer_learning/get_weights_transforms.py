import torch
import torchvision


def get_model_transforms():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms
