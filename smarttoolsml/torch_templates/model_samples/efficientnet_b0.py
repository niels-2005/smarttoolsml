import timm
import torch
from torch import nn


def print_efficientnet_resolutions():
    print("EfficientNetB0: (224, 224)")
    print("EfficientNetB1: (240, 240)")
    print("EfficientNetB2: (260, 260)")
    print("EfficientNetB3: (300, 300)")
    print("EfficientNetB4: (380, 380)")
    print("EfficientNetB5: (456, 456)")
    print("EfficientNetB6: (528, 528)")
    print("EfficientNetB7: (600, 600)")
    print("https://keras.io/api/applications/")


class EfficientNetClassifier(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_

    Example usage:
        model_b0 = EfficientNetClassifier(version='b0', num_classes=53, dropout_rate=0.5)
        model_b1 = EfficientNetClassifier(version='b1', num_classes=53, dropout_rate=0.5)
        model_b2 = EfficientNetClassifier(version='b2', num_classes=53, dropout_rate=0.5)
        model_b3 = EfficientNetClassifier(version='b3', num_classes=53, dropout_rate=0.5)
        model_b4 = EfficientNetClassifier(version='b4', num_classes=53, dropout_rate=0.5)
        model_b5 = EfficientNetClassifier(version='b5', num_classes=53, dropout_rate=0.5)
        model_b6 = EfficientNetClassifier(version='b6', num_classes=53, dropout_rate=0.5)
        model_b7 = EfficientNetClassifier(version='b7', num_classes=53, dropout_rate=0.5)

    """

    def __init__(self, version="b0", num_classes=53, dropout_rate=0.5):
        super(EfficientNetClassifier, self).__init__()
        model_name = f"efficientnet_{version}"
        self.base_model = timm.create_model(model_name, pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Output-Größen entsprechend der spezifischen EfficientNet-Version
        feature_sizes = {
            "b0": 1280,
            "b1": 1280,
            "b2": 1408,
            "b3": 1536,
            "b4": 1792,
            "b5": 2048,
            "b6": 2304,
            "b7": 2560,
        }

        enet_out_size = feature_sizes[version]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# basic
class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# with one more dense and relu activation
class SimpleCardClassifierV2(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifierV2, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Erweiterte Klassifizierer-Architektur
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# with dropout
class SimpleCardClassifierV3(nn.Module):
    def __init__(self, num_classes=53, dropout_rate=0.5):
        super(SimpleCardClassifierV3, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
