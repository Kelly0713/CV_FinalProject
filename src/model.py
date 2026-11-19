import torch
import torch.nn as nn
from torchvision import models

CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def build_model(
    arch: str,
    strategy: str = "full_fine_tuning",
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # 預設先凍結 backbone，若為 full_fine_tuning 會在 warmup 後解凍
    set_backbone_trainable(model, arch, trainable=False)
    return model


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool):
    arch = arch.lower()
    classifier = model.fc if arch == "resnet34" else model.classifier
    classifier_params = set(classifier.parameters())

    for param in model.parameters():
        if param in classifier_params:
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params