import torch.nn as nn
from torchvision import models

# ------------------------
# ResNet18
# ------------------------
def get_resnet18(num_classes=37, pretrained=True):
    """
    Returns a ResNet18 model with a custom classifier head.
    
    Parameters:
    - num_classes: Number of output classes (37 for Galaxy Zoo)
    - pretrained: Whether to use ImageNet pretrained weights
    
    By default, all layers are trainable. To freeze the backbone, uncomment the block below.
    """
    # Load ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    
    # ------------------------
    # Optional: Freeze backbone
    # ------------------------
    # for param in model.parameters():
    #     param.requires_grad = False
    # Only uncomment to train only the classifier head
    
    # Replace final FC layer with custom classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
        # nn.Sigmoid()  # Uncomment if you want outputs between 0 and 1
    )
    
    return model
