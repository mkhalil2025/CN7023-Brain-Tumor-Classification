import torch
import torch.nn as nn
from torchvision import models


def create_resnet50_model(num_classes=4):
    """
    Creates a ResNet-50 model for transfer learning.

    Args:
        num_classes (int): The number of output classes for the new classifier head.

    Returns:
        torch.nn.Module: The modified ResNet-50 model.
    """
    # Load the pre-trained ResNet-50 model with the latest recommended weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all the parameters in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer (the 'classifier')
    # The new nn.Linear layer's parameters will have requires_grad=True by default
    model.fc = nn.Linear(num_ftrs, num_classes)

    print(f"ResNet-50 model adapted for {num_classes} classes.")
    print("All base layers are frozen. Only the final 'fc' layer will be trained.")

    return model


if __name__ == '__main__':
    # Example of how to create the model
    model = create_resnet50_model(num_classes=4)

    # Verify which parameters are trainable
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # Create a dummy input tensor to test the model
    # Note the input size (224, 224) which is standard for ImageNet models
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape for a dummy input: {output.shape}")  # Should be [1, 4]