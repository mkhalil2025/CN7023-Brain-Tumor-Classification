import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """A custom Convolutional Neural Network for image classification."""

    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- THIS IS THE FIX ---
        # The input features depend on the output size of the last conv/pool layer.
        # Input image: 200x200
        # After pool 1: 100x100
        # After pool 2: 50x50
        # After pool 3: 25x25
        # The flattened size is therefore 128 (channels) * 25 * 25
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout Layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # --- THIS IS THE FIX ---
        # Flatten the image for the fully connected layer using the correct dimensions
        x = x.view(-1, 128 * 25 * 25)

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation here, as CrossEntropyLoss will apply it

        return x