import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from src.dataset import get_dataloaders
from src.resnet50_model import create_resnet50_model


def main():
    # --- Configuration ---
    DATA_DIR = '../data'
    RESULTS_DIR = 'results'
    BATCH_SIZE = 32
    NUM_CLASSES = 4
    IMG_SIZE = 224
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2  # Define the validation split percentage

    # --- Setup ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transformations ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Data Loaders ---
    print("Loading datasets...")
    # --- THE FIX IS HERE ---
    # Added the required 'val_split' argument to the function call.
    train_loader, val_loader, _ = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        transform=data_transforms,
        val_split=VAL_SPLIT  # Pass the validation split value
    )

    # --- Model, Optimizer, Loss ---
    model = create_resnet50_model(num_classes=NUM_CLASSES)
    model.to(device)

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    best_val_accuracy = 0.0
    best_model_path = os.path.join(RESULTS_DIR, 'resnet50_best_model.pth')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nStarting ResNet-50 model training...")
    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_predictions.double() / total_samples
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())

        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        epoch_val_loss = running_loss / total_samples
        epoch_val_acc = correct_predictions.double() / total_samples
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} -> "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save the best model
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved with validation accuracy: {best_val_accuracy:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best model saved to {best_model_path} with accuracy: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    main()