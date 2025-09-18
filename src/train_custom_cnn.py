import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import copy

# Make sure these files are in the same directory or accessible
from dataset import get_dataloaders
from custom_cnn_model import CustomCNN


def main():
    # --- Configuration ---
    # Correctly points to the 'data' folder, which is one level up from 'src'
    DATA_DIR = '../data'
    RESULTS_DIR = 'results'
    MODEL_TYPE = 'custom_cnn'
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training model: Custom CNN")

    # --- Data Transformations for Custom CNN ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
        ]),
    }

    # --- Data Loaders ---
    # Use your original function to ensure consistent data splitting
    train_loader, val_loader, _ = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        transform=data_transforms
    )
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}
    print("Data loaders created successfully.")

    # --- Model, Loss, Optimizer ---
    model = CustomCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Paths for saving results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f'{MODEL_TYPE}_best_model.pth')
    log_path = os.path.join(RESULTS_DIR, f'{MODEL_TYPE}_training_log.csv')

    # --- Training Loop ---
    best_val_accuracy = 0.0
    log_data = []

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("-" * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log data after each phase
            if phase == 'train':
                train_epoch_loss = epoch_loss
                train_epoch_acc = epoch_acc.item()
            else:  # val phase
                val_epoch_loss = epoch_loss
                val_epoch_acc = epoch_acc.item()

            # Save model if validation accuracy has improved
            if phase == 'val' and val_epoch_acc > best_val_accuracy:
                best_val_accuracy = val_epoch_acc
                torch.save(model.state_dict(), model_path)
                print(f"*** New best model saved with val acc: {best_val_accuracy:.4f} ***")

        log_data.append({
            'epoch': epoch + 1,
            'train_loss': train_epoch_loss,
            'train_accuracy': train_epoch_acc,
            'val_loss': val_epoch_loss,
            'val_accuracy': val_epoch_acc
        })
        pd.DataFrame(log_data).to_csv(log_path, index=False)

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to {model_path}")


if __name__ == '__main__':
    main()