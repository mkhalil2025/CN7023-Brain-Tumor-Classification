import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from dataset import get_dataloaders
from custom_cnn_model import CustomCNN


def main():
    # --- Configuration ---
    data_dir = 'data'
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    val_split = 0.2

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loaders ---
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir, batch_size=batch_size, val_split=val_split
        )
        print("Data loaders created successfully.")
    except FileNotFoundError as e:
        print(f"Error creating data loaders: {e}")
        print(
            "Please ensure your dataset is structured correctly in the 'data' directory (e.g., data/Training, data/Testing).")
        return

    # --- Model, Loss, Optimizer ---
    model = CustomCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Paths for saving results ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'best_model.pth')
    log_path = os.path.join(results_dir, 'training_log.csv')

    # --- Training Loop ---
    best_val_accuracy = 0.0
    log_data = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        print("-" * 20)
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("-" * 20)

        # **MODIFIED LOOP TO ADD BATCH-LEVEL PRINTING**
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy for the batch
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f'Train Batch: {i + 1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100 * train_correct / train_total:.2f}%')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # --- Logging and Saving ---
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * val_correct / val_total

        print(f"\nEpoch Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%\n")

        log_data.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_accuracy': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_accuracy': epoch_val_accuracy
        })

        pd.DataFrame(log_data).to_csv(log_path, index=False)

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"*** New best model saved with validation accuracy: {best_val_accuracy:.2f}% ***\n")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model saved to {model_path}")
    print(f"Training log saved to {log_path}")


if __name__ == '__main__':
    main()