import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import copy
from tqdm import tqdm

# Assuming your custom dataset is in a file named 'dataset.py'
# You may need to adjust the import if your file structure is different
from dataset import BrainTumorDataset
from resnet50_model import create_resnet50_model

# --- Configuration ---
# Paths
DATA_DIR = '../data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = 'resnet50_brain_tumor_classifier.pth'

# Hyperparameters
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 25 # Start with a reasonable number, can be adjusted
LEARNING_RATE = 0.001

# --- Data Transformations ---
# Pre-trained models expect specific input transformations
# 1. Resize images to 224x224
# 2. Normalize using ImageNet mean and standard deviation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Datasets and DataLoaders ---
print("Loading datasets...")
image_datasets = {
    'train': BrainTumorDataset(root_dir=TRAIN_DIR, transform=data_transforms['train']),
    'val': BrainTumorDataset(root_dir=VAL_DIR, transform=data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Class names: {class_names}")
print(f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")

# --- Model, Loss, and Optimizer ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = create_resnet50_model(num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# We only want to optimize the parameters of the new classifier head
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
def train_model(model, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Validation accuracy improved. Model saved to {MODEL_SAVE_PATH}")


    print(f'\nBest val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    print("Starting model training...")
    trained_model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
    print("Training complete.")
