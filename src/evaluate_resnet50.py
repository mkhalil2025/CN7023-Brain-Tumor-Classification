import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Adjust these imports to match your project structure
from dataset import BrainTumorDataset
from resnet50_model import create_resnet50_model

# --- Configuration ---
# Paths
DATA_DIR = 'data/'
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = 'resnet50_brain_tumor_classifier.pth' # Path to the trained ResNet-50 model

# Output files
CONFUSION_MATRIX_FILE = 'resnet50_confusion_matrix.png'
CLASSIFICATION_REPORT_FILE = 'resnet50_classification_report.txt'

# Parameters
NUM_CLASSES = 4
BATCH_SIZE = 32

# --- Data Transformations for ResNet-50 ---
# This must match the validation transforms used during training
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Dataset and DataLoader ---
print("Loading test dataset...")
test_dataset = BrainTumorDataset(root_dir=TEST_DIR, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class_names = test_dataset.classes
print(f"Class names: {class_names}")
print(f"Test set size: {len(test_dataset)}")

# --- Model Loading ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the model architecture
model = create_resnet50_model(num_classes=NUM_CLASSES)

# Load the saved state dictionary
# Use map_location to ensure it works whether you are on CPU or GPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# --- Evaluation ---
print("Running evaluation on the test set...")
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# --- Generate and Save Results ---
print("Generating classification report and confusion matrix...")

# 1. Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

with open(CLASSIFICATION_REPORT_FILE, 'w') as f:
    f.write("ResNet-50 Classification Report\n")
    f.write("="*30 + "\n")
    f.write(report)
print(f"Classification report saved to {CLASSIFICATION_REPORT_FILE}")

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet-50 Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

# Save the figure
plt.savefig(CONFUSION_MATRIX_FILE)
print(f"Confusion matrix saved to {CONFUSION_MATRIX_FILE}")
plt.show()

print("\nEvaluation complete.")
