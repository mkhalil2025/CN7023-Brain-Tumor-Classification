import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- IMPORTANT ---
# You will need to import your custom CNN model's class here.
# For example, if your model is in 'custom_cnn_model.py' and the class is named 'CustomCNN':
# from custom_cnn_model import CustomCNN
#
# Replace 'CustomCNN' below with the actual name of your model class.
# If the file is named differently, adjust the import accordingly.
from custom_cnn_model import CustomCNN  # <--- PLEASE VERIFY THIS LINE

# --- Configuration ---
# Paths
DATA_DIR = 'data/'
TEST_DIR = os.path.join(DATA_DIR, 'test')
# Make sure this path points to your trained custom CNN model weights
MODEL_PATH = 'custom_cnn_model.pth' # <--- PLEASE VERIFY THIS FILENAME

# Output files
CONFUSION_MATRIX_FILE = 'custom_cnn_confusion_matrix.png'
CLASSIFICATION_REPORT_FILE = 'custom_cnn_classification_report.txt'

# Parameters
NUM_CLASSES = 4
BATCH_SIZE = 32

# --- Data Transformations for Custom CNN ---
# These should match the transformations you originally used for your custom model
eval_transform = transforms.Compose([
    transforms.Resize((200, 200)), # Assuming 200x200 for your custom model
    transforms.ToTensor(),
])

# --- Dataset and DataLoader ---
# Assuming you use the same BrainTumorDataset class
from dataset import BrainTumorDataset

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
# Replace 'CustomCNN' if your class name is different
model = CustomCNN(num_classes=NUM_CLASSES) # <--- PLEASE VERIFY THIS LINE

# Load the saved state dictionary
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# --- Evaluation ---
print("Running evaluation on the test set for the Custom CNN...")
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating Custom CNN"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# --- Generate and Save Results ---
print("Generating classification report and confusion matrix for Custom CNN...")

# 1. Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

with open(CLASSIFICATION_REPORT_FILE, 'w') as f:
    f.write("Custom CNN Classification Report\n")
    f.write("="*30 + "\n")
    f.write(report)
print(f"Classification report saved to {CLASSIFICATION_REPORT_FILE}")

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Custom CNN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

# Save the figure
plt.savefig(CONFUSION_MATRIX_FILE)
print(f"Confusion matrix saved to {CONFUSION_MATRIX_FILE}")
plt.show()

print("\nCustom CNN evaluation complete.")
