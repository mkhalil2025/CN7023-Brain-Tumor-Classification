import sys
import os
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import get_dataloaders

# --- Start of The Fix ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of The Fix ---

# Correctly import the function from your model file
from src.resnet50_model import create_resnet50_model


def main():
    """Main function to run the evaluation for the ResNet-50 model."""
    # --- Configuration ---
    DATA_DIR = '../data'
    RESULTS_DIR = 'results'
    BATCH_SIZE = 32
    NUM_CLASSES = 4
    IMG_SIZE = 224
    MODEL_PATH = os.path.join(RESULTS_DIR, 'resnet50_best_model.pth')

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transformations ---
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # --- Get Test Loader ---
    print("Loading test dataset...")
    try:
        _, _, test_loader = get_dataloaders(
            DATA_DIR,
            batch_size=BATCH_SIZE,
            val_split=0.2,
            transform={'train': None, 'val': data_transforms['val']}
        )
        print(f"Test set size: {len(test_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    # Correctly call the function to create the model
    model = create_resnet50_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # --- Evaluation ---
    print("\nRunning evaluation on the test set for the ResNet-50 model...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating ResNet-50"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Display Results ---
    class_names = test_loader.dataset.classes
    class_names = [name.replace('notumor', 'no_tumor') for name in class_names]
    print(f"Class names: {class_names}")

    print("\n--- Final Test Results (ResNet-50) ---")

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\nConfusion Matrix:")
    print(cm_df)

    # Plot and save Confusion Matrix
    cm_plot_path = os.path.join(RESULTS_DIR, 'test_results_resnet50_confusion_matrix.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - ResNet-50 on Test Set')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(cm_plot_path)
    plt.show()

    print(f"\nConfusion matrix plot saved to: {cm_plot_path}")


if __name__ == '__main__':
    main()