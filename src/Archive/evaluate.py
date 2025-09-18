import os
import torch
from sklearn.metrics import classification_report
import pandas as pd

from dataset import get_dataloaders
from custom_cnn_model import CustomCNN

def evaluate():
    """
    Loads the best trained model and evaluates its performance on the test dataset.
    """
    # --- Configuration ---
    data_dir = 'data'
    model_path = os.path.join('results', 'best_model.pth')
    batch_size = 32

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loader ---
    # We only need the test_loader for evaluation
    try:
        _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
        print("Test data loader created successfully.")
    except FileNotFoundError as e:
        print(f"Error creating data loader: {e}")
        return

    # --- Load Model ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train.py first to train and save a model.")
        return

    model = CustomCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- Evaluation ---
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculation for efficiency
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Results ---
    # Get the class names from the dataset object
    class_names = test_loader.dataset.classes

    print("\n" + "="*50)
    print("           Final Evaluation Results on Test Set")
    print("="*50 + "\n")

    # Generate and print the classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # You can also save this report to a file
    report_path = os.path.join('results', 'test_set_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nClassification report saved to {report_path}")

if __name__ == '__main__':
    evaluate()