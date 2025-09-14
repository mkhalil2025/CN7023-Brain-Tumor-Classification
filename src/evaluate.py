"""
Brain Tumor Classification Evaluation Script

This script evaluates the trained model on test data and generates
comprehensive performance metrics including classification report and confusion matrix.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import argparse

from model import BrainTumorCNN
from dataset import create_data_loaders, get_class_names_from_directory


class ModelEvaluator:
    """
    Model evaluation class for Brain Tumor Classification
    """
    
    def __init__(self, model, test_loader, device, class_names):
        """
        Initialize the evaluator
        
        Args:
            model: Trained neural network model
            test_loader: DataLoader for test data
            device: torch.device for computation
            class_names: List of class names
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        
    def evaluate(self):
        """
        Evaluate the model on test data
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("Evaluating model on test data...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Get predictions
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f'Processed {batch_idx+1}/{len(self.test_loader)} batches')
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Generate classification report
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return results
    
    def print_classification_report(self, results):
        """
        Print detailed classification report
        
        Args:
            results (dict): Results from evaluate() method
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = results['classification_report']
        
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<15} {metrics['precision']:<10.3f} "
                      f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} "
                      f"{int(metrics['support']):<10}")
        
        print("-" * 60)
        print(f"{'Accuracy':<15} {'':<10} {'':<10} {results['accuracy']:<10.3f} "
              f"{len(results['labels']):<10}")
        print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<10.3f} "
              f"{report['macro avg']['recall']:<10.3f} {report['macro avg']['f1-score']:<10.3f} "
              f"{int(report['macro avg']['support']):<10}")
        print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<10.3f} "
              f"{report['weighted avg']['recall']:<10.3f} {report['weighted avg']['f1-score']:<10.3f} "
              f"{int(report['weighted avg']['support']):<10}")
    
    def plot_confusion_matrix(self, results, save_path='results/confusion_matrix.png'):
        """
        Plot and save confusion matrix
        
        Args:
            results (dict): Results from evaluate() method
            save_path (str): Path to save the plot
        """
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_class_distribution(self, results, save_path='results/class_distribution.png'):
        """
        Plot class distribution in test set
        
        Args:
            results (dict): Results from evaluate() method
            save_path (str): Path to save the plot
        """
        labels = results['labels']
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        class_names_subset = [self.class_names[i] for i in unique]
        plt.bar(class_names_subset, counts)
        plt.title('Class Distribution in Test Set')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Class distribution plot saved to: {save_path}")
    
    def save_detailed_results(self, results, save_path='results/detailed_results.txt'):
        """
        Save detailed results to text file
        
        Args:
            results (dict): Results from evaluate() method
            save_path (str): Path to save the results
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("Brain Tumor Classification - Detailed Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Total Test Samples: {len(results['labels'])}\n\n")
            
            f.write("Class-wise Performance:\n")
            f.write("-" * 30 + "\n")
            
            report = results['classification_report']
            for class_name in self.class_names:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {int(metrics['support'])}\n")
            
            f.write(f"\nMacro Average:\n")
            f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
            f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
            f.write(f"  F1-Score: {report['macro avg']['f1-score']:.4f}\n")
            
            f.write(f"\nWeighted Average:\n")
            f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
            f.write(f"  Recall: {report['weighted avg']['recall']:.4f}\n")
            f.write(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(str(results['confusion_matrix']))
        
        print(f"Detailed results saved to: {save_path}")


def load_model(model_path, num_classes=4, device='cpu'):
    """
    Load trained model from checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        num_classes (int): Number of output classes
        device (str): Device to load the model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    model = BrainTumorCNN(num_classes=num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from: {model_path}")
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation accuracy: {checkpoint.get('best_val_accuracy', 'unknown'):.2f}%")
    
    return model


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Classification Model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory containing test folder')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get class names
    train_dir = os.path.join(args.data_dir, 'train')
    class_names = get_class_names_from_directory(train_dir)
    
    if not class_names:
        # Fallback class names
        class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        print("Warning: Could not detect class names from directory structure.")
        print(f"Using default class names: {class_names}")
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Load model
    try:
        model = load_model(args.model_path, num_classes, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have trained a model first by running train.py")
        return
    
    # Create data loaders (we only need the test loader)
    _, _, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    if test_loader is None or len(test_loader.dataset) == 0:
        print("Warning: No test data found. Please check your data directory structure.")
        print("Expected structure:")
        print("data/")
        print("  test/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class2/")
        print("      ...")
        return
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, test_loader, device, class_names)
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_classification_report(results)
    
    # Create visualizations
    evaluator.plot_confusion_matrix(results, 
                                   os.path.join(args.results_dir, 'confusion_matrix.png'))
    evaluator.plot_class_distribution(results, 
                                     os.path.join(args.results_dir, 'class_distribution.png'))
    
    # Save detailed results
    evaluator.save_detailed_results(results, 
                                   os.path.join(args.results_dir, 'detailed_results.txt'))
    
    print(f"\nEvaluation completed!")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Results saved in: {args.results_dir}/")


if __name__ == '__main__':
    main()