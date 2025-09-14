"""
Brain Tumor Classification Training Script

This script handles the training of the BrainTumorCNN model with proper
validation and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import argparse

from model import BrainTumorCNN
from dataset import create_data_loaders, get_class_names_from_directory


class Trainer:
    """
    Training class for Brain Tumor Classification
    """
    
    def __init__(self, model, train_loader, val_loader, device, num_classes=4):
        """
        Initialize the trainer
        
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: torch.device for computation
            num_classes: Number of output classes
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training tracking
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """
        Train the model for one epoch
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Train Batch: {batch_idx}/{len(self.train_loader)} '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """
        Validate the model for one epoch
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        os.makedirs('models', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'models/latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'models/best_model.pth')
            print(f'New best model saved with validation accuracy: {self.best_val_accuracy:.2f}%')
    
    def train(self, num_epochs):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs (int): Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 20)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Check if this is the best model
            is_best = val_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_acc
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {self.best_val_accuracy:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f}s')
            print(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f}s')
        print(f'Best validation accuracy: {self.best_val_accuracy:.2f}%')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Root directory containing train and test folders')
    parser.add_argument('--epochs', type=int, default=25, 
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8, 
                       help='Fraction of training data for training (rest for validation)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get class names to determine number of classes
    train_dir = os.path.join(args.data_dir, 'train')
    class_names = get_class_names_from_directory(train_dir)
    num_classes = len(class_names) if class_names else 4  # Default to 4 classes
    
    print(f"Detected {num_classes} classes: {class_names}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        train_split=args.train_split
    )
    
    # Create model
    model = BrainTumorCNN(num_classes=num_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Check if we have data
    if train_loader is None or len(train_loader.dataset) == 0:
        print("Warning: No training data found. Please check your data directory structure.")
        print("Expected structure:")
        print("data/")
        print("  train/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class2/")
        print("      ...")
        print("  test/")
        print("    class1/")
        print("      ...")
        return
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device, num_classes)
    trainer.train(args.epochs)
    
    print("Training completed!")
    print(f"Best model saved as 'models/best_model.pth'")


if __name__ == '__main__':
    main()