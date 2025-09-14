"""
Brain Tumor Dataset Module

This module implements PyTorch Dataset classes for loading and processing
brain tumor MRI images with proper train/validation/test splits.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import Tuple, List, Optional


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI Images
    
    Handles loading images from directory structure and applies transformations
    """
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Path to the data directory
            transform (transforms.Compose, optional): Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load images and labels
        self._load_data()
        
    def _load_data(self):
        """Load image paths and labels from directory structure"""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return
            
        # Get class directories (tumor types)
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        class_dirs.sort()  # Ensure consistent ordering
        
        self.class_names = class_dirs
        
        # If no class directories found, treat as flat directory
        if not class_dirs:
            # Handle flat directory structure
            image_files = [f for f in os.listdir(self.data_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files:
                self.images.append(os.path.join(self.data_dir, img_file))
                self.labels.append(0)  # Default class
            self.class_names = ['tumor'] if image_files else []
        else:
            # Handle hierarchical directory structure
            for class_idx, class_name in enumerate(class_dirs):
                class_path = os.path.join(self.data_dir, class_name)
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_file in image_files:
                    self.images.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
            
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (128, 128), (0, 0, 0)))
            else:
                image = Image.new('RGB', (128, 128), (0, 0, 0))
        
        return image, label
    
    def get_class_names(self):
        """Get the list of class names"""
        return self.class_names
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        if not self.labels:
            return {}
            
        class_counts = {}
        for label in self.labels:
            class_name = self.class_names[label] if label < len(self.class_names) else f"class_{label}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get image transformations for training or validation/testing
    
    Args:
        is_training (bool): Whether to apply training augmentations
        
    Returns:
        transforms.Compose: Composed transformations
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_root: str, batch_size: int = 32, train_split: float = 0.8, 
                       num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_root (str): Root directory containing train and test folders
        batch_size (int): Batch size for data loaders
        train_split (float): Fraction of training data to use for training (rest for validation)
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    
    # Create full training dataset
    full_train_dataset = BrainTumorDataset(
        train_dir, 
        transform=get_transforms(is_training=True)
    )
    
    # Split training dataset into train and validation
    if len(full_train_dataset) > 0:
        train_size = int(train_split * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # Update validation dataset transform
        val_transform = get_transforms(is_training=False)
        val_dataset.dataset.transform = val_transform
    else:
        # Handle empty dataset
        train_dataset = full_train_dataset
        val_dataset = BrainTumorDataset(train_dir, transform=get_transforms(is_training=False))
    
    # Create test dataset
    test_dataset = BrainTumorDataset(
        test_dir,
        transform=get_transforms(is_training=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)) if len(test_dataset) > 0 else 1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader


def get_class_names_from_directory(data_dir: str) -> List[str]:
    """
    Get class names from directory structure
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        list: List of class names
    """
    if not os.path.exists(data_dir):
        return []
        
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    return sorted(class_dirs)