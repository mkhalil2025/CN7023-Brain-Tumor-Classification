import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """Custom Dataset for Brain Tumor MRI images."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class folders (e.g., 'data/train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if not self.classes:
            raise FileNotFoundError(f"No class folders found in directory {root_dir}.")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for target_class in self.classes:
            target_dir = os.path.join(self.root_dir, target_class)
            for fname in os.listdir(target_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    path = os.path.join(target_dir, fname)
                    item = (path, self.class_to_idx[target_class])
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        # Use PIL to open the image, ensuring it's in RGB format
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target


# --- THIS IS THE UPDATED FUNCTION ---
def get_dataloaders(data_dir, batch_size, val_split, transform):
    """
    Creates and returns the data loaders for training, validation, and testing.
    This version correctly handles separate transforms for train and val sets.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Testing directory not found at {test_dir}")

    # Create two separate dataset instances for the full training data;
    # one for training transforms and one for validation transforms.
    train_dataset_with_aug = BrainTumorDataset(root_dir=train_dir, transform=transform['train'])
    val_dataset_no_aug = BrainTumorDataset(root_dir=train_dir, transform=transform['val'])

    # --- Split the dataset ---
    dataset_size = len(train_dataset_with_aug)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Use a fixed generator for reproducibility of the split
    indices = torch.randperm(len(train_dataset_with_aug), generator=torch.Generator().manual_seed(42)).tolist()

    # Use the indices to create subsets from the appropriate dataset instance
    train_subset = Subset(train_dataset_with_aug, indices[:train_size])
    val_subset = Subset(val_dataset_no_aug, indices[train_size:])

    print(f"Full training dataset size: {dataset_size}")
    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Test DataLoader ---
    # The test set always uses the 'val' transforms (no augmentation)
    test_dataset = BrainTumorDataset(root_dir=test_dir, transform=transform['val'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader