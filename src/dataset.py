import os
import torch  # <--- THIS IS THE FIX
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """Brain Tumor MRI Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'data/Training').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Filter out system files like .DS_Store
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if not self.classes:
            raise FileNotFoundError(
                f"No class folders found in directory {root_dir}. Please check the dataset structure.")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        images = []
        for target_class in self.classes:
            target_dir = os.path.join(self.root_dir, target_class)
            for fname in os.listdir(target_dir):
                # Ensure we are only picking up image files
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    path = os.path.join(target_dir, fname)
                    item = (path, self.class_to_idx[target_class])
                    images.append(item)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # Return a placeholder tensor or skip
            return torch.randn(3, 128, 128), -1  # Return dummy data

        if self.transform:
            image = self.transform(image)
        return image, target


def get_dataloaders(data_dir, batch_size=32, val_split=0.2):
    """Creates and returns the train, validation, and test dataloaders."""

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_train_dataset = BrainTumorDataset(root_dir=os.path.join(data_dir, 'train'), transform=train_transform)

    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # The validation set should not have augmentation, so using test_transform is correct.
    # The dataset object within val_dataset is a Subset, we need to access its underlying dataset.
    val_dataset.dataset.transform = test_transform

    test_dataset = BrainTumorDataset(root_dir=os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader