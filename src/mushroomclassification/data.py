from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class MushroomDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform: Optional[callable] = None,
                 subset: Optional[List[int]] = None):
        """
        Args:
            data_path (str): Path to the data directory.
            transform (callable, optional): Optional transform to be applied to the images.
            subset (List[int], optional): Optional list of indices to define a subset of the data.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.classes = ["conditionally_edible", "deadly", "edible", "poisonous"]
        self.image_files = self._prepare_image_list()

        # Apply subset if provided
        if subset is not None:
            self.image_files = [self.image_files[i] for i in subset]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Retrieve a single sample (image and label) from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple: A transformed image and its corresponding label as an integer.
        """
        img_path, label = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, label

    def _prepare_image_list(self) -> List[Tuple[Path, int]]:
        """
        Prepare a list of image paths and their corresponding class labels.

        Returns:
            List[Tuple[Path, int]]: List of tuples with image paths and their labels.
        """
        image_files = []
        for cls_idx, cls_name in enumerate(self.classes):
            class_path = self.data_path / "Classes" / cls_name
            if class_path.exists() and class_path.is_dir():
                image_files.extend(
                    (file_path, cls_idx)
                    for file_path in class_path.glob("**/*")
                    if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                )
        return image_files

class MushroomDatamodule(L.LightningDataModule):
    """Mushroom data module"""

    def __init__(self, data_path: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transformations for each dataset
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.RandomRotation(30),  # Rotate images by up to 30 degrees
            transforms.ToTensor(),          # Convert to a PyTorch tensor
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def prepare_data(self) -> None:
        """Download data"""
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the datasets for different stages: train, validation, test, predict."""
        # Create full dataset (no transforms here; transforms applied during splitting)
        data_full = MushroomDataset(self.data_path)

        # Split the data into train, validation, test, and predict sets
        train_size = int(0.8 * len(data_full))  # 80% for training
        val_size = int(0.1 * len(data_full))    # 10% for validation
        test_size = len(data_full) - train_size - val_size  # Remainder for test

        # Use a fixed seed for reproducible splits
        train_data, val_data, test_data = random_split(
            data_full, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Apply transforms to each split
        self.data_train = MushroomDataset(
            self.data_path, transform=self.train_transform, subset=train_data.indices
        )
        self.data_val = MushroomDataset(
            self.data_path, transform=self.val_transform, subset=val_data.indices
        )
        self.data_test = MushroomDataset(
            self.data_path, transform=self.test_transform, subset=test_data.indices
        )

        # For prediction, use the same dataset as test for now
        self.data_predict = self.data_test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Use pinned memory for faster GPU transfers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # Use pinned memory for faster GPU transfers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # Use pinned memory for faster GPU transfers
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # Use pinned memory for faster GPU transfers
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        return
