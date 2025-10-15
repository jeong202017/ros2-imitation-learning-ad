"""
PyTorch Dataset for Imitation Learning.

Loads extracted bag data for training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImitationLearningDataset(Dataset):
    """PyTorch Dataset for behavioral cloning.
    
    Loads images and corresponding control commands from extracted bag data.
    Supports data augmentation and temporal sequences.
    
    Args:
        data_dir: Directory containing extracted data
        sequence_length: Number of frames in a sequence (1 for single frame)
        transform: Albumentations transform for data augmentation
        image_size: Target image size (height, width)
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 1,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (224, 224),
        mode: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        # Load data
        self.images_dir = self.data_dir / "images"
        self.csv_dir = self.data_dir / "csv"
        
        if not self.images_dir.exists() or not self.csv_dir.exists():
            raise ValueError(f"Data directory {data_dir} must contain 'images' and 'csv' subdirectories")
        
        # Load control commands
        cmd_vel_path = self.csv_dir / "cmd_vel.csv"
        if not cmd_vel_path.exists():
            raise ValueError(f"cmd_vel.csv not found in {self.csv_dir}")
            
        self.cmd_vel_df = pd.read_csv(cmd_vel_path)
        self.cmd_vel_df = self.cmd_vel_df.sort_values("timestamp")
        
        # Get list of images
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Build mapping from timestamp to image and control
        self._build_data_mapping()
        
        logger.info(
            f"Loaded {len(self.samples)} samples from {data_dir} "
            f"(mode: {mode}, sequence_length: {sequence_length})"
        )
        
    def _build_data_mapping(self) -> None:
        """Build mapping between images and control commands based on timestamps."""
        self.samples = []
        
        for img_path in self.image_files:
            # Extract timestamp from filename
            timestamp_str = img_path.stem.split("_")[-1]
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                logger.warning(f"Could not parse timestamp from {img_path.name}")
                continue
                
            # Find closest control command
            time_diffs = np.abs(self.cmd_vel_df["timestamp"].values - timestamp)
            closest_idx = np.argmin(time_diffs)
            
            # Only use if time difference is reasonable (< 100ms = 100,000,000 ns)
            if time_diffs[closest_idx] < 100_000_000:
                cmd = self.cmd_vel_df.iloc[closest_idx]
                self.samples.append({
                    "image_path": str(img_path),
                    "timestamp": timestamp,
                    "linear_x": cmd["linear_x"],
                    "angular_z": cmd["angular_z"],
                })
                
        # Sort by timestamp
        self.samples = sorted(self.samples, key=lambda x: x["timestamp"])
        
        # Remove samples that can't form complete sequences
        if self.sequence_length > 1:
            self.samples = self.samples[: -(self.sequence_length - 1)]
            
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (images, controls)
            - images: Tensor of shape (seq_len, C, H, W) or (C, H, W) if seq_len=1
            - controls: Tensor of shape (2,) with [linear_x, angular_z]
        """
        if self.sequence_length == 1:
            sample = self.samples[idx]
            image = self._load_image(sample["image_path"])
            controls = torch.tensor(
                [sample["linear_x"], sample["angular_z"]],
                dtype=torch.float32
            )
            return image, controls
        else:
            # Load sequence
            images = []
            for i in range(self.sequence_length):
                sample = self.samples[idx + i]
                image = self._load_image(sample["image_path"])
                images.append(image)
                
            # Stack images along first dimension
            images = torch.stack(images, dim=0)
            
            # Use control from last frame
            last_sample = self.samples[idx + self.sequence_length - 1]
            controls = torch.tensor(
                [last_sample["linear_x"], last_sample["angular_z"]],
                dtype=torch.float32
            )
            
            return images, controls
            
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor of shape (C, H, W)
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        # Convert to tensor and normalize
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # HWC -> CHW
        image = image / 255.0  # Normalize to [0, 1]
        
        return image
        
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics for normalization.
        
        Returns:
            Dictionary with mean and std for control values
        """
        linear_x = [s["linear_x"] for s in self.samples]
        angular_z = [s["angular_z"] for s in self.samples]
        
        return {
            "linear_x_mean": np.mean(linear_x),
            "linear_x_std": np.std(linear_x),
            "angular_z_mean": np.mean(angular_z),
            "angular_z_std": np.std(angular_z),
        }


def get_augmentation_transform(config: Dict) -> Optional[A.Compose]:
    """Create augmentation transform from config.
    
    Args:
        config: Augmentation configuration dictionary
        
    Returns:
        Albumentations Compose transform or None
    """
    if not config.get("enabled", False):
        return None
        
    transforms = []
    
    if config.get("horizontal_flip", 0) > 0:
        transforms.append(
            A.HorizontalFlip(p=config["horizontal_flip"])
        )
        
    if config.get("brightness", 0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=config["brightness"],
                contrast_limit=config["brightness"],
                p=0.5
            )
        )
        
    if config.get("rotation", 0) > 0:
        transforms.append(
            A.Rotate(
                limit=config["rotation"],
                p=0.5
            )
        )
        
    if transforms:
        return A.Compose(transforms)
    return None


def create_data_splits(
    data_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    sequence_length: int = 1,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train, validation, and test datasets.
    
    Args:
        data_dir: Directory containing extracted data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        sequence_length: Number of frames in a sequence
        image_size: Target image size
        augmentation_config: Configuration for data augmentation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load full dataset to split
    full_dataset = ImitationLearningDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        transform=None,
        image_size=image_size,
        mode="train",
    )
    
    # Calculate split sizes
    n_samples = len(full_dataset.samples)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    # Split samples
    train_samples = full_dataset.samples[:n_train]
    val_samples = full_dataset.samples[n_train:n_train + n_val]
    test_samples = full_dataset.samples[n_train + n_val:]
    
    # Create augmentation transform for training
    train_transform = None
    if augmentation_config:
        train_transform = get_augmentation_transform(augmentation_config)
    
    # Create datasets
    train_dataset = ImitationLearningDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        transform=train_transform,
        image_size=image_size,
        mode="train",
    )
    train_dataset.samples = train_samples
    
    val_dataset = ImitationLearningDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        transform=None,
        image_size=image_size,
        mode="val",
    )
    val_dataset.samples = val_samples
    
    test_dataset = ImitationLearningDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        transform=None,
        image_size=image_size,
        mode="test",
    )
    test_dataset.samples = test_samples
    
    logger.info(
        f"Data splits - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    
    return train_dataset, val_dataset, test_dataset
