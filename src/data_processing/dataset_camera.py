"""
Camera-Only Dataset for Imitation Learning.

카메라 이미지 전용 Dataset 클래스.

Loads preprocessed camera images and control commands for training.
전처리된 카메라 이미지와 제어 명령을 학습을 위해 로드합니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraOnlyDataset(Dataset):
    """Camera-only dataset for behavioral cloning.
    
    카메라 이미지 전용 Dataset.
    
    Loads images and corresponding control commands from preprocessed data.
    전처리된 데이터에서 이미지와 해당 제어 명령을 로드합니다.
    
    Args:
        data_dir: Directory with preprocessed data
        split: Data split ('train', 'val', or 'test')
        transform: Optional transform to apply to images
        normalize: Whether to normalize control values
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Load dataset manifest
        manifest_path = self.data_dir / "dataset.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        if split not in manifest:
            raise ValueError(f"Split '{split}' not found in manifest")
            
        self.samples = manifest[split]
        
        # Load statistics for normalization
        self.stats = None
        if normalize:
            stats_path = self.data_dir / "statistics.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.stats = json.load(f)
            else:
                logger.warning(f"Statistics file not found: {stats_path}")
                
        self.images_dir = self.data_dir / "images"
        
        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split "
            f"(normalize={normalize})"
        )
        
    def __len__(self) -> int:
        """Return number of samples.
        
        샘플 수를 반환합니다.
        """
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample.
        
        샘플을 가져옵니다.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, controls)
            - image: Tensor of shape (C, H, W)
            - controls: Tensor of shape (2,) with [speed, steering]
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = self.images_dir / sample["image"]
        image = self._load_image(image_path)
        
        # Get controls
        speed = sample["speed"]
        steering = sample["steering"]
        
        # Normalize controls if requested
        if self.normalize and self.stats:
            speed = (speed - self.stats["speed"]["mean"]) / (
                self.stats["speed"]["std"] + 1e-8
            )
            steering = (steering - self.stats["steering"]["mean"]) / (
                self.stats["steering"]["std"] + 1e-8
            )
            
        controls = torch.tensor([speed, steering], dtype=torch.float32)
        
        return image, controls
        
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image.
        
        이미지를 로드하고 전처리합니다.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor of shape (C, H, W)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
            
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            if image.ndim == 3:
                image = image.permute(2, 0, 1)  # HWC -> CHW
                
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
            
        return image
        
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics.
        
        데이터셋 통계를 가져옵니다.
        
        Returns:
            Dictionary with statistics
        """
        if self.stats is None:
            # Compute from samples
            speeds = [s["speed"] for s in self.samples]
            steerings = [s["steering"] for s in self.samples]
            
            return {
                "num_samples": len(self.samples),
                "speed_mean": float(np.mean(speeds)),
                "speed_std": float(np.std(speeds)),
                "steering_mean": float(np.mean(steerings)),
                "steering_std": float(np.std(steerings)),
            }
        else:
            return self.stats
            
    def denormalize_controls(
        self, 
        controls: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize control values.
        
        제어 값을 역정규화합니다.
        
        Args:
            controls: Normalized control tensor of shape (N, 2) or (2,)
            
        Returns:
            Denormalized control tensor
        """
        if not self.normalize or self.stats is None:
            return controls
            
        controls_denorm = controls.clone()
        
        # Handle both single sample and batch
        if controls.ndim == 1:
            controls_denorm[0] = (
                controls[0] * self.stats["speed"]["std"] + 
                self.stats["speed"]["mean"]
            )
            controls_denorm[1] = (
                controls[1] * self.stats["steering"]["std"] + 
                self.stats["steering"]["mean"]
            )
        else:
            controls_denorm[:, 0] = (
                controls[:, 0] * self.stats["speed"]["std"] + 
                self.stats["speed"]["mean"]
            )
            controls_denorm[:, 1] = (
                controls[:, 1] * self.stats["steering"]["std"] + 
                self.stats["steering"]["mean"]
            )
            
        return controls_denorm
