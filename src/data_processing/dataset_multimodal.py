"""
Multimodal Dataset for Imitation Learning.

Camera + LiDAR 멀티모달 Dataset 클래스.

Loads preprocessed camera images, LiDAR data, and control commands.
전처리된 카메라 이미지, LiDAR 데이터, 제어 명령을 로드합니다.
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


class MultimodalDataset(Dataset):
    """Multimodal dataset with camera and LiDAR for behavioral cloning.
    
    Camera + LiDAR 멀티모달 Dataset.
    
    Loads images, point clouds, laser scans, and control commands.
    이미지, 포인트 클라우드, 레이저 스캔, 제어 명령을 로드합니다.
    
    Args:
        data_dir: Directory with preprocessed data
        split: Data split ('train', 'val', or 'test')
        use_pointcloud: Whether to use point cloud data
        use_laserscan: Whether to use laser scan data
        transform: Optional transform to apply to images
        normalize: Whether to normalize control values
        max_points: Maximum number of points to sample from point cloud
        laserscan_bins: Number of bins for laser scan
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        use_pointcloud: bool = True,
        use_laserscan: bool = True,
        transform: Optional[callable] = None,
        normalize: bool = True,
        max_points: int = 2048,
        laserscan_bins: int = 360,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_pointcloud = use_pointcloud
        self.use_laserscan = use_laserscan
        self.transform = transform
        self.normalize = normalize
        self.max_points = max_points
        self.laserscan_bins = laserscan_bins
        
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
        self.lidar_dir = self.data_dir / "lidar"
        
        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split "
            f"(pointcloud={use_pointcloud}, laserscan={use_laserscan})"
        )
        
    def __len__(self) -> int:
        """Return number of samples.
        
        샘플 수를 반환합니다.
        """
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample.
        
        샘플을 가져옵니다.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'image': Tensor of shape (C, H, W)
            - 'pointcloud': Tensor of shape (N, 3) if use_pointcloud
            - 'laserscan': Tensor of shape (bins,) if use_laserscan
            - 'target': Tensor of shape (2,) with [speed, steering]
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = self.images_dir / sample["image"]
        image = self._load_image(image_path)
        
        result = {"image": image}
        
        # Load point cloud if requested
        if self.use_pointcloud:
            pc_path = self.lidar_dir / sample["pointcloud"]
            pointcloud = self._load_pointcloud(pc_path)
            result["pointcloud"] = pointcloud
            
        # Load laser scan if requested
        if self.use_laserscan:
            scan_path = self.lidar_dir / sample["laserscan"]
            laserscan = self._load_laserscan(scan_path)
            result["laserscan"] = laserscan
            
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
            
        result["target"] = torch.tensor([speed, steering], dtype=torch.float32)
        
        return result
        
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
        
    def _load_pointcloud(self, pc_path: Path) -> torch.Tensor:
        """Load and process point cloud.
        
        포인트 클라우드를 로드하고 처리합니다.
        
        Args:
            pc_path: Path to point cloud file (.npy)
            
        Returns:
            Point cloud tensor of shape (max_points, 3)
        """
        # Load point cloud
        points = np.load(str(pc_path))
        
        # Remove invalid points (inf, nan)
        valid_mask = np.all(np.isfinite(points), axis=1)
        points = points[valid_mask]
        
        # Sample or pad to max_points
        if len(points) > self.max_points:
            # Random sampling
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        elif len(points) < self.max_points:
            # Pad with zeros
            padding = np.zeros((self.max_points - len(points), 3), dtype=np.float32)
            points = np.vstack([points, padding])
            
        return torch.from_numpy(points.astype(np.float32))
        
    def _load_laserscan(self, scan_path: Path) -> torch.Tensor:
        """Load and process laser scan.
        
        레이저 스캔을 로드하고 처리합니다.
        
        Args:
            scan_path: Path to laser scan file (.npz)
            
        Returns:
            Laser scan tensor of shape (laserscan_bins,)
        """
        # Load laser scan
        data = np.load(str(scan_path))
        ranges = data["ranges"]
        
        # Replace inf with max range
        range_max = data.get("range_max", 100.0)
        ranges = np.nan_to_num(ranges, nan=range_max, posinf=range_max, neginf=0.0)
        
        # Resample to desired number of bins if needed
        if len(ranges) != self.laserscan_bins:
            # Simple linear interpolation
            indices = np.linspace(0, len(ranges) - 1, self.laserscan_bins)
            ranges = np.interp(indices, np.arange(len(ranges)), ranges)
            
        return torch.from_numpy(ranges.astype(np.float32))
        
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
