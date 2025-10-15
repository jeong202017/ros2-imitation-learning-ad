#!/usr/bin/env python3
"""
Multimodal Data Preprocessing Script.

Preprocesses extracted ERP42 data for multimodal experiment:
- Synchronizes images, LiDAR (PointCloud & LaserScan), and control commands
- All three sensor modalities must be temporally aligned
- Resizes images and processes LiDAR data
- Splits data into train/val/test sets
- Generates dataset.json manifest

멀티모달 실험을 위한 추출된 ERP42 데이터 전처리:
- 이미지, LiDAR (PointCloud & LaserScan), 제어 명령 동기화
- 세 가지 센서 모달리티가 모두 시간적으로 정렬되어야 함
- 이미지 리사이징 및 LiDAR 데이터 처리
- 데이터를 train/val/test로 분할
- dataset.json 매니페스트 생성
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultimodalPreprocessor:
    """Preprocess multimodal data for training.
    
    멀티모달 데이터를 학습을 위해 전처리합니다.
    
    Args:
        input_dir: Directory with extracted data
        output_dir: Directory to save preprocessed data
        image_size: Target image size (height, width)
        train_split: Training data ratio
        val_split: Validation data ratio
        time_threshold: Maximum time difference for synchronization (seconds)
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        train_split: float = 0.8,
        val_split: float = 0.1,
        time_threshold: float = 0.1,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        self.time_threshold = time_threshold
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.lidar_dir = self.output_dir / "lidar"
        self.csv_dir = self.output_dir / "csv"
        
        for directory in [self.images_dir, self.lidar_dir, self.csv_dir]:
            directory.mkdir(exist_ok=True)
            
    def preprocess(self) -> None:
        """Run preprocessing pipeline.
        
        전처리 파이프라인을 실행합니다.
        """
        logger.info("Starting multimodal preprocessing")
        
        # Load control data
        control_path = self.input_dir / "csv" / "control_data.csv"
        if not control_path.exists():
            raise FileNotFoundError(f"Control data not found: {control_path}")
            
        control_df = pd.read_csv(control_path)
        logger.info(f"Loaded {len(control_df)} control commands")
        
        # Get sensor files
        images_path = self.input_dir / "images"
        lidar_path = self.input_dir / "lidar"
        
        image_files = sorted(list(images_path.glob("*.jpg")))
        pointcloud_files = sorted(list(lidar_path.glob("*velodyne*.npy")))
        laserscan_files = sorted(list(lidar_path.glob("*scan*.npz")))
        
        logger.info(f"Found {len(image_files)} images")
        logger.info(f"Found {len(pointcloud_files)} point clouds")
        logger.info(f"Found {len(laserscan_files)} laser scans")
        
        # Synchronize all modalities
        logger.info("Synchronizing camera, LiDAR, and control data...")
        synchronized_data = self._synchronize_multimodal(
            image_files, pointcloud_files, laserscan_files, control_df
        )
        logger.info(f"Synchronized {len(synchronized_data)} samples")
        
        if len(synchronized_data) == 0:
            raise ValueError(
                "No synchronized samples found. "
                "Check time threshold or sensor availability."
            )
        
        # Split data
        train_data, val_data, test_data = self._split_data(synchronized_data)
        
        logger.info(
            f"Data split - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )
        
        # Process and save all modalities
        logger.info("Processing multimodal data...")
        dataset_manifest = {
            "train": self._process_split(train_data, "train"),
            "val": self._process_split(val_data, "val"),
            "test": self._process_split(test_data, "test"),
        }
        
        # Save dataset manifest
        manifest_path = self.output_dir / "dataset.json"
        with open(manifest_path, "w") as f:
            json.dump(dataset_manifest, f, indent=2)
        logger.info(f"Saved dataset manifest to {manifest_path}")
        
        # Compute and save statistics
        self._save_statistics(synchronized_data)
        
        logger.info("✅ Preprocessing complete!")
        
    def _synchronize_multimodal(
        self,
        image_files: List[Path],
        pointcloud_files: List[Path],
        laserscan_files: List[Path],
        control_df: pd.DataFrame
    ) -> List[Dict]:
        """Synchronize all sensor modalities by timestamp.
        
        타임스탬프로 모든 센서 모달리티를 동기화합니다.
        
        Args:
            image_files: List of image file paths
            pointcloud_files: List of point cloud file paths
            laserscan_files: List of laser scan file paths
            control_df: DataFrame with control commands
            
        Returns:
            List of synchronized data dictionaries
        """
        # Extract timestamps from filenames
        def get_timestamp(path: Path) -> Optional[int]:
            try:
                return int(path.stem.split("_")[-1])
            except (ValueError, IndexError):
                return None
                
        image_ts = {get_timestamp(f): f for f in image_files if get_timestamp(f)}
        pc_ts = {get_timestamp(f): f for f in pointcloud_files if get_timestamp(f)}
        scan_ts = {get_timestamp(f): f for f in laserscan_files if get_timestamp(f)}
        
        control_timestamps = control_df["timestamp"].values
        threshold_ns = int(self.time_threshold * 1e9)
        
        synchronized = []
        
        # Use images as reference
        for img_timestamp, img_path in tqdm(image_ts.items(), desc="Synchronizing"):
            # Find closest point cloud
            pc_diffs = {ts: abs(ts - img_timestamp) for ts in pc_ts.keys()}
            if not pc_diffs:
                continue
            closest_pc_ts = min(pc_diffs, key=pc_diffs.get)
            
            # Find closest laser scan
            scan_diffs = {ts: abs(ts - img_timestamp) for ts in scan_ts.keys()}
            if not scan_diffs:
                continue
            closest_scan_ts = min(scan_diffs, key=scan_diffs.get)
            
            # Find closest control
            ctrl_diffs = np.abs(control_timestamps - img_timestamp)
            closest_ctrl_idx = np.argmin(ctrl_diffs)
            
            # Check if all modalities are within threshold
            if (pc_diffs[closest_pc_ts] <= threshold_ns and
                scan_diffs[closest_scan_ts] <= threshold_ns and
                ctrl_diffs[closest_ctrl_idx] <= threshold_ns):
                
                control_row = control_df.iloc[closest_ctrl_idx]
                synchronized.append({
                    "image_path": str(img_path),
                    "pointcloud_path": str(pc_ts[closest_pc_ts]),
                    "laserscan_path": str(scan_ts[closest_scan_ts]),
                    "timestamp": img_timestamp,
                    "speed": float(control_row["speed"]),
                    "steering": float(control_row["steering"]),
                })
                
        return synchronized
        
    def _split_data(
        self,
        data: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets.
        
        데이터를 train/val/test로 분할합니다.
        
        Args:
            data: List of synchronized data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Sort by timestamp to maintain temporal order
        data = sorted(data, key=lambda x: x["timestamp"])
        
        n_samples = len(data)
        n_train = int(n_samples * self.train_split)
        n_val = int(n_samples * self.val_split)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        return train_data, val_data, test_data
        
    def _process_split(
        self,
        split_data: List[Dict],
        split_name: str
    ) -> List[Dict]:
        """Process and save all modalities for a data split.
        
        데이터 분할에 대한 모든 모달리티를 처리하고 저장합니다.
        
        Args:
            split_data: List of data for this split
            split_name: Name of split ('train', 'val', 'test')
            
        Returns:
            List of processed sample dictionaries
        """
        processed_samples = []
        
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Process image
            img = cv2.imread(sample["image_path"])
            if img is None:
                logger.warning(f"Could not load image: {sample['image_path']}")
                continue
                
            img_resized = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            
            # Save resized image
            img_filename = f"{split_name}_{idx:06d}.jpg"
            img_output_path = self.images_dir / img_filename
            cv2.imwrite(str(img_output_path), img_resized)
            
            # Copy point cloud
            pc_filename = f"{split_name}_{idx:06d}_pc.npy"
            pc_output_path = self.lidar_dir / pc_filename
            shutil.copy2(sample["pointcloud_path"], pc_output_path)
            
            # Copy laser scan
            scan_filename = f"{split_name}_{idx:06d}_scan.npz"
            scan_output_path = self.lidar_dir / scan_filename
            shutil.copy2(sample["laserscan_path"], scan_output_path)
            
            # Create sample entry
            processed_samples.append({
                "image": img_filename,
                "pointcloud": pc_filename,
                "laserscan": scan_filename,
                "timestamp": sample["timestamp"],
                "speed": sample["speed"],
                "steering": sample["steering"],
            })
            
        return processed_samples
        
    def _save_statistics(self, data: List[Dict]) -> None:
        """Compute and save dataset statistics.
        
        데이터셋 통계를 계산하고 저장합니다.
        
        Args:
            data: List of synchronized data
        """
        speeds = [d["speed"] for d in data]
        steerings = [d["steering"] for d in data]
        
        stats = {
            "num_samples": len(data),
            "modalities": ["image", "pointcloud", "laserscan"],
            "speed": {
                "mean": float(np.mean(speeds)),
                "std": float(np.std(speeds)),
                "min": float(np.min(speeds)),
                "max": float(np.max(speeds)),
            },
            "steering": {
                "mean": float(np.mean(steerings)),
                "std": float(np.std(steerings)),
                "min": float(np.min(steerings)),
                "max": float(np.max(steerings)),
            },
            "image_size": self.image_size,
        }
        
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
        
        # Log statistics
        logger.info(f"Dataset statistics:")
        logger.info(f"  Speed: {stats['speed']['mean']:.3f} ± {stats['speed']['std']:.3f} m/s")
        logger.info(f"  Steering: {stats['steering']['mean']:.3f} ± {stats['steering']['std']:.3f} rad")


def parse_args():
    """Parse command line arguments.
    
    명령줄 인수를 파싱합니다.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess multimodal data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with extracted data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (height width)"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training data ratio"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation data ratio"
    )
    
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=0.1,
        help="Maximum time difference for synchronization (seconds)"
    )
    
    return parser.parse_args()


def main():
    """Main function.
    
    메인 함수.
    """
    args = parse_args()
    
    try:
        preprocessor = MultimodalPreprocessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image_size=tuple(args.image_size),
            train_split=args.train_split,
            val_split=args.val_split,
            time_threshold=args.time_threshold,
        )
        
        preprocessor.preprocess()
        
    except KeyboardInterrupt:
        logger.info("Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
        sys.exit(1)


if __name__ == "__main__":
    main()
