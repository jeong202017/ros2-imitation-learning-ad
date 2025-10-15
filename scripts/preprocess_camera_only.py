#!/usr/bin/env python3
"""
Camera-Only Data Preprocessing Script.

Preprocesses extracted ERP42 data for camera-only experiment:
- Synchronizes images with control commands by timestamp
- Resizes images to target size
- Splits data into train/val/test sets
- Generates dataset.json manifest

카메라 전용 실험을 위한 추출된 ERP42 데이터 전처리:
- 타임스탬프로 이미지와 제어 명령 동기화
- 이미지를 목표 크기로 리사이징
- 데이터를 train/val/test로 분할
- dataset.json 매니페스트 생성
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CameraOnlyPreprocessor:
    """Preprocess camera-only data for training.
    
    카메라 전용 데이터를 학습을 위해 전처리합니다.
    
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
        self.csv_dir = self.output_dir / "csv"
        
        for directory in [self.images_dir, self.csv_dir]:
            directory.mkdir(exist_ok=True)
            
    def preprocess(self) -> None:
        """Run preprocessing pipeline.
        
        전처리 파이프라인을 실행합니다.
        """
        logger.info("Starting camera-only preprocessing")
        
        # Load control data
        control_path = self.input_dir / "csv" / "control_data.csv"
        if not control_path.exists():
            raise FileNotFoundError(f"Control data not found: {control_path}")
            
        control_df = pd.read_csv(control_path)
        logger.info(f"Loaded {len(control_df)} control commands")
        
        # Get image files
        images_path = self.input_dir / "images"
        image_files = sorted(list(images_path.glob("*.jpg")))
        logger.info(f"Found {len(image_files)} images")
        
        # Synchronize images with controls
        logger.info("Synchronizing images with control commands...")
        synchronized_data = self._synchronize_data(image_files, control_df)
        logger.info(f"Synchronized {len(synchronized_data)} samples")
        
        if len(synchronized_data) == 0:
            raise ValueError("No synchronized samples found. Check time threshold.")
        
        # Split data
        train_data, val_data, test_data = self._split_data(synchronized_data)
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Process and save images
        logger.info("Processing images...")
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
        
    def _synchronize_data(
        self, 
        image_files: List[Path], 
        control_df: pd.DataFrame
    ) -> List[Dict]:
        """Synchronize images with control commands by timestamp.
        
        타임스탬프로 이미지와 제어 명령을 동기화합니다.
        
        Args:
            image_files: List of image file paths
            control_df: DataFrame with control commands
            
        Returns:
            List of synchronized data dictionaries
        """
        synchronized = []
        control_timestamps = control_df["timestamp"].values
        threshold_ns = int(self.time_threshold * 1e9)  # Convert to nanoseconds
        
        for img_path in tqdm(image_files, desc="Synchronizing"):
            # Extract timestamp from filename
            try:
                timestamp_str = img_path.stem.split("_")[-1]
                img_timestamp = int(timestamp_str)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse timestamp from {img_path.name}")
                continue
                
            # Find closest control command
            time_diffs = np.abs(control_timestamps - img_timestamp)
            closest_idx = np.argmin(time_diffs)
            
            # Only include if within threshold
            if time_diffs[closest_idx] <= threshold_ns:
                control_row = control_df.iloc[closest_idx]
                synchronized.append({
                    "image_path": str(img_path),
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
        """Process and save images for a data split.
        
        데이터 분할에 대한 이미지를 처리하고 저장합니다.
        
        Args:
            split_data: List of data for this split
            split_name: Name of split ('train', 'val', 'test')
            
        Returns:
            List of processed sample dictionaries
        """
        processed_samples = []
        
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Load and resize image
            img = cv2.imread(sample["image_path"])
            if img is None:
                logger.warning(f"Could not load image: {sample['image_path']}")
                continue
                
            img_resized = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            
            # Save resized image
            output_filename = f"{split_name}_{idx:06d}.jpg"
            output_path = self.images_dir / output_filename
            cv2.imwrite(str(output_path), img_resized)
            
            # Create sample entry
            processed_samples.append({
                "image": output_filename,
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
        description="Preprocess camera-only data for training",
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
        preprocessor = CameraOnlyPreprocessor(
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
