"""
Data Preprocessing Module.

Provides utilities for data preprocessing including:
- Image resizing and normalization
- Control command normalization
- Data filtering and outlier removal
- Timestamp synchronization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for imitation learning data.
    
    Args:
        image_size: Target image size (height, width)
        normalize_controls: Whether to normalize control commands
        filter_outliers: Whether to remove outliers
        outlier_threshold: Number of std deviations for outlier detection
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize_controls: bool = True,
        filter_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ):
        self.image_size = image_size
        self.normalize_controls = normalize_controls
        self.filter_outliers = filter_outliers
        self.outlier_threshold = outlier_threshold
        
        # Statistics for normalization
        self.control_stats: Optional[Dict[str, float]] = None
        
    def preprocess_image(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Preprocess image.
        
        Args:
            image: Input image (BGR format)
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed image
        """
        # Resize
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            
        # Convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Normalize
        if normalize:
            image = image.astype(np.float32) / 255.0
            
        return image
        
    def compute_control_statistics(
        self,
        cmd_vel_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute statistics for control normalization.
        
        Args:
            cmd_vel_df: DataFrame with control commands
            
        Returns:
            Dictionary with mean and std for each control dimension
        """
        stats = {}
        
        for column in ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"]:
            if column in cmd_vel_df.columns:
                stats[f"{column}_mean"] = cmd_vel_df[column].mean()
                stats[f"{column}_std"] = cmd_vel_df[column].std()
                
        self.control_stats = stats
        return stats
        
    def normalize_controls(
        self,
        controls: np.ndarray,
        inverse: bool = False
    ) -> np.ndarray:
        """Normalize or denormalize control commands.
        
        Args:
            controls: Array of shape (N, 2) with [linear_x, angular_z]
            inverse: If True, denormalize instead of normalize
            
        Returns:
            Normalized/denormalized controls
        """
        if self.control_stats is None:
            raise ValueError("Control statistics not computed. Call compute_control_statistics first.")
            
        linear_x_mean = self.control_stats.get("linear_x_mean", 0.0)
        linear_x_std = self.control_stats.get("linear_x_std", 1.0)
        angular_z_mean = self.control_stats.get("angular_z_mean", 0.0)
        angular_z_std = self.control_stats.get("angular_z_std", 1.0)
        
        controls_normalized = controls.copy()
        
        if inverse:
            # Denormalize
            controls_normalized[:, 0] = controls[:, 0] * linear_x_std + linear_x_mean
            controls_normalized[:, 1] = controls[:, 1] * angular_z_std + angular_z_mean
        else:
            # Normalize
            controls_normalized[:, 0] = (controls[:, 0] - linear_x_mean) / (linear_x_std + 1e-8)
            controls_normalized[:, 1] = (controls[:, 1] - angular_z_mean) / (angular_z_std + 1e-8)
            
        return controls_normalized
        
    def filter_control_outliers(
        self,
        cmd_vel_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove outliers from control commands.
        
        Args:
            cmd_vel_df: DataFrame with control commands
            
        Returns:
            Filtered DataFrame
        """
        if not self.filter_outliers:
            return cmd_vel_df
            
        df_filtered = cmd_vel_df.copy()
        
        for column in ["linear_x", "angular_z"]:
            if column not in df_filtered.columns:
                continue
                
            mean = df_filtered[column].mean()
            std = df_filtered[column].std()
            
            # Remove outliers beyond threshold
            mask = np.abs(df_filtered[column] - mean) <= (self.outlier_threshold * std)
            removed_count = len(df_filtered) - mask.sum()
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outliers from {column}")
                df_filtered = df_filtered[mask]
                
        return df_filtered
        
    def synchronize_timestamps(
        self,
        image_timestamps: List[int],
        cmd_vel_df: pd.DataFrame,
        max_time_diff: int = 100_000_000,  # 100ms in nanoseconds
    ) -> List[Tuple[int, int]]:
        """Synchronize image timestamps with control commands.
        
        Args:
            image_timestamps: List of image timestamps (nanoseconds)
            cmd_vel_df: DataFrame with control commands
            max_time_diff: Maximum allowed time difference (nanoseconds)
            
        Returns:
            List of tuples (image_timestamp, cmd_vel_index)
        """
        synchronized = []
        cmd_timestamps = cmd_vel_df["timestamp"].values
        
        for img_ts in image_timestamps:
            # Find closest command timestamp
            time_diffs = np.abs(cmd_timestamps - img_ts)
            closest_idx = np.argmin(time_diffs)
            
            # Only include if within threshold
            if time_diffs[closest_idx] <= max_time_diff:
                synchronized.append((img_ts, closest_idx))
            else:
                logger.debug(
                    f"Image timestamp {img_ts} has no close command "
                    f"(min diff: {time_diffs[closest_idx] / 1e6:.1f}ms)"
                )
                
        logger.info(
            f"Synchronized {len(synchronized)}/{len(image_timestamps)} images "
            f"with control commands"
        )
        
        return synchronized
        
    def filter_low_speed_samples(
        self,
        cmd_vel_df: pd.DataFrame,
        min_speed: float = 0.01
    ) -> pd.DataFrame:
        """Filter out samples with very low speed (stationary robot).
        
        Args:
            cmd_vel_df: DataFrame with control commands
            min_speed: Minimum linear speed threshold
            
        Returns:
            Filtered DataFrame
        """
        # Calculate speed magnitude
        speed = np.sqrt(
            cmd_vel_df["linear_x"]**2 + 
            cmd_vel_df["linear_y"]**2 if "linear_y" in cmd_vel_df.columns else 0
        )
        
        mask = speed >= min_speed
        filtered_df = cmd_vel_df[mask]
        
        removed_count = len(cmd_vel_df) - len(filtered_df)
        logger.info(f"Removed {removed_count} low-speed samples (< {min_speed} m/s)")
        
        return filtered_df
        
    def clip_controls(
        self,
        cmd_vel_df: pd.DataFrame,
        linear_x_range: Tuple[float, float] = (-1.0, 1.0),
        angular_z_range: Tuple[float, float] = (-2.0, 2.0),
    ) -> pd.DataFrame:
        """Clip control values to specified ranges.
        
        Args:
            cmd_vel_df: DataFrame with control commands
            linear_x_range: (min, max) for linear velocity
            angular_z_range: (min, max) for angular velocity
            
        Returns:
            DataFrame with clipped values
        """
        df_clipped = cmd_vel_df.copy()
        
        if "linear_x" in df_clipped.columns:
            df_clipped["linear_x"] = np.clip(
                df_clipped["linear_x"],
                linear_x_range[0],
                linear_x_range[1]
            )
            
        if "angular_z" in df_clipped.columns:
            df_clipped["angular_z"] = np.clip(
                df_clipped["angular_z"],
                angular_z_range[0],
                angular_z_range[1]
            )
            
        return df_clipped
        
    def smooth_controls(
        self,
        cmd_vel_df: pd.DataFrame,
        window_size: int = 5
    ) -> pd.DataFrame:
        """Apply moving average smoothing to control commands.
        
        Args:
            cmd_vel_df: DataFrame with control commands
            window_size: Size of smoothing window
            
        Returns:
            DataFrame with smoothed values
        """
        df_smoothed = cmd_vel_df.copy()
        
        for column in ["linear_x", "angular_z"]:
            if column in df_smoothed.columns:
                df_smoothed[column] = (
                    df_smoothed[column]
                    .rolling(window=window_size, center=True, min_periods=1)
                    .mean()
                )
                
        return df_smoothed
        
    def process_dataset(
        self,
        data_dir: Path,
        output_dir: Path,
        filter_config: Optional[Dict] = None
    ) -> None:
        """Process entire dataset with filtering and preprocessing.
        
        Args:
            data_dir: Input data directory
            output_dir: Output directory for processed data
            filter_config: Configuration for filtering (optional)
        """
        logger.info(f"Processing dataset from {data_dir}")
        
        # Load control commands
        csv_dir = data_dir / "csv"
        cmd_vel_path = csv_dir / "cmd_vel.csv"
        
        if not cmd_vel_path.exists():
            raise ValueError(f"cmd_vel.csv not found in {csv_dir}")
            
        cmd_vel_df = pd.read_csv(cmd_vel_path)
        original_size = len(cmd_vel_df)
        
        # Apply filters
        if filter_config:
            if filter_config.get("filter_outliers", True):
                cmd_vel_df = self.filter_control_outliers(cmd_vel_df)
                
            if filter_config.get("filter_low_speed", False):
                min_speed = filter_config.get("min_speed", 0.01)
                cmd_vel_df = self.filter_low_speed_samples(cmd_vel_df, min_speed)
                
            if filter_config.get("clip_controls", False):
                cmd_vel_df = self.clip_controls(cmd_vel_df)
                
            if filter_config.get("smooth_controls", False):
                window_size = filter_config.get("smooth_window", 5)
                cmd_vel_df = self.smooth_controls(cmd_vel_df, window_size)
                
        # Compute statistics
        stats = self.compute_control_statistics(cmd_vel_df)
        
        # Save processed data
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv_dir = output_dir / "csv"
        output_csv_dir.mkdir(exist_ok=True)
        
        cmd_vel_df.to_csv(output_csv_dir / "cmd_vel.csv", index=False)
        
        # Save statistics
        import json
        stats_path = output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
            
        logger.info(
            f"Processing complete. {len(cmd_vel_df)}/{original_size} samples retained. "
            f"Data saved to {output_dir}"
        )
