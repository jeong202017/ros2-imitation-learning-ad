"""
Model Evaluation Module.

Evaluates trained models on test data with metrics and visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for imitation learning models.
    
    Args:
        model: Trained model
        device: Device to run evaluation on
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_targets = []
        
        logger.info("Running evaluation...")
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, controls = batch
            images = images.to(self.device)
            controls = controls.to(self.device)
            
            # Predict
            predictions = self.model(images)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(controls.cpu().numpy())
            
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)
        
        # Log results
        logger.info("Evaluation Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
        return metrics
        
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Predicted values of shape (N, output_dim)
            targets: Ground truth values of shape (N, output_dim)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics["mae"] = mean_absolute_error(targets, predictions)
        metrics["mse"] = mean_squared_error(targets, predictions)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        
        # Per-action metrics (assuming [linear_x, angular_z])
        if predictions.shape[1] >= 2:
            # Linear velocity
            metrics["linear_x_mae"] = mean_absolute_error(
                targets[:, 0],
                predictions[:, 0]
            )
            metrics["linear_x_mse"] = mean_squared_error(
                targets[:, 0],
                predictions[:, 0]
            )
            metrics["linear_x_rmse"] = np.sqrt(metrics["linear_x_mse"])
            
            try:
                metrics["linear_x_r2"] = r2_score(
                    targets[:, 0],
                    predictions[:, 0]
                )
            except:
                metrics["linear_x_r2"] = 0.0
                
            # Angular velocity
            metrics["angular_z_mae"] = mean_absolute_error(
                targets[:, 1],
                predictions[:, 1]
            )
            metrics["angular_z_mse"] = mean_squared_error(
                targets[:, 1],
                predictions[:, 1]
            )
            metrics["angular_z_rmse"] = np.sqrt(metrics["angular_z_mse"])
            
            try:
                metrics["angular_z_r2"] = r2_score(
                    targets[:, 1],
                    predictions[:, 1]
                )
            except:
                metrics["angular_z_r2"] = 0.0
                
        return metrics
        
    @torch.no_grad()
    def get_predictions(
        self,
        test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get all predictions and targets.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(test_loader, desc="Getting predictions"):
            images, controls = batch
            images = images.to(self.device)
            
            # Predict
            predictions = self.model(images)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(controls.numpy())
            
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets
        
    def evaluate_realtime_performance(
        self,
        test_loader: DataLoader,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Evaluate real-time inference performance.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with timing metrics
        """
        import time
        
        inference_times = []
        
        logger.info("Evaluating real-time performance...")
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_samples:
                    break
                    
                images, _ = batch
                images = images.to(self.device)
                
                # Measure inference time
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                start_time = time.time()
                _ = self.model(images)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
        inference_times = np.array(inference_times)
        
        metrics = {
            "mean_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "fps": 1.0 / np.mean(inference_times),
        }
        
        logger.info("Real-time Performance:")
        logger.info(f"  Mean inference time: {metrics['mean_inference_time']*1000:.2f} ms")
        logger.info(f"  FPS: {metrics['fps']:.2f}")
        
        return metrics
        
    def save_results(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: str
    ) -> None:
        """Save evaluation results to file.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            output_path: Path to save results
        """
        import pandas as pd
        
        # Create DataFrame
        data = {
            "pred_linear_x": predictions[:, 0],
            "pred_angular_z": predictions[:, 1],
            "true_linear_x": targets[:, 0],
            "true_angular_z": targets[:, 1],
        }
        
        df = pd.DataFrame(data)
        
        # Compute errors
        df["error_linear_x"] = df["pred_linear_x"] - df["true_linear_x"]
        df["error_angular_z"] = df["pred_angular_z"] - df["true_angular_z"]
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")


def load_model_for_evaluation(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cuda"
) -> nn.Module:
    """Load trained model from checkpoint for evaluation.
    
    Args:
        model: Model architecture (uninitialized)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    
    return model
