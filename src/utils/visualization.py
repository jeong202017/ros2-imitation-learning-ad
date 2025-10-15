"""
Visualization Utilities for Imitation Learning.

Provides functions for:
- Training curve plots
- Data distribution visualization
- Prediction comparison plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def plot_training_curves(
    log_dir: str,
    output_path: Optional[str] = None,
    metrics: List[str] = ["loss"]
) -> None:
    """Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_path: Path to save figure (None to display)
        metrics: List of metrics to plot
    """
    from tensorboard.backend.event_processing import event_accumulator
    
    log_dir = Path(log_dir)
    
    # Find event files
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    
    if not event_files:
        logger.warning(f"No event files found in {log_dir}")
        return
        
    # Load data
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()["scalars"]
    
    # Plot metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
        
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot train and val
        for phase in ["train", "val"]:
            tag = f"{phase}/{metric}"
            if tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=phase.capitalize(), linewidth=2)
                
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training curves saved to {output_path}")
    else:
        plt.show()
        
    plt.close()


def plot_data_distribution(
    data_dir: str,
    output_path: Optional[str] = None
) -> None:
    """Plot distribution of control commands in dataset.
    
    Args:
        data_dir: Directory containing extracted data
        output_path: Path to save figure
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "csv" / "cmd_vel.csv"
    
    if not csv_path.exists():
        logger.error(f"cmd_vel.csv not found in {data_dir}")
        return
        
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear velocity histogram
    axes[0, 0].hist(df["linear_x"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Linear Velocity (m/s)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Linear Velocity Distribution")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Angular velocity histogram
    axes[0, 1].hist(df["angular_z"], bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Angular Velocity (rad/s)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Angular Velocity Distribution")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1, 0].scatter(
        df["linear_x"],
        df["angular_z"],
        alpha=0.3,
        s=10
    )
    axes[1, 0].set_xlabel("Linear Velocity (m/s)")
    axes[1, 0].set_ylabel("Angular Velocity (rad/s)")
    axes[1, 0].set_title("Velocity Command Space")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Time series
    if len(df) > 1000:
        sample_df = df.sample(1000).sort_values("timestamp")
    else:
        sample_df = df.sort_values("timestamp")
        
    time = np.arange(len(sample_df))
    axes[1, 1].plot(time, sample_df["linear_x"].values, label="Linear", alpha=0.7)
    axes[1, 1].plot(time, sample_df["angular_z"].values, label="Angular", alpha=0.7)
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Velocity")
    axes[1, 1].set_title("Velocity Commands Over Time")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Data distribution plot saved to {output_path}")
    else:
        plt.show()
        
    plt.close()


def plot_predictions_vs_ground_truth(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    max_samples: int = 500
) -> None:
    """Plot predicted vs ground truth control commands.
    
    Args:
        predictions: Predicted values of shape (N, 2)
        targets: Ground truth values of shape (N, 2)
        output_path: Path to save figure
        max_samples: Maximum number of samples to plot
    """
    # Sample if too many points
    if len(predictions) > max_samples:
        indices = np.random.choice(len(predictions), max_samples, replace=False)
        predictions = predictions[indices]
        targets = targets[indices]
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear velocity scatter
    axes[0, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=20)
    axes[0, 0].plot(
        [targets[:, 0].min(), targets[:, 0].max()],
        [targets[:, 0].min(), targets[:, 0].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction"
    )
    axes[0, 0].set_xlabel("Ground Truth Linear Velocity")
    axes[0, 0].set_ylabel("Predicted Linear Velocity")
    axes[0, 0].set_title("Linear Velocity Predictions")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Angular velocity scatter
    axes[0, 1].scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=20, color="orange")
    axes[0, 1].plot(
        [targets[:, 1].min(), targets[:, 1].max()],
        [targets[:, 1].min(), targets[:, 1].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction"
    )
    axes[0, 1].set_xlabel("Ground Truth Angular Velocity")
    axes[0, 1].set_ylabel("Predicted Angular Velocity")
    axes[0, 1].set_title("Angular Velocity Predictions")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Linear velocity error distribution
    linear_error = predictions[:, 0] - targets[:, 0]
    axes[1, 0].hist(linear_error, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Prediction Error")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"Linear Velocity Error (MAE: {np.abs(linear_error).mean():.4f})")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Angular velocity error distribution
    angular_error = predictions[:, 1] - targets[:, 1]
    axes[1, 1].hist(angular_error, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Prediction Error")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"Angular Velocity Error (MAE: {np.abs(angular_error).mean():.4f})")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Prediction comparison plot saved to {output_path}")
    else:
        plt.show()
        
    plt.close()


def plot_trajectory_comparison(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    num_steps: int = 100
) -> None:
    """Plot trajectory comparison over time.
    
    Args:
        predictions: Predicted values of shape (N, 2)
        targets: Ground truth values of shape (N, 2)
        output_path: Path to save figure
        num_steps: Number of time steps to plot
    """
    # Take first num_steps
    predictions = predictions[:num_steps]
    targets = targets[:num_steps]
    
    time = np.arange(len(predictions))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Linear velocity
    axes[0].plot(time, targets[:, 0], label="Ground Truth", linewidth=2, alpha=0.7)
    axes[0].plot(time, predictions[:, 0], label="Prediction", linewidth=2, alpha=0.7)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Linear Velocity (m/s)")
    axes[0].set_title("Linear Velocity Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Angular velocity
    axes[1].plot(time, targets[:, 1], label="Ground Truth", linewidth=2, alpha=0.7)
    axes[1].plot(time, predictions[:, 1], label="Prediction", linewidth=2, alpha=0.7)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].set_title("Angular Velocity Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Trajectory comparison plot saved to {output_path}")
    else:
        plt.show()
        
    plt.close()


def plot_confusion_matrix_bins(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 5,
    output_path: Optional[str] = None
) -> None:
    """Plot confusion matrix for binned velocity commands.
    
    Args:
        predictions: Predicted values of shape (N, 2)
        targets: Ground truth values of shape (N, 2)
        num_bins: Number of bins for discretization
        output_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, ax) in enumerate(zip(["Linear", "Angular"], axes)):
        # Bin the values
        pred_binned = np.digitize(predictions[:, idx], np.linspace(
            targets[:, idx].min(),
            targets[:, idx].max(),
            num_bins
        ))
        target_binned = np.digitize(targets[:, idx], np.linspace(
            targets[:, idx].min(),
            targets[:, idx].max(),
            num_bins
        ))
        
        # Compute confusion matrix
        cm = confusion_matrix(target_binned, pred_binned)
        
        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Count"}
        )
        ax.set_xlabel("Predicted Bin")
        ax.set_ylabel("True Bin")
        ax.set_title(f"{name} Velocity Confusion Matrix")
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {output_path}")
    else:
        plt.show()
        
    plt.close()
