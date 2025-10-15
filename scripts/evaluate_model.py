#!/usr/bin/env python3
"""
Model Evaluation Script.

Evaluates trained models on test data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.dataset import create_data_splits
from evaluation.evaluator import Evaluator, load_model_for_evaluation
from models.cnn_policy import create_cnn_policy
from utils.visualization import (
    plot_predictions_vs_ground_truth,
    plot_trajectory_comparison,
    plot_confusion_matrix_bins,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate imitation learning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file (YAML)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing extracted bag data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (cuda/cpu)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Evaluate inference timing"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
        
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading data from {args.data_dir}")
    
    train_dataset, val_dataset, test_dataset = create_data_splits(
        data_dir=args.data_dir,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
        sequence_length=config["data"].get("sequence_length", 1),
        image_size=tuple(config["model"].get("input_size", [224, 224])),
        augmentation_config=None,  # No augmentation for evaluation
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    temporal = config["data"].get("sequence_length", 1) > 1
    model = create_cnn_policy(
        config=config["model"],
        temporal=temporal
    )
    
    model = load_model_for_evaluation(
        model=model,
        checkpoint_path=args.model_path,
        device=device
    )
    
    # Create evaluator
    evaluator = Evaluator(model=model, device=device)
    
    # Run evaluation
    logger.info("Evaluating model on test set")
    metrics = evaluator.evaluate(test_loader)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Get predictions for visualization
    if args.visualize:
        logger.info("Generating visualizations")
        predictions, targets = evaluator.get_predictions(test_loader)
        
        # Save predictions
        evaluator.save_results(
            predictions=predictions,
            targets=targets,
            output_path=str(output_dir / "predictions.csv")
        )
        
        # Generate plots
        plot_predictions_vs_ground_truth(
            predictions=predictions,
            targets=targets,
            output_path=str(output_dir / "predictions_vs_ground_truth.png")
        )
        
        plot_trajectory_comparison(
            predictions=predictions,
            targets=targets,
            output_path=str(output_dir / "trajectory_comparison.png"),
            num_steps=100
        )
        
        plot_confusion_matrix_bins(
            predictions=predictions,
            targets=targets,
            output_path=str(output_dir / "confusion_matrix.png")
        )
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    # Evaluate timing
    if args.timing:
        logger.info("Evaluating inference timing")
        timing_metrics = evaluator.evaluate_realtime_performance(
            test_loader=test_loader,
            num_samples=100
        )
        
        # Save timing metrics
        timing_path = output_dir / "timing_metrics.json"
        with open(timing_path, "w") as f:
            json.dump(timing_metrics, f, indent=2)
        logger.info(f"Timing metrics saved to {timing_path}")
        
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
