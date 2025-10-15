#!/usr/bin/env python3
"""
Model Training Script.

Trains imitation learning models using extracted bag data.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.dataset import create_data_splits, get_augmentation_transform
from models.cnn_policy import create_cnn_policy
from training.bc_trainer import create_bc_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train imitation learning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default="outputs",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu, overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
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
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
        
    # Set device
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
        
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading data from {args.data_dir}")
    
    train_dataset, val_dataset, test_dataset = create_data_splits(
        data_dir=args.data_dir,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
        sequence_length=config["data"].get("sequence_length", 1),
        image_size=tuple(config["model"].get("input_size", [224, 224])),
        augmentation_config=config.get("augmentation"),
    )
    
    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    
    # Create data loaders
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Create model
    logger.info("Creating model")
    temporal = config["data"].get("sequence_length", 1) > 1
    model = create_cnn_policy(
        config=config["model"],
        temporal=temporal
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Create trainer
    logger.info("Creating trainer")
    
    training_config = config["training"].copy()
    training_config["log_dir"] = str(log_dir)
    training_config["checkpoint_dir"] = str(checkpoint_dir)
    
    trainer = create_bc_trainer(
        model=model,
        config=training_config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        
    # Train
    logger.info("Starting training")
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["training"].get("epochs", 100),
            save_interval=config["training"].get("save_interval", 10),
            early_stopping_patience=config["training"].get("early_stopping_patience"),
        )
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Models saved to {checkpoint_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted.pth")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
