#!/usr/bin/env python3
"""
Multimodal Model Training Script.

Trains multimodal model using camera and LiDAR for control prediction.
카메라와 LiDAR를 사용하여 제어 예측을 위한 멀티모달 모델을 학습합니다.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.dataset_multimodal import MultimodalDataset
from models.multimodal_policy import (
    MultimodalPolicy, 
    PointNetSimple, 
    LaserScanCNN1D,
    CameraEncoder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedMultimodalPolicy(nn.Module):
    """Enhanced multimodal policy with separate encoders for each LiDAR type.
    
    각 LiDAR 유형에 대한 별도 인코더가 있는 향상된 멀티모달 정책.
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Camera encoder
        camera_config = config["camera_encoder"]
        self.camera_encoder = CameraEncoder(
            backbone=camera_config["backbone"],
            pretrained=camera_config["pretrained"],
            output_dim=camera_config["output_dim"],
        )
        
        # LiDAR encoders
        lidar_config = config["lidar_encoder"]
        self.pointnet = PointNetSimple(
            output_dim=lidar_config["output_dim"],
            hidden_dims=tuple(lidar_config.get("hidden_dims", [64, 128, 256])),
        )
        
        laserscan_config = config.get("laserscan_encoder", {})
        laserscan_bins = config.get("laserscan_bins", 360)
        self.laserscan_cnn = LaserScanCNN1D(
            input_dim=laserscan_bins,
            output_dim=laserscan_config.get("output_dim", 128),
            hidden_dims=tuple(laserscan_config.get("hidden_dims", [64, 128, 256])),
        )
        
        # Fusion
        fusion_config = config["fusion"]
        total_feature_dim = (
            camera_config["output_dim"] + 
            lidar_config["output_dim"] + 
            laserscan_config.get("output_dim", 128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_config["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(fusion_config.get("dropout", 0.3)),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_config["hidden_dim"], fusion_config["hidden_dim"] // 2),
            nn.ReLU(),
            nn.Dropout(fusion_config.get("dropout", 0.3)),
            nn.Linear(fusion_config["hidden_dim"] // 2, config["output_dim"]),
        )
        
    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.
        
        Args:
            batch: Dictionary with 'image', 'pointcloud', 'laserscan'
            
        Returns:
            Control predictions of shape (B, output_dim)
        """
        # Encode each modality
        camera_features = self.camera_encoder(batch["image"])
        pointcloud_features = self.pointnet(batch["pointcloud"])
        laserscan_features = self.laserscan_cnn(batch["laserscan"])
        
        # Concatenate features
        fused = torch.cat([camera_features, pointcloud_features, laserscan_features], dim=1)
        
        # Apply fusion layer
        fused = self.fusion(fused)
        
        # Predict controls
        controls = self.policy_head(fused)
        
        return controls


class MultimodalTrainer:
    """Trainer for multimodal model.
    
    멀티모달 모델 학습기.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        output_dir: Directory to save models and logs
        device: Device to train on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: Path,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.device = device
        
        # Create output directories
        self.checkpoint_dir = output_dir / "checkpoints"
        self.log_dir = output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        lr = config["training"]["learning_rate"]
        weight_decay = config["training"].get("weight_decay", 0.0)
        
        if config["training"]["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif config["training"]["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        elif config["training"]["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
            
        # Setup learning rate scheduler
        self.scheduler = None
        if config["training"].get("lr_scheduler") == "step":
            step_size = config["training"].get("lr_step_size", 30)
            gamma = config["training"].get("lr_gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif config["training"].get("lr_scheduler") == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["training"]["epochs"]
            )
            
        # Setup loss function
        loss_type = config["loss"]["type"]
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.gradient_clip = config["training"].get("gradient_clip", None)
        
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Optimizer: {config['training']['optimizer']}, LR: {lr}")
        logger.info(f"Loss: {loss_type}")
        
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        한 에폭 동안 학습합니다.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch in pbar:
            # Move batch to device
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            controls = batch_device.pop("target")
            batch = batch_device
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            loss = self.criterion(predictions, controls)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def validate(self) -> float:
        """Validate model.
        
        모델을 검증합니다.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
            
            for batch in pbar:
                # Move batch to device
                batch_device = {k: v.to(self.device) for k, v in batch.items()}
                controls = batch_device.pop("target")
                batch = batch_device
                
                # Forward pass
                predictions = self.model(batch)
                loss = self.criterion(predictions, controls)
                
                # Update stats
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def train(self, epochs: int, save_interval: int = 10, early_stopping_patience: int = None):
        """Train model for multiple epochs.
        
        여러 에폭 동안 모델을 학습합니다.
        
        Args:
            epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
            early_stopping_patience: Stop if val loss doesn't improve for N epochs
        """
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log metrics
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar(
                "learning_rate", 
                self.optimizer.param_groups[0]["lr"], 
                epoch
            )
            
            logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
                logger.info(f"✓ New best model saved (val_loss={val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
                
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {early_stopping_patience} "
                    f"epochs without improvement"
                )
                break
                
        # Save final model
        self.save_checkpoint("last_model.pth")
        logger.info("Training complete!")
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint.
        
        모델 체크포인트를 저장합니다.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
    def close(self):
        """Close resources.
        
        리소스를 닫습니다.
        """
        self.writer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multimodal model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with preprocessed data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
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
    
    # Save config to output directory
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {config_save_path}")
    
    # Load datasets
    logger.info(f"Loading datasets from {args.data_dir}")
    
    max_points = config["data"].get("max_points", 2048)
    laserscan_bins = config["data"].get("laserscan_bins", 360)
    
    train_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="train",
        use_pointcloud=config["data"].get("use_pointcloud", True),
        use_laserscan=config["data"].get("use_laserscan", True),
        normalize=True,
        max_points=max_points,
        laserscan_bins=laserscan_bins,
    )
    
    val_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="val",
        use_pointcloud=config["data"].get("use_pointcloud", True),
        use_laserscan=config["data"].get("use_laserscan", True),
        normalize=True,
        max_points=max_points,
        laserscan_bins=laserscan_bins,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data loaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    
    # Create model
    logger.info("Creating multimodal model...")
    # Create model config with laserscan_bins
    model_config = config["model"].copy()
    model_config["laserscan_bins"] = laserscan_bins
    model = EnhancedMultimodalPolicy(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        device=device,
    )
    
    # Train
    epochs = args.epochs or config["training"]["epochs"]
    save_interval = config["training"].get("save_interval", 10)
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    try:
        trainer.train(
            epochs=epochs,
            save_interval=save_interval,
            early_stopping_patience=early_stopping_patience,
        )
        
        logger.info("✅ Training complete!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Models saved to {output_dir / 'checkpoints'}")
        
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
