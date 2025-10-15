"""
Generic Trainer for Imitation Learning.

Provides base training loop with:
- Training and validation
- TensorBoard logging
- Model checkpointing
- Early stopping
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for imitation learning.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on ('cuda' or 'cpu')
        log_dir: Directory for TensorBoard logs
        checkpoint_dir: Directory to save model checkpoints
        scheduler: Optional learning rate scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        scheduler: Optional[_LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Setup checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        
        logger.info(f"Trainer initialized on device: {device}")
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        clip_grad: Optional[float] = None
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            clip_grad: Gradient clipping value (None to disable)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        for batch in pbar:
            images, controls = batch
            images = images.to(self.device)
            controls = controls.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, controls)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    clip_grad
                )
                
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to tensorboard
            self.writer.add_scalar(
                "train/batch_loss",
                loss.item(),
                self.global_step
            )
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / num_batches
        
        return {"loss": avg_loss}
        
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        for batch in pbar:
            images, controls = batch
            images = images.to(self.device)
            controls = controls.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, controls)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / num_batches
        
        return {"loss": avg_loss}
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_interval: int = 10,
        early_stopping_patience: Optional[int] = None,
        clip_grad: Optional[float] = None,
    ) -> None:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            clip_grad: Gradient clipping value
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, clip_grad=clip_grad)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/learning_rate", current_lr, epoch)
                
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss = {train_metrics['loss']:.4f}, "
                f"Val Loss = {val_metrics['loss']:.4f}"
            )
            
            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
                
            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pth")
                logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                
            # Early stopping
            if early_stopping_patience is not None:
                if self.epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                    break
                    
        # Save final model
        self.save_checkpoint("final_model.pth")
        logger.info("Training complete")
        
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def close(self) -> None:
        """Close trainer and cleanup resources."""
        self.writer.close()
