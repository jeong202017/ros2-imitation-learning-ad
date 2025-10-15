"""
Behavioral Cloning Trainer.

Specialized trainer for behavioral cloning with MSE loss.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from .trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCTrainer(Trainer):
    """Behavioral Cloning trainer with MSE loss.
    
    Extends base Trainer with BC-specific features:
    - MSE loss for continuous actions
    - Adam/AdamW optimizer
    - Learning rate scheduling
    - Gradient clipping
    
    Args:
        model: Policy network
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        optimizer_type: Optimizer type ('adam' or 'adamw')
        scheduler_type: Scheduler type ('step' or 'cosine')
        scheduler_params: Scheduler parameters
        device: Device to train on
        log_dir: TensorBoard log directory
        checkpoint_dir: Checkpoint directory
        clip_grad: Gradient clipping value
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = "step",
        scheduler_params: Optional[Dict] = None,
        device: str = "cuda",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        clip_grad: Optional[float] = 1.0,
    ):
        # Create optimizer
        if optimizer_type.lower() == "adam":
            optimizer = Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        # Create criterion (MSE loss for continuous actions)
        criterion = nn.MSELoss()
        
        # Create scheduler
        scheduler = None
        if scheduler_type is not None:
            if scheduler_params is None:
                scheduler_params = {}
                
            if scheduler_type.lower() == "step":
                step_size = scheduler_params.get("step_size", 30)
                gamma = scheduler_params.get("gamma", 0.1)
                scheduler = StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
            elif scheduler_type.lower() == "cosine":
                T_max = scheduler_params.get("T_max", 100)
                eta_min = scheduler_params.get("eta_min", 1e-6)
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}")
                
        # Initialize base trainer
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            scheduler=scheduler,
        )
        
        self.clip_grad = clip_grad
        
        logger.info(
            f"BCTrainer initialized with {optimizer_type} optimizer "
            f"(lr={learning_rate}, weight_decay={weight_decay})"
        )
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with gradient clipping.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        return super().train_epoch(train_loader, clip_grad=self.clip_grad)
        
    @torch.no_grad()
    def compute_action_metrics(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Compute detailed action prediction metrics.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with per-action MAE and MSE
        """
        self.model.eval()
        
        total_linear_mae = 0.0
        total_angular_mae = 0.0
        total_linear_mse = 0.0
        total_angular_mse = 0.0
        num_samples = 0
        
        for batch in val_loader:
            images, controls = batch
            images = images.to(self.device)
            controls = controls.to(self.device)
            
            # Predict
            predictions = self.model(images)
            
            # Compute per-action errors
            linear_error = (predictions[:, 0] - controls[:, 0]).abs()
            angular_error = (predictions[:, 1] - controls[:, 1]).abs()
            
            total_linear_mae += linear_error.sum().item()
            total_angular_mae += angular_error.sum().item()
            total_linear_mse += (linear_error ** 2).sum().item()
            total_angular_mse += (angular_error ** 2).sum().item()
            num_samples += images.size(0)
            
        metrics = {
            "linear_x_mae": total_linear_mae / num_samples,
            "angular_z_mae": total_angular_mae / num_samples,
            "linear_x_mse": total_linear_mse / num_samples,
            "angular_z_mse": total_angular_mse / num_samples,
        }
        
        return metrics
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_interval: int = 10,
        early_stopping_patience: Optional[int] = None,
        log_metrics_interval: int = 5,
    ) -> None:
        """Training loop with BC-specific logging.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_interval: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
            log_metrics_interval: Log detailed metrics every N epochs
        """
        logger.info(f"Starting BC training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Compute detailed metrics periodically
            if (epoch + 1) % log_metrics_interval == 0:
                action_metrics = self.compute_action_metrics(val_loader)
                
                logger.info(
                    f"Epoch {epoch} - Action Metrics: "
                    f"Linear MAE={action_metrics['linear_x_mae']:.4f}, "
                    f"Angular MAE={action_metrics['angular_z_mae']:.4f}"
                )
                
                # Log to tensorboard
                for key, value in action_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                    
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
        logger.info("BC training complete")


def create_bc_trainer(
    model: nn.Module,
    config: Dict,
    device: str = "cuda"
) -> BCTrainer:
    """Factory function to create BC trainer from config.
    
    Args:
        model: Policy network
        config: Training configuration
        device: Device to train on
        
    Returns:
        BCTrainer instance
    """
    return BCTrainer(
        model=model,
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
        optimizer_type=config.get("optimizer", "adam"),
        scheduler_type=config.get("scheduler_type", "step"),
        scheduler_params=config.get("scheduler_params"),
        device=device,
        log_dir=config.get("log_dir", "logs"),
        checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
        clip_grad=config.get("clip_grad", 1.0),
    )
