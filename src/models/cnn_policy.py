"""
CNN-based Policy Networks for Imitation Learning.

Implements:
- CNNPolicy: ResNet-based policy for single frame input
- TemporalCNNPolicy: CNN + LSTM for temporal sequences
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNPolicy(nn.Module):
    """CNN-based policy network using ResNet backbone.
    
    Maps images to control commands [linear_x, angular_z].
    
    Args:
        backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        input_size: Input image size (height, width)
        output_dim: Output dimension (default: 2 for [linear_x, angular_z])
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        input_size: Tuple[int, int] = (224, 224),
        output_dim: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.input_size = input_size
        self.output_dim = output_dim
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        logger.info(
            f"CNNPolicy initialized with {backbone} backbone "
            f"(pretrained={pretrained})"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Control commands of shape (B, output_dim)
        """
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Predict controls
        controls = self.policy_head(features)
        
        return controls
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict with output range constraints.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Control commands with constrained ranges
        """
        controls = self.forward(x)
        
        # Apply range constraints
        # linear_x: typically [0, max_speed] or [-max_speed, max_speed]
        # angular_z: typically [-max_angular, max_angular]
        
        # Using tanh for bounded outputs
        # Adjust scale as needed based on your robot's limits
        controls[:, 0] = torch.sigmoid(controls[:, 0])  # linear_x in [0, 1]
        controls[:, 1] = torch.tanh(controls[:, 1])     # angular_z in [-1, 1]
        
        return controls


class TemporalCNNPolicy(nn.Module):
    """CNN + LSTM policy for temporal sequences.
    
    Processes sequences of images with CNN encoder and LSTM temporal module.
    
    Args:
        backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        input_size: Input image size (height, width)
        output_dim: Output dimension (default: 2 for [linear_x, angular_z])
        hidden_dim: Hidden dimension for LSTM and FC layers
        lstm_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        input_size: Tuple[int, int] = (224, 224),
        output_dim: int = 2,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.input_size = input_size
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Load CNN backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        logger.info(
            f"TemporalCNNPolicy initialized with {backbone} backbone "
            f"and {lstm_layers}-layer LSTM"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input image sequences of shape (B, T, C, H, W)
               where T is sequence length
            
        Returns:
            Control commands of shape (B, output_dim)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each frame
        # Reshape to (B*T, C, H, W) for batch processing
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Reshape back to (B, T, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Process temporal sequence with LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Predict controls
        controls = self.policy_head(last_output)
        
        return controls
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict with output range constraints.
        
        Args:
            x: Input image sequences of shape (B, T, C, H, W)
            
        Returns:
            Control commands with constrained ranges
        """
        controls = self.forward(x)
        
        # Apply range constraints
        controls[:, 0] = torch.sigmoid(controls[:, 0])  # linear_x in [0, 1]
        controls[:, 1] = torch.tanh(controls[:, 1])     # angular_z in [-1, 1]
        
        return controls


def create_cnn_policy(
    config: dict,
    temporal: bool = False
) -> nn.Module:
    """Factory function to create CNN policy from config.
    
    Args:
        config: Configuration dictionary with model parameters
        temporal: Whether to create temporal policy
        
    Returns:
        CNN policy network
    """
    backbone = config.get("backbone", "resnet18")
    pretrained = config.get("pretrained", True)
    input_size = tuple(config.get("input_size", [224, 224]))
    output_dim = config.get("output_dim", 2)
    hidden_dim = config.get("hidden_dim", 256)
    dropout = config.get("dropout", 0.3)
    
    if temporal:
        lstm_layers = config.get("lstm_layers", 2)
        return TemporalCNNPolicy(
            backbone=backbone,
            pretrained=pretrained,
            input_size=input_size,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
    else:
        return CNNPolicy(
            backbone=backbone,
            pretrained=pretrained,
            input_size=input_size,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
