"""
Multimodal Policy Networks for Imitation Learning.

Implements sensor fusion architectures combining:
- Camera images
- LiDAR/LaserScan data
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraEncoder(nn.Module):
    """CNN encoder for camera images.
    
    Args:
        backbone: ResNet backbone ('resnet18', 'resnet34')
        pretrained: Whether to use pretrained weights
        output_dim: Output feature dimension
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        output_dim: int = 256,
    ):
        super().__init__()
        
        # Load backbone
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, output_dim)
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        features = self.projection(features)
        return features


class LiDAREncoder(nn.Module):
    """Encoder for LiDAR/LaserScan data.
    
    Processes 1D range scans or 2D point clouds.
    
    Args:
        input_dim: Input dimension (e.g., 360 for 360-degree scan)
        output_dim: Output feature dimension
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int = 360,
        output_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input LiDAR data of shape (B, input_dim)
            
        Returns:
            Features of shape (B, output_dim)
        """
        return self.encoder(x)


class PointNetSimple(nn.Module):
    """Simplified PointNet encoder for 3D point clouds.
    
    3D 포인트 클라우드를 위한 간소화된 PointNet 인코더.
    
    Processes unordered point sets with permutation invariance.
    순서가 없는 포인트 집합을 순열 불변성으로 처리합니다.
    
    Args:
        output_dim: Output feature dimension
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        
        # Shared MLP for point features
        layers = []
        prev_dim = 3  # xyz coordinates
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
            
        self.point_mlp = nn.Sequential(*layers)
        
        # Global feature aggregation
        self.global_mlp = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with max pooling for permutation invariance.
        
        순열 불변성을 위한 max pooling을 사용한 순전파.
        
        Args:
            x: Point cloud of shape (B, N, 3) where N is number of points
            
        Returns:
            Global features of shape (B, output_dim)
        """
        batch_size, num_points, _ = x.shape
        
        # Reshape for shared MLP: (B*N, 3) -> (B*N, hidden_dim)
        x = x.reshape(-1, 3)
        x = self.point_mlp(x)
        
        # Reshape back and max pool over points
        x = x.reshape(batch_size, num_points, -1)  # (B, N, hidden_dim)
        x = torch.max(x, dim=1)[0]  # (B, hidden_dim) - max pooling
        
        # Global features
        x = self.global_mlp(x)
        
        return x


class LaserScanCNN1D(nn.Module):
    """1D CNN encoder for laser scan data.
    
    레이저 스캔 데이터를 위한 1D CNN 인코더.
    
    Args:
        input_dim: Input dimension (number of range readings)
        output_dim: Output feature dimension
        hidden_dims: List of channel dimensions
    """
    
    def __init__(
        self,
        input_dim: int = 360,
        output_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        
        layers = []
        in_channels = 1  # Single channel for range values
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_channels = hidden_dim
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after conv layers
        self.flatten_size = hidden_dims[-1] * (input_dim // (2 ** len(hidden_dims)))
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Laser scan of shape (B, input_dim)
            
        Returns:
            Features of shape (B, output_dim)
        """
        # Add channel dimension: (B, input_dim) -> (B, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Flatten and apply FC
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x


class MultimodalPolicy(nn.Module):
    """Multimodal policy network with camera and LiDAR fusion.
    
    Uses late fusion: separate encoders for each modality, then fusion.
    
    Args:
        camera_backbone: CNN backbone for camera ('resnet18', 'resnet34')
        camera_pretrained: Whether to use pretrained weights for camera
        lidar_input_dim: LiDAR input dimension
        camera_feature_dim: Camera encoder output dimension
        lidar_feature_dim: LiDAR encoder output dimension
        fusion_dim: Dimension after fusion
        output_dim: Output dimension (default: 2 for [linear_x, angular_z])
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        camera_backbone: str = "resnet18",
        camera_pretrained: bool = True,
        lidar_input_dim: int = 360,
        camera_feature_dim: int = 256,
        lidar_feature_dim: int = 256,
        fusion_dim: int = 256,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Camera encoder
        self.camera_encoder = CameraEncoder(
            backbone=camera_backbone,
            pretrained=camera_pretrained,
            output_dim=camera_feature_dim,
        )
        
        # LiDAR encoder
        self.lidar_encoder = LiDAREncoder(
            input_dim=lidar_input_dim,
            output_dim=lidar_feature_dim,
        )
        
        # Fusion layer
        total_feature_dim = camera_feature_dim + lidar_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, output_dim),
        )
        
        logger.info(
            f"MultimodalPolicy initialized with camera ({camera_backbone}) "
            f"and LiDAR encoders"
        )
        
    def forward(
        self,
        camera: torch.Tensor,
        lidar: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            camera: Camera images of shape (B, C, H, W)
            lidar: LiDAR data of shape (B, lidar_dim)
            
        Returns:
            Control commands of shape (B, output_dim)
        """
        # Encode each modality
        camera_features = self.camera_encoder(camera)
        lidar_features = self.lidar_encoder(lidar)
        
        # Concatenate features (late fusion)
        fused = torch.cat([camera_features, lidar_features], dim=1)
        
        # Apply fusion layer
        fused = self.fusion(fused)
        
        # Predict controls
        controls = self.policy_head(fused)
        
        return controls
        
    def predict(
        self,
        camera: torch.Tensor,
        lidar: torch.Tensor
    ) -> torch.Tensor:
        """Predict with output range constraints.
        
        Args:
            camera: Camera images of shape (B, C, H, W)
            lidar: LiDAR data of shape (B, lidar_dim)
            
        Returns:
            Control commands with constrained ranges
        """
        controls = self.forward(camera, lidar)
        
        # Apply range constraints
        controls[:, 0] = torch.sigmoid(controls[:, 0])  # linear_x in [0, 1]
        controls[:, 1] = torch.tanh(controls[:, 1])     # angular_z in [-1, 1]
        
        return controls


class TemporalMultimodalPolicy(nn.Module):
    """Temporal multimodal policy with LSTM for sequence processing.
    
    Args:
        camera_backbone: CNN backbone for camera
        camera_pretrained: Whether to use pretrained weights
        lidar_input_dim: LiDAR input dimension
        camera_feature_dim: Camera encoder output dimension
        lidar_feature_dim: LiDAR encoder output dimension
        fusion_dim: Dimension after fusion
        lstm_hidden_dim: LSTM hidden dimension
        lstm_layers: Number of LSTM layers
        output_dim: Output dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        camera_backbone: str = "resnet18",
        camera_pretrained: bool = True,
        lidar_input_dim: int = 360,
        camera_feature_dim: int = 256,
        lidar_feature_dim: int = 256,
        fusion_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Camera encoder
        self.camera_encoder = CameraEncoder(
            backbone=camera_backbone,
            pretrained=camera_pretrained,
            output_dim=camera_feature_dim,
        )
        
        # LiDAR encoder
        self.lidar_encoder = LiDAREncoder(
            input_dim=lidar_input_dim,
            output_dim=lidar_feature_dim,
        )
        
        # Fusion layer
        total_feature_dim = camera_feature_dim + lidar_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, output_dim),
        )
        
        logger.info(
            f"TemporalMultimodalPolicy initialized with "
            f"{camera_backbone} and {lstm_layers}-layer LSTM"
        )
        
    def forward(
        self,
        camera: torch.Tensor,
        lidar: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            camera: Camera sequences of shape (B, T, C, H, W)
            lidar: LiDAR sequences of shape (B, T, lidar_dim)
            
        Returns:
            Control commands of shape (B, output_dim)
        """
        batch_size, seq_len = camera.size(0), camera.size(1)
        
        # Process each timestep
        fused_sequence = []
        for t in range(seq_len):
            camera_t = camera[:, t]
            lidar_t = lidar[:, t]
            
            # Encode
            camera_features = self.camera_encoder(camera_t)
            lidar_features = self.lidar_encoder(lidar_t)
            
            # Fuse
            fused_t = torch.cat([camera_features, lidar_features], dim=1)
            fused_t = self.fusion(fused_t)
            fused_sequence.append(fused_t)
            
        # Stack into sequence (B, T, fusion_dim)
        fused_sequence = torch.stack(fused_sequence, dim=1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(fused_sequence)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        
        # Predict controls
        controls = self.policy_head(last_output)
        
        return controls
        
    def predict(
        self,
        camera: torch.Tensor,
        lidar: torch.Tensor
    ) -> torch.Tensor:
        """Predict with output range constraints.
        
        Args:
            camera: Camera sequences of shape (B, T, C, H, W)
            lidar: LiDAR sequences of shape (B, T, lidar_dim)
            
        Returns:
            Control commands with constrained ranges
        """
        controls = self.forward(camera, lidar)
        
        # Apply range constraints
        controls[:, 0] = torch.sigmoid(controls[:, 0])  # linear_x in [0, 1]
        controls[:, 1] = torch.tanh(controls[:, 1])     # angular_z in [-1, 1]
        
        return controls


def create_multimodal_policy(
    config: dict,
    temporal: bool = False
) -> nn.Module:
    """Factory function to create multimodal policy from config.
    
    Args:
        config: Configuration dictionary
        temporal: Whether to create temporal policy
        
    Returns:
        Multimodal policy network
    """
    camera_backbone = config.get("camera_backbone", "resnet18")
    camera_pretrained = config.get("camera_pretrained", True)
    lidar_input_dim = config.get("lidar_input_dim", 360)
    output_dim = config.get("output_dim", 2)
    dropout = config.get("dropout", 0.3)
    
    if temporal:
        return TemporalMultimodalPolicy(
            camera_backbone=camera_backbone,
            camera_pretrained=camera_pretrained,
            lidar_input_dim=lidar_input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    else:
        return MultimodalPolicy(
            camera_backbone=camera_backbone,
            camera_pretrained=camera_pretrained,
            lidar_input_dim=lidar_input_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
