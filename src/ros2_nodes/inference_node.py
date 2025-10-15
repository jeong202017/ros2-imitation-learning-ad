"""
ROS2 Inference Node for Imitation Learning.

Subscribes to camera images and publishes control commands.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
import torch.nn as nn
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceNode(Node):
    """ROS2 node for real-time policy inference.
    
    Args:
        model: Trained policy network
        device: Device to run inference on
        image_topic: Camera image topic to subscribe to
        cmd_vel_topic: Control command topic to publish to
        image_size: Input image size for model
        max_linear_x: Maximum linear velocity
        max_angular_z: Maximum angular velocity
        publish_rate: Publishing rate in Hz
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        image_topic: str = "/camera/image_raw",
        cmd_vel_topic: str = "/cmd_vel",
        image_size: tuple = (224, 224),
        max_linear_x: float = 1.0,
        max_angular_z: float = 2.0,
        publish_rate: float = 10.0,
    ):
        super().__init__("inference_node")
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.image_size = image_size
        self.max_linear_x = max_linear_x
        self.max_angular_z = max_angular_z
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Latest received image
        self.latest_image: Optional[np.ndarray] = None
        self.image_received = False
        
        # Create subscriber
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        # Create publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )
        
        # Create timer for inference and publishing
        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.inference_callback
        )
        
        self.get_logger().info(
            f"Inference node initialized "
            f"(image_topic: {image_topic}, cmd_vel_topic: {cmd_vel_topic})"
        )
        
    def image_callback(self, msg: Image) -> None:
        """Callback for receiving camera images.
        
        Args:
            msg: Image message
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_image
            self.image_received = True
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image.to(self.device)
        
    def inference_callback(self) -> None:
        """Callback for running inference and publishing commands."""
        if not self.image_received or self.latest_image is None:
            self.get_logger().warn(
                "No image received yet, publishing zero velocity",
                throttle_duration_sec=5.0
            )
            self.publish_zero_velocity()
            return
            
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(self.latest_image)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
                
            # Get control commands
            linear_x = predictions[0, 0].item()
            angular_z = predictions[0, 1].item()
            
            # Scale to real values (assuming model outputs [0, 1] and [-1, 1])
            linear_x = linear_x * self.max_linear_x
            angular_z = angular_z * self.max_angular_z
            
            # Publish command
            self.publish_cmd_vel(linear_x, angular_z)
            
        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")
            self.publish_zero_velocity()
            
    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        """Publish velocity command.
        
        Args:
            linear_x: Linear velocity
            angular_z: Angular velocity
        """
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        
        self.cmd_vel_pub.publish(msg)
        
        self.get_logger().debug(
            f"Published cmd_vel: linear_x={linear_x:.3f}, angular_z={angular_z:.3f}"
        )
        
    def publish_zero_velocity(self) -> None:
        """Publish zero velocity command."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)


def load_model(
    model_path: str,
    device: str = "cuda"
) -> nn.Module:
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Import here to avoid circular dependencies
    from models.cnn_policy import CNNPolicy
    
    # Create model
    # Note: Model configuration should ideally be saved with checkpoint
    model = CNNPolicy(
        backbone="resnet18",
        pretrained=False,
        output_dim=2
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    
    return model


def main(args=None):
    """Main function for ROS2 node."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Inference node for imitation learning")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default="/camera/image_raw",
        help="Camera image topic"
    )
    parser.add_argument(
        "--cmd-vel-topic",
        type=str,
        default="/cmd_vel",
        help="Velocity command topic"
    )
    parser.add_argument(
        "--max-linear-x",
        type=float,
        default=1.0,
        help="Maximum linear velocity"
    )
    parser.add_argument(
        "--max-angular-z",
        type=float,
        default=2.0,
        help="Maximum angular velocity"
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=10.0,
        help="Publishing rate in Hz"
    )
    
    parsed_args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Load model
        model = load_model(parsed_args.model_path, parsed_args.device)
        
        # Create node
        node = InferenceNode(
            model=model,
            device=parsed_args.device,
            image_topic=parsed_args.image_topic,
            cmd_vel_topic=parsed_args.cmd_vel_topic,
            max_linear_x=parsed_args.max_linear_x,
            max_angular_z=parsed_args.max_angular_z,
            publish_rate=parsed_args.publish_rate,
        )
        
        # Spin
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        logger.info("Shutting down inference node")
    except Exception as e:
        logger.error(f"Error in inference node: {e}")
    finally:
        # Cleanup
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
