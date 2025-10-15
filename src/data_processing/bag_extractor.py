"""
ROS2 Bag Extractor Module.

Extracts sensor data and control commands from ROS2 bag files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BagExtractor:
    """Extract data from ROS2 bag files.
    
    Supports extraction of:
    - Camera images (sensor_msgs/Image)
    - LiDAR/LaserScan data (sensor_msgs/LaserScan, sensor_msgs/PointCloud2)
    - Control commands (geometry_msgs/Twist)
    - Odometry (nav_msgs/Odometry)
    - IMU data (sensor_msgs/Imu)
    
    Args:
        bag_path: Path to ROS2 bag file or directory
        output_dir: Directory to save extracted data
        topics: List of topics to extract (if None, extracts all)
    """
    
    def __init__(
        self,
        bag_path: str,
        output_dir: str,
        topics: Optional[List[str]] = None
    ):
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.topics = set(topics) if topics else None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.lidar_dir = self.output_dir / "lidar"
        self.csv_dir = self.output_dir / "csv"
        
        for directory in [self.images_dir, self.lidar_dir, self.csv_dir]:
            directory.mkdir(exist_ok=True)
            
        self.metadata: Dict = {
            "bag_path": str(self.bag_path),
            "topics": {},
            "message_counts": {},
            "time_range": {},
        }
        
    def extract(self) -> None:
        """Extract all data from the bag file."""
        logger.info(f"Extracting data from {self.bag_path}")
        
        # Storage for different data types
        cmd_vel_data = []
        odom_data = []
        imu_data = []
        
        try:
            with Reader(self.bag_path) as reader:
                # Get topic information
                topic_types = {conn.topic: conn.msgtype for conn in reader.connections}
                
                # Filter topics if specified
                topics_to_process = (
                    self.topics.intersection(topic_types.keys())
                    if self.topics
                    else set(topic_types.keys())
                )
                
                logger.info(f"Processing topics: {topics_to_process}")
                
                # Store metadata
                for topic in topics_to_process:
                    self.metadata["topics"][topic] = topic_types[topic]
                    self.metadata["message_counts"][topic] = 0
                
                # Count total messages for progress bar
                total_messages = sum(
                    1 for conn, timestamp, rawdata in reader.messages()
                    if conn.topic in topics_to_process
                )
                
                # Process messages
                with tqdm(total=total_messages, desc="Extracting messages") as pbar:
                    for conn, timestamp, rawdata in reader.messages():
                        if conn.topic not in topics_to_process:
                            continue
                            
                        msg = deserialize_cdr(rawdata, conn.msgtype)
                        self.metadata["message_counts"][conn.topic] += 1
                        
                        # Process based on message type
                        if "Image" in conn.msgtype:
                            self._extract_image(msg, timestamp, conn.topic)
                        elif "LaserScan" in conn.msgtype:
                            self._extract_laserscan(msg, timestamp, conn.topic)
                        elif "PointCloud2" in conn.msgtype:
                            self._extract_pointcloud(msg, timestamp, conn.topic)
                        elif "Twist" in conn.msgtype:
                            cmd_vel_data.append(self._extract_twist(msg, timestamp))
                        elif "Odometry" in conn.msgtype:
                            odom_data.append(self._extract_odometry(msg, timestamp))
                        elif "Imu" in conn.msgtype:
                            imu_data.append(self._extract_imu(msg, timestamp))
                            
                        pbar.update(1)
                        
        except Exception as e:
            logger.error(f"Error reading bag file: {e}")
            raise
            
        # Save CSV data
        self._save_csv_data("cmd_vel", cmd_vel_data)
        self._save_csv_data("odometry", odom_data)
        self._save_csv_data("imu", imu_data)
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Extraction complete. Data saved to {self.output_dir}")
        
    def _extract_image(self, msg, timestamp: int, topic: str) -> None:
        """Extract and save image data as JPG.
        
        Args:
            msg: Image message
            timestamp: Message timestamp in nanoseconds
            topic: Topic name
        """
        try:
            # Convert ROS image to numpy array
            if msg.encoding in ["rgb8", "bgr8"]:
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                if msg.encoding == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "mono8":
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width
                )
            else:
                logger.warning(f"Unsupported image encoding: {msg.encoding}")
                return
                
            # Save image
            topic_name = topic.replace("/", "_")
            filename = f"{topic_name}_{timestamp}.jpg"
            cv2.imwrite(str(self.images_dir / filename), image)
            
        except Exception as e:
            logger.error(f"Error extracting image: {e}")
            
    def _extract_laserscan(self, msg, timestamp: int, topic: str) -> None:
        """Extract and save LaserScan data as numpy array.
        
        Args:
            msg: LaserScan message
            timestamp: Message timestamp in nanoseconds
            topic: Topic name
        """
        try:
            data = {
                "ranges": np.array(msg.ranges),
                "intensities": np.array(msg.intensities) if msg.intensities else None,
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
            }
            
            topic_name = topic.replace("/", "_")
            filename = f"{topic_name}_{timestamp}.npz"
            np.savez_compressed(self.lidar_dir / filename, **data)
            
        except Exception as e:
            logger.error(f"Error extracting LaserScan: {e}")
            
    def _extract_pointcloud(self, msg, timestamp: int, topic: str) -> None:
        """Extract and save PointCloud2 data as numpy array.
        
        Args:
            msg: PointCloud2 message
            timestamp: Message timestamp in nanoseconds
            topic: Topic name
        """
        try:
            # Simple point cloud extraction (x, y, z)
            # Note: Full PointCloud2 parsing is complex and depends on field layout
            point_step = msg.point_step
            num_points = len(msg.data) // point_step
            
            # Assuming xyz float32 format
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, point_step // 4)
            xyz = points[:, :3]  # Extract x, y, z
            
            topic_name = topic.replace("/", "_")
            filename = f"{topic_name}_{timestamp}.npy"
            np.save(self.lidar_dir / filename, xyz)
            
        except Exception as e:
            logger.error(f"Error extracting PointCloud2: {e}")
            
    def _extract_twist(self, msg, timestamp: int) -> Dict:
        """Extract control command (Twist) data.
        
        Args:
            msg: Twist message
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary with timestamp and control values
        """
        return {
            "timestamp": timestamp,
            "linear_x": msg.linear.x,
            "linear_y": msg.linear.y,
            "linear_z": msg.linear.z,
            "angular_x": msg.angular.x,
            "angular_y": msg.angular.y,
            "angular_z": msg.angular.z,
        }
        
    def _extract_odometry(self, msg, timestamp: int) -> Dict:
        """Extract odometry data.
        
        Args:
            msg: Odometry message
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary with timestamp and odometry values
        """
        return {
            "timestamp": timestamp,
            "pos_x": msg.pose.pose.position.x,
            "pos_y": msg.pose.pose.position.y,
            "pos_z": msg.pose.pose.position.z,
            "orient_x": msg.pose.pose.orientation.x,
            "orient_y": msg.pose.pose.orientation.y,
            "orient_z": msg.pose.pose.orientation.z,
            "orient_w": msg.pose.pose.orientation.w,
            "linear_x": msg.twist.twist.linear.x,
            "linear_y": msg.twist.twist.linear.y,
            "linear_z": msg.twist.twist.linear.z,
            "angular_x": msg.twist.twist.angular.x,
            "angular_y": msg.twist.twist.angular.y,
            "angular_z": msg.twist.twist.angular.z,
        }
        
    def _extract_imu(self, msg, timestamp: int) -> Dict:
        """Extract IMU data.
        
        Args:
            msg: IMU message
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary with timestamp and IMU values
        """
        return {
            "timestamp": timestamp,
            "orient_x": msg.orientation.x,
            "orient_y": msg.orientation.y,
            "orient_z": msg.orientation.z,
            "orient_w": msg.orientation.w,
            "angular_vel_x": msg.angular_velocity.x,
            "angular_vel_y": msg.angular_velocity.y,
            "angular_vel_z": msg.angular_velocity.z,
            "linear_acc_x": msg.linear_acceleration.x,
            "linear_acc_y": msg.linear_acceleration.y,
            "linear_acc_z": msg.linear_acceleration.z,
        }
        
    def _save_csv_data(self, name: str, data: List[Dict]) -> None:
        """Save data as CSV file.
        
        Args:
            name: Filename (without extension)
            data: List of dictionaries with data
        """
        if not data:
            logger.warning(f"No data to save for {name}")
            return
            
        try:
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            csv_path = self.csv_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} rows to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving CSV {name}: {e}")
            
    def _save_metadata(self) -> None:
        """Save extraction metadata as JSON."""
        try:
            metadata_path = self.output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
