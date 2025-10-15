#!/usr/bin/env python3
"""
ERP42 ROS2 Bag Extraction Script.

Extracts sensor data and control commands from ERP42 vehicle bag files.
Specifically handles Float32MultiArray control data format.

ERP42 차량의 bag 파일에서 센서 데이터와 제어 명령을 추출합니다.
Float32MultiArray 제어 데이터 형식을 특별히 처리합니다.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ERP42BagExtractor:
    """Extract data from ERP42 ROS2 bag files.
    
    ERP42 차량의 ROS2 bag 파일에서 데이터를 추출합니다.
    
    Handles:
    - Camera images from /videoN topics
    - LiDAR PointCloud2 from /velodyne_points_filtered
    - LaserScan from /scan
    - Control commands from /Control/serial_data (Float32MultiArray)
    
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
        }
        
    def extract(self) -> None:
        """Extract all data from the bag file.
        
        bag 파일에서 모든 데이터를 추출합니다.
        """
        logger.info(f"Extracting data from {self.bag_path}")
        
        # Storage for different data types
        control_data = []
        
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
                        elif "Float32MultiArray" in conn.msgtype:
                            # ERP42 control data
                            control_data.append(
                                self._extract_erp42_control(msg, timestamp)
                            )
                            
                        pbar.update(1)
                        
        except Exception as e:
            logger.error(f"Error reading bag file: {e}")
            raise
            
        # Save CSV data
        self._save_control_data(control_data)
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Extraction complete. Data saved to {self.output_dir}")
        logger.info(f"Message counts: {self.metadata['message_counts']}")
        
    def _extract_image(self, msg, timestamp: int, topic: str) -> None:
        """Extract and save image data as JPG.
        
        이미지 데이터를 추출하여 JPG로 저장합니다.
        
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
            logger.error(f"Error extracting image from {topic}: {e}")
            
    def _extract_laserscan(self, msg, timestamp: int, topic: str) -> None:
        """Extract and save LaserScan data as numpy array.
        
        LaserScan 데이터를 추출하여 numpy 배열로 저장합니다.
        
        Args:
            msg: LaserScan message
            timestamp: Message timestamp in nanoseconds
            topic: Topic name
        """
        try:
            data = {
                "ranges": np.array(msg.ranges, dtype=np.float32),
                "intensities": (
                    np.array(msg.intensities, dtype=np.float32) 
                    if msg.intensities else np.array([], dtype=np.float32)
                ),
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
            logger.error(f"Error extracting LaserScan from {topic}: {e}")
            
    def _extract_pointcloud(self, msg, timestamp: int, topic: str) -> None:
        """Extract and save PointCloud2 data as numpy array.
        
        PointCloud2 데이터를 추출하여 numpy 배열로 저장합니다.
        
        Args:
            msg: PointCloud2 message
            timestamp: Message timestamp in nanoseconds
            topic: Topic name
        """
        try:
            # Simple point cloud extraction (x, y, z)
            point_step = msg.point_step
            num_points = len(msg.data) // point_step
            
            # Assuming xyz float32 format (standard for Velodyne)
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(
                num_points, point_step // 4
            )
            xyz = points[:, :3]  # Extract x, y, z
            
            topic_name = topic.replace("/", "_")
            filename = f"{topic_name}_{timestamp}.npy"
            np.save(self.lidar_dir / filename, xyz)
            
        except Exception as e:
            logger.error(f"Error extracting PointCloud2 from {topic}: {e}")
            
    def _extract_erp42_control(self, msg, timestamp: int) -> Dict:
        """Extract ERP42 control command data.
        
        ERP42 제어 명령 데이터를 추출합니다.
        
        The Float32MultiArray contains control data:
        - Index 3: speed (m/s)
        - Index 4: steering (radian)
        
        Args:
            msg: Float32MultiArray message
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary with timestamp, speed, and steering
        """
        try:
            data = msg.data
            
            # Check if array has enough elements
            if len(data) < 5:
                logger.warning(
                    f"Control array too short: {len(data)} elements, "
                    f"expected at least 5"
                )
                return None
                
            return {
                "timestamp": timestamp,
                "speed": float(data[3]),      # Index 3: speed (m/s)
                "steering": float(data[4]),   # Index 4: steering (radian)
            }
            
        except Exception as e:
            logger.error(f"Error extracting control data: {e}")
            return None
            
    def _save_control_data(self, control_data: List[Dict]) -> None:
        """Save control data as CSV file.
        
        제어 데이터를 CSV 파일로 저장합니다.
        
        Args:
            control_data: List of control dictionaries
        """
        if not control_data:
            logger.warning("No control data to save")
            return
            
        try:
            # Filter out None values
            control_data = [d for d in control_data if d is not None]
            
            if not control_data:
                logger.warning("No valid control data after filtering")
                return
                
            df = pd.DataFrame(control_data)
            df = df.sort_values("timestamp")
            csv_path = self.csv_dir / "control_data.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} control commands to {csv_path}")
            
            # Log statistics
            logger.info(f"Speed range: [{df['speed'].min():.3f}, {df['speed'].max():.3f}] m/s")
            logger.info(f"Steering range: [{df['steering'].min():.3f}, {df['steering'].max():.3f}] rad")
            
        except Exception as e:
            logger.error(f"Error saving control data: {e}")
            
    def _save_metadata(self) -> None:
        """Save extraction metadata as JSON.
        
        추출 메타데이터를 JSON으로 저장합니다.
        """
        try:
            metadata_path = self.output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")


def parse_args():
    """Parse command line arguments.
    
    명령줄 인수를 파싱합니다.
    """
    parser = argparse.ArgumentParser(
        description="Extract data from ERP42 ROS2 bag files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--bag-path",
        type=str,
        required=True,
        help="Path to ROS2 bag file (.db3) or directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for extracted data"
    )
    
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="List of topics to extract (default: all topics)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple bag files in batch mode"
    )
    
    return parser.parse_args()


def extract_single_bag(bag_path: str, output_dir: str, topics: list = None):
    """Extract data from a single bag file.
    
    단일 bag 파일에서 데이터를 추출합니다.
    
    Args:
        bag_path: Path to bag file
        output_dir: Output directory
        topics: List of topics to extract
    """
    logger.info(f"Processing bag: {bag_path}")
    
    try:
        extractor = ERP42BagExtractor(
            bag_path=bag_path,
            output_dir=output_dir,
            topics=topics
        )
        
        extractor.extract()
        
        logger.info(f"Successfully extracted data to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to extract bag {bag_path}: {e}")
        raise


def extract_batch(bag_dir: str, output_base_dir: str, topics: list = None):
    """Extract data from multiple bag files.
    
    여러 bag 파일에서 데이터를 추출합니다.
    
    Args:
        bag_dir: Directory containing bag files
        output_base_dir: Base output directory
        topics: List of topics to extract
    """
    bag_dir = Path(bag_dir)
    output_base_dir = Path(output_base_dir)
    
    # Find all bag files
    bag_files = list(bag_dir.glob("**/*.db3"))
    
    if not bag_files:
        logger.error(f"No bag files found in {bag_dir}")
        return
        
    logger.info(f"Found {len(bag_files)} bag files to process")
    
    # Process each bag
    for idx, bag_file in enumerate(bag_files, 1):
        logger.info(f"Processing bag {idx}/{len(bag_files)}: {bag_file.name}")
        
        # Create output directory for this bag
        output_dir = output_base_dir / bag_file.stem
        
        try:
            extract_single_bag(
                bag_path=str(bag_file),
                output_dir=str(output_dir),
                topics=topics
            )
        except Exception as e:
            logger.error(f"Failed to process {bag_file.name}, continuing...")
            continue
            
    logger.info(f"Batch processing complete. Processed {len(bag_files)} bags")


def main():
    """Main function.
    
    메인 함수.
    """
    args = parse_args()
    
    try:
        if args.batch:
            # Batch mode
            extract_batch(
                bag_dir=args.bag_path,
                output_base_dir=args.output_dir,
                topics=args.topics
            )
        else:
            # Single file mode
            extract_single_bag(
                bag_path=args.bag_path,
                output_dir=args.output_dir,
                topics=args.topics
            )
            
        logger.info("✅ Extraction complete!")
        
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
