#!/usr/bin/env python3
"""
ROS2 Bag Extraction Script.

Extracts sensor data and control commands from ROS2 bag files.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.bag_extractor import BagExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract data from ROS2 bag files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--bag-path",
        type=str,
        required=True,
        help="Path to ROS2 bag file or directory"
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
    
    Args:
        bag_path: Path to bag file
        output_dir: Output directory
        topics: List of topics to extract
    """
    logger.info(f"Processing bag: {bag_path}")
    
    try:
        extractor = BagExtractor(
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
    """Main function."""
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
            
        logger.info("Extraction complete!")
        
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
