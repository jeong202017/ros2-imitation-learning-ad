import argparse
import matplotlib.pyplot as plt
import rosbag2_py
import numpy as np

"""
Analyze control data statistics from ROS2 bag files.

This script analyzes the /Control/serial_data topic with 8 indices, showing min/max/mean/std for each index,
generating histogram plots for speed, steering, brake, and a scatter plot for speed vs steering.
"""

def analyze_control_data(bag_path: str):
    # Load the bag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(storage_type='sqlite3', uri=bag_path)
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)

    # Initialize statistics
    stats = {'speed': [], 'steering': [], 'brake': []}
    while reader.has_next():
        (topic, msg, t) = reader.read_next()
        if topic == "/Control/serial_data":
            data = np.frombuffer(msg.data, dtype=np.float32)
            stats['speed'].append(data[3])
            stats['steering'].append(data[7])
            stats['brake'].append(data[4])

    # Convert lists to numpy arrays
    speed = np.array(stats['speed'])
    steering = np.array(stats['steering'])
    brake = np.array(stats['brake'])

    # Calculate statistics
    for key in stats:
        print(f'{key.capitalize()}: Min={np.min(stats[key])}, Max={np.max(stats[key])}, Mean={np.mean(stats[key])}, Std={np.std(stats[key])}')

    # Generate histograms
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.hist(speed, bins=30, color='blue', alpha=0.7)
    plt.title('Speed Histogram')
    plt.subplot(3, 1, 2)
    plt.hist(steering, bins=30, color='green', alpha=0.7)
    plt.title('Steering Histogram')
    plt.subplot(3, 1, 3)
    plt.hist(brake, bins=30, color='red', alpha=0.7)
    plt.title('Brake Histogram')
    plt.tight_layout()
    plt.savefig('control_data_analysis.png')

    # Generate scatter plot
    plt.figure()
    plt.scatter(speed, steering, alpha=0.5)
    plt.title('Speed vs Steering')
    plt.xlabel('Speed')
    plt.ylabel('Steering')
    plt.savefig('speed_steering_scatter.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze control data from ROS2 bag files.')
    parser.add_argument('bag_path', type=str, help='Path to the ROS2 bag file')
    args = parser.parse_args()
    analyze_control_data(args.bag_path)
