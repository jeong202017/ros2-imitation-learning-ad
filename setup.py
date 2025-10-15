from setuptools import setup, find_packages

package_name = 'ros2_imitation_learning'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeong202017',
    maintainer_email='jeong202017@users.noreply.github.com',
    description='ROS2 Bag-based Imitation Learning for Autonomous Driving',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = ros2_nodes.inference_node:main',
            'data_collector_node = ros2_nodes.data_collector_node:main',
        ],
    },
)
