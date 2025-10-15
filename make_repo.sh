# setup_project.sh
#!/bin/bash

echo "🚀 Setting up ROS2 Imitation Learning project..."

# README.md 생성
cat > README.md << 'EOF'
# 🚗 ROS2 Bag-based Imitation Learning for Autonomous Driving

ROS2 bag 파일에서 성공한 주행 데이터를 추출하여 학습하는 모방학습 기반 자율주행 시스템

## 📋 프로젝트 개요

이 프로젝트는 ROS2 bag 파일에 기록된 성공적인 주행 데이터를 학습하여 자율주행을 수행하는 Behavioral Cloning 기반 시스템입니다.

## 🎯 주요 기능

- **ROS2 Bag 데이터 추출**: bag 파일에서 센서 데이터 및 제어 명령 추출
- **Behavioral Cloning**: 성공한 주행 경험 기반 지도학습
- **다중 센서 융합**: 카메라, LiDAR, IMU 등 다양한 센서 데이터 활용
- **실시간 ROS2 통합**: 학습된 모델을 ROS2 노드로 배포
- **데이터 필터링**: 성공/실패 주행 데이터 자동 분류

## 🚀 빠른 시작

### 설치
\`\`\`bash
git clone https://github.com/jeong202017/ros2-imitation-learning-ad.git
cd ros2-imitation-learning-ad
pip install -r requirements.txt
\`\`\`

### ROS2 Bag 데이터 추출
\`\`\`bash
python scripts/extract_rosbag.py \
    --bag-path data/rosbags/successful/drive_001.db3 \
    --output-dir data/extracted/drive_001 \
    --topics /camera/image_raw /cmd_vel /odom
\`\`\`

### 모델 학습
\`\`\`bash
python scripts/train_model.py \
    --config configs/training_config.yaml \
    --data-dir data/processed \
    --epochs 100
\`\`\`

### ROS2 노드 실행
\`\`\`bash
ros2 run ros2_imitation_learning inference_node \
    --model-path models/bc_cnn_best.pth
\`\`\`

## 📁 프로젝트 구조

\`\`\`
ros2-imitation-learning-ad/
├── data/                      # 데이터 디렉토리
├── models/                    # 학습된 모델
├── src/                       # 소스 코드
│   ├── data_processing/       # 데이터 처리
│   ├── models/                # 모델 정의
│   ├── training/              # 학습 로직
│   └── ros2_nodes/            # ROS2 노드
├── scripts/                   # 실행 스크립트
├── configs/                   # 설정 파일
└── notebooks/                 # 분석 노트북
\`\`\`

## 📚 문서

자세한 사용법은 [Wiki](https://github.com/jeong202017/ros2-imitation-learning-ad/wiki)를 참고하세요.

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR을 환영합니다!
EOF

# requirements.txt 생성
cat > requirements.txt << 'EOF'
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
pillow>=10.0.0

# Point Cloud Processing
open3d>=0.17.0

# ROS Bag Processing
rosbags>=0.9.0

# Data Augmentation
albumentations>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0

# Configuration
pyyaml>=6.0
omegaconf>=2.3.0

# Utils
tqdm>=4.65.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
EOF

# package.xml 생성 (ROS2 패키지 정의)
cat > package.xml << 'EOF'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>ros2_imitation_learning</name>
  <version>0.1.0</version>
  <description>ROS2 Bag-based Imitation Learning for Autonomous Driving</description>
  <maintainer email="jeong202017@users.noreply.github.com">jeong202017</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>cv_bridge</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
EOF

# setup.py 생성
cat > setup.py << 'EOF'
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
EOF

# 기본 config 파일 생성
cat > configs/training_config.yaml << 'EOF'
# Training Configuration
data:
  input_topics:
    - /camera/image_raw
    - /scan
  output_topic: /cmd_vel
  sequence_length: 1
  train_split: 0.8
  val_split: 0.1

model:
  type: cnn
  backbone: resnet18
  pretrained: true
  input_size: [224, 224]
  output_dim: 2  # [linear_x, angular_z]
  hidden_dim: 256

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  loss: mse
  save_interval: 10
  
augmentation:
  enabled: true
  horizontal_flip: 0.5
  brightness: 0.2
  rotation: 5

device: cuda  # or cpu
EOF

# __init__.py 파일 생성
touch src/__init__.py
touch src/data_processing/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/ros2_nodes/__init__.py
touch src/utils/__init__.py

# resource 디렉토리 생성
mkdir -p resource
touch resource/ros2_imitation_learning

echo "✅ Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Review and commit the initial files:"
echo "   git add ."
echo "   git commit -m 'Initial project structure'"
echo "   git push -u origin main"
echo ""
echo "2. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Start adding your ROS2 bag files to data/rosbags/successful/"
