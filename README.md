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
