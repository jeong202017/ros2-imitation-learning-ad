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

## 🧪 Experiments / 실험

이 프로젝트는 ERP42 차량의 모방학습을 위한 두 가지 주요 실험을 제공합니다:

### Experiment 1: Camera-Only Baseline / 실험 1: 카메라 전용 베이스라인
- **Input**: Front camera images only (전방 카메라 이미지만)
- **Model**: ResNet-18 CNN encoder
- **Purpose**: Establish baseline performance (베이스라인 성능 설정)

### Experiment 2: Multimodal Sensor Fusion / 실험 2: 멀티모달 센서 융합
- **Input**: Camera + LiDAR (PointCloud & LaserScan)
- **Model**: CNN + PointNet + 1D CNN fusion
- **Purpose**: Improve spatial awareness and robustness (공간 인식 및 강건성 향상)

### Quick Start / 빠른 시작

```bash
# Run all experiments automatically
# 모든 실험 자동 실행
./scripts/run_experiment.sh

# Or run individually
# 또는 개별 실행

# Experiment 1: Camera-only
python scripts/extract_erp42_multiview.py --bag-path data/rosbags/001_0.db3 --output-dir data/extracted/erp42_exp1 --topics /video1 /Control/serial_data
python scripts/preprocess_camera_only.py --input-dir data/extracted/erp42_exp1 --output-dir data/processed/experiment1
python scripts/train_camera_only.py --config configs/experiment1_camera_only.yaml --data-dir data/processed/experiment1 --output-dir models/experiment1

# Experiment 2: Multimodal
python scripts/extract_erp42_multiview.py --bag-path data/rosbags/001_0.db3 --output-dir data/extracted/erp42_exp2 --topics /video1 /velodyne_points_filtered /scan /Control/serial_data
python scripts/preprocess_multimodal.py --input-dir data/extracted/erp42_exp2 --output-dir data/processed/experiment2
python scripts/train_multimodal.py --config configs/experiment2_multimodal.yaml --data-dir data/processed/experiment2 --output-dir models/experiment2

# Compare results
# 결과 비교
python scripts/compare_experiments.py --exp1-dir models/experiment1 --exp2-dir models/experiment2 --output-dir results/comparison
```

### Documentation / 문서
- 📖 [Experiments Guide](docs/EXPERIMENTS.md) - Detailed experiment comparison / 상세 실험 비교
- 📦 [Data Extraction Guide](docs/DATA_EXTRACTION.md) - ERP42 bag file processing / ERP42 bag 파일 처리
- 🎓 [Training Guide](docs/TRAINING.md) - Training tips and troubleshooting / 학습 팁 및 문제 해결

## 📁 프로젝트 구조

\`\`\`
ros2-imitation-learning-ad/
├── data/                      # 데이터 디렉토리
│   ├── rosbags/              # ROS2 bag 파일
│   ├── extracted/            # 추출된 센서 데이터
│   └── processed/            # 전처리된 학습 데이터
├── models/                    # 학습된 모델
│   ├── experiment1/          # 카메라 전용 모델
│   └── experiment2/          # 멀티모달 모델
├── results/                   # 실험 결과
│   └── comparison/           # 실험 비교 결과
├── src/                       # 소스 코드
│   ├── data_processing/       # 데이터 처리
│   │   ├── dataset_camera.py       # 카메라 전용 Dataset
│   │   └── dataset_multimodal.py   # 멀티모달 Dataset
│   ├── models/                # 모델 정의
│   │   ├── cnn_policy.py           # CNN 정책
│   │   └── multimodal_policy.py    # 멀티모달 정책 (PointNet + CNN)
│   ├── training/              # 학습 로직
│   └── ros2_nodes/            # ROS2 노드
├── scripts/                   # 실행 스크립트
│   ├── extract_erp42_multiview.py  # ERP42 데이터 추출
│   ├── preprocess_camera_only.py   # 카메라 전처리
│   ├── preprocess_multimodal.py    # 멀티모달 전처리
│   ├── train_camera_only.py        # 카메라 학습
│   ├── train_multimodal.py         # 멀티모달 학습
│   ├── compare_experiments.py      # 실험 비교
│   └── run_experiment.sh           # 자동 실험 실행
├── configs/                   # 설정 파일
│   ├── experiment1_camera_only.yaml
│   └── experiment2_multimodal.yaml
├── docs/                      # 문서
│   ├── EXPERIMENTS.md         # 실험 가이드
│   ├── DATA_EXTRACTION.md     # 데이터 추출 가이드
│   └── TRAINING.md            # 학습 가이드
└── notebooks/                 # 분석 노트북
\`\`\`

## 📚 문서

자세한 사용법은 [Wiki](https://github.com/jeong202017/ros2-imitation-learning-ad/wiki)를 참고하세요.

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR을 환영합니다!
