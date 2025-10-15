# 🧪 Experiments / 실험 가이드

This document describes the two main experiments comparing camera-only baseline and multimodal sensor fusion approaches.

이 문서는 카메라 전용 베이스라인과 멀티모달 센서 융합 접근 방식을 비교하는 두 가지 주요 실험을 설명합니다.

## Experiment Overview / 실험 개요

### Experiment 1: Camera-Only Baseline
**목적 (Purpose)**: Establish a baseline performance using only front camera images for control prediction.
- Single sensor modality (front camera)
- Simple CNN architecture (ResNet-18)
- Lightweight and fast inference

**방법 (Method)**:
- Input: Front camera images from `/video1` topic
- Output: Speed (m/s) and steering (radian)
- Model: ResNet-18 based CNN encoder + MLP head
- Training: Standard behavioral cloning with MSE loss

**예상 결과 (Expected Results)**:
- Faster training and inference
- Good performance in simple scenarios
- May struggle with complex driving situations or occlusions
- Limited spatial understanding

### Experiment 2: Camera + LiDAR Multimodal
**목적 (Purpose)**: Improve robustness and spatial awareness by fusing camera and LiDAR data.
- Multi-sensor fusion (camera + LiDAR)
- Enhanced spatial perception
- More robust to challenging conditions

**방법 (Method)**:
- Input: 
  - Front camera images from `/video1`
  - LiDAR PointCloud from `/velodyne_points_filtered`
  - LaserScan from `/scan`
- Output: Speed (m/s) and steering (radian)
- Model: 
  - Camera encoder (ResNet-18)
  - LiDAR encoder (PointNet-style or 1D CNN)
  - Late fusion with concatenation
- Training: Behavioral cloning with MSE loss

**예상 결과 (Expected Results)**:
- Better spatial awareness
- More robust obstacle detection
- Improved performance in complex scenarios
- Slightly slower inference due to additional processing

## Data Specifications / 데이터 사양

### ERP42 Vehicle Control Data Format
```python
# /Control/serial_data (Float32MultiArray)
# Index 3: speed (m/s) - 속도
# Index 4: steering (radian) - 조향각
control_array = [1.0, 0.0, 0.0, 5.932, -0.0708, 35.0, 0.0, 4.0]
```

### Topic Information / 토픽 정보
- **Camera**: `/video1` (2,404 messages) - 전방 카메라
- **LiDAR PointCloud**: `/velodyne_points_filtered` (965 messages)
- **LaserScan**: `/scan` (965 messages)
- **Control**: `/Control/serial_data` (9,728 messages)

### Expected Dataset Size / 예상 데이터셋 크기
After timestamp synchronization, approximately 900-1000 samples:
- Train: ~720-800 samples (80%)
- Validation: ~90-100 samples (10%)
- Test: ~90-100 samples (10%)

## Step-by-Step Execution Guide / 실행 가이드

### Prerequisites / 사전 준비
```bash
# Install dependencies / 의존성 설치
pip install -r requirements.txt

# Verify bag file / bag 파일 확인
ls data/rosbags/001_0.db3
```

### Experiment 1: Camera-Only / 실험 1: 카메라 전용

#### Step 1: Extract Data / 데이터 추출
```bash
python scripts/extract_erp42_multiview.py \
    --bag-path data/rosbags/001_0.db3 \
    --output-dir data/extracted/erp42_exp1 \
    --topics /video1 /Control/serial_data
```

#### Step 2: Preprocess Data / 데이터 전처리
```bash
python scripts/preprocess_camera_only.py \
    --input-dir data/extracted/erp42_exp1 \
    --output-dir data/processed/experiment1 \
    --train-split 0.8 \
    --val-split 0.1 \
    --time-threshold 0.1
```

#### Step 3: Train Model / 모델 학습
```bash
python scripts/train_camera_only.py \
    --config configs/experiment1_camera_only.yaml \
    --data-dir data/processed/experiment1 \
    --output-dir models/experiment1 \
    --epochs 100
```

#### Step 4: Monitor Training / 학습 모니터링
```bash
tensorboard --logdir models/experiment1/logs
```

### Experiment 2: Multimodal / 실험 2: 멀티모달

#### Step 1: Extract Data / 데이터 추출
```bash
python scripts/extract_erp42_multiview.py \
    --bag-path data/rosbags/001_0.db3 \
    --output-dir data/extracted/erp42_exp2 \
    --topics /video1 /velodyne_points_filtered /scan /Control/serial_data
```

#### Step 2: Preprocess Data / 데이터 전처리
```bash
python scripts/preprocess_multimodal.py \
    --input-dir data/extracted/erp42_exp2 \
    --output-dir data/processed/experiment2 \
    --train-split 0.8 \
    --val-split 0.1 \
    --time-threshold 0.1
```

#### Step 3: Train Model / 모델 학습
```bash
python scripts/train_multimodal.py \
    --config configs/experiment2_multimodal.yaml \
    --data-dir data/processed/experiment2 \
    --output-dir models/experiment2 \
    --epochs 100
```

#### Step 4: Monitor Training / 학습 모니터링
```bash
tensorboard --logdir models/experiment2/logs
```

### Compare Results / 결과 비교

```bash
python scripts/compare_experiments.py \
    --exp1-dir models/experiment1 \
    --exp2-dir models/experiment2 \
    --output-dir results/comparison
```

This will generate:
- Training curves comparison (학습 곡선 비교)
- Evaluation metrics table (평가 지표 테이블)
- Visualization plots (시각화 그래프)

## Evaluation Metrics / 평가 지표

Both experiments are evaluated using:

**Regression Metrics**:
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and ground truth
- **MSE (Mean Squared Error)**: Average squared difference (penalizes large errors)
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **R² Score**: Coefficient of determination (closer to 1 is better)

**Per-Variable Metrics**:
- Speed MAE/MSE/R²
- Steering MAE/MSE/R²

## Results Interpretation / 결과 해석

### What to Look For / 확인 사항

1. **Training Convergence / 학습 수렴**
   - Does the loss decrease smoothly?
   - Is there overfitting (train loss << val loss)?

2. **Prediction Accuracy / 예측 정확도**
   - Lower MAE/MSE is better
   - Higher R² is better (closer to 1)

3. **Model Comparison / 모델 비교**
   - Does multimodal outperform camera-only?
   - What is the performance vs. complexity trade-off?

4. **Failure Cases / 실패 케이스**
   - Where does each model struggle?
   - Are errors correlated with specific scenarios?

## Troubleshooting / 문제 해결

### Common Issues / 일반적인 문제

**Problem**: Not enough synchronized samples
**Solution**: Increase `--time-threshold` in preprocessing

**Problem**: Training loss not decreasing
**Solution**: 
- Reduce learning rate
- Check data normalization
- Verify data loading is correct

**Problem**: Overfitting (val loss increases)
**Solution**:
- Increase dropout
- Enable data augmentation
- Reduce model complexity

**Problem**: Out of memory during training
**Solution**:
- Reduce batch size
- Use smaller images (e.g., 128x128 instead of 224x224)
- Use gradient accumulation

## Next Steps / 다음 단계

After completing both experiments:

1. **Analyze Results**: Compare metrics and identify strengths/weaknesses
2. **Hyperparameter Tuning**: Experiment with different learning rates, architectures
3. **Data Augmentation**: Add more augmentation for robustness
4. **Temporal Modeling**: Try LSTM or temporal convolutions
5. **Real Vehicle Testing**: Deploy best model on ERP42 vehicle

## References / 참고 자료

- Original Behavioral Cloning paper: [End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- Multimodal Fusion: [MultiNet](https://arxiv.org/abs/1809.08009)
- PointNet: [Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
