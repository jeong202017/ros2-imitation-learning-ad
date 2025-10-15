# 🎓 Training Guide / 학습 가이드

This guide explains how to train imitation learning models for the ERP42 vehicle.

이 가이드는 ERP42 차량을 위한 모방학습 모델 학습 방법을 설명합니다.

## Overview / 개요

The training process involves:
1. Loading preprocessed data
2. Configuring model and training parameters
3. Training with automatic checkpointing
4. Monitoring with TensorBoard
5. Evaluating the trained model

학습 과정은 다음을 포함합니다:
1. 전처리된 데이터 로드
2. 모델 및 학습 매개변수 설정
3. 자동 체크포인트 저장과 함께 학습
4. TensorBoard로 모니터링
5. 학습된 모델 평가

## Hyperparameter Configuration / 하이퍼파라미터 설정

### Experiment 1: Camera-Only / 실험 1: 카메라 전용

Configuration file: `configs/experiment1_camera_only.yaml`

```yaml
experiment:
  name: "experiment1_camera_only"
  description: "Baseline using front camera only"

data:
  camera_topic: /video1
  control_topic: /Control/serial_data
  control_indices:
    speed: 3
    steering: 4
  image_size: [224, 224]
  use_lidar: false

model:
  type: cnn
  backbone: resnet18        # or resnet34, resnet50
  pretrained: true          # Use ImageNet pretrained weights
  output_dim: 2             # [speed, steering]

training:
  batch_size: 32            # Reduce if OOM
  epochs: 100
  learning_rate: 0.001      # Try 0.0001 for fine-tuning
  optimizer: adam           # or sgd, adamw
  weight_decay: 0.0001
  lr_scheduler: step        # step, cosine, or none
  lr_step_size: 30          # For step scheduler
  lr_gamma: 0.1             # Multiply LR by this every step
  early_stopping_patience: 15
  gradient_clip: 1.0        # Gradient clipping value

loss:
  type: mse                 # or mae, huber
  
augmentation:
  enabled: true
  horizontal_flip: 0.0      # Don't flip for driving
  brightness: 0.2
  contrast: 0.2
  rotation: 0               # No rotation for driving
  blur: 0.1

device: cuda                # or cpu
```

### Experiment 2: Multimodal / 실험 2: 멀티모달

Configuration file: `configs/experiment2_multimodal.yaml`

```yaml
experiment:
  name: "experiment2_multimodal"
  description: "Camera + LiDAR sensor fusion"

data:
  camera_topic: /video1
  lidar_pointcloud: /velodyne_points_filtered
  lidar_scan: /scan
  control_topic: /Control/serial_data
  use_lidar: true
  use_pointcloud: true      # Use 3D point cloud
  use_laserscan: true       # Use 2D laser scan

model:
  type: multimodal
  camera_encoder:
    backbone: resnet18
    pretrained: true
    output_dim: 256
  lidar_encoder:
    type: pointnet_simple   # or cnn1d for laser scan
    output_dim: 256
    hidden_dims: [512, 256]
  fusion:
    type: concatenate       # or attention, bilinear
    hidden_dim: 512
    dropout: 0.3
  output_dim: 2

training:
  batch_size: 16            # Smaller due to point clouds
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 0.0001
  lr_scheduler: step
  lr_step_size: 30
  lr_gamma: 0.1
  early_stopping_patience: 15
  gradient_clip: 1.0

loss:
  type: mse
  
device: cuda
```

## Training Scripts / 학습 스크립트

### Camera-Only Training / 카메라 전용 학습

```bash
python scripts/train_camera_only.py \
    --config configs/experiment1_camera_only.yaml \
    --data-dir data/processed/experiment1 \
    --output-dir models/experiment1 \
    --epochs 100 \
    --device cuda
```

**Parameters**:
- `--config`: Path to configuration file
- `--data-dir`: Directory with preprocessed data
- `--output-dir`: Where to save models and logs
- `--epochs`: Number of training epochs (overrides config)
- `--device`: 'cuda' or 'cpu'
- `--resume`: Path to checkpoint to resume training

### Multimodal Training / 멀티모달 학습

```bash
python scripts/train_multimodal.py \
    --config configs/experiment2_multimodal.yaml \
    --data-dir data/processed/experiment2 \
    --output-dir models/experiment2 \
    --epochs 100 \
    --device cuda
```

## Training Monitoring / 학습 모니터링

### TensorBoard Usage / TensorBoard 사용법

Start TensorBoard to monitor training in real-time:

실시간으로 학습을 모니터링하려면 TensorBoard를 시작하세요:

```bash
tensorboard --logdir models/experiment1/logs --port 6006
```

Then open your browser at: `http://localhost:6006`

브라우저에서 다음 주소를 여세요: `http://localhost:6006`

### What to Monitor / 모니터링 항목

**Scalars / 스칼라**:
- `loss/train`: Training loss per epoch (에폭당 학습 손실)
- `loss/val`: Validation loss per epoch (에폭당 검증 손실)
- `metrics/speed_mae`: Speed prediction MAE
- `metrics/steering_mae`: Steering prediction MAE
- `learning_rate`: Current learning rate (현재 학습률)

**Images / 이미지** (if enabled):
- Sample predictions vs. ground truth
- Input images with overlays

**Histograms / 히스토그램**:
- Model weights distribution
- Gradient distribution

### Training Curves / 학습 곡선

**Healthy Training** / 정상 학습:
- Loss decreases smoothly
- Train and val loss decrease together
- Validation loss plateaus or slightly increases at the end

**Overfitting** / 과적합:
- Train loss << Val loss
- Val loss increases while train loss decreases
- **Solutions**:
  - Increase dropout
  - Add data augmentation
  - Reduce model capacity
  - Early stopping

**Underfitting** / 과소적합:
- Both train and val loss remain high
- Loss plateaus early
- **Solutions**:
  - Increase model capacity
  - Train longer
  - Reduce regularization
  - Check data quality

## Checkpoint Management / 체크포인트 관리

### Automatic Saving / 자동 저장

The training script automatically saves:
- `best_model.pth`: Model with lowest validation loss (최저 검증 손실 모델)
- `last_model.pth`: Most recent model (최신 모델)
- `checkpoint_epoch_N.pth`: Periodic checkpoints (주기적 체크포인트)

### Checkpoint Contents / 체크포인트 내용

Each checkpoint contains:
```python
{
    'epoch': 50,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'train_loss': 0.0234,
    'val_loss': 0.0456,
    'config': {...}
}
```

### Resume Training / 학습 재개

To resume training from a checkpoint:

체크포인트에서 학습을 재개하려면:

```bash
python scripts/train_camera_only.py \
    --config configs/experiment1_camera_only.yaml \
    --data-dir data/processed/experiment1 \
    --output-dir models/experiment1 \
    --resume models/experiment1/checkpoint_epoch_50.pth
```

## Training Tips / 학습 팁

### Learning Rate Selection / 학습률 선택

**Too High** (너무 높음):
- Loss oscillates or diverges
- Training unstable

**Too Low** (너무 낮음):
- Very slow convergence
- Gets stuck in local minima

**Good Starting Points** (좋은 시작점):
- Pretrained model: `1e-4` to `1e-3`
- From scratch: `1e-3` to `1e-2`

### Batch Size Effects / 배치 크기 효과

**Larger Batch Size** (큰 배치 크기):
- More stable gradients
- Faster training (better GPU utilization)
- Requires more memory
- May need higher learning rate

**Smaller Batch Size** (작은 배치 크기):
- More noise in gradients (can help escape local minima)
- Less memory usage
- Slower training
- May need lower learning rate

### Data Augmentation / 데이터 증강

**Recommended for Driving** (주행에 권장):
- ✅ Brightness/contrast adjustment
- ✅ Blur (simulates motion blur)
- ❌ Horizontal flip (changes driving direction)
- ❌ Rotation (unrealistic for driving)
- ❌ Vertical flip (never happens in driving)

### Transfer Learning / 전이 학습

**Benefits** (이점):
- Faster convergence
- Better performance with limited data
- Pretrained on ImageNet (natural images)

**Usage**:
- Set `pretrained: true` in config
- Use lower learning rate (e.g., 1e-4)
- Optionally freeze early layers initially

## Common Issues / 일반적인 문제

### Out of Memory (OOM) / 메모리 부족

**Symptoms** (증상):
- CUDA out of memory error
- Training crashes

**Solutions** (해결책):
1. Reduce batch size
2. Use smaller images (e.g., 128x128 instead of 224x224)
3. Use gradient accumulation
4. Use mixed precision training (FP16)

```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 2  # Effective batch size = 32
```

### Loss Not Decreasing / 손실이 감소하지 않음

**Check**:
1. Data loading is correct
2. Data normalization is applied
3. Learning rate is not too low
4. Model architecture matches data

**Solutions**:
1. Visualize input data
2. Check label distribution
3. Try different learning rate
4. Simplify model first

### NaN Loss / NaN 손실

**Causes** (원인):
- Learning rate too high
- Numerical instability
- Invalid data (inf, nan values)

**Solutions** (해결책):
1. Reduce learning rate
2. Enable gradient clipping
3. Check data preprocessing
4. Use mixed precision carefully

## Performance Optimization / 성능 최적화

### Speed Up Training / 학습 가속화

1. **Use GPU** (GPU 사용):
   ```yaml
   device: cuda
   ```

2. **Increase Batch Size** (배치 크기 증가):
   - Up to GPU memory limit

3. **Use More Workers** (더 많은 워커 사용):
   ```python
   DataLoader(..., num_workers=4)
   ```

4. **Pin Memory** (메모리 고정):
   ```python
   DataLoader(..., pin_memory=True)
   ```

5. **Mixed Precision Training** (혼합 정밀도 학습):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

### Reduce Memory Usage / 메모리 사용 감소

1. **Gradient Accumulation** (그래디언트 누적)
2. **Checkpoint Gradients** (그래디언트 체크포인트)
3. **Smaller Batch Size** (작은 배치 크기)
4. **Reduce Image Resolution** (이미지 해상도 감소)

## Model Evaluation / 모델 평가

After training, evaluate the model:

학습 후 모델을 평가하세요:

```bash
python scripts/evaluate_model.py \
    --config configs/experiment1_camera_only.yaml \
    --data-dir data/processed/experiment1 \
    --model-path models/experiment1/best_model.pth \
    --output-dir results/experiment1
```

This generates:
- Metrics JSON file (지표 JSON 파일)
- Prediction vs. ground truth plots (예측 대 실제 플롯)
- Error distribution histograms (오류 분포 히스토그램)

## Next Steps / 다음 단계

1. **Compare Experiments**: Use `compare_experiments.py` to compare results
2. **Hyperparameter Tuning**: Experiment with different settings
3. **Ensemble Models**: Combine multiple models for better performance
4. **Deploy**: Use the trained model in ROS2 inference node

1. **실험 비교**: `compare_experiments.py`를 사용하여 결과 비교
2. **하이퍼파라미터 튜닝**: 다양한 설정 실험
3. **앙상블 모델**: 여러 모델을 결합하여 성능 향상
4. **배포**: 학습된 모델을 ROS2 추론 노드에서 사용

## References / 참고 자료

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started)
- [Learning Rate Scheduling](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
