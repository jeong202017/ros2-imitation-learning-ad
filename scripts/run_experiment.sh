#!/bin/bash
#
# Automated Experiment Runner
# 자동화된 실험 실행 스크립트
#
# Runs both experiments (camera-only and multimodal) and compares results.
# 두 실험(카메라 전용 및 멀티모달)을 실행하고 결과를 비교합니다.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

print_msg "${BLUE}" "========================================"
print_msg "${BLUE}" "ERP42 Imitation Learning Experiments"
print_msg "${BLUE}" "ERP42 모방학습 실험"
print_msg "${BLUE}" "========================================"
echo ""

# Configuration
BAG_FILE="${BAG_FILE:-data/rosbags/001_0.db3}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda}"

print_msg "${GREEN}" "Configuration / 설정:"
echo "  Bag file: ${BAG_FILE}"
echo "  Epochs: ${EPOCHS}"
echo "  Device: ${DEVICE}"
echo ""

# Check if bag file exists
if [ ! -f "${BAG_FILE}" ]; then
    print_msg "${RED}" "Error: Bag file not found: ${BAG_FILE}"
    print_msg "${RED}" "오류: Bag 파일을 찾을 수 없습니다: ${BAG_FILE}"
    exit 1
fi

# ============================================================================
# EXPERIMENT 1: Camera-Only
# 실험 1: 카메라 전용
# ============================================================================

print_msg "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_msg "${YELLOW}" "Experiment 1: Camera-Only / 실험 1: 카메라 전용"
print_msg "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1.1: Extract data
print_msg "${GREEN}" "Step 1.1: Extracting data / 데이터 추출 중..."
python scripts/extract_erp42_multiview.py \
    --bag-path "${BAG_FILE}" \
    --output-dir data/extracted/erp42_exp1 \
    --topics /video1 /Control/serial_data

# Step 1.2: Preprocess data
print_msg "${GREEN}" "Step 1.2: Preprocessing data / 데이터 전처리 중..."
python scripts/preprocess_camera_only.py \
    --input-dir data/extracted/erp42_exp1 \
    --output-dir data/processed/experiment1 \
    --train-split 0.8 \
    --val-split 0.1 \
    --time-threshold 0.1

# Step 1.3: Train model
print_msg "${GREEN}" "Step 1.3: Training model / 모델 학습 중..."
python scripts/train_camera_only.py \
    --config configs/experiment1_camera_only.yaml \
    --data-dir data/processed/experiment1 \
    --output-dir models/experiment1 \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}"

print_msg "${GREEN}" "✓ Experiment 1 complete! / 실험 1 완료!"
echo ""

# ============================================================================
# EXPERIMENT 2: Multimodal
# 실험 2: 멀티모달
# ============================================================================

print_msg "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_msg "${YELLOW}" "Experiment 2: Multimodal / 실험 2: 멀티모달"
print_msg "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 2.1: Extract data
print_msg "${GREEN}" "Step 2.1: Extracting data / 데이터 추출 중..."
python scripts/extract_erp42_multiview.py \
    --bag-path "${BAG_FILE}" \
    --output-dir data/extracted/erp42_exp2 \
    --topics /video1 /velodyne_points_filtered /scan /Control/serial_data

# Step 2.2: Preprocess data
print_msg "${GREEN}" "Step 2.2: Preprocessing data / 데이터 전처리 중..."
python scripts/preprocess_multimodal.py \
    --input-dir data/extracted/erp42_exp2 \
    --output-dir data/processed/experiment2 \
    --train-split 0.8 \
    --val-split 0.1 \
    --time-threshold 0.1

# Step 2.3: Train model
print_msg "${GREEN}" "Step 2.3: Training model / 모델 학습 중..."
python scripts/train_multimodal.py \
    --config configs/experiment2_multimodal.yaml \
    --data-dir data/processed/experiment2 \
    --output-dir models/experiment2 \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}"

print_msg "${GREEN}" "✓ Experiment 2 complete! / 실험 2 완료!"
echo ""

# ============================================================================
# COMPARISON
# 결과 비교
# ============================================================================

print_msg "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_msg "${YELLOW}" "Comparing Results / 결과 비교"
print_msg "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python scripts/compare_experiments.py \
    --exp1-dir models/experiment1 \
    --exp2-dir models/experiment2 \
    --output-dir results/comparison

print_msg "${GREEN}" "✓ Comparison complete! / 비교 완료!"
echo ""

# ============================================================================
# SUMMARY
# 요약
# ============================================================================

print_msg "${BLUE}" "========================================"
print_msg "${BLUE}" "All Experiments Complete! / 모든 실험 완료!"
print_msg "${BLUE}" "========================================"
echo ""
print_msg "${GREEN}" "Results / 결과:"
echo "  - Experiment 1 models: models/experiment1/"
echo "  - Experiment 2 models: models/experiment2/"
echo "  - Comparison results: results/comparison/"
echo ""
print_msg "${GREEN}" "Next Steps / 다음 단계:"
echo "  1. Review comparison_report.txt in results/comparison/"
echo "     results/comparison/comparison_report.txt 검토"
echo "  2. View training curves and metrics plots"
echo "     학습 곡선 및 메트릭 플롯 보기"
echo "  3. Run TensorBoard to analyze training:"
echo "     TensorBoard를 실행하여 학습 분석:"
echo "     tensorboard --logdir models/"
echo ""
print_msg "${BLUE}" "Thank you! / 감사합니다!"
