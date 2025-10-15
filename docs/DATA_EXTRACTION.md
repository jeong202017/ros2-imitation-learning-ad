# 📦 Data Extraction Guide / 데이터 추출 가이드

This guide explains how to extract data from ERP42 ROS2 bag files for imitation learning.

이 가이드는 모방학습을 위해 ERP42 ROS2 bag 파일에서 데이터를 추출하는 방법을 설명합니다.

## Overview / 개요

The `extract_erp42_multiview.py` script extracts sensor data and control commands from ROS2 bag files recorded on the ERP42 autonomous vehicle platform.

`extract_erp42_multiview.py` 스크립트는 ERP42 자율주행 차량 플랫폼에 기록된 ROS2 bag 파일에서 센서 데이터와 제어 명령을 추출합니다.

## ERP42 Bag File Structure / ERP42 Bag 파일 구조

### Available Topics / 사용 가능한 토픽

The ERP42 bag files typically contain the following topics:

#### Camera Topics / 카메라 토픽
- `/video1`: Front camera (주 전방 카메라) - **2,404 messages**
- `/video2`: Additional camera view (추가 카메라 뷰)
- `/video3`: Additional camera view (추가 카메라 뷰)
- `/video4`: Additional camera view (추가 카메라 뷰)

**Message Type**: `sensor_msgs/msg/Image`
**Encoding**: RGB8 or BGR8
**Resolution**: Varies by camera

#### LiDAR Topics / LiDAR 토픽
- `/velodyne_points_filtered`: Filtered 3D point cloud (필터링된 3D 포인트 클라우드) - **965 messages**
- `/scan`: 2D laser scan (2D 레이저 스캔) - **965 messages**

**PointCloud Type**: `sensor_msgs/msg/PointCloud2`
**LaserScan Type**: `sensor_msgs/msg/LaserScan`

#### Control Topics / 제어 토픽
- `/Control/serial_data`: Vehicle control commands (차량 제어 명령) - **9,728 messages**

**Message Type**: `std_msgs/msg/Float32MultiArray`

**Array Structure**:
```python
# Index mapping for control data
# 인덱스 매핑
data[0]: Unknown (미사용)
data[1]: Unknown (미사용)
data[2]: Unknown (미사용)
data[3]: speed (m/s) - 속도
data[4]: steering (radian) - 조향각
data[5]: Unknown (미사용)
data[6]: Unknown (미사용)
data[7]: Unknown (미사용)

# Example / 예시
[1.0, 0.0, 0.0, 5.932, -0.0708, 35.0, 0.0, 4.0]
#                 ^^^^^ speed  ^^^^^^^ steering
```

## Script Usage / 스크립트 사용법

### Basic Usage / 기본 사용법

```bash
python scripts/extract_erp42_multiview.py \
    --bag-path <path_to_bag_file> \
    --output-dir <output_directory> \
    --topics <topic1> <topic2> ...
```

### Parameters / 매개변수

- `--bag-path`: Path to ROS2 bag file (`.db3`) or directory
  - ROS2 bag 파일 경로 (`.db3`) 또는 디렉토리
- `--output-dir`: Output directory for extracted data
  - 추출된 데이터 저장 디렉토리
- `--topics`: List of topics to extract (optional, defaults to all)
  - 추출할 토픽 목록 (선택사항, 기본값은 모든 토픽)
- `--batch`: Process multiple bag files in a directory
  - 디렉토리 내 여러 bag 파일 일괄 처리

### Example Commands / 예제 명령어

#### Extract Camera Only / 카메라만 추출
```bash
python scripts/extract_erp42_multiview.py \
    --bag-path data/rosbags/001_0.db3 \
    --output-dir data/extracted/camera_only \
    --topics /video1 /Control/serial_data
```

#### Extract Camera + LiDAR / 카메라 + LiDAR 추출
```bash
python scripts/extract_erp42_multiview.py \
    --bag-path data/rosbags/001_0.db3 \
    --output-dir data/extracted/multimodal \
    --topics /video1 /velodyne_points_filtered /scan /Control/serial_data
```

#### Extract All Topics / 모든 토픽 추출
```bash
python scripts/extract_erp42_multiview.py \
    --bag-path data/rosbags/001_0.db3 \
    --output-dir data/extracted/full
```

#### Batch Processing / 일괄 처리
```bash
python scripts/extract_erp42_multiview.py \
    --bag-path data/rosbags/ \
    --output-dir data/extracted/ \
    --batch
```

## Output Structure / 출력 구조

The extraction script creates the following directory structure:

추출 스크립트는 다음 디렉토리 구조를 생성합니다:

```
output_dir/
├── images/                          # Camera images / 카메라 이미지
│   ├── _video1_<timestamp>.jpg     # Front camera / 전방 카메라
│   ├── _video2_<timestamp>.jpg     # Additional views / 추가 뷰
│   └── ...
├── lidar/                           # LiDAR data / LiDAR 데이터
│   ├── _velodyne_points_filtered_<timestamp>.npy    # Point clouds
│   ├── _scan_<timestamp>.npz                        # Laser scans
│   └── ...
├── csv/                             # Tabular data / 테이블 데이터
│   ├── control_data.csv            # Control commands / 제어 명령
│   └── metadata.csv                # Additional metadata / 추가 메타데이터
└── metadata.json                    # Extraction summary / 추출 요약
```

### File Formats / 파일 형식

#### Images / 이미지
- **Format**: JPEG
- **Naming**: `<topic_name>_<timestamp_ns>.jpg`
- **Color**: RGB (converted from bag encoding)

#### LiDAR PointCloud / 포인트 클라우드
- **Format**: NumPy `.npy` file
- **Shape**: `(N, 3)` where N is number of points
- **Columns**: `[x, y, z]` in meters

#### LaserScan / 레이저 스캔
- **Format**: NumPy `.npz` compressed archive
- **Contents**:
  - `ranges`: Distance measurements (거리 측정값)
  - `intensities`: Intensity values (강도 값)
  - `angle_min`: Minimum scan angle (최소 스캔 각도)
  - `angle_max`: Maximum scan angle (최대 스캔 각도)
  - `angle_increment`: Angular resolution (각도 해상도)
  - `range_min`: Minimum valid range (최소 유효 거리)
  - `range_max`: Maximum valid range (최대 유효 거리)

#### Control Data CSV / 제어 데이터 CSV
```csv
timestamp,speed,steering
1234567890000000000,5.932,-0.0708
1234567890100000000,5.845,-0.0650
...
```

Columns:
- `timestamp`: Message timestamp in nanoseconds (나노초 단위 타임스탬프)
- `speed`: Vehicle speed in m/s (차량 속도, m/s)
- `steering`: Steering angle in radians (조향각, 라디안)

### Metadata JSON / 메타데이터 JSON

```json
{
  "bag_path": "data/rosbags/001_0.db3",
  "topics": {
    "/video1": "sensor_msgs/msg/Image",
    "/Control/serial_data": "std_msgs/msg/Float32MultiArray",
    "/velodyne_points_filtered": "sensor_msgs/msg/PointCloud2",
    "/scan": "sensor_msgs/msg/LaserScan"
  },
  "message_counts": {
    "/video1": 2404,
    "/Control/serial_data": 9728,
    "/velodyne_points_filtered": 965,
    "/scan": 965
  },
  "time_range": {
    "start": 1234567890000000000,
    "end": 1234567990000000000,
    "duration_seconds": 100.0
  }
}
```

## Data Statistics / 데이터 통계

Based on the example bag file `001_0.db3`:

예제 bag 파일 `001_0.db3` 기준:

| Topic | Message Count | Frequency | Duration |
|-------|--------------|-----------|----------|
| `/video1` | 2,404 | ~24 Hz | ~100s |
| `/velodyne_points_filtered` | 965 | ~10 Hz | ~100s |
| `/scan` | 965 | ~10 Hz | ~100s |
| `/Control/serial_data` | 9,728 | ~97 Hz | ~100s |

**Expected Synchronized Samples**: ~900-1,000 samples after timestamp matching
(with 0.1s threshold)

**예상 동기화 샘플 수**: 타임스탬프 매칭 후 약 900-1,000개 샘플 (0.1초 임계값 기준)

## Troubleshooting / 문제 해결

### Common Issues / 일반적인 문제

#### Issue: "Bag file not found"
**Solution**: Check the path to your bag file. Make sure the `.db3` file exists.

**해결책**: bag 파일 경로를 확인하세요. `.db3` 파일이 존재하는지 확인하세요.

#### Issue: "No messages found for topic"
**Solution**: Verify the topic name using:
```bash
ros2 bag info data/rosbags/001_0.db3
```

**해결책**: 다음 명령으로 토픽 이름을 확인하세요:

#### Issue: "Out of disk space"
**Solution**: 
- Extract only necessary topics
- Use compression for images (already JPEG)
- Remove extracted data after preprocessing

**해결책**:
- 필요한 토픽만 추출
- 이미지 압축 사용 (이미 JPEG 형식)
- 전처리 후 추출 데이터 제거

#### Issue: "Image encoding error"
**Solution**: The script supports rgb8, bgr8, and mono8. Other encodings may need manual handling.

**해결책**: 스크립트는 rgb8, bgr8, mono8을 지원합니다. 다른 인코딩은 수동 처리가 필요할 수 있습니다.

## Performance Tips / 성능 팁

1. **Selective Extraction / 선택적 추출**: Only extract topics you need to save time and space
2. **Batch Processing / 일괄 처리**: Use `--batch` flag for multiple bag files
3. **Storage / 저장소**: Use SSD for faster I/O operations
4. **Parallel Processing / 병렬 처리**: Process multiple bags in parallel on different machines

## Next Steps / 다음 단계

After extraction, proceed to data preprocessing:

추출 후 데이터 전처리 진행:

1. **Camera-Only**: Use `scripts/preprocess_camera_only.py`
2. **Multimodal**: Use `scripts/preprocess_multimodal.py`

See [TRAINING.md](TRAINING.md) for training instructions.

학습 가이드는 [TRAINING.md](TRAINING.md)를 참조하세요.

## Additional Resources / 추가 자료

- [ROS2 Bag Format](https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)
- [rosbags Python Library](https://gitlab.com/ternaris/rosbags)
- [ERP42 Vehicle Documentation](https://github.com/ERPLab/ERP42)
