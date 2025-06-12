# WiFi Fingerprinting Indoor Localization

## 프로젝트 개요

이 프로젝트는 WiFi 신호 강도를 기반으로 실내 위치를 추정하는 시스템입니다. CNN-Transformer 아키텍처를 사용하여 WiFi 신호의 패턴을 학습하고, 실내 위치를 정확하게 예측합니다.

## 주요 기능

- WiFi 신호 기반 실내 위치 추정
- FastAPI 기반 RESTful API 제공
- CNN-Transformer 딥러닝 모델 사용
- Kalman Filter, Log-distance, KNN Imputer 전처리
- 실시간 예측 API 제공
- 온라인 학습 지원
- 데이터 증강 및 전처리 도구
- Weights & Biases를 통한 하이퍼파라미터 튜닝

## 기술 스택

- FastAPI: REST API 서버
- PyTorch: 딥러닝 프레임워크
- Nginx: 리버스 프록시
- Docker & Docker Compose: 컨테이너화
- Weights & Biases: 실험 관리 및 하이퍼파라미터 튜닝

## 시스템 아키텍처

```
├── app/
│   ├── main.py              # FastAPI 서버
│   ├── predict.py           # 예측 로직
│   ├── model_CNNTransformer.py  # 모델 정의
│   └── online_trainer.py    # 온라인 학습
├── finger_printing/
│   ├── datasets_manager/    # 데이터셋 관리
│   │   ├── data_augmentation_*.py
│   │   ├── merge_datasets.py
│   │   └── ...
│   ├─ train/              # 모델 학습
│   │   ├── train_CNNTransformer.py
│   │   ├── train_CNNTransformer_sweep.py
│   │   └── ...
│   └── datasets/ # Wi-Fi RSSI 데이터셋
├── nginx/
│   └── nginx.conf         # Nginx 설정
├── docker-compose.yml     # Docker Compose 설정
├── Dockerfile             # Docker 이미지 설정
└── requirements.txt       # Python 의존성 패키지

```

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. Docker를 사용한 실행

```bash
# Docker Compose로 서비스 실행
docker-compose up -d
```

### 3. API 사용

```bash
# 위치 예측 API 호출
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"mac_rssi": {"location1": {"mac1": -50, "mac2": -60}, "location2": {"mac1": -55, "mac2": -65}, "location3": {"mac1": -45, "mac2": -70}}}'
```

## 모델 학습

### 1. 데이터 전처리

```bash
# 데이터셋 병합
python finger_printing/datasets_manager/merge_datasets.py

# 데이터 증강
python finger_printing/datasets_manager/data_augmentation_selective.py
python finger_printing/datasets_manager/data_augmentation_all.py
```

### 2. 모델 학습

```bash
# 기본 학습
python finger_printing/train/train_CNNTransformer.py

# Weights & Biases를 사용한 하이퍼파라미터 튜닝
python finger_printing/train/train_CNNTransformer_sweep.py
```

## 모델 아키텍처

### CNN-Transformer 모델 구조
- **입력 데이터**:
  - RSSI 값 (정규화된 신호 강도)
  - MAC 주소 (임베딩된 AP 식별자)
- **모델 구성**:
  1. MAC 주소 임베딩 레이어
  2. 1D CNN 레이어 (특징 추출)
  3. Transformer 인코더 (시퀀스 처리)
  4. 완전연결 레이어 (분류)
- **주요 하이퍼파라미터**:
  - 임베딩 차원: 16
  - Transformer 헤드 수: 4
  - Transformer 레이어 수: 4
  - 드롭아웃 비율: 0.3

### 데이터 전처리 및 정규화
1. **Kalman Filter**:
   - RSSI 노이즈 제거
   - 신호 강도 안정화
2. **Log-distance Path Loss**:
   - 거리에 따른 신호 감쇠 모델링
   - RSSI 값 정규화
3. **KNN Imputer**:
   - 결측값 보간
   - 데이터 완전성 보장
4. **정규화**:
   - 평균/표준편차 기반 RSSI 정규화
   - 배치 정규화 (BatchNorm) 적용

## Weights & Biases 설정

1. Weights & Biases 계정 생성 및 로그인
2. API 키 설정

```bash
wandb login
```

3. 하이퍼파라미터 튜닝 실행

```bash
python finger_printing/train/train_CNNTransformer_sweep.py
```

## Nginx 설정

- 리버스 프록시로 FastAPI 서버를 프록시
- SSL/TLS 종료
- 로드 밸런싱
- 정적 파일 서빙

## Docker Compose 구성

- FastAPI 서비스
- Nginx 서비스
- 볼륨 마운트
- 네트워크 설정
- 환경 변수 관리

## 👥 팀 소개

**팀명: 핑프 (Ping-FP)**  
WiFi 핑거프린팅 기술을 통해 정확한 실내 위치 추정을 목표로 개발하고 있습니다.

| 역할                | 담당자    |
| ------------------- | --------- |
| 데이터 수집         | Team 핑프 |
| 모델 개발           | 박기찬    |
| API 서버            | 박기찬    |
| 백엔드 서버         | 신민규    |
| 프론트엔드/대시보드 | 박성재    |
| Android             | 김동욱    |
