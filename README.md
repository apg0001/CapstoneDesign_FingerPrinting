# CapstoneDesign_FingerPrinting
# 📍 WiFi 핑거프린팅 실내 위치 추정 API

WiFi AP의 MAC 주소 및 RSSI 데이터를 기반으로 실내 위치를 예측하는 FastAPI 기반 REST API입니다.  
사전에 학습된 CNN + Embedding 모델을 통해 WiFi 스캔 데이터를 입력받아 위치를 반환합니다.

---

## 🚀 주요 기능

- FastAPI 기반 RESTful API 제공
- WiFi 스캔 데이터로 실내 위치 예측
- CNN + Embedding 모델 (PyTorch 기반)
- Kalman Filter, Log-distance, KNN Imputer 전처리
- 최신 모델 & 인코더 자동 불러오기
- Docker, AWS 배포 지원 예정

---

## 🏗️ 프로젝트 구조

```
├── app/
│   ├── main.py          # FastAPI 서버 실행 파일
│   ├── model.py         # PyTorch CNN + Embedding 모델 정의
│   ├── utils.py         # 모델, 인코더 불러오기 및 예측 함수
│   └── models/          # 학습된 모델 및 인코더 저장 폴더
│       ├── fp_model_YYYYMMDD.pt
│       ├── location_encoder_YYYYMMDD.pkl
│       └── mac_encoder_YYYYMMDD.pkl
├── datasets/            # 수집된 WiFi RSSI 데이터
├── requirements.txt     # Python 의존성 패키지
└── README.md
```

---

## ⚙️ 설치 방법

```bash
git clone https://github.com/apg0001/wifi-fingerprinting-api.git
cd wifi-fingerprinting-api
pip install -r requirements.txt
```

---

## 🔥 실행 방법

### 1️⃣ 서버 실행 (로컬 환경)

```bash
uvicorn app.main:app --reload
```

- 기본 실행: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI 문서: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API 사용법

### 위치 예측 API

**POST /predict**

| 파라미터 | 타입  | 설명 |
|----------|------|------------|
| data     | JSON | `{ "MAC": RSSI }` 형태의 WiFi 스캔 데이터 |

**요청 예시:**

```json
{
  "data": {
    "26:3f:0b:e2:66:14": -54,
    "2c:3f:0b:e2:66:3f": -59,
    "da:55:a8:6e:4b:9c": -86
  }
}
```

**응답 예시:**

```json
{
  "predicted_location": "hallway_1"
}
```

---

## 🧠 모델 정보

- **구조**: CNN + Embedding (PyTorch 기반)
- **입력 데이터**: MAC 주소 + RSSI 값 최대 70개 AP
- **전처리 과정**:
  - Kalman Filter: RSSI 노이즈 제거
  - Log-distance Path Loss 모델로 거리 변환
  - KNN Imputer로 결측치 처리
- **출력**: 위치 인덱스 (LabelEncoder로 디코딩하여 위치명 반환)

---

## 📦 향후 계획

- Docker + Nginx + AWS 배포 가이드 제공 예정
- 대시보드 구축 (React 기반)
- 실시간 데이터 기반 재학습 지원
- 사용자 인증 & 관리 기능 추가
- Android 앱 연동 (WiFi 스캔 데이터 실시간 전송)

---

## 👥 팀 소개

**팀명: 핑프 (Ping-FP)**  
WiFi 핑거프린팅 기술을 통해 정확한 실내 위치 추정을 목표로 개발하고 있습니다.

| 역할           | 담당자 |
|----------------|--------|
| 데이터 수집    | Team 핑프    |
| 모델 개발      | 박기찬    |
| API 서버 구축  | 박기찬    |
| 프론트엔드/대시보드 | 누군가 하겠지 |

---

## 📝 참고

- PyTorch
- FastAPI
- scikit-learn
- filterpy (Kalman Filter)
- uvicorn
