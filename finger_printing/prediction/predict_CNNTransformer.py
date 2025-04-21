import torch
import joblib
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from finger_printing.models.model_CNNTransformer import WifiCNNTransformer
from finger_printing.train.train_CNNTransformer_sweep import create_dataset

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 예측 함수
def predict_rssi(model_path, encoder_path, norm_path, input_data):
    # 모델, 인코더, 정규화 파라미터 불러오기
    model = WifiCNNTransformer(num_ap=100, num_classes=10, num_mac=300)  # 모델 파라미터 맞게 설정
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # 예측 모드로 설정
    
    # 인코더 및 정규화 파라미터 불러오기
    encoders = joblib.load(encoder_path)
    location_encoder = encoders['location_encoder']
    mac_encoder = encoders['mac_encoder']
    norm_params = joblib.load(norm_path)
    rssi_mean = norm_params['mean']
    rssi_std = norm_params['std']

    # 입력 데이터를 처리 (이곳에서 필요한 전처리 적용)
    input_data['rssi_filtered'] = input_data.groupby('MAC')['RSSI'].transform(apply_kalman_filter)
    input_data['rssi_weighted'] = input_data.groupby('MAC')['rssi_filtered'].transform(lambda x: np.average(x, weights=np.abs(x)))
    
    # 정규화 처리
    input_data['rssi_norm'] = (input_data['rssi_weighted'] - rssi_mean) / rssi_std

    # 데이터셋 준비 (기존 데이터셋 생성 코드 참고)
    X, _ = create_dataset(input_data, mac_encoder)  # 실제 데이터로 X만 추출

    # 예측
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        predictions = model(X_tensor)  # 모델 예측
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

    # 예측된 레이블을 위치로 변환
    predicted_locations = location_encoder.inverse_transform(predicted_labels)

    return predicted_locations


# 데이터 전처리 함수 (칼만 필터 적용)
def apply_kalman_filter(rssi_values):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[rssi_values.iloc[0]]])
    kf.F, kf.H = np.array([[1]]), np.array([[1]])
    kf.P *= 1000
    kf.R = 5
    kf.Q = 1e-5
    filtered = []
    for r in rssi_values:
        kf.predict()
        kf.update(np.array([[r]]))
        filtered.append(kf.x[0, 0])
    return np.array(filtered)


# 예시: 모델 경로, 인코더 경로, 정규화 파라미터 경로 및 예측을 위한 데이터
model_path = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250409_000650.pt"
encoder_path = "./finger_printing/checkpoints/encoders/encoders_20250409_000650.pkl"
norm_path = "./finger_printing/checkpoints/norm/norm_20250409_000650.pkl"
input_data = pd.read_csv('your_input_data.csv')  # 예측에 사용할 데이터 로드

# 예측 수행
predicted_locations = predict_rssi(model_path, encoder_path, norm_path, input_data)

print("예측된 위치:", predicted_locations)