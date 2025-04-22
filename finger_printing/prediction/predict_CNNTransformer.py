import sys
import os
import torch
import joblib
import numpy as np
from filterpy.kalman import KalmanFilter

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

from finger_printing.train.train_CNNTransformer_sweep import create_dataset
from finger_printing.models.model_CNNTransformer import WifiCNNTransformer
# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 예측 함수


def predict_rssi(model_path, encoder_path, norm_path, input_data):
    # 모델, 인코더, 정규화 파라미터 불러오기
    model = WifiCNNTransformer(
        num_ap=100, num_classes=10, num_mac=300)  # 모델 파라미터 맞게 설정
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

    # 입력 데이터 처리 (칼만 필터 적용 및 가중치 계산)
    input_data['rssi_filtered'] = input_data.groupby(
        'MAC')['RSSI'].transform(apply_kalman_filter)
    input_data['rssi_weighted'] = input_data.groupby(
        'MAC')['rssi_filtered'].transform(lambda x: np.average(x, weights=np.abs(x)))

    # 정규화 처리
    input_data['rssi_norm'] = (
        input_data['rssi_weighted'] - rssi_mean) / rssi_std

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

input_data = {'26:3f:0b:e2:66:14': -54, '2c:3f:0b:e2:66:3f': -59, 'da:55:a8:6e:4b:9c': -86, '26:3f:0b:e2:64:73': -77, '58:86:94:de:1b:78': -78, '70:5d:cc:a3:b1:fa': -90, '78:db:2f:0f:fb:06': -77, '26:3f:0b:e2:66:15': -86, '78:db:2f:10:01:dd': -77, '78:db:2f:0f:f2:fd': -88, '82:e8:2c:60:50:3a': -90, '2e:3f:1b:e2:66:b0': -85, '36:3f:0b:59:2e:20': -77, '26:3f:0b:59:2e:56': -74, '78:db:2f:0f:fe:a3': -79, '22:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:65:87': -79, '2a:3f:1b:e2:65:61': -87, '26:3f:1b:e2:65:61': -88, '22:3f:1b:e2:65:87': -88, '12:3f:0b:e2:66:3f': -58, '26:3f:1b:59:2e:56': -79, '2a:3f:0b:e2:64:73': -77, '12:3f:0b:e2:65:75': -83, '26:3f:0b:e2:66:b0': -69, '2a:3f:0b:e2:66:15': -87, '2a:3f:0b:e2:66:14': -55, '90:9f:33:69:18:98': -89, '12:3f:0b:e2:67:13': -81, '58:86:94:de:10:e4': -53, '58:86:94:99:68:8e': -74, '58:86:94:ae:06:de': -85, '22:3f:0b:59:2e:56': -74, '82:de:4b:62:c7:ed': -66, '2a:3f:1b:59:2e:56': -79, '12:3f:1b:59:2e:56': -79, '82:de:4b:62:c7:ec': -66, '2a:3f:1b:e2:66:3f': -70, '36:3f:0b:e2:67:13': -73, '36:3f:0b:e2:65:75': -79, '00:26:66:fe:8d:74': -80, '22:3f:0b:e2:66:b0': -68, '78:db:2f:0f:fa:ac': -79, '12:3f:1b:e2:66:b0': -84, '26:3f:1b:e2:66:3f': -71, 'e2:bb:9e:fd:51:93': -88, '78:db:2f:10:04:59': -79, '2a:3f:1b:e2:65:87': -89, 'e2:55:a8:6e:4b:9c': -85, '36:3f:1b:e2:64:73': -82, '36:3f:0b:e2:66:3f': -59, '78:db:2f:0f:ff:d2': -86, '58:86:94:de:6c:84': -72, '36:3f:1b:e2:66:14': -61, '2a:3f:0b:e2:67:13': -73, '2c:3f:0b:59:2e:20': -77, 'fe:55:a8:6e:4b:9c': -85, '22:3f:1b:e2:65:61': -88, '36:3f:1b:e2:66:b0': -85, '78:db:2f:0f:fa:9a': -65, '36:3f:1b:59:2e:56': -79, '1c:28:af:af:58:a2': -85, '26:3f:1b:e2:65:87': -89, '2c:3f:0b:e2:67:13': -73, '22:3f:0b:e2:66:15': -86, '22:3f:0b:e2:66:14': -
              55, '28:ee:52:de:60:eb': -64, '28:ee:52:de:60:ea': -88, '22:3f:0b:e2:64:73': -77, '2e:3f:1b:59:2e:56': -79, '26:3f:0b:e2:65:87': -78, 'b2:ee:52:de:60:ed': -73, '36:3f:0b:e2:66:21': -81, 'ee:55:a8:6e:4b:9c': -84, '2a:3f:0b:59:2e:20': -77, 'f6:9e:38:74:f9:08': -86, '36:3f:1b:e2:65:61': -88, '2c:3f:0b:e2:66:15': -86, '22:3f:1b:e2:66:b0': -85, 'c6:9e:38:74:f9:08': -86, '2e:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:64:73': -77, '26:3f:0b:e2:67:13': -73, '22:3f:1b:e2:64:73': -82, '26:3f:0b:e2:65:75': -79, '12:3f:0b:59:2e:56': -74, '2a:3f:0b:e2:66:3f': -59, 'fa:9e:38:74:f9:08': -87, '22:3f:1b:e2:66:14': -61, '36:3f:0b:e2:65:61': -78, '00:04:ea:45:8a:14': -85, '26:3f:0b:e2:66:3f': -59, '78:db:2f:0f:fb:66': -74, '7a:de:4b:62:c7:ec': -88, '36:3f:0b:59:2e:56': -74, '7a:de:4b:62:c7:eb': -88, '70:5d:cc:ca:e6:f4': -74, '2a:3f:1b:e2:66:b0': -85, '12:3f:1b:e2:66:3f': -70, '22:3f:0b:e2:66:3f': -59, 'e4:55:a8:6e:4b:9c': -85, '2a:3f:1b:e2:64:73': -82, '70:5d:cc:cb:e6:f4': -67, '2a:3f:0b:e2:66:a9': -86, '36:3f:1b:e2:65:87': -88, '2a:3f:1b:e2:66:14': -61, '22:3f:0b:e2:67:13': -73, '22:3f:0b:e2:65:75': -78, '26:3f:1b:e2:66:b0': -85, '36:3f:0b:e2:66:b0': -70, '26:3f:1b:e2:66:14': -61, '12:3f:1b:59:2e:20': -91, '36:3f:0b:e2:66:15': -86, '36:3f:0b:e2:66:14': -55, '22:3f:0b:59:2e:20': -77, '36:3f:1b:e2:66:3f': -70, '2a:3f:0b:59:2e:56': -74, '26:3f:1b:e2:64:73': -82, 'aa:ee:52:de:60:ec': -87, 'ea:55:a8:6e:4b:9c': -85, '70:5d:cc:60:cd:b0': -70, '78:db:2f:0f:fe:9b': -67, '22:3f:0b:e2:65:87': -79, 'd4:6e:0e:a5:71:3c': -81, '58:86:94:db:27:e4': -62, '88:36:6c:69:7a:b2': -78, '1c:28:af:ae:f9:62': -88, '88:36:6c:87:0b:30': -94, '22:3f:1b:59:2e:56': -79, '40:ca:63:4e:41:b5': -80, '58:86:94:c9:8c:fa': -85, 'b8:62:1f:50:6e:5c': -87}


# 예측 수행
predicted_locations = predict_rssi(
    model_path, encoder_path, norm_path, input_data)

print("예측된 위치:", predicted_locations)
