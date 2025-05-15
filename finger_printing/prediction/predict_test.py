import torch
import numpy as np
import joblib
import yaml
import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

from filterpy.kalman import KalmanFilter
from finger_printing.models.model_CNNTransformer import WifiCNNTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 칼만 필터 함수 (학습 때와 동일하게 사용)


def apply_kalman_filter(rssi_values):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[rssi_values[0]]])
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

# 전처리 함수 (학습 때와 동일하게 맞춤)


def preprocess_input(input_mac_rssi, mac_encoder, rssi_mean, rssi_std, max_ap=100, rssi_threshold=-95):
    macs = []
    rssis = []

    # 1. rssi_threshold 필터링
    for mac, rssi in input_mac_rssi.items():
        if rssi > rssi_threshold:
            macs.append(mac)
            rssis.append(rssi)
    if len(rssis) == 0:
        raise ValueError("No RSSI values above threshold")

    # 2. 칼만 필터 적용
    rssis_filtered = apply_kalman_filter(np.array(rssis))

    # 3. 거리 기반 가중 평균 (예시)
    weights = np.abs(rssis_filtered)
    rssi_weighted = np.average(rssis_filtered, weights=weights)

    # 4. 정규화
    rssi_norm = (rssi_weighted - rssi_mean) / rssi_std

    # 5. MAC 인코딩 (존재하지 않는 MAC 무시)
    mac_indices = []
    rssi_values = []
    for mac, rssi in zip(macs, [rssi_norm]*len(macs)):
        try:
            idx = mac_encoder.transform([mac])[0] + 1  # 0은 패딩용
            mac_indices.append(idx)
            rssi_values.append(rssi)
        except ValueError:
            # 인코딩 불가능한 MAC 무시
            continue

    # 6. 패딩 또는 자르기
    if len(mac_indices) < max_ap:
        pad_len = max_ap - len(mac_indices)
        mac_indices = np.pad(mac_indices, (0, pad_len), constant_values=0)
        rssi_values = np.pad(rssi_values, (0, pad_len), constant_values=0)
    else:
        mac_indices = mac_indices[:max_ap]
        rssi_values = rssi_values[:max_ap]

    # 7. 입력 배열 생성
    X = np.column_stack([mac_indices, rssi_values])
    return np.expand_dims(X, axis=0)  # 배치 차원 추가

# 모델 로드 및 추론 함수


def load_model_and_predict(input_mac_rssi, model_path, encoder_path, norm_path, config_path):
    # 설정 로드
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 인코더, 정규화 파라미터 로드
    encoders = joblib.load(encoder_path)
    location_encoder = encoders["location_encoder"]
    mac_encoder = encoders["mac_encoder"]

    norm_params = joblib.load(norm_path)
    rssi_mean = norm_params["mean"]
    rssi_std = norm_params["std"]

    # 전처리
    X = preprocess_input(input_mac_rssi, mac_encoder,
                         rssi_mean, rssi_std, max_ap=config["num_ap"])

    # 모델 초기화
    model = WifiCNNTransformer(
        num_ap=config["num_ap"],
        num_classes=config["num_classes"],
        num_mac=config["num_mac"],
        embedding_dim=config["embedding_dim"],
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"],
        dropout_rate=config["dropout_rate"],
    ).to(device)

    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 입력을 tensor로 변환
    rssi_tensor = torch.tensor(X[:, :, 1], dtype=torch.float32).to(device)
    mac_tensor = torch.tensor(X[:, :, 0], dtype=torch.long).to(device)

    # 추론
    with torch.no_grad():
        outputs = model(rssi_tensor, mac_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).cpu().item()
        pred_location = location_encoder.inverse_transform([pred_idx])[0]

    return pred_location, probs.cpu().numpy()


# 사용 예제
if __name__ == "__main__":
    input_mac_rssi = {'26:3f:0b:e2:66:14': -54, '2c:3f:0b:e2:66:3f': -59, 'da:55:a8:6e:4b:9c': -86, '26:3f:0b:e2:64:73': -77, '58:86:94:de:1b:78': -78, '70:5d:cc:a3:b1:fa': -90, '78:db:2f:0f:fb:06': -77, '26:3f:0b:e2:66:15': -86, '78:db:2f:10:01:dd': -77, '78:db:2f:0f:f2:fd': -88, '82:e8:2c:60:50:3a': -90, '2e:3f:1b:e2:66:b0': -85, '36:3f:0b:59:2e:20': -77, '26:3f:0b:59:2e:56': -74, '78:db:2f:0f:fe:a3': -79, '22:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:65:87': -79, '2a:3f:1b:e2:65:61': -87, '26:3f:1b:e2:65:61': -88, '22:3f:1b:e2:65:87': -88, '12:3f:0b:e2:66:3f': -58, '26:3f:1b:59:2e:56': -79, '2a:3f:0b:e2:64:73': -77, '12:3f:0b:e2:65:75': -83, '26:3f:0b:e2:66:b0': -69, '2a:3f:0b:e2:66:15': -87, '2a:3f:0b:e2:66:14': -55, '90:9f:33:69:18:98': -89, '12:3f:0b:e2:67:13': -81, '58:86:94:de:10:e4': -53, '58:86:94:99:68:8e': -74, '58:86:94:ae:06:de': -85, '22:3f:0b:59:2e:56': -74, '82:de:4b:62:c7:ed': -66, '2a:3f:1b:59:2e:56': -79, '12:3f:1b:59:2e:56': -79, '82:de:4b:62:c7:ec': -66, '2a:3f:1b:e2:66:3f': -70, '36:3f:0b:e2:67:13': -73, '36:3f:0b:e2:65:75': -79, '00:26:66:fe:8d:74': -80, '22:3f:0b:e2:66:b0': -68, '78:db:2f:0f:fa:ac': -79, '12:3f:1b:e2:66:b0': -84, '26:3f:1b:e2:66:3f': -71, 'e2:bb:9e:fd:51:93': -88, '78:db:2f:10:04:59': -79, '2a:3f:1b:e2:65:87': -89, 'e2:55:a8:6e:4b:9c': -85, '36:3f:1b:e2:64:73': -82, '36:3f:0b:e2:66:3f': -59, '78:db:2f:0f:ff:d2': -86, '58:86:94:de:6c:84': -72, '36:3f:1b:e2:66:14': -61, '2a:3f:0b:e2:67:13': -73, '2c:3f:0b:59:2e:20': -77, 'fe:55:a8:6e:4b:9c': -85, '22:3f:1b:e2:65:61': -88, '36:3f:1b:e2:66:b0': -85, '78:db:2f:0f:fa:9a': -65, '36:3f:1b:59:2e:56': -79, '1c:28:af:af:58:a2': -85, '26:3f:1b:e2:65:87': -89, '2c:3f:0b:e2:67:13': -73, '22:3f:0b:e2:66:15': -86, '22:3f:0b:e2:66:14': -
                      55, '28:ee:52:de:60:eb': -64, '28:ee:52:de:60:ea': -88, '22:3f:0b:e2:64:73': -77, '2e:3f:1b:59:2e:56': -79, '26:3f:0b:e2:65:87': -78, 'b2:ee:52:de:60:ed': -73, '36:3f:0b:e2:66:21': -81, 'ee:55:a8:6e:4b:9c': -84, '2a:3f:0b:59:2e:20': -77, 'f6:9e:38:74:f9:08': -86, '36:3f:1b:e2:65:61': -88, '2c:3f:0b:e2:66:15': -86, '22:3f:1b:e2:66:b0': -85, 'c6:9e:38:74:f9:08': -86, '2e:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:64:73': -77, '26:3f:0b:e2:67:13': -73, '22:3f:1b:e2:64:73': -82, '26:3f:0b:e2:65:75': -79, '12:3f:0b:59:2e:56': -74, '2a:3f:0b:e2:66:3f': -59, 'fa:9e:38:74:f9:08': -87, '22:3f:1b:e2:66:14': -61, '36:3f:0b:e2:65:61': -78, '00:04:ea:45:8a:14': -85, '26:3f:0b:e2:66:3f': -59, '78:db:2f:0f:fb:66': -74, '7a:de:4b:62:c7:ec': -88, '36:3f:0b:59:2e:56': -74, '7a:de:4b:62:c7:eb': -88, '70:5d:cc:ca:e6:f4': -74, '2a:3f:1b:e2:66:b0': -85, '12:3f:1b:e2:66:3f': -70, '22:3f:0b:e2:66:3f': -59, 'e4:55:a8:6e:4b:9c': -85, '2a:3f:1b:e2:64:73': -82, '70:5d:cc:cb:e6:f4': -67, '2a:3f:0b:e2:66:a9': -86, '36:3f:1b:e2:65:87': -88, '2a:3f:1b:e2:66:14': -61, '22:3f:0b:e2:67:13': -73, '22:3f:0b:e2:65:75': -78, '26:3f:1b:e2:66:b0': -85, '36:3f:0b:e2:66:b0': -70, '26:3f:1b:e2:66:14': -61, '12:3f:1b:59:2e:20': -91, '36:3f:0b:e2:66:15': -86, '36:3f:0b:e2:66:14': -55, '22:3f:0b:59:2e:20': -77, '36:3f:1b:e2:66:3f': -70, '2a:3f:0b:59:2e:56': -74, '26:3f:1b:e2:64:73': -82, 'aa:ee:52:de:60:ec': -87, 'ea:55:a8:6e:4b:9c': -85, '70:5d:cc:60:cd:b0': -70, '78:db:2f:0f:fe:9b': -67, '22:3f:0b:e2:65:87': -79, 'd4:6e:0e:a5:71:3c': -81, '58:86:94:db:27:e4': -62, '88:36:6c:69:7a:b2': -78, '1c:28:af:ae:f9:62': -88, '88:36:6c:87:0b:30': -94, '22:3f:1b:59:2e:56': -79, '40:ca:63:4e:41:b5': -80, '58:86:94:c9:8c:fa': -85, 'b8:62:1f:50:6e:5c': -87}

    model_path = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
    encoder_path = "./finger_printing/checkpoints/encoders/encoders_20250423_181508.pkl"
    norm_path = "./finger_printing/checkpoints/norm/norm_20250423_181508.pkl"
    config_path = "./finger_printing/config/hyperparameters_20250423_181508.yaml"

    location, probabilities = load_model_and_predict(
        input_mac_rssi, model_path, encoder_path, norm_path, config_path)
    print(f"Predicted location: {location}")
    # print(f"Probabilities: {probabilities}")
