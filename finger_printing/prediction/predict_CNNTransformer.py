import sys
import os
import torch
import joblib
import numpy as np
import yaml
import pandas as pd
from filterpy.kalman import KalmanFilter

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

from finger_printing.models.model_CNNTransformer import WifiCNNTransformer

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# YAML 파일에서 하이퍼파라미터 불러오기
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 예측 함수
def predict_rssi(model_path, encoder_path, norm_path, input_data, config_path):
    config = load_config(config_path)

    # 모델, 인코더, 정규화 파라미터 불러오기
    batch_size = config['batch_size']
    data_path = config['data_path']
    dropout_rate = config['dropout_rate']
    early_stopping = config['early_stopping']
    embedding_dim = config['embedding_dim']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    num_ap = config['num_ap']
    num_classes = config['num_classes']
    num_mac = config['num_mac']
    scheduler = config['scheduler']
    transformer_heads = config['transformer_heads']
    transformer_layers = config['transformer_layers']

    # 3. 모델 초기화
    model = WifiCNNTransformer(
        num_ap=num_ap,
        num_classes=num_classes,
        num_mac=num_mac,
        embedding_dim=embedding_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()  # 예측 모드로 설정

    # 인코더 및 정규화 파라미터 불러오기
    encoders = joblib.load(encoder_path)
    location_encoder = encoders['location_encoder']
    mac_encoder = encoders['mac_encoder']
    norm_params = joblib.load(norm_path)
    rssi_mean = norm_params['mean']
    rssi_std = norm_params['std']

    max_ap = 100
    P_0 = -40
    n = 3
    num_classes = len(location_encoder.classes_)
    num_mac = len(mac_encoder.classes_)

    # 입력 데이터 처리 (가중치 계산)
    input_vector = np.zeros((max_ap, 2))
    for i, (mac, rssi) in enumerate(input_data.items()):
        if i >= max_ap:
            break
        if mac in mac_encoder.classes_:
            mac_index = mac_encoder.transform([mac])[0]
            distance = 10**((P_0-rssi)/(10*n))
            input_vector[i] = [mac_index, distance]
            
    input_tensor = torch.tensor(
        input_vector[:, 1], dtype=torch.float32
    ).unsqueeze(0)
    mac_tensor = torch.tensor(
        input_vector[:, 0], dtype=torch.long
    ).unsqueeze(0)
    
    input_tensor = (input_tensor - rssi_mean) / rssi_std


    # 예측
    with torch.no_grad():
        predictions= model(input_tensor, mac_tensor)  # 모델 예측
        predicted_labels= torch.argmax(predictions, dim=1).cpu().numpy()

    # 예측된 레이블을 위치로 변환
    predicted_locations= location_encoder.inverse_transform(predicted_labels)

    return predicted_locations


# 데이터 전처리 함수 (칼만 필터 적용)
def apply_kalman_filter(rssi_values):
    kf= KalmanFilter(dim_x=1, dim_z=1)
    kf.x= np.array([[rssi_values.iloc[0]]])
    kf.F, kf.H= np.array([[1]]), np.array([[1]])
    kf.P *= 1000
    kf.R= 5
    kf.Q= 1e-5
    filtered= []
    for r in rssi_values:
        kf.predict()
        kf.update(np.array([[r]]))
        filtered.append(kf.x[0, 0])
    return np.array(filtered)


if __name__ == "__main__":

    # 예시: 모델 경로, 인코더 경로, 정규화 파라미터 경로 및 예측을 위한 데이터
    model_path= "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250423_181508.pt"
    encoder_path= "./finger_printing/checkpoints/encoders/encoders_20250423_181508.pkl"
    norm_path= "./finger_printing/checkpoints/norm/norm_20250423_181508.pkl"
    config_path= "./finger_printing/config/hyperparameters_20250423_215818.yaml"

    input_data= {'26:3f:0b:e2:66:14': -54, '2c:3f:0b:e2:66:3f': -59, 'da:55:a8:6e:4b:9c': -86, '26:3f:0b:e2:64:73': -77, '58:86:94:de:1b:78': -78, '70:5d:cc:a3:b1:fa': -90, '78:db:2f:0f:fb:06': -77, '26:3f:0b:e2:66:15': -86, '78:db:2f:10:01:dd': -77, '78:db:2f:0f:f2:fd': -88, '82:e8:2c:60:50:3a': -90, '2e:3f:1b:e2:66:b0': -85, '36:3f:0b:59:2e:20': -77, '26:3f:0b:59:2e:56': -74, '78:db:2f:0f:fe:a3': -79, '22:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:65:87': -79, '2a:3f:1b:e2:65:61': -87, '26:3f:1b:e2:65:61': -88, '22:3f:1b:e2:65:87': -88, '12:3f:0b:e2:66:3f': -58, '26:3f:1b:59:2e:56': -79, '2a:3f:0b:e2:64:73': -77, '12:3f:0b:e2:65:75': -83, '26:3f:0b:e2:66:b0': -69, '2a:3f:0b:e2:66:15': -87, '2a:3f:0b:e2:66:14': -55, '90:9f:33:69:18:98': -89, '12:3f:0b:e2:67:13': -81, '58:86:94:de:10:e4': -53, '58:86:94:99:68:8e': -74, '58:86:94:ae:06:de': -85, '22:3f:0b:59:2e:56': -74, '82:de:4b:62:c7:ed': -66, '2a:3f:1b:59:2e:56': -79, '12:3f:1b:59:2e:56': -79, '82:de:4b:62:c7:ec': -66, '2a:3f:1b:e2:66:3f': -70, '36:3f:0b:e2:67:13': -73, '36:3f:0b:e2:65:75': -79, '00:26:66:fe:8d:74': -80, '22:3f:0b:e2:66:b0': -68, '78:db:2f:0f:fa:ac': -79, '12:3f:1b:e2:66:b0': -84, '26:3f:1b:e2:66:3f': -71, 'e2:bb:9e:fd:51:93': -88, '78:db:2f:10:04:59': -79, '2a:3f:1b:e2:65:87': -89, 'e2:55:a8:6e:4b:9c': -85, '36:3f:1b:e2:64:73': -82, '36:3f:0b:e2:66:3f': -59, '78:db:2f:0f:ff:d2': -86, '58:86:94:de:6c:84': -72, '36:3f:1b:e2:66:14': -61, '2a:3f:0b:e2:67:13': -73, '2c:3f:0b:59:2e:20': -77, 'fe:55:a8:6e:4b:9c': -85, '22:3f:1b:e2:65:61': -88, '36:3f:1b:e2:66:b0': -85, '78:db:2f:0f:fa:9a': -65, '36:3f:1b:59:2e:56': -79, '1c:28:af:af:58:a2': -85, '26:3f:1b:e2:65:87': -89, '2c:3f:0b:e2:67:13': -73, '22:3f:0b:e2:66:15': -86, '22:3f:0b:e2:66:14': -
                55, '28:ee:52:de:60:eb': -64, '28:ee:52:de:60:ea': -88, '22:3f:0b:e2:64:73': -77, '2e:3f:1b:59:2e:56': -79, '26:3f:0b:e2:65:87': -78, 'b2:ee:52:de:60:ed': -73, '36:3f:0b:e2:66:21': -81, 'ee:55:a8:6e:4b:9c': -84, '2a:3f:0b:59:2e:20': -77, 'f6:9e:38:74:f9:08': -86, '36:3f:1b:e2:65:61': -88, '2c:3f:0b:e2:66:15': -86, '22:3f:1b:e2:66:b0': -85, 'c6:9e:38:74:f9:08': -86, '2e:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:64:73': -77, '26:3f:0b:e2:67:13': -73, '22:3f:1b:e2:64:73': -82, '26:3f:0b:e2:65:75': -79, '12:3f:0b:59:2e:56': -74, '2a:3f:0b:e2:66:3f': -59, 'fa:9e:38:74:f9:08': -87, '22:3f:1b:e2:66:14': -61, '36:3f:0b:e2:65:61': -78, '00:04:ea:45:8a:14': -85, '26:3f:0b:e2:66:3f': -59, '78:db:2f:0f:fb:66': -74, '7a:de:4b:62:c7:ec': -88, '36:3f:0b:59:2e:56': -74, '7a:de:4b:62:c7:eb': -88, '70:5d:cc:ca:e6:f4': -74, '2a:3f:1b:e2:66:b0': -85, '12:3f:1b:e2:66:3f': -70, '22:3f:0b:e2:66:3f': -59, 'e4:55:a8:6e:4b:9c': -85, '2a:3f:1b:e2:64:73': -82, '70:5d:cc:cb:e6:f4': -67, '2a:3f:0b:e2:66:a9': -86, '36:3f:1b:e2:65:87': -88, '2a:3f:1b:e2:66:14': -61, '22:3f:0b:e2:67:13': -73, '22:3f:0b:e2:65:75': -78, '26:3f:1b:e2:66:b0': -85, '36:3f:0b:e2:66:b0': -70, '26:3f:1b:e2:66:14': -61, '12:3f:1b:59:2e:20': -91, '36:3f:0b:e2:66:15': -86, '36:3f:0b:e2:66:14': -55, '22:3f:0b:59:2e:20': -77, '36:3f:1b:e2:66:3f': -70, '2a:3f:0b:59:2e:56': -74, '26:3f:1b:e2:64:73': -82, 'aa:ee:52:de:60:ec': -87, 'ea:55:a8:6e:4b:9c': -85, '70:5d:cc:60:cd:b0': -70, '78:db:2f:0f:fe:9b': -67, '22:3f:0b:e2:65:87': -79, 'd4:6e:0e:a5:71:3c': -81, '58:86:94:db:27:e4': -62, '88:36:6c:69:7a:b2': -78, '1c:28:af:ae:f9:62': -88, '88:36:6c:87:0b:30': -94, '22:3f:1b:59:2e:56': -79, '40:ca:63:4e:41:b5': -80, '58:86:94:c9:8c:fa': -85, 'b8:62:1f:50:6e:5c': -87}


    # 예측 수행
    predicted_locations = predict_rssi(
        model_path, encoder_path, norm_path, input_data, config_path)

    print("예측된 위치:", predicted_locations)
