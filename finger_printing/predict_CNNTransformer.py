import torch
import numpy as np
import glob
import os
from model_CNNTransformer import WifiCNNTransformer  # 모델 정의
import joblib

# 최신 파일 가져오기 (모델 & 인코더 공통 사용)


def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"{pattern} 에 맞는 파일이 없습니다.")
    return max(files, key=os.path.getctime)

# 최신 모델 및 인코더 불러오기


def load_latest_resources(model_dir="./finger_printing/models"):
    model_path = get_latest_file(os.path.join(
        model_dir + "/finger_printing", "fp_model_CNNTransformer_*.pt"))
    location_encoder_path = get_latest_file(os.path.join(
        model_dir + "/location_encoder", "location_encoder_*.pkl"))
    mac_encoder_path = get_latest_file(os.path.join(
        model_dir + "/mac_encoder", "mac_encoder_*.pkl"))

    location_encoder = joblib.load(location_encoder_path)
    mac_encoder = joblib.load(mac_encoder_path)

    return model_path, location_encoder, mac_encoder

# 예측 함수


def predict_location(mac_rssi_dict, model_path, location_encoder, mac_encoder):
    P_0 = -40
    n = 3
    max_ap = 70  # 모델 학습 시 사용한 max_ap와 동일하게 맞춰야 함

    num_classes = len(location_encoder.classes_)
    num_mac = len(mac_encoder.classes_)

    model = WifiCNNTransformer(max_ap, num_classes, num_mac)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 입력 데이터 전처리
    input_vector = np.zeros((max_ap, 2))
    for i, (mac, rssi) in enumerate(mac_rssi_dict.items()):
        if i >= max_ap:
            break
        if mac in mac_encoder.classes_:
            mac_index = mac_encoder.transform([mac])[0]
            distance = 10 ** ((P_0 - rssi) / (10 * n))
            input_vector[i] = [mac_index, distance]

    input_tensor = torch.tensor(
        input_vector[:, 1], dtype=torch.float32).unsqueeze(0)
    mac_tensor = torch.tensor(
        input_vector[:, 0], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor, mac_tensor)
        _, predicted_index = torch.max(output, 1)

    return location_encoder.inverse_transform([predicted_index.item()])[0]


# 테스트 예시
if __name__ == "__main__":
    # 테스트 입력
    test_input = {'26:3f:0b:e2:66:14': -54, '2c:3f:0b:e2:66:3f': -59, 'da:55:a8:6e:4b:9c': -86, '26:3f:0b:e2:64:73': -77, '58:86:94:de:1b:78': -78, '70:5d:cc:a3:b1:fa': -90, '78:db:2f:0f:fb:06': -77, '26:3f:0b:e2:66:15': -86, '78:db:2f:10:01:dd': -77, '78:db:2f:0f:f2:fd': -88, '82:e8:2c:60:50:3a': -90, '2e:3f:1b:e2:66:b0': -85, '36:3f:0b:59:2e:20': -77, '26:3f:0b:59:2e:56': -74, '78:db:2f:0f:fe:a3': -79, '22:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:65:87': -79, '2a:3f:1b:e2:65:61': -87, '26:3f:1b:e2:65:61': -88, '22:3f:1b:e2:65:87': -88, '12:3f:0b:e2:66:3f': -58, '26:3f:1b:59:2e:56': -79, '2a:3f:0b:e2:64:73': -77, '12:3f:0b:e2:65:75': -83, '26:3f:0b:e2:66:b0': -69, '2a:3f:0b:e2:66:15': -87, '2a:3f:0b:e2:66:14': -55, '90:9f:33:69:18:98': -89, '12:3f:0b:e2:67:13': -81, '58:86:94:de:10:e4': -53, '58:86:94:99:68:8e': -74, '58:86:94:ae:06:de': -85, '22:3f:0b:59:2e:56': -74, '82:de:4b:62:c7:ed': -66, '2a:3f:1b:59:2e:56': -79, '12:3f:1b:59:2e:56': -79, '82:de:4b:62:c7:ec': -66, '2a:3f:1b:e2:66:3f': -70, '36:3f:0b:e2:67:13': -73, '36:3f:0b:e2:65:75': -79, '00:26:66:fe:8d:74': -80, '22:3f:0b:e2:66:b0': -68, '78:db:2f:0f:fa:ac': -79, '12:3f:1b:e2:66:b0': -84, '26:3f:1b:e2:66:3f': -71, 'e2:bb:9e:fd:51:93': -88, '78:db:2f:10:04:59': -79, '2a:3f:1b:e2:65:87': -89, 'e2:55:a8:6e:4b:9c': -85, '36:3f:1b:e2:64:73': -82, '36:3f:0b:e2:66:3f': -59, '78:db:2f:0f:ff:d2': -86, '58:86:94:de:6c:84': -72, '36:3f:1b:e2:66:14': -61, '2a:3f:0b:e2:67:13': -73, '2c:3f:0b:59:2e:20': -77, 'fe:55:a8:6e:4b:9c': -85, '22:3f:1b:e2:65:61': -88, '36:3f:1b:e2:66:b0': -85, '78:db:2f:0f:fa:9a': -65, '36:3f:1b:59:2e:56': -79, '1c:28:af:af:58:a2': -85, '26:3f:1b:e2:65:87': -89, '2c:3f:0b:e2:67:13': -73, '22:3f:0b:e2:66:15': -86, '22:3f:0b:e2:66:14': -
                  55, '28:ee:52:de:60:eb': -64, '28:ee:52:de:60:ea': -88, '22:3f:0b:e2:64:73': -77, '2e:3f:1b:59:2e:56': -79, '26:3f:0b:e2:65:87': -78, 'b2:ee:52:de:60:ed': -73, '36:3f:0b:e2:66:21': -81, 'ee:55:a8:6e:4b:9c': -84, '2a:3f:0b:59:2e:20': -77, 'f6:9e:38:74:f9:08': -86, '36:3f:1b:e2:65:61': -88, '2c:3f:0b:e2:66:15': -86, '22:3f:1b:e2:66:b0': -85, 'c6:9e:38:74:f9:08': -86, '2e:3f:1b:e2:66:3f': -70, '2c:3f:0b:e2:64:73': -77, '26:3f:0b:e2:67:13': -73, '22:3f:1b:e2:64:73': -82, '26:3f:0b:e2:65:75': -79, '12:3f:0b:59:2e:56': -74, '2a:3f:0b:e2:66:3f': -59, 'fa:9e:38:74:f9:08': -87, '22:3f:1b:e2:66:14': -61, '36:3f:0b:e2:65:61': -78, '00:04:ea:45:8a:14': -85, '26:3f:0b:e2:66:3f': -59, '78:db:2f:0f:fb:66': -74, '7a:de:4b:62:c7:ec': -88, '36:3f:0b:59:2e:56': -74, '7a:de:4b:62:c7:eb': -88, '70:5d:cc:ca:e6:f4': -74, '2a:3f:1b:e2:66:b0': -85, '12:3f:1b:e2:66:3f': -70, '22:3f:0b:e2:66:3f': -59, 'e4:55:a8:6e:4b:9c': -85, '2a:3f:1b:e2:64:73': -82, '70:5d:cc:cb:e6:f4': -67, '2a:3f:0b:e2:66:a9': -86, '36:3f:1b:e2:65:87': -88, '2a:3f:1b:e2:66:14': -61, '22:3f:0b:e2:67:13': -73, '22:3f:0b:e2:65:75': -78, '26:3f:1b:e2:66:b0': -85, '36:3f:0b:e2:66:b0': -70, '26:3f:1b:e2:66:14': -61, '12:3f:1b:59:2e:20': -91, '36:3f:0b:e2:66:15': -86, '36:3f:0b:e2:66:14': -55, '22:3f:0b:59:2e:20': -77, '36:3f:1b:e2:66:3f': -70, '2a:3f:0b:59:2e:56': -74, '26:3f:1b:e2:64:73': -82, 'aa:ee:52:de:60:ec': -87, 'ea:55:a8:6e:4b:9c': -85, '70:5d:cc:60:cd:b0': -70, '78:db:2f:0f:fe:9b': -67, '22:3f:0b:e2:65:87': -79, 'd4:6e:0e:a5:71:3c': -81, '58:86:94:db:27:e4': -62, '88:36:6c:69:7a:b2': -78, '1c:28:af:ae:f9:62': -88, '88:36:6c:87:0b:30': -94, '22:3f:1b:59:2e:56': -79, '40:ca:63:4e:41:b5': -80, '58:86:94:c9:8c:fa': -85, 'b8:62:1f:50:6e:5c': -87}

    # 최신 모델과 인코더 불러오기
    model_path, location_encoder, mac_encoder = load_latest_resources()

    # 예측
    predicted_location = predict_location(
        test_input, model_path, location_encoder, mac_encoder)
    print(f"예측된 위치: {predicted_location}")
