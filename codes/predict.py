import torch
import numpy as np
import glob
import os
from model import WifiCNN  # 모델 정의
import joblib

# 최신 파일 가져오기 (모델 & 인코더 공통 사용)


def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"{pattern} 에 맞는 파일이 없습니다.")
    return max(files, key=os.path.getctime)

# 최신 모델 및 인코더 불러오기


def load_latest_resources(model_dir="./models"):
    model_path = get_latest_file(os.path.join(
        model_dir + "/finger_printing", "fp_model_*.pt"))
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

    model = WifiCNN(max_ap, num_classes, num_mac)
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
    test_input = {'26:3f:0b:e2:66:14': -54,
                  '2c:3f:0b:e2:66:3f': -59, 'da:55:a8:6e:4b:9c': -86}

    # 최신 모델과 인코더 불러오기
    model_path, location_encoder, mac_encoder = load_latest_resources()

    # 예측
    predicted_location = predict_location(
        test_input, model_path, location_encoder, mac_encoder)
    print(f"예측된 위치: {predicted_location}")
