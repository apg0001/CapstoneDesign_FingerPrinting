import torch
import numpy as np
import glob
import os
import joblib
from .model import WifiCNN  # 같은 폴더 내 model.py 사용

# 최신 파일 가져오기
def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"{pattern} 에 맞는 파일이 없습니다.")
    return max(files, key=os.path.getctime)

# 모델과 인코더 불러오기
def load_latest_resources(model_dir="./app/models"):
    model_path = get_latest_file(os.path.join(
        model_dir, "fp_model_*.pt"))
    location_encoder_path = get_latest_file(os.path.join(
        model_dir, "location_encoder_*.pkl"))
    mac_encoder_path = get_latest_file(os.path.join(
        model_dir, "mac_encoder_*.pkl"))

    location_encoder = joblib.load(location_encoder_path)
    mac_encoder = joblib.load(mac_encoder_path)

    # 모델 파라미터
    num_classes = len(location_encoder.classes_)
    num_mac = len(mac_encoder.classes_)
    max_ap = 70

    # 모델 불러오기
    model = WifiCNN(max_ap, num_classes, num_mac)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, location_encoder, mac_encoder

# 예측 함수
def predict_location(mac_rssi_dict, model, location_encoder, mac_encoder):
    P_0 = -40
    n = 3
    max_ap = 70  # 모델 학습 시 사용한 값

    input_vector = np.zeros((max_ap, 2))

    for i, (mac, rssi) in enumerate(mac_rssi_dict.items()):
        if i >= max_ap:
            break
        if mac in mac_encoder.classes_:
            mac_idx = mac_encoder.transform([mac])[0]
            distance = 10 ** ((P_0 - rssi) / (10 * n))
            input_vector[i] = [mac_idx, distance]

    input_tensor = torch.tensor(input_vector[:, 1], dtype=torch.float32).unsqueeze(0)
    mac_tensor = torch.tensor(input_vector[:, 0], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor, mac_tensor)
        _, predicted_idx = torch.max(output, 1)

    predicted_location = location_encoder.inverse_transform([predicted_idx.item()])[0]
    return predicted_location