import torch
import numpy as np
import joblib
import yaml
from filterpy.kalman import KalmanFilter
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

from finger_printing.models.model_CNNTransformer import WifiCNNTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 칼만 필터 함수


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

# 전처리 함수


def preprocess_input(input_mac_rssi, mac_encoder, rssi_mean, rssi_std, max_ap=100, rssi_threshold=-95):
    macs = []
    rssis = []

    for mac, rssi in input_mac_rssi.items():
        if rssi > rssi_threshold:
            macs.append(mac)
            rssis.append(rssi)
    if len(rssis) == 0:
        raise ValueError("No RSSI values above threshold")

    rssis_filtered = apply_kalman_filter(np.array(rssis))

    weights = np.abs(rssis_filtered)
    rssi_weighted = np.average(rssis_filtered, weights=weights)

    rssi_norm = (rssi_weighted - rssi_mean) / rssi_std

    mac_indices = []
    rssi_values = []
    for mac, rssi in zip(macs, [rssi_norm]*len(macs)):
        try:
            idx = mac_encoder.transform([mac])[0] + 1
            mac_indices.append(idx)
            rssi_values.append(rssi)
        except ValueError:
            continue

    if len(mac_indices) < max_ap:
        pad_len = max_ap - len(mac_indices)
        mac_indices = np.pad(mac_indices, (0, pad_len), constant_values=0)
        rssi_values = np.pad(rssi_values, (0, pad_len), constant_values=0)
    else:
        mac_indices = mac_indices[:max_ap]
        rssi_values = rssi_values[:max_ap]

    X = np.column_stack([mac_indices, rssi_values])
    return np.expand_dims(X, axis=0)

# 클래스형으로 감싸서 상태 유지


class Predictor:
    def __init__(self, model_path, encoder_path, norm_path, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        encoders = joblib.load(encoder_path)
        self.location_encoder = encoders["location_encoder"]
        self.mac_encoder = encoders["mac_encoder"]

        norm_params = joblib.load(norm_path)
        self.rssi_mean = norm_params["mean"]
        self.rssi_std = norm_params["std"]

        self.model = WifiCNNTransformer(
            num_ap=self.config["num_ap"],
            num_classes=self.config["num_classes"],
            num_mac=self.config["num_mac"],
            embedding_dim=self.config["embedding_dim"],
            transformer_heads=self.config["transformer_heads"],
            transformer_layers=self.config["transformer_layers"],
            dropout_rate=self.config["dropout_rate"],
        ).to(device)

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, input_mac_rssi):
        X = preprocess_input(input_mac_rssi, self.mac_encoder,
                             self.rssi_mean, self.rssi_std, max_ap=self.config["num_ap"])

        rssi_tensor = torch.tensor(X[:, :, 1], dtype=torch.float32).to(device)
        mac_tensor = torch.tensor(X[:, :, 0], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = self.model(rssi_tensor, mac_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).cpu().item()
            pred_location = self.location_encoder.inverse_transform([pred_idx])[
                0]
        return pred_location, probs.cpu().numpy()


def predict_and_save_csv(input_data_list, predictor, output_csv_path):
    result_list = []

    # for item in input_data_list:
    #     location = item['location']
    #     input_data = item['input_data']

    #     predicted_location = predictor.predict(input_data)

    #     if(location!=predicted_location[0]):
    #         result_list.append({
    #             'Predicted Location': predicted_location[0],  # 예측된 위치
    #             'Actual Location': location  # 실제 위치
    #         })
    #         print(result_list[-1])

    #     result_df = pd.DataFrame(result_list)
    #     result_df.to_csv(output_csv_path, index=False)
    #     # print(f"Results saved to {output_csv_path}")

    total = len(input_data_list)
    wrong_room = 0
    wrong_total = 0

    for i, item in enumerate(input_data_list):
        location = item['location']
        input_data = item['input_data']

        predicted_location = predictor.predict(input_data)[0]

        label = location.split("_")
        pred = predicted_location.split("_")
        
        if label[2] == 'r': # 13011
            label[2] = '-1'
        if label[2] == 'l':
            label[2] = '-4'
        if pred[2] == 'r':
            pred[2] = '-1'
        if pred[2] == 'l':
            pred[2] = '-4'

        # 완전히 다른 장소를 예측함 hall_5143 <-> room_5143 or room_5141 <-> room_5143
        condition1 = label[0] != pred[0] or label[1] != pred[1]
        # 같은 장소지만 좌표의 오차가 양옆 2칸 이상 차이남
        condition2 =  (label[0] == pred[0] and label[1] == pred[1]) and \
            abs(int(label[2]) - int(pred[2])) > 2 or abs(int(label[3]) - int(pred[3])) > 2

        if (condition1):
            wrong_room += 1
            wrong_total += 1
            result_list.append({
                'Predicted Location': predicted_location,  # 예측된 위치
                'Actual Location': location  # 실제 위치
            })
            print(
                f"{i}/{total} | ({wrong_room/i}% | {wrong_total/i}) : {result_list[-1]}")
        elif (condition2):
            wrong_total += 1
            # result_list.append({
            #     'Predicted Location': predicted_location,  # 예측된 위치
            #     'Actual Location': location  # 실제 위치
            # })
            # print(
            #     f"{i}/{total} | ({wrong_room/i}% | {wrong_total/i}) : {result_list[-1]}")

        result_df = pd.DataFrame(result_list)
        result_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    print(f"Accuracy(같은 방) = {1-wrong_room/total}%")
    print(f"Accuracy(오차 범위가 1칸 이내) = {1-wrong_total/total}%")


if __name__ == "__main__":
    # 데이터 증강 전
    # MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250607_155046.pt"
    # ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250607_155046.pkl"
    # NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250607_155046.pkl"
    # CONFIG_PATH = "./finger_printing/config/hyperparameters_20250607_155046.yaml"
    
    # 데이터 증강 후
    # MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250605_194159.pt"
    # ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250605_194159.pkl"
    # NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250605_194159.pkl"
    # CONFIG_PATH = "./finger_printing/config/hyperparameters_20250605_194159.yaml"
    
    # 파인튜닝 데이터 추가 및 데이터 증강 후
    # MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250608_001029.pt"
    # ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250608_001029.pkl"
    # NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250608_001029.pkl"
    # CONFIG_PATH = "./finger_printing/config/hyperparameters_20250608_001029.yaml"
    
    # 파인튜닝 데이터 추가 및 데이터 증강 후 - early_stopping
    MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250608_143837.pt"
    ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250608_143837.pkl"
    NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250608_143837.pkl"
    CONFIG_PATH = "./finger_printing/config/hyperparameters_20250608_143837.yaml"



    predictor = Predictor(MODEL_PATH, ENCODER_PATH, NORM_PATH, CONFIG_PATH)

    # location, _ = predictor.predict(input_data)
    # print(location)

    # CSV 파일에서 Time, Location 기준으로 그룹화한 input_data 리스트 생성
    df = pd.read_csv('./finger_printing/datasets/train_dataset.csv')
    grouped = df.groupby(['Time', 'Location'])

    input_data_list = []
    for (_, location), group in grouped:
        input_data = {}
        for _, row in group.iterrows():
            mac = row['MAC']
            rssi = row['RSSI']
            input_data[mac] = rssi

        input_data_list.append({
            'location': location,
            'input_data': input_data
        })

    output_csv_path = './prediction_results.csv'
    predict_and_save_csv(input_data_list, predictor, output_csv_path)
