import torch
import numpy as np
import joblib
import yaml
from filterpy.kalman import KalmanFilter
import sys
import os

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


if __name__ == "__main__":
    MODEL_PATH = "./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_20250520_172109.pt"
    ENCODER_PATH = "./finger_printing/checkpoints/encoders/encoders_20250520_172109.pkl"
    NORM_PATH = "./finger_printing/checkpoints/norm/norm_20250520_172109.pkl"
    CONFIG_PATH = "./finger_printing/config/hyperparameters_20250520_172109.yaml"

    predictor = Predictor(MODEL_PATH, ENCODER_PATH, NORM_PATH, CONFIG_PATH)
    # input_data =  { # hall_5143_1_3
    #     "72:5d:cc:4d:b2:a8": -62,
    #     "2a:3f:1b:59:2e:74": -82,
    #     "22:3f:1b:e2:63:60": -68,
    #     "2e:3f:1b:f3:73:d7": -78,
    #     "5a:86:94:46:3d:88": -88,
    #     "2a:3f:0b:59:2e:74": -71,
    #     "00:08:9f:7c:c9:3c": -68,
    #     "34:fc:b9:c8:c5:12": -96,
    #     "22:3f:1b:59:2e:74": -82,
    #     "58:86:94:16:4a:a8": -57,
    #     "58:86:94:ae:06:df": -72,
    #     "58:86:94:ae:06:de": -59,
    #     "70:5d:cc:d2:35:6c": -95,
    #     "78:db:2f:10:04:65": -60,
    #     "68:27:19:3d:b8:2d": -46,
    #     "f6:9e:28:74:f8:92": -66,
    #     "fe:55:b8:6e:63:8f": -81,
    #     "2a:3f:1b:e2:63:60": -68,
    #     "88:36:6c:0c:b6:4c": -85,
    #     "26:3f:1b:e2:63:60": -68,
    #     "34:fc:b9:c9:05:d1": -92,
    #     "34:fc:b9:c9:05:d0": -92,
    #     "34:fc:b9:c8:20:93": -94,
    #     "34:fc:b9:c9:05:d2": -92,
    #     "da:55:b8:6e:63:8f": -80,
    #     "ea:55:a8:6e:54:1e": -69,
    #     "2e:3f:1b:e2:63:60": -68,
    #     "58:86:94:16:3d:88": -59,
    #     "e2:55:b8:6e:63:8f": -81,
    #     "70:5d:cc:1d:b2:a8": -68,
    #     "36:3f:1b:59:2e:74": -82,
    #     "ea:55:b8:6e:63:8f": -81,
    #     "e6:55:b8:6e:63:8f": -81,
    #     "7a:de:4b:62:c7:ec": -91,
    #     "7a:de:4b:62:c7:eb": -90,
    #     "22:3f:1b:f3:73:d7": -78,
    #     "58:86:94:7e:ea:78": -52,
    #     "88:36:6c:42:04:36": -86,
    #     "5a:86:94:4e:ea:78": -87,
    #     "36:3f:1b:e2:63:60": -68,
    #     "12:3f:1b:e2:67:10": -91,
    #     "58:86:94:ea:f7:4c": -91,
    #     "f6:9e:38:74:f8:92": -68,
    #     "34:fc:b9:c8:e6:a3": -81,
    #     "ee:55:b8:6e:63:8f": -81,
    #     "22:3f:0b:f3:73:d7": -68,
    #     "fe:55:a8:6e:54:1e": -69,
    #     "12:3f:1b:e2:63:60": -68,
    #     "22:3f:0b:e2:63:60": -60,
    #     "fa:9e:38:74:f8:92": -68,
    #     "26:3f:1b:59:2e:74": -82
    # }
    
    input_data = { # hall_5143_1_1
        "2a:3f:1b:59:2e:74": -80,
        "26:3f:0b:e2:63:60": -45,
        "22:3f:0b:59:2e:74": -70,
        "50:46:ae:e4:ef:b0": -76,
        "2c:3f:0b:e2:63:60": -45,
        "2c:3f:0b:f3:73:d7": -68,
        "70:5d:cc:1d:b2:a8": -75,
        "12:3f:0b:e2:63:60": -49,
        "70:5d:cc:a3:b4:aa": -86,
        "f8:9e:28:74:f8:92": -55,
        "36:3f:1b:59:2e:74": -81,
        "26:3f:0b:e2:65:73": -74,
        "22:3f:1b:e2:63:60": -59,
        "2e:3f:1b:f3:73:d7": -71,
        "22:3f:1b:e2:63:fb": -67,
        "2a:3f:0b:e2:63:60": -45,
        "22:3f:1b:f3:73:d7": -71,
        "12:3f:0b:e2:65:73": -76,
        "22:3f:1b:59:2e:74": -81,
        "58:86:94:ae:06:df": -80,
        "2a:3f:0b:e2:63:fb": -64,
        "b4:5d:50:6a:1f:10": -87,
        "58:86:94:7e:ea:78": -68,
        "b4:5d:50:6a:1f:11": -86,
        "b4:5d:50:6a:1f:12": -87,
        "b4:5d:50:6a:1e:70": -85,
        "b4:5d:50:6a:1e:71": -86,
        "da:55:a8:6e:63:02": -79,
        "b4:5d:50:6a:1e:72": -85,
        "b4:5d:50:6a:17:72": -86,
        "78:db:2f:10:04:65": -48,
        "36:3f:0b:e2:65:73": -74,
        "b4:5d:50:6a:17:71": -86,
        "68:27:19:3d:b8:2d": -49,
        "b4:5d:50:6a:17:70": -86,
        "2a:3f:1b:e2:63:fb": -67,
        "36:3f:1b:e2:63:60": -59,
        "2a:3f:0b:e2:65:73": -72,
        "36:3f:1b:e2:63:fb": -67,
        "22:3f:0b:e2:65:73": -73,
        "f6:9e:28:74:f8:92": -53,
        "32:cd:a7:32:0e:12": -82,
        "2a:3f:1b:e2:63:60": -59,
        "da:55:a8:6e:51:f5": -81,
        "78:db:2f:0f:f4:4e": -58,
        "f6:9e:38:74:f8:92": -61,
        "26:3f:1b:e2:63:60": -59,
        "da:55:b8:6e:53:4a": -86,
        "26:3f:1b:e2:63:fb": -67,
        "12:3f:1b:59:2e:60": -86,
        "22:3f:0b:f3:73:d7": -70,
        "36:3f:0b:e2:63:60": -46,
        "88:36:6c:d1:7d:90": -85,
        "12:3f:1b:f3:73:d7": -71,
        "28:ee:52:de:60:eb": -80,
        "22:3f:0b:e2:63:60": -46,
        "fa:9e:38:74:f8:92": -61,
        "da:55:b8:6e:54:1e": -86,
        "2c:3f:0b:e2:65:73": -73,
        "26:3f:1b:59:2e:74": -81,
        "78:db:2f:0f:ee:39": -57,
        "88:36:6c:d1:7d:92": -70,
        "78:db:2f:0f:ec:59": -66
    }


    location, _ = predictor.predict(input_data)
    print(location)
