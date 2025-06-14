import torch
import numpy as np
import joblib
import yaml
from filterpy.kalman import KalmanFilter
from .model_CNNTransformer import WifiCNNTransformer

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

    for mac, rssi in input_mac_rssi.get("location1").items():
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
class Predictor_temp:
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

    def predict_temp(self, input_mac_rssi):
        X = preprocess_input(input_mac_rssi, self.mac_encoder, self.rssi_mean, self.rssi_std, max_ap=self.config["num_ap"])

        rssi_tensor = torch.tensor(X[:, :, 1], dtype=torch.float32).to(device)
        mac_tensor = torch.tensor(X[:, :, 0], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = self.model(rssi_tensor, mac_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).cpu().item()
            pred_location = self.location_encoder.inverse_transform([pred_idx])[0]
        return pred_location, probs.cpu().numpy()