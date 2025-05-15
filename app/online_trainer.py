import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from filterpy.kalman import KalmanFilter
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IncrementalDataset(Dataset):
    def __init__(self, data_list, mac_encoder, rssi_mean, rssi_std, max_ap=100):
        self.data_list = data_list
        self.mac_encoder = mac_encoder
        self.rssi_mean = rssi_mean
        self.rssi_std = rssi_std
        self.max_ap = max_ap
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_mac_rssi, label = self.data_list[idx]  # label이 있다면 포함

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

        macs = []
        rssis = []
        for mac, rssi in input_mac_rssi.items():
            if rssi > -95:
                macs.append(mac)
                rssis.append(rssi)
        rssis_filtered = apply_kalman_filter(np.array(rssis))
        weights = np.abs(rssis_filtered)
        rssi_weighted = np.average(rssis_filtered, weights=weights)
        rssi_norm = (rssi_weighted - self.rssi_mean) / self.rssi_std

        mac_indices = []
        rssi_values = []
        for mac, rssi in zip(macs, [rssi_norm]*len(macs)):
            try:
                idx = self.mac_encoder.transform([mac])[0] + 1
                mac_indices.append(idx)
                rssi_values.append(rssi)
            except ValueError:
                continue

        if len(mac_indices) < self.max_ap:
            pad_len = self.max_ap - len(mac_indices)
            mac_indices = np.pad(mac_indices, (0, pad_len), constant_values=0)
            rssi_values = np.pad(rssi_values, (0, pad_len), constant_values=0)
        else:
            mac_indices = mac_indices[:self.max_ap]
            rssi_values = rssi_values[:self.max_ap]

        X = np.column_stack([mac_indices, rssi_values])

        rssi_tensor = torch.tensor(X[:, 1], dtype=torch.float32)
        mac_tensor = torch.tensor(X[:, 0], dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return rssi_tensor, mac_tensor, label_tensor

class OnlineTrainer:
    def __init__(self, model, mac_encoder, location_encoder, rssi_mean, rssi_std,
                 max_ap=100, lr=1e-4,
                 model_save_path="./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_latest.pt",
                 encoder_save_path="./finger_printing/checkpoints/encoders/encoders_latest.pkl",
                 norm_save_path="./finger_printing/checkpoints/norm/norm_latest.pkl"):
        self.model = model
        self.mac_encoder = mac_encoder
        self.location_encoder = location_encoder
        self.rssi_mean = rssi_mean
        self.rssi_std = rssi_std
        self.max_ap = max_ap
        self.lr = lr

        self.model_save_path = model_save_path
        self.encoder_save_path = encoder_save_path
        self.norm_save_path = norm_save_path

        self.new_data_buffer = []  # (input_mac_rssi, label) 쌍 저장

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def add_new_sample(self, input_mac_rssi, label):
        self.new_data_buffer.append((input_mac_rssi, label))

    def train_incrementally(self, epochs=1, batch_size=8):
        if len(self.new_data_buffer) == 0:
            print("No new data to train on.")
            return

        dataset = IncrementalDataset(self.new_data_buffer, self.mac_encoder, self.rssi_mean, self.rssi_std, self.max_ap)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for rssi_batch, mac_batch, labels_batch in dataloader:
                rssi_batch = rssi_batch.to(device)
                mac_batch = mac_batch.to(device)
                labels_batch = labels_batch.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(rssi_batch, mac_batch)
                loss = self.criterion(outputs, labels_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        # 학습 후 버퍼 비우기
        self.new_data_buffer.clear()

    def save_model_and_encoders(self):
        # 모델 저장 (덮어쓰기)
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

        # 인코더 저장
        joblib.dump({
            "location_encoder": self.location_encoder,
            "mac_encoder": self.mac_encoder
        }, self.encoder_save_path)
        print(f"Encoders saved to {self.encoder_save_path}")

        # 정규화 파라미터 저장
        joblib.dump({
            "mean": self.rssi_mean,
            "std": self.rssi_std
        }, self.norm_save_path)
        print(f"Normalization params saved to {self.norm_save_path}")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()