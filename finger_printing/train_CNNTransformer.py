import joblib
import pandas as pd
import chardet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
import datetime
import glob
import os
from model_CNNTransformer import WifiCNNTransformer

# 데이터 로드 및 인코딩 감지


def load_data(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())

    encoding_type = result['encoding']
    print(f"감지된 인코딩: {encoding_type}")

    df = pd.read_csv(file_path, encoding=encoding_type)
    return df

# 칼만 필터 적용 (RSSI 노이즈 제거)


def apply_kalman_filter(rssi_values):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[rssi_values.iloc[0]]])
    kf.F = np.array([[1]])
    kf.H = np.array([[1]])
    kf.P *= 1000
    kf.R = 5
    kf.Q = 1e-5

    filtered_rssi = []
    for rssi in rssi_values:
        kf.predict()
        kf.update(np.array([[rssi]]))
        filtered_rssi.append(kf.x[0, 0])

    return np.array(filtered_rssi)

# 데이터 전처리 함수


def preprocess_data(df):
    location_encoder = LabelEncoder()
    df["location_encoded"] = location_encoder.fit_transform(df["Location"])
    # df["location_encoded"] = location_encoder.fit_transform(df["Location"]) + 1

    mac_encoder = LabelEncoder()
    df["mac_encoded"] = mac_encoder.fit_transform(df["MAC"])

    df["rssi_filtered"] = df.groupby("MAC")["RSSI"].transform(
        lambda x: apply_kalman_filter(x))
    df["rssi_weighted"] = df.groupby("MAC")["rssi_filtered"].transform(
        lambda x: np.average(x, weights=np.abs(x)))

    P_0 = -40
    n = 3
    df["distance"] = 10 ** ((P_0 - df["rssi_weighted"]) / (10 * n))

    imputer = KNNImputer(n_neighbors=3)
    df["rssi_imputed"] = imputer.fit_transform(df[["rssi_weighted"]])

    return df, location_encoder, mac_encoder

# 데이터셋 변환


def create_dataset(df, mac_encoder):
    grouped = df.groupby(["Time", "location_encoded"])
    X_list, y_list = [], []

    max_ap = 100  # 최대 AP 수
    for (time, location), group in grouped:
        mac_indices = mac_encoder.transform(group["MAC"]) + 1  # +1 padding 대비
        distances = group["distance"].values

        if len(distances) < max_ap:
            pad_size = max_ap - len(distances)
            mac_indices = np.pad(mac_indices, (0, pad_size),
                                 constant_values=0)  # padding = 0
            distances = np.pad(distances, (0, pad_size), constant_values=0)
        elif len(distances) > max_ap:
            mac_indices = mac_indices[:max_ap]
            distances = distances[:max_ap]

        feature_vector = np.column_stack([mac_indices, distances])
        X_list.append(feature_vector)
        y_list.append(location)

        X = np.array(X_list)
        y = np.array(y_list)
    return X, y

# PyTorch Dataset 정의


class WifiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, :, 1], dtype=torch.float32)
        self.mac = torch.tensor(X[:, :, 0], dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mac[idx], self.y[idx]


# 모델 학습 함수


def train_model(model, train_loader, test_loader, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    progress_bar = tqdm(range(num_epochs), desc="Training")

    for epoch in progress_bar:
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, mac_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch, mac_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_accuracy = 100 * correct / total
        progress_bar.set_postfix(
            Loss=total_loss / len(train_loader), Accuracy=f"{train_accuracy:.2f}%")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp_model_path = f"./finger_printing/models/finger_printing/fp_model_CNNTransformer_{timestamp}.pt"
    le_model_path = f"./finger_printing/models/location_encoder/location_encoder_{timestamp}.pkl"
    me_model_path = f"./finger_printing/models/mac_encoder/mac_encoder_{timestamp}.pkl"
    torch.save(model.state_dict(), fp_model_path)
    # 학습 코드에서
    joblib.dump(location_encoder, le_model_path)
    joblib.dump(mac_encoder, me_model_path)
    print(f"모델 가중치 저장됨: {fp_model_path}")

# 모델 평가 함수


def evaluate_model(model, test_loader, location_encoder):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, mac_batch, y_batch in test_loader:
            outputs = model(X_batch, mac_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            for p, y in zip(predicted, y_batch):
                if (p != y):
                    pred_label = location_encoder.inverse_transform([p.item()])[0]
                    true_label = location_encoder.inverse_transform([y.item()])[0]
                    print(f"predicted: {pred_label}, y: {true_label}")

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# 메인 실행 코드
if __name__ == "__main__":
    # file_path = "./finger_printing/datasets/filtered/wifi_rssi_log_filtered_20250315_233054.csv"
    file_path = "./finger_printing/datasets/merged/wifi_rssi_log_merged_20250329_224816.csv"
    df = load_data(file_path)
    df, location_encoder, mac_encoder = preprocess_data(df)
    X, y = create_dataset(df, mac_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    train_dataset = WifiDataset(X_train, y_train)
    test_dataset = WifiDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    num_ap = X.shape[1]
    num_mac = len(mac_encoder.classes_)
    num_classes = len(set(y))

    model = WifiCNNTransformer(num_ap, num_classes, num_mac)
    train_model(model, train_loader, test_loader, num_epochs=100)
    evaluate_model(model, test_loader, location_encoder)
