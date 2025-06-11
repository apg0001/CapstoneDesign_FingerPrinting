import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import wandb
import datetime
import joblib
import sys
import os
import yaml
from transformers import get_linear_schedule_with_warmup

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from finger_printing.models.model_CNNTransformer import WifiCNNTransformer


# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 칼만 필터 ----------


def apply_kalman_filter(rssi_values):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[rssi_values.iloc[0]]])
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

# ---------- 전처리 ----------


def preprocess_data(df, rssi_threshold=-95):
    df = df[df["RSSI"] > rssi_threshold].copy()

    location_encoder = LabelEncoder()
    df["location_encoded"] = location_encoder.fit_transform(df["Location"])

    mac_encoder = LabelEncoder()
    df["mac_encoded"] = mac_encoder.fit_transform(df["MAC"])

    df["rssi_filtered"] = df.groupby(
        "MAC")["RSSI"].transform(apply_kalman_filter)
    df["rssi_weighted"] = df.groupby("MAC")["rssi_filtered"].transform(
        lambda x: np.average(x, weights=np.abs(x))
    )

    rssi_mean = df["rssi_weighted"].mean()
    rssi_std = df["rssi_weighted"].std()
    df["rssi_norm"] = (df["rssi_weighted"] - rssi_mean) / rssi_std

    df["distance"] = 10 ** ((-40 - df["rssi_weighted"]) / (10 * 3))

    return df, location_encoder, mac_encoder, rssi_mean, rssi_std

# ---------- Dataset ----------


class WifiDataset(Dataset):
    def __init__(self, X, y):
        self.rssi = torch.tensor(X[:, :, 1], dtype=torch.float32)
        self.mac = torch.tensor(X[:, :, 0], dtype=torch.long)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.rssi[idx], self.mac[idx], self.labels[idx]

    def __len__(self):
        return len(self.rssi)

# ---------- Dataset 생성 ----------


def create_dataset(df, mac_encoder, max_ap=100):
    grouped = df.groupby(["Time", "location_encoded"])
    X_list, y_list = [], []

    for (_, location), group in grouped:
        mac_indices = mac_encoder.transform(group["MAC"]) + 1
        values = group["rssi_norm"].values

        if len(values) < max_ap:
            pad = max_ap - len(values)
            mac_indices = np.pad(mac_indices, (0, pad), constant_values=0)
            values = np.pad(values, (0, pad), constant_values=0)
        elif len(values) > max_ap:
            mac_indices = mac_indices[:max_ap]
            values = values[:max_ap]

        X_list.append(np.column_stack([mac_indices, values]))
        y_list.append(location)

    return np.array(X_list), np.array(y_list)

# ---------- 모델 학습 ----------

def train_model():
    # 하이퍼파라미터 입력
    embedding_dim = 16
    transformer_heads = 4
    transformer_layers = 3
    dropout_rate = 0.1
    batch_size = 32
    learning_rate = 1e-4
    epochs = 50
    scheduler = True
    early_stopping = True
    data_path = "./finger_printing/datasets/train_dataset.csv"

    # 모델 관련 파라미터 계산
    df = pd.read_csv(data_path)
    df, location_encoder, mac_encoder, rssi_mean, rssi_std = preprocess_data(df)
    X, y = create_dataset(df, mac_encoder)

    num_ap = X.shape[1]  # AP 수
    num_classes = len(set(y))  # 클래스 수 (위치 수)
    num_mac = len(mac_encoder.classes_)  # MAC 주소 개수

    # config 파라미터를 YAML 파일로 저장
    config_dict = {
        "embedding_dim": embedding_dim,
        "transformer_heads": transformer_heads,
        "transformer_layers": transformer_layers,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "scheduler": scheduler,
        "early_stopping": early_stopping,
        "data_path": data_path,
        "num_ap": num_ap,  # AP 수 추가
        "num_classes": num_classes,  # 클래스 수 추가
        "num_mac": num_mac  # MAC 주소 개수 추가
    }

    with open(f"./finger_printing/config/hyperparameters_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml", 'w') as f:
        yaml.dump(config_dict, f)

    # 모델 학습 데이터셋 준비
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5)

    train_loader = DataLoader(WifiDataset(train_X, train_y), batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(WifiDataset(val_X, val_y), batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(WifiDataset(test_X, test_y), batch_size=batch_size, drop_last=True)

    # 모델 정의
    model = WifiCNNTransformer(
        num_ap=num_ap,
        num_classes=num_classes,
        num_mac=num_mac,
        embedding_dim=embedding_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        dropout_rate=dropout_rate,
    ).to(device)

    # 학습 관련 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    if scheduler:
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

    best_val_loss = float('inf')
    best_model_state = None
    patience, patience_counter = 15, 0

    epoch_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        total_loss, correct, total = 0, 0, 0
        for rssi_batch, mac_batch, labels_batch in train_loader:
            rssi_batch, mac_batch, labels_batch = rssi_batch.to(device), mac_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(rssi_batch, mac_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # Validation loss 계산
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for rssi_val, mac_val, labels_val in val_loader:
                rssi_val, mac_val, labels_val = rssi_val.to(device), mac_val.to(device), labels_val.to(device)
                outputs = model(rssi_val, mac_val)
                loss = criterion(outputs, labels_val)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels_val.size(0)
                val_correct += (predicted == labels_val).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        wandb.log({"Train Loss": train_loss, "Train Acc": train_acc,
                   "Val Loss": val_loss, "Val Acc": val_acc, "epoch": epoch + 1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

        # 모델 저장
        model_dir = "./finger_printing/checkpoints/checkpoints"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/fp_model_CNNTransformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ 모델 저장 완료: {model_path}")

        # 인코더 저장
        encoder_dir = "./finger_printing/checkpoints/encoders"
        os.makedirs(encoder_dir, exist_ok=True)
        encoder_path = f"{encoder_dir}/encoders_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump({
            "location_encoder": location_encoder,
            "mac_encoder": mac_encoder
        }, encoder_path)
        print(f"✅ 인코더 저장 완료: {encoder_path}")

        # 정규화 파라미터 저장
        norm_dir = "./finger_printing/checkpoints/norm"
        os.makedirs(norm_dir, exist_ok=True)
        norm_path = f"{norm_dir}/norm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump({"mean": rssi_mean, "std": rssi_std}, norm_path)
        print(f"✅ 정규화 파라미터 저장 완료: {norm_path}")

    evaluate_model(model, test_loader, location_encoder)


# ---------- 평가 ----------


def evaluate_model(model, loader, location_encoder):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for rssi, mac, labels in loader:
            rssi, mac, labels = rssi.to(device), mac.to(device), labels.to(device)
            _, pred = torch.max(model(rssi, mac), 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    wandb.log({"Test Accuracy": acc})


# ---------- Main 실행 ----------
if __name__ == "__main__":
    train_model()