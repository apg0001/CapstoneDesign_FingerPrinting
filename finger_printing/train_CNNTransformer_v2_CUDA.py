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
import os
from transformers import get_linear_schedule_with_warmup

from model_CNNTransformer import WifiCNNTransformer

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

    norm_path = "./finger_printing/models/rssi_stats.pkl"
    joblib.dump({"mean": rssi_mean, "std": rssi_std}, norm_path)

    df["distance"] = 10 ** ((-40 - df["rssi_weighted"]) / (10 * 3))

    return df, location_encoder, mac_encoder

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


def train_model(model, train_loader, val_loader, test_loader, location_encoder, num_epochs=100, early_stop=True, use_wandb=True, use_scheduler=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if use_scheduler:
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    best_val_loss = float('inf')
    best_model_state = None
    patience, patience_counter = 15, 0

    if use_wandb:
        wandb.init(project="wifi-fingerprinting", name="CNNTransformer")
        wandb.config.update({
            "epochs": num_epochs,
            "early_stopping": early_stop,
            "model": "WifiCNNTransformer",
            "scheduler": use_scheduler
        })

    progress_bar = tqdm(range(num_epochs), desc="Training")

    for epoch in progress_bar:
        model.train()
        total_loss, correct, total = 0, 0, 0

        for rssi_batch, mac_batch, labels_batch in train_loader:
            rssi_batch = rssi_batch.to(device)
            mac_batch = mac_batch.to(device)
            labels_batch = labels_batch.to(device)

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

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for rssi_val, mac_val, labels_val in val_loader:
                rssi_val = rssi_val.to(device)
                mac_val = mac_val.to(device)
                labels_val = labels_val.to(device)

                outputs = model(rssi_val, mac_val)
                loss = criterion(outputs, labels_val)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels_val.size(0)
                val_correct += (predicted == labels_val).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        if use_wandb:
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc,
                       "Val Loss": val_loss, "Val Acc": val_acc, "epoch": epoch + 1})

        tqdm.write(
            f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write("💡 Early stopping triggered.")
                    break

    if early_stop and best_model_state:
        model.load_state_dict(best_model_state)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./finger_printing/models/finger_printing/fp_model_CNNTransformer_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    tqdm.write(f"모델 저장 완료: {model_path}")

    if use_wandb:
        wandb.finish()

    evaluate_model(model, test_loader, location_encoder)

# ---------- 평가 ----------


def evaluate_model(model, loader, location_encoder):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for rssi, mac, labels in loader:
            rssi, mac, labels = rssi.to(device), mac.to(
                device), labels.to(device)
            _, pred = torch.max(model(rssi, mac), 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            for p, y in zip(pred, labels):
                if p != y:
                    pred_label = location_encoder.inverse_transform([p.item()])[
                        0]
                    true_label = location_encoder.inverse_transform([y.item()])[
                        0]
                    print(f"predicted: {pred_label}, y: {true_label}")

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# ---------- 실행 ----------
if __name__ == "__main__":
    file_path = "./finger_printing/datasets/merged/wifi_rssi_log_merged_20250329_224816.csv"
    file_path = "./finger_printing/datasets/augmented/wifi_rssi_log_augmented_20250331_000650.csv"
    df = pd.read_csv(file_path)
    df, location_encoder, mac_encoder = preprocess_data(df)

    X, y = create_dataset(df, mac_encoder)
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2)
    val_X, test_X, val_y, test_y = train_test_split(
        temp_X, temp_y, test_size=0.5)

    default_batch_size = 4

    train_loader = DataLoader(WifiDataset(
        train_X, train_y), batch_size=default_batch_size, drop_last=True)
    val_loader = DataLoader(WifiDataset(val_X, val_y),
                            batch_size=default_batch_size, drop_last=True)
    test_loader = DataLoader(WifiDataset(
        test_X, test_y), batch_size=default_batch_size, drop_last=True)

    model = WifiCNNTransformer(X.shape[1], len(
        set(y)), len(mac_encoder.classes_)).to(device)
    train_model(model, train_loader, val_loader, test_loader,
                location_encoder=location_encoder, num_epochs=200, early_stop=True, use_scheduler=True)
