# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import KNNImputer
# from filterpy.kalman import KalmanFilter
# from tqdm import tqdm
# import wandb
# import datetime
# import joblib
# import os

# from model_CNNTransformer import WifiCNNTransformer

# # ---------- 칼만 필터 ----------


# def apply_kalman_filter(rssi_values):
#     kf = KalmanFilter(dim_x=1, dim_z=1)
#     kf.x = np.array([[rssi_values.iloc[0]]])
#     kf.F, kf.H = np.array([[1]]), np.array([[1]])
#     kf.P *= 1000
#     kf.R = 5
#     kf.Q = 1e-5
#     return np.array([kf.update(np.array([[r]])) or kf.predict() or kf.x[0, 0] for r in rssi_values])

# # ---------- 전처리 ----------


# def preprocess_data(df, rssi_threshold=-95):
#     df = df[df["RSSI"] > rssi_threshold]

#     location_encoder = LabelEncoder()
#     df["location_encoded"] = location_encoder.fit_transform(df["Location"])

#     mac_encoder = LabelEncoder()
#     df["mac_encoded"] = mac_encoder.fit_transform(df["MAC"])

#     df["rssi_filtered"] = df.groupby(
#         "MAC")["RSSI"].transform(apply_kalman_filter)
#     df["rssi_weighted"] = df.groupby("MAC")["rssi_filtered"].transform(
#         lambda x: np.average(x, weights=np.abs(x)))

#     # 정규화
#     rssi_mean = df["rssi_weighted"].mean()
#     rssi_std = df["rssi_weighted"].std()
#     df["rssi_norm"] = (df["rssi_weighted"] - rssi_mean) / rssi_std

#     # 정규화 정보 저장
#     with open("./finger_printing/models/rssi_stats.pkl", "wb") as f:
#         joblib.dump({"mean": rssi_mean, "std": rssi_std}, f)

#     df["distance"] = 10 ** ((-40 - df["rssi_weighted"]) / (10 * 3))

#     return df, location_encoder, mac_encoder

# # ---------- Dataset ----------


# class WifiDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X[:, :, 1], dtype=torch.float32)
#         self.mac = torch.tensor(X[:, :, 0], dtype=torch.long)
#         self.y = torch.tensor(y, dtype=torch.long)

#     def __getitem__(self, idx):
#         return self.X[idx], self.mac[idx], self.y[idx]

#     def __len__(self):
#         return len(self.X)

# # ---------- Dataset 생성 ----------


# def create_dataset(df, mac_encoder, max_ap=100):
#     grouped = df.groupby(["Time", "location_encoded"])
#     X_list, y_list = [], []

#     for (time, location), group in grouped:
#         mac_indices = mac_encoder.transform(group["MAC"]) + 1
#         values = group["rssi_norm"].values

#         if len(values) < max_ap:
#             pad = max_ap - len(values)
#             mac_indices = np.pad(mac_indices, (0, pad), constant_values=0)
#             values = np.pad(values, (0, pad), constant_values=0)
#         elif len(values) > max_ap:
#             mac_indices = mac_indices[:max_ap]
#             values = values[:max_ap]

#         X_list.append(np.column_stack([mac_indices, values]))
#         y_list.append(location)

#     return np.array(X_list), np.array(y_list)

# # ---------- 모델 학습 ----------


# def train_model(model, train_loader, val_loader, test_loader, num_epochs=100,
#                 early_stop=True, use_wandb=True):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     best_val_loss = float('inf')
#     best_model_state = None
#     patience, patience_counter = 10, 0

#     if use_wandb:
#         import wandb
#         wandb.init(project="wifi-fingerprinting", name="CNNTransformer")
#         wandb.config.update({
#             "epochs": num_epochs,
#             "early_stopping": early_stop,
#             "model": "WifiCNNTransformer"
#         })

#     progress_bar = tqdm(range(num_epochs), desc="Training")

#     for epoch in progress_bar:
#         model.train()
#         total_loss, correct, total = 0, 0, 0

#         for X_batch, mac_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch, mac_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += y_batch.size(0)
#             correct += (predicted == y_batch).sum().item()

#         train_acc = 100 * correct / total
#         train_loss = total_loss / len(train_loader)

#         # Validation
#         model.eval()
#         val_loss, val_correct, val_total = 0, 0, 0
#         with torch.no_grad():
#             for X_val, mac_val, y_val in val_loader:
#                 outputs = model(X_val, mac_val)
#                 loss = criterion(outputs, y_val)
#                 val_loss += loss.item()

#                 _, predicted = torch.max(outputs, 1)
#                 val_total += y_val.size(0)
#                 val_correct += (predicted == y_val).sum().item()

#         val_acc = 100 * val_correct / val_total
#         val_loss /= len(val_loader)

#         # 로깅
#         if use_wandb:
#             wandb.log({
#                 "Train Loss": train_loss,
#                 "Train Acc": train_acc,
#                 "Val Loss": val_loss,
#                 "Val Acc": val_acc,
#                 "epoch": epoch + 1
#             })

#         tqdm.write(
#             f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

#         # Early Stopping
#         if early_stop:
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_state = model.state_dict()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     tqdm.write("💡 Early stopping triggered.")
#                     break

#     if early_stop and best_model_state:
#         model.load_state_dict(best_model_state)

#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = f"./finger_printing/models/finger_printing/fp_model_CNNTransformer_{timestamp}.pt"
#     torch.save(model.state_dict(), model_path)
#     tqdm.write(f"✅ 모델 저장 완료: {model_path}")

#     if use_wandb:
#         wandb.finish()

#     evaluate_model(model, test_loader)

# # ---------- 모델 평가 ----------


# def evaluate_model(model, loader):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for X, M, y in loader:
#             _, pred = torch.max(model(X, M), 1)
#             correct += (pred == y).sum().item()
#             total += y.size(0)
#     print(f"Test Accuracy: {100 * correct / total:.2f}%")


# # ---------- 실행 ----------
# if __name__ == "__main__":
#     file_path = "./finger_printing/datasets/merged/wifi_rssi_log_merged_20250329_224816.csv"
#     df = pd.read_csv(file_path)
#     df, le, me = preprocess_data(df)

#     X, y = create_dataset(df, me)
#     train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2)
#     val_X, test_X, val_y, test_y = train_test_split(
#         temp_X, temp_y, test_size=0.5)

#     train_loader = DataLoader(WifiDataset(train_X, train_y), batch_size=4)
#     val_loader = DataLoader(WifiDataset(val_X, val_y), batch_size=4)
#     test_loader = DataLoader(WifiDataset(test_X, test_y), batch_size=4)

#     model = WifiCNNTransformer(X.shape[1], len(set(y)), len(me.classes_))
#     train_model(model, train_loader, val_loader, test_loader, early_stop=False)

#     # 저장
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     torch.save(model.state_dict(
#     ), f"./finger_printing/models/finger_printing/fp_model_CNNTransformer_{timestamp}.pt")
#     joblib.dump(
#         le, f"./finger_printing/models/location_encoder/location_encoder_{timestamp}.pkl")
#     joblib.dump(
#         me, f"./finger_printing/models/mac_encoder/mac_encoder_{timestamp}.pkl")


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
import wandb
import datetime
import joblib
import os

from model_CNNTransformer import WifiCNNTransformer

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
    df.loc[:, "location_encoded"] = location_encoder.fit_transform(
        df["Location"])

    mac_encoder = LabelEncoder()
    df.loc[:, "mac_encoded"] = mac_encoder.fit_transform(df["MAC"])

    df.loc[:, "rssi_filtered"] = df.groupby(
        "MAC")["RSSI"].transform(apply_kalman_filter)
    df.loc[:, "rssi_weighted"] = df.groupby("MAC")["rssi_filtered"].transform(
        lambda x: np.average(x, weights=np.abs(x))
    )

    # 정규화
    rssi_mean = df["rssi_weighted"].mean()
    rssi_std = df["rssi_weighted"].std()
    df.loc[:, "rssi_norm"] = (df["rssi_weighted"] - rssi_mean) / rssi_std

    # 정규화 값 저장
    norm_path = "./finger_printing/models/rssi_stats.pkl"
    joblib.dump({"mean": rssi_mean, "std": rssi_std}, norm_path)

    df.loc[:, "distance"] = 10 ** ((-40 - df["rssi_weighted"]) / (10 * 3))

    return df, location_encoder, mac_encoder

# ---------- Dataset ----------


class WifiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, :, 1], dtype=torch.float32)
        self.mac = torch.tensor(X[:, :, 0], dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.mac[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

# ---------- Dataset 생성 ----------


def create_dataset(df, mac_encoder, max_ap=100):
    grouped = df.groupby(["Time", "location_encoded"])
    X_list, y_list = [], []

    for (time, location), group in grouped:
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


def train_model(model, train_loader, val_loader, test_loader, num_epochs=100,
                early_stop=True, use_wandb=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_state = None
    patience, patience_counter = 10, 0

    if use_wandb:
        wandb.init(project="wifi-fingerprinting", name="CNNTransformer")
        wandb.config.update({
            "epochs": num_epochs,
            "early_stopping": early_stop,
            "model": "WifiCNNTransformer"
        })

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

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        # with torch.no_grad():
        #     sample_X, sample_mac, _ = next(iter(test_loader))
        #     out = model(sample_X, sample_mac)
        #     print(torch.softmax(out, dim=1))  # 확률 분포 확인
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_val, mac_val, y_val in val_loader:
                outputs = model(X_val, mac_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        if use_wandb:
            wandb.log({
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Val Loss": val_loss,
                "Val Acc": val_acc,
                "epoch": epoch + 1
            })

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
    tqdm.write(f"✅ 모델 저장 완료: {model_path}")

    if use_wandb:
        wandb.finish()

    evaluate_model(model, test_loader)

# ---------- 평가 ----------


def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, M, y in loader:
            _, pred = torch.max(model(X, M), 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# ---------- 실행 ----------
if __name__ == "__main__":
    file_path = "./finger_printing/datasets/merged/wifi_rssi_log_merged_20250329_224816.csv"
    df = pd.read_csv(file_path)
    df, le, me = preprocess_data(df)

    X, y = create_dataset(df, me)
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.2)
    val_X, test_X, val_y, test_y = train_test_split(
        temp_X, temp_y, test_size=0.5)

    train_loader = DataLoader(WifiDataset(train_X, train_y), batch_size=4)
    val_loader = DataLoader(WifiDataset(val_X, val_y), batch_size=4)
    test_loader = DataLoader(WifiDataset(test_X, test_y), batch_size=4)

    model = WifiCNNTransformer(X.shape[1], len(set(y)), len(me.classes_))
    train_model(model, train_loader, val_loader, test_loader, early_stop=False, num_epochs=10000)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(
    ), f"./finger_printing/models/finger_printing/fp_model_CNNTransformer_{timestamp}.pt")
    joblib.dump(
        le, f"./finger_printing/models/location_encoder/location_encoder_{timestamp}.pkl")
    joblib.dump(
        me, f"./finger_printing/models/mac_encoder/mac_encoder_{timestamp}.pkl")
