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
    # max_ap = 70

    # for (time, location), group in grouped:
    #     mac_indices = mac_encoder.transform(group["MAC"])
    #     distances = group["distance"].values

    #     if len(distances) < max_ap:
    #         pad_size = max_ap - len(distances)
    #         mac_indices = np.pad(
    #             mac_indices, (0, pad_size), constant_values=-1)
    #         distances = np.pad(distances, (0, pad_size), constant_values=0)

    #     elif len(distances) > max_ap:
    #         mac_indices = mac_indices[:max_ap]
    #         distances = distances[:max_ap]

    #     feature_vector = np.column_stack([mac_indices, distances])
    #     X_list.append(feature_vector)
    #     y_list.append(location)

    max_ap = 70  # 최대 AP 수
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

# CNN + Embedding 모델 정의


# class WifiCNN(nn.Module):
#     def __init__(self, num_ap, num_classes, num_mac):
#         super(WifiCNN, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings=num_mac, embedding_dim=8, padding_idx=-1)
#         self.conv1 = nn.Conv1d(
#             in_channels=9, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * num_ap, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x, mac):
#         mac_embed = self.embedding(mac)
#         x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)
#         x = x.permute(0, 2, 1)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class WifiCNN(nn.Module):
    def __init__(self, num_ap, num_classes, num_mac):
        super(WifiCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_mac + 1,  # padding 포함
            embedding_dim=8,
            padding_idx=0  # padding index
        )
        self.conv1 = nn.Conv1d(
            in_channels=9, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * num_ap, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, mac):
        mac_embed = self.embedding(mac)
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
    model_path = f"./models/fp_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"모델 가중치 저장됨: {model_path}")

# 모델 평가 함수


def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, mac_batch, y_batch in test_loader:
            outputs = model(X_batch, mac_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# 모델 불러와서 예측


def predict_location(mac_rssi_dict, model_path):
    P_0 = -40
    n = 3

    model = WifiCNN(num_ap, num_classes, num_mac)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_vector = np.zeros((num_ap, 2))

    for i, (mac, rssi) in enumerate(mac_rssi_dict.items()):
        if i >= num_ap:
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

# 최신 모델 불러오기


def get_latest_model():
    model_files = glob.glob("./models/fp_model_*.pt")
    if not model_files:
        raise FileNotFoundError("저장된 모델이 없습니다.")
    return max(model_files, key=os.path.getctime)


# 메인 실행 코드
if __name__ == "__main__":
    file_path = "./datasets/filtered/wifi_rssi_log_filtered_20250315_233054.csv"
    df = load_data(file_path)
    df, location_encoder, mac_encoder = preprocess_data(df)
    X, y = create_dataset(df, mac_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    train_dataset = WifiDataset(X_train, y_train)
    test_dataset = WifiDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    num_ap = X.shape[1]
    num_mac = len(mac_encoder.classes_)
    num_classes = len(set(y))

    model = WifiCNN(num_ap, num_classes, num_mac)
    train_model(model, train_loader, test_loader, num_epochs=100)
    evaluate_model(model, test_loader)

    # 예측 실행
    test_input = {'26:3f:0b:e2:66:14': -40, '2c:3f:0b:e2:66:3f': -74, '2e:3f:1b:e2:64:73': -86, '26:3f:0b:e2:64:73': -75, '58:86:94:de:1b:78': -73, '2e:3f:1b:e2:66:14': -59, '22:3f:1b:e2:67:13': -89, '78:db:2f:0f:fb:06': -87, '12:3f:0b:59:2e:20': -87, '70:5d:cc:a3:86:58': -87, '78:db:2f:10:01:dd': -86, '78:db:2f:0f:f2:fd': -91, '2e:3f:1b:e2:66:b0': -77, '78:db:2f:0f:fe:a3': -85, '22:3f:1b:e2:66:3f': -77, '2a:3f:0b:e2:66:b0': -65, '26:3f:1b:59:2e:56': -87, '2a:3f:1b:e2:67:13': -89, 'ea:55:b8:6e:4b:9c': -87, '26:3f:0b:e2:66:b0': -65, '2a:3f:0b:e2:66:14': -40, '12:3f:0b:e2:67:13': -78, '58:86:94:16:4a:a8': -84, '58:86:94:de:10:e4': -59, 'e6:55:b8:6e:4b:9c': -87, '26:3f:0b:e2:65:61': -77, '82:de:4b:62:c7:ed': -76, '2a:3f:1b:59:2e:56': -87, '12:3f:1b:59:2e:56': -88, '82:de:4b:62:c7:ec': -76, '2a:3f:1b:e2:66:3f': -77, '36:3f:0b:e2:67:13': -67, '00:26:66:fe:8d:74': -77, '22:3f:0b:e2:66:b0': -66, '12:3f:1b:e2:66:b0': -77, '26:3f:1b:e2:66:3f': -77, '78:db:2f:10:04:59': -81, '36:3f:1b:e2:64:73': -86, '36:3f:0b:e2:66:3f': -75, '78:db:2f:0f:ff:d2': -92, '58:86:94:de:6c:84': -70, '22:3f:1b:59:2e:20': -84, '36:3f:1b:e2:66:14': -59, '2a:3f:0b:e2:67:13': -66, '36:3f:1b:e2:66:b0': -77, 'ee:55:b8:6e:4b:9c': -87, '78:db:2f:0f:fa:9a': -78, '36:3f:1b:59:2e:56': -88, '2c:3f:0b:e2:67:13': -70, '12:3f:1b:e2:66:14': -58, '22:3f:0b:e2:66:14': -40, '28:ee:52:de:60:ea': -
                  85, '22:3f:0b:e2:64:73': -75, '2e:3f:1b:59:2e:56': -88, 'b2:ee:52:de:60:ed': -71, '2c:3f:0b:e2:66:b0': -68, 'f6:9e:38:74:f9:08': -88, '22:3f:1b:e2:66:b0': -77, 'c6:9e:38:74:f9:08': -88, '2e:3f:1b:e2:66:3f': -77, '2e:3f:1b:59:2e:20': -89, '26:3f:1b:59:2e:20': -83, '26:3f:0b:e2:67:13': -66, '12:3f:0b:e2:66:b0': -72, '2a:3f:1b:59:2e:20': -83, '22:3f:1b:e2:64:73': -86, '2a:3f:0b:e2:66:3f': -75, '22:3f:1b:e2:66:14': -59, '36:3f:0b:e2:65:61': -78, 'b4:fc:75:16:12:4f': -86, '78:db:2f:0f:fb:66': -74, '26:3f:0b:e2:66:3f': -75, '7a:de:4b:62:c7:ec': -89, '7a:de:4b:62:c7:eb': -88, '70:5d:cc:ca:e6:f4': -73, '12:3f:1b:e2:65:61': -92, '58:86:94:7e:ea:78': -91, '2a:3f:1b:e2:66:b0': -77, 'fe:55:b8:6e:4b:9c': -87, '12:3f:1b:e2:66:3f': -77, '22:3f:0b:e2:66:3f': -74, '58:86:94:55:20:d0': -88, '36:3f:1b:59:2e:20': -83, '2a:3f:1b:e2:64:73': -87, '70:5d:cc:cb:e6:f4': -67, '2a:3f:1b:e2:66:14': -59, '22:3f:0b:e2:67:13': -66, '26:3f:1b:e2:66:b0': -77, '36:3f:0b:e2:66:b0': -66, '26:3f:1b:e2:66:14': -59, '12:3f:1b:59:2e:20': -88, 'da:55:b8:6e:4b:9c': -86, '36:3f:0b:e2:66:14': -40, '36:3f:1b:e2:66:3f': -77, '26:3f:1b:e2:64:73': -87, '36:3f:0b:e2:64:73': -77, 'aa:ee:52:de:60:ec': -85, '70:5d:cc:60:cd:b0': -71, '78:db:2f:0f:fe:9b': -87, '36:3f:1b:e2:67:13': -90, '58:86:94:db:27:e4': -60, 'e2:55:b8:6e:4b:9c': -86, '22:3f:1b:59:2e:56': -88, '70:5d:cc:60:cd:b2': -89}
    latest_model = get_latest_model()
    predicted_location = predict_location(test_input, latest_model)
    print(f"예측된 위치: {predicted_location}")
