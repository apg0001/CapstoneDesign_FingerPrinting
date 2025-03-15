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
import matplotlib.pyplot as plt
from tqdm import tqdm

# 데이터 로드 및 인코딩 감지
file_path = "./datasets/wifi_rssi_log.csv"

with open(file_path, "rb") as f:
    result = chardet.detect(f.read())

encoding_type = result['encoding']
print(f"감지된 인코딩: {encoding_type}")

df = pd.read_csv(file_path, encoding=encoding_type)

# Location을 Label Encoding
location_encoder = LabelEncoder()
df["location_encoded"] = location_encoder.fit_transform(df["Location"])

# MAC 주소를 Label Encoding (Embedding 사용)
mac_encoder = LabelEncoder()
df["mac_encoded"] = mac_encoder.fit_transform(df["MAC"])

# RSSI 가중 평균값 계산 (RSSI 절대값이 클수록 높은 신뢰도)
df["rssi_weighted"] = df.groupby("MAC")["RSSI"].transform(lambda x: np.average(x, weights=np.abs(x)))

# 로그 거리 경로 손실 모델 적용
# d = 10^((P_0 - RSSI) / (10 * n))
P_0 = -40  # 기준 RSSI 값
n = 3  # 경로 손실 계수

df["distance"] = 10 ** ((P_0 - df["rssi_weighted"]) / (10 * n))

# KNN 보간법 적용 (누락된 RSSI 값 보정)
imputer = KNNImputer(n_neighbors=3)
df["rssi_imputed"] = imputer.fit_transform(df[["rssi_weighted"]])

# 데이터 그룹화 (Time + Location)
# grouped = df.groupby(["Time", "location_encoded"])
grouped = df.groupby(["Time", "location_encoded"])

# 데이터셋 변환
X_list, y_list = [], []
max_ap = 10  # 최대 AP 개수 (환경에 맞게 조절 가능)

for (time, location), group in grouped:
    mac_indices = mac_encoder.transform(group["MAC"])
    distances = group["distance"].values

    # 데이터 개수가 max_ap보다 작을 경우 패딩 추가
    if len(distances) < max_ap:
        pad_size = max_ap - len(distances)
        mac_indices = np.pad(mac_indices, (0, pad_size), constant_values=-1)  # MAC 인코딩 패딩
        distances = np.pad(distances, (0, pad_size), constant_values=0)  # 거리 패딩

    # 데이터 개수가 max_ap보다 클 경우 잘라내기
    elif len(distances) > max_ap:
        mac_indices = mac_indices[:max_ap]
        distances = distances[:max_ap]

    feature_vector = np.column_stack([mac_indices, distances])
    X_list.append(feature_vector)
    y_list.append(location)

X = np.array(X_list)  # 패딩 후에는 정상적으로 변환됨
y = np.array(y_list)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# PyTorch Dataset 정의
class WifiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, :, 1], dtype=torch.float32)  # 거리 정보만 학습
        self.mac = torch.tensor(X[:, :, 0], dtype=torch.long)  # MAC 주소
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mac[idx], self.y[idx]


train_dataset = WifiDataset(X_train, y_train)
test_dataset = WifiDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# CNN + Embedding 모델 정의
class WifiCNN(nn.Module):
    def __init__(self, num_ap, num_classes, num_mac):
        super(WifiCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_mac, embedding_dim=8, padding_idx=-1)
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, padding=1)  # 8 + 1 (거리)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * num_ap, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, mac):
        mac_embed = self.embedding(mac)  # MAC 임베딩
        x = torch.cat([x.unsqueeze(2), mac_embed], dim=2)  # 거리 + MAC 임베딩 결합
        x = x.permute(0, 2, 1)  # (batch, feature, time)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 모델 초기화
num_ap = X.shape[1]
num_mac = len(mac_encoder.classes_)
num_classes = len(set(y))

model = WifiCNN(num_ap, num_classes, num_mac)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 학습 루프
num_epochs = 100
progress_bar = tqdm(range(num_epochs), desc="Training")

for epoch in progress_bar:
    model.train()
    total_loss = 0

    for X_batch, mac_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch, mac_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 정확도 계산
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, mac_batch, y_batch in test_loader:
            outputs = model(X_batch, mac_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total

    # ✅ tqdm 인스턴스에서 `set_postfix()` 호출
    progress_bar.set_postfix(Loss=total_loss / len(train_loader), Accuracy=f"{accuracy}%")

# 모델 평가
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


# 예측 함수
def predict_location(mac_rssi_dict):
    input_vector = np.zeros((num_ap, 2))  # MAC + 거리 정보 저장

    for i, (mac, rssi) in enumerate(mac_rssi_dict.items()):
        if i >= num_ap:
            break
        if mac in mac_encoder.classes_:
            mac_index = mac_encoder.transform([mac])[0]
            distance = 10 ** ((P_0 - rssi) / (10 * n))
            input_vector[i] = [mac_index, distance]

    input_tensor = torch.tensor(input_vector[:, 1], dtype=torch.float32).unsqueeze(0)  # 거리 정보
    mac_tensor = torch.tensor(input_vector[:, 0], dtype=torch.long).unsqueeze(0)  # MAC 정보

    model.eval()
    with torch.no_grad():
        output = model(input_tensor, mac_tensor)
        _, predicted_index = torch.max(output, 1)

    return location_encoder.inverse_transform([predicted_index.item()])[0]


# 예측 예시
test_input = {
    "26:3f:0b:e2:66:14": -40,
    "2c:3f:0b:e2:66:3f": -74,
    "2e:3f:1b:e2:64:73": -86
}

predicted_location = predict_location(test_input)
print(f"예측된 위치: {predicted_location}")

# print(X)