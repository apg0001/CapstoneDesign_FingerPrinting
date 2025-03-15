import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import chardet
from pykalman import KalmanFilter
import numpy as np

def apply_kalman_filter(rssi_values):
    """
    칼만 필터를 적용하여 RSSI 값을 보정하는 함수
    """
    kf = KalmanFilter(initial_state_mean=rssi_values[0], n_dim_obs=1)
    filtered_rssi, _ = kf.smooth(rssi_values)
    return filtered_rssi


# 📌 데이터 로드
file_path = "./wifi_rssi_log.csv"

# 🔍 파일 인코딩 감지 후 로드
with open(file_path, "rb") as f:
    result = chardet.detect(f.read())
encoding_type = result['encoding']
print(f"감지된 인코딩: {encoding_type}")

df = pd.read_csv(file_path, encoding=encoding_type)

# 📌 MAC 주소를 One-Hot Encoding
label_encoder_mac = LabelEncoder()
df["mac_encoded"] = label_encoder_mac.fit_transform(df["MAC"])
one_hot_encoder = OneHotEncoder(sparse=False)
mac_one_hot = one_hot_encoder.fit_transform(df["mac_encoded"].values.reshape(-1, 1))

# 📌 RSSI 정규화
df["rssi_norm"] = (df["RSSI"] + 100) / 100
# 📌 적용 (전처리 단계에서 실행)
# df["rssi_norm"] = apply_kalman_filter(df["rssi_norm"])

# 📌 Location을 Label Encoding (문자열 → 숫자 변환)
label_encoder_loc = LabelEncoder()
df["loc_encoded"] = label_encoder_loc.fit_transform(df["Location"])  # 원래 값을 숫자로 변환

# 📌 타임스탬프와 위치 기준으로 그룹화
grouped = df.groupby(["Time", "loc_encoded"])

# 📌 데이터 변환
X_list, y_list = [], []
max_ap = 10  # 한 번의 측정에서 최대 10개 AP 사용

for (time, location), group in grouped:
    mac_one_hot_group = one_hot_encoder.transform(label_encoder_mac.transform(group["MAC"]).reshape(-1, 1))
    rssi_values = group["rssi_norm"].values

    # MAC 주소 + RSSI 결합
    feature_vector = np.hstack([mac_one_hot_group, rssi_values.reshape(-1, 1)])

    # 패딩 추가 (AP 개수가 max_ap보다 적으면 0으로 채우기)
    if feature_vector.shape[0] < max_ap:
        pad_size = max_ap - feature_vector.shape[0]
        feature_vector = np.vstack([feature_vector, np.zeros((pad_size, feature_vector.shape[1]))])
    elif feature_vector.shape[0] > max_ap:
        feature_vector = feature_vector[:max_ap]  # 초과하면 잘라버림

    X_list.append(feature_vector)
    y_list.append(location)  # 이미 LabelEncoder로 변환한 값 사용

# 📌 최종 데이터셋 변환
X = np.array(X_list)
y = np.array(y_list)

# 데이터 크기 출력
print(f"총 행 수: {df.shape[0]}")
print(f"총 데이터 샘플 개수: {len(X)}")
print(f"입력 데이터 크기: {X.shape}")  # (샘플 수, AP 개수, MAC + RSSI)
print(f"출력 데이터 크기: {y.shape}")  # (샘플 수,)
print(f"사용된 Location 라벨: {list(label_encoder_loc.classes_)}")  # 원래 문자열 Location 값

# 📌 Train/Test Split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


# 📌 데이터셋 클래스 정의
class WifiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 📌 데이터 로더 생성
train_dataset = WifiDataset(X_train, y_train)
test_dataset = WifiDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# 📌 CNN 모델 정의
class DeepWifiCNN(nn.Module):
    def __init__(self, num_ap, num_features, num_classes):
        super(DeepWifiCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)  # Dropout 추가
        self.fc1 = nn.Linear(256 * num_ap, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


# 📌 모델 초기화
num_ap = X.shape[1]
num_features = X.shape[2]
num_classes = len(set(y))

model = DeepWifiCNN(num_ap, num_features, num_classes)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing 적용
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 📌 모델 학습
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 🔥 매 Epoch 후 정확도 계산
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")


# 📌 예측 함수
def predict_location(model, label_encoder_mac, one_hot_encoder, label_encoder_loc, mac_rssi_dict, num_ap, num_features):
    input_vector = np.zeros((num_ap, num_features))

    for i, (mac, rssi) in enumerate(mac_rssi_dict.items()):
        if i >= num_ap:
            break
        if mac in label_encoder_mac.classes_:
            mac_one_hot = one_hot_encoder.transform([[label_encoder_mac.transform([mac])[0]]])[0]
            rssi_norm = (rssi + 100) / 100
            input_vector[i] = np.hstack([mac_one_hot, rssi_norm])

    input_tensor = torch.tensor(input_vector.reshape(1, num_ap, num_features), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_index = torch.max(output, 1)

    predicted_location = label_encoder_loc.inverse_transform([predicted_index.item()])[0]

    return predicted_location


# 📌 예측 예시
test_input = {
    "26:3f:0b:e2:66:14": -40,
    "26:3f:0b:e2:64:73": -75,
    "26:3f:1b:59:2e:56": -87,
    "26:3f:0b:e2:66:b0": -65,
    "26:3f:0b:e2:65:61": -77,
    "26:3f:1b:e2:66:3f": -77,
    "ee:55:b8:6e:4b:9c": -87,
    "26:3f:1b:59:2e:20": -83,
    "26:3f:0b:e2:67:13": -66,
    "26:3f:0b:e2:66:3f": -75,
    "26:3f:1b:e2:66:b0": -77,
}

predicted_location = predict_location(model, label_encoder_mac, one_hot_encoder, label_encoder_loc, test_input, num_ap, num_features)
print(f"예측된 위치: {predicted_location}")