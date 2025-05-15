import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import joblib
import logging
import os

# GPU 사용 가능하면 GPU, 아니면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로거 설정: OnlineTrainer용 전용 로거 생성
logger = logging.getLogger("OnlineTrainer")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.propagate = False  # 로그 중복 전파 방지


def detect_new_macs(input_mac_rssi, mac_encoder):
    """
    새로운 MAC 주소 탐지 함수
    :param input_mac_rssi: dict, MAC 주소(key)와 RSSI(value) 쌍
    :param mac_encoder: LabelEncoder, 이미 학습된 MAC 인코더
    :return: list, mac_encoder에 없는 새 MAC 주소 리스트
    """
    new_macs = []
    for mac in input_mac_rssi.keys():
        try:
            mac_encoder.transform([mac])  # 기존에 인코딩 가능한지 검사
        except ValueError:
            new_macs.append(mac)  # 인코딩 불가하면 새 MAC으로 간주
    return new_macs


class IncrementalDataset(Dataset):
    """
    온라인 학습용 데이터셋 클래스

    :param data_list: list of tuples, [(input_mac_rssi_dict, label), ...]
    :param mac_encoder: LabelEncoder, MAC 주소 인코딩용
    :param rssi_mean: float, RSSI 정규화에 사용할 평균
    :param rssi_std: float, RSSI 정규화에 사용할 표준편차
    :param max_ap: int, 최대 AP 수 (패딩 길이)
    """

    def __init__(self, data_list, mac_encoder, rssi_mean, rssi_std, max_ap=100):
        self.data_list = data_list
        self.mac_encoder = mac_encoder
        self.rssi_mean = rssi_mean
        self.rssi_std = rssi_std
        self.max_ap = max_ap

    def __len__(self):
        """
        데이터셋 크기 반환
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플 반환

        :param idx: int, 인덱스
        :return: tuple(torch.Tensor, torch.Tensor, torch.Tensor),
                 (rssi_tensor, mac_tensor, label_tensor)
        """
        input_mac_rssi, label = self.data_list[idx]

        def apply_kalman_filter(rssi_values):
            """
            칼만 필터를 적용하여 RSSI 값 노이즈 제거

            :param rssi_values: numpy array, RSSI 값 배열
            :return: numpy array, 필터링된 RSSI 값 배열
            """
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([[rssi_values[0]]])  # 초기 상태
            kf.F, kf.H = np.array([[1]]), np.array([[1]])  # 상태전이, 관측 모델
            kf.P *= 1000  # 초기 공분산
            kf.R = 5  # 관측 노이즈 공분산
            kf.Q = 1e-5  # 프로세스 노이즈 공분산
            filtered = []
            for r in rssi_values:
                kf.predict()
                kf.update(np.array([[r]]))
                filtered.append(kf.x[0, 0])
            return np.array(filtered)

        # RSSI가 -95 이상인 MAC과 RSSI만 선별
        macs = []
        rssis = []
        for mac, rssi in input_mac_rssi.items():
            if rssi > -95:
                macs.append(mac)
                rssis.append(rssi)

        # 칼만 필터 적용
        rssis_filtered = apply_kalman_filter(np.array(rssis))
        # 가중치: RSSI 절대값 사용
        weights = np.abs(rssis_filtered)
        # 가중 평균 RSSI 계산
        rssi_weighted = np.average(rssis_filtered, weights=weights)
        # 정규화 (평균, 표준편차 이용)
        rssi_norm = (rssi_weighted - self.rssi_mean) / self.rssi_std

        mac_indices = []
        rssi_values = []

        # MAC 주소를 인코딩하고 RSSI 정규화값으로 대응
        for mac, rssi in zip(macs, [rssi_norm] * len(macs)):
            try:
                idx = self.mac_encoder.transform([mac])[0] + 1  # 0은 패딩용 인덱스
                mac_indices.append(idx)
                rssi_values.append(rssi)
            except ValueError:
                # 인코딩 실패 시 무시
                continue

        # AP 개수 맞추기 위해 패딩 또는 잘라내기
        if len(mac_indices) < self.max_ap:
            pad_len = self.max_ap - len(mac_indices)
            mac_indices = np.pad(mac_indices, (0, pad_len), constant_values=0)
            rssi_values = np.pad(rssi_values, (0, pad_len), constant_values=0)
        else:
            mac_indices = mac_indices[:self.max_ap]
            rssi_values = rssi_values[:self.max_ap]

        # 최종 입력 배열 생성 (shape: max_ap x 2)
        X = np.column_stack([mac_indices, rssi_values])

        # 텐서 변환
        rssi_tensor = torch.tensor(X[:, 1], dtype=torch.float32)
        mac_tensor = torch.tensor(X[:, 0], dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return rssi_tensor, mac_tensor, label_tensor


class OnlineTrainer:
    """
    온라인 학습 관리 클래스

    :param model: PyTorch 모델 인스턴스
    :param mac_encoder: MAC 주소 인코더 (LabelEncoder)
    :param location_encoder: 위치 인코더 (LabelEncoder)
    :param rssi_mean: RSSI 정규화 평균값
    :param rssi_std: RSSI 정규화 표준편차
    :param max_ap: 최대 AP 개수 (입력 크기)
    :param lr: 학습률
    :param model_save_path: 모델 가중치 저장 경로
    :param encoder_save_path: 인코더 저장 경로
    :param norm_save_path: 정규화 파라미터 저장 경로
    :param csv_path: 온라인 학습 데이터 CSV 저장 경로
    :param retrain_batch_size: 재학습 실행을 위한 버퍼 크기 임계값
    """

    def __init__(self, model, mac_encoder, location_encoder, rssi_mean, rssi_std,
                 max_ap=100, lr=1e-4,
                 model_save_path="./finger_printing/checkpoints/checkpoints/fp_model_CNNTransformer_latest.pt",
                 encoder_save_path="./finger_printing/checkpoints/encoders/encoders_latest.pkl",
                 norm_save_path="./finger_printing/checkpoints/norm/norm_latest.pkl",
                 csv_path="./finger_printing/online_training_data.csv",
                 retrain_batch_size=100):
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
        self.csv_path = csv_path
        self.retrain_batch_size = retrain_batch_size

        self.new_data_buffer = []

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def add_new_sample(self, input_mac_rssi, label):
        """
        새 샘플을 버퍼에 추가하고
        버퍼가 retrain_batch_size 이상이면 학습, 저장, CSV 기록 수행

        :param input_mac_rssi: dict, MAC-RSSI 쌍
        :param label: int, 위치 인코딩 라벨
        """
        self.new_data_buffer.append((input_mac_rssi, label))
        logger.info(
            f"New sample added. Buffer size: {len(self.new_data_buffer)}")

        if len(self.new_data_buffer) >= self.retrain_batch_size:
            logger.info(
                f"Buffer size reached {self.retrain_batch_size}. Starting incremental training.")
            self.train_incrementally(epochs=1, batch_size=8)
            self.save_model_and_encoders()
            self.save_data_to_csv()
            logger.info(
                "Incremental training, saving and data logging completed.")

    def train_incrementally(self, epochs=1, batch_size=8):
        """
        버퍼 내 데이터를 이용해 온라인 학습 수행

        :param epochs: int, 학습 에폭 수
        :param batch_size: int, 배치 크기
        """
        if len(self.new_data_buffer) < self.retrain_batch_size:
            logger.info(
                f"Buffer size ({len(self.new_data_buffer)}) less than retrain_batch_size ({self.retrain_batch_size}). Skipping training.")
            return

        dataset = IncrementalDataset(
            self.new_data_buffer, self.mac_encoder, self.rssi_mean, self.rssi_std, self.max_ap)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for rssi_batch, mac_batch, labels_batch in dataloader:
                rssi_batch = rssi_batch.to(device)
                mac_batch = mac_batch.to(device)
                labels_batch = labels_batch.to(device)

                # 배치 크기 1일 때 BatchNorm 문제 회피
                if rssi_batch.size(0) == 1:
                    self.model.eval()
                    outputs = self.model(rssi_batch, mac_batch)
                    self.model.train()
                    loss = self.criterion(outputs, labels_batch)
                else:
                    outputs = self.model(rssi_batch, mac_batch)
                    loss = self.criterion(outputs, labels_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 학습 완료 후 버퍼 비움
        self.new_data_buffer.clear()

    def save_model_and_encoders(self):
        """
        모델과 인코더, 정규화 파라미터를 파일에 저장
        """
        torch.save(self.model.state_dict(), self.model_save_path)
        joblib.dump({
            "location_encoder": self.location_encoder,
            "mac_encoder": self.mac_encoder
        }, self.encoder_save_path)
        joblib.dump({
            "mean": self.rssi_mean,
            "std": self.rssi_std
        }, self.norm_save_path)
        logger.info(f"Model and encoders saved to disk.")

    def save_data_to_csv(self):
        """
        학습에 사용된 버퍼 데이터를 CSV 파일에 추가 저장
        파일이 존재하지 않으면 헤더 포함 새로 생성

        CSV 컬럼: MAC, RSSI, LOCATION
        """
        records = []
        for input_mac_rssi, label in self.new_data_buffer:
            location = self.location_encoder.inverse_transform([label])[0]
            for mac, rssi in input_mac_rssi.items():
                records.append({
                    "MAC": mac,
                    "RSSI": rssi,
                    "LOCATION": location
                })

        df_new = pd.DataFrame(records)

        if os.path.exists(self.csv_path):
            df_new.to_csv(self.csv_path, mode='a', index=False, header=False)
        else:
            df_new.to_csv(self.csv_path, mode='w', index=False, header=True)

        logger.info(
            f"Appended {len(records)} records to CSV at {self.csv_path}")

    def load_model(self, model_path):
        """
        모델 가중치를 파일에서 불러와 로드
        """
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
