import os
import pandas as pd
import numpy as np
from datetime import datetime

# 설정값
INPUT_PATH = "./finger_printing/datasets/train_dataset.csv"
OUTPUT_DIR = "./finger_printing/datasets/"
AUGMENT_COUNT = 2  # 원본 대비 추가로 몇 배 생성할지
NOISE_STD = 1.0
DROP_RATIO = 0.1  # 각 세션별 10% dropout
SEED = 42

# 특정 Location 값 리스트 (여기에 원하는 Location 값을 추가)
TARGET_LOCATIONS = ["Location_1", "Location_2"]  # 예시로 Location_1, Location_2에 대해서만 증강

np.random.seed(SEED)

# 원본 데이터 로드
original_df = pd.read_csv(INPUT_PATH)
print(f"🧪 원본 데이터 행 수: {len(original_df)}")

# Time → 정수 timestamp
original_df["Time"] = pd.to_datetime(original_df["Time"])
original_df["Time"] = original_df["Time"].astype(np.int64) // 10**9

# 증강용 시작 타임
new_time_start = original_df["Time"].max() + 1

# 증강 데이터 생성
augmented_list = []

# 'Location' 값에 해당하는 데이터만 증강
for location in TARGET_LOCATIONS:
    location_df = original_df[original_df["Location"] == location].copy()

    for i in range(AUGMENT_COUNT):
        session = location_df.copy()

        # 노이즈 추가
        session["RSSI"] = session["RSSI"] + np.random.normal(0, NOISE_STD, size=len(session))

        # 10% 무작위 dropout
        drop_indices = np.random.choice(session.index, size=int(len(session) * DROP_RATIO), replace=False)
        session = session.drop(index=drop_indices).reset_index(drop=True)

        # 새로운 timestamp 부여
        session["Time"] += (new_time_start + i)

        augmented_list.append(session)

# 병합 및 저장
augmented_df = pd.concat(augmented_list, ignore_index=True)
final_df = pd.concat([original_df, augmented_df], ignore_index=True)
print(f"✨ 최종 데이터 행 수: {len(final_df)}")

# 저장
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"train_dataset_augmented_{NOISE_STD}.csv")
final_df.to_csv(output_path, index=False)
print(f"📁 저장 완료: {output_path}")