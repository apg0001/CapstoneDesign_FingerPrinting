import os
import pandas as pd
import numpy as np
from datetime import datetime

# 설정값
INPUT_PATH = "./finger_printing/datasets/train_dataset.csv"
OUTPUT_DIR = "./finger_printing/datasets/"
AUGMENT_COUNT = 1  # 원본 대비 추가로 몇 배 생성할지
NOISE_STD = 1.0
DROP_RATIO = 0.1  # 각 세션별 10% dropout
SEED = 42

# 특정 Location 값 리스트 (여기에 원하는 Location 값을 추가)
TARGET_LOCATIONS = ["toilet_5_7_0", "toilet_5_6_0", "toilet_5_5_0", "toilet_5_4_0", "toilet_5_3_0", "toilet_5_2_0", "toilet_5_1_0", "stair_6_r_0", "stair_5_r_3", "stair_5_r_2", "stair_5_l_3", "stair_5_l_2", "stair_5_l_1", "room_6141_9_3", "room_6141_7_8", "room_6141_7_7", "room_6141_7_5", "room_6141_7_4", "room_6141_7_3", "room_6141_6_8", "room_6141_6_7", "room_6141_6_6", "room_6141_6_5", "room_6141_6_3", "room_6141_6_2", "room_6141_6_1", "room_6141_5_1", "room_6141_4_2", "room_6141_2_8", "room_6141_2_7", "room_6141_2_5", "room_6141_2_3", "room_6141_2_2", "room_6141_2_1", "room_6141_1_8", "room_6141_1_7", "room_6141_1_6", "room_6141_1_5", "room_6141_1_4", "room_6141_1_3", "room_6141_1_2", "room_6141_1_1", "room_5143_9_3", "room_5143_9_2", "room_5143_8_5", "room_5143_8_4", "room_5143_8_3", "room_5143_8_2", "room_5143_7_4", "room_5143_7_3", "room_5143_7_2", "room_5143_7_1", "room_5143_5_2", "room_5143_4_1", "room_5143_3_5", "room_5143_3_1", "room_5143_2_5", "room_5143_2_4", "room_5143_2_3", "room_5143_2_2", "room_5143_2_1", "room_5143_1_5", "room_5143_1_4", "room_5143_1_1", "room_5141_9_5", "room_5141_8_5", "room_5141_8_4", "room_5141_8_2", "room_5141_8_1", "room_5141_6_5", "room_5141_6_2", "room_5141_6_1", "room_5141_5_5", "room_5141_5_4", "room_5141_5_3", "room_5141_5_2", "room_5141_5_1", "room_5141_3_5", "room_5141_3_4", "room_5141_2_5", "room_5141_2_4", "room_5141_2_3", "room_5141_2_2", "room_5141_2_1", "room_5141_1_5", "room_5141_1_4", "room_5141_1_3", "room_5141_1_2", "room_5141_1_1", "hall_toilet5_2_1", "hall_toilet5_1_5", "hall_toilet5_1_4", "hall_toilet5_1_3", "hall_toilet5_1_2", "hall_toilet5_1_1", "hall_6147_2_7", "hall_6147_2_2", "hall_6147_1_2", "hall_6147_1_1", "hall_6144_2_2", "hall_6144_2_1", "hall_6144_1_8", "hall_6144_1_7", "hall_6144_1_4", "hall_6144_1_1", "hall_6141_4_11", "hall_6141_4_10", "hall_6141_3_11", "hall_6141_2_9", "hall_6141_2_8", "hall_6141_2_7", "hall_6141_2_6", "hall_6141_2_5", "hall_6141_2_4", "hall_6141_2_3", "hall_6141_2_2",
                    "hall_6141_2_12", "hall_6141_2_11", "hall_6141_2_10", "hall_6141_2_1", "hall_6141_1_9", "hall_6141_1_8", "hall_6141_1_7", "hall_6141_1_6", "hall_6141_1_5", "hall_6141_1_4", "hall_6141_1_3", "hall_6141_1_2", "hall_6141_1_11", "hall_6141_1_10", "hall_6141_1_1", "hall_6105_5_2", "hall_6105_5_1", "hall_6105_4_2", "hall_6105_4_1", "hall_6105_3_2", "hall_6105_3_1", "hall_6105_2_2", "hall_6105_2_1", "hall_6105_1_2", "hall_6105_1_1", "hall_6103_5_2", "hall_6103_5_1", "hall_6103_4_2", "hall_6103_4_1", "hall_6103_3_1", "hall_6103_2_2", "hall_6103_2_1", "hall_6103_1_2", "hall_5147_2_7", "hall_5147_2_6", "hall_5147_2_4", "hall_5147_2_3", "hall_5147_2_2", "hall_5147_2_1", "hall_5147_1_7", "hall_5147_1_6", "hall_5147_1_5", "hall_5147_1_4", "hall_5147_1_3", "hall_5147_1_2", "hall_5147_1_1", "hall_5145_2_6", "hall_5145_2_5", "hall_5145_2_4", "hall_5145_2_3", "hall_5145_2_2", "hall_5145_2_1", "hall_5145_1_5", "hall_5145_1_4", "hall_5145_1_3", "hall_5145_1_2", "hall_5145_1_1", "hall_5143_2_3", "hall_5143_2_2", "hall_5143_2_1", "hall_5143_1_5", "hall_5143_1_3", "hall_5143_1_2", "hall_5143_1_1", "hall_5141_4_7", "hall_5141_4_6", "hall_5141_3_7", "hall_5141_2_8", "hall_5141_2_7", "hall_5141_1_8", "hall_5141_1_7", "hall_5134_1_1", "hall_5132_1_1", "hall_5130_3_1", "hall_5130_2_1", "hall_5130_1_1", "hall_5128_9_1", "hall_5128_3_1", "hall_5128_2_1", "hall_5128_1_1", "hall_5109_5_1", "hall_5109_4_1", "hall_5109_3_1", "hall_5109_2_1", "hall_5109_1_1", "hall_5107_5_2", "hall_5107_5_1", "hall_5107_4_2", "hall_5107_4_1", "hall_5107_1_2", "hall_5107_1_1", "hall_5105_5_2", "hall_5105_5_1", "hall_5105_4_2", "hall_5105_3_2", "hall_5105_2_1", "hall_5105_1_1", "hall_5103_5_2", "hall_5103_5_1", "hall_5103_4_1", "hall_5103_2_1", "hall_5103_1_2", "hall_5103_1_1", "ev_6_3_6", "ev_6_3_5", "ev_6_3_4", "ev_6_3_2", "ev_6_2_6", "ev_6_2_5", "ev_6_2_4", "ev_6_2_3", "ev_6_2_1", "ev_6_1_6", "ev_6_1_5", "ev_6_1_4", "ev_5_3_4", "ev_5_3_1", "ev_5_2_2", "ev_5_2_1", "ev_5_1_1", ]  # 예시로 Location_1, Location_2에 대해서만 증강

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
        session["RSSI"] = session["RSSI"] + \
            np.random.normal(0, NOISE_STD, size=len(session))

        # 10% 무작위 dropout
        drop_indices = np.random.choice(session.index, size=int(
            len(session) * DROP_RATIO), replace=False)
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
output_path = os.path.join(
    OUTPUT_DIR, f"train_dataset_augmented_selective_{NOISE_STD}.csv")
final_df.to_csv(output_path, index=False)
print(f"📁 저장 완료: {output_path}")
