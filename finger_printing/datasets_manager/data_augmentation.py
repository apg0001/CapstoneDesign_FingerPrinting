# # import os
# # import pandas as pd
# # import numpy as np
# # from datetime import datetime

# # # 설정값
# # INPUT_PATH = "./finger_printing/datasets/train_dataset.csv"
# # OUTPUT_DIR = "./finger_printing/datasets/augmented"
# # AUGMENT_RATIO = 2.0  # 총 5배 만들기 (원본 1 + 증강 4)
# # NOISE_STD = 3.0  # RSSI에 추가할 노이즈 표준편차
# # SEED = 42

# # np.random.seed(SEED)

# # # 데이터 로드
# # original_df = pd.read_csv(INPUT_PATH)
# # print(f"🧪 원본 데이터 행 수: {len(original_df)}")

# # # Time 기준으로 세션 그룹화
# # grouped = original_df.groupby("Time")
# # all_keys = list(grouped.groups.keys())
# # num_available = len(all_keys)
# # num_augmented_groups = int(num_available * AUGMENT_RATIO)
# # print(f"✨ 증강할 세션 수: {num_augmented_groups}")

# # # 증강할 세션 무작위 선택 (중복 허용)
# # sampled_keys = np.random.choice(
# #     all_keys, size=num_augmented_groups, replace=True)

# # augmented_list = []
# # # new_time_start = max(original_df["Time"]) + 1
# # # new_time_start = original_df["Time"].astype(int).max() + 1
# # # datetime → timestamp (초 단위 정수)
# # original_df["Time"] = pd.to_datetime(original_df["Time"])
# # original_df["Time"] = original_df["Time"].astype(np.int64) // 10**9  # 초 단위로 변환

# # new_time_start = original_df["Time"].max() + 1

# # for i, key in enumerate(sampled_keys):
# #     session = grouped.get_group(key).copy()

# #     # RSSI에 정규분포 노이즈 추가
# #     session["RSSI"] = session["RSSI"] + \
# #         np.random.normal(0, NOISE_STD, size=len(session))

# #     # Time 값 새로 부여 (겹치지 않게)
# #     session["Time"] = new_time_start + i
# #     augmented_list.append(session)

# # # 증강 데이터 병합 및 저장
# # augmented_df = pd.concat(augmented_list, ignore_index=True)
# # final_df = pd.concat([original_df, augmented_df], ignore_index=True)
# # print(f"증강 후 총 데이터 행 수: {len(final_df)}")

# # # 저장
# # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # output_path = os.path.join(
# #     OUTPUT_DIR, f"wifi_rssi_log_augmented_{timestamp}.csv")
# # final_df.to_csv(output_path, index=False, encoding="utf-8")
# # print(f"저장 완료: {output_path}")

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime

# # 설정값
# INPUT_PATH = "./finger_printing/datasets/train_dataset.csv"
# OUTPUT_DIR = "./finger_printing/datasets/augmented"
# AUGMENT_COUNT = 2  # 원본 대비 추가로 몇 배 생성할지 (2면 → 총 3배)
# NOISE_STD = 3.0
# SEED = 42

# np.random.seed(SEED)

# # 원본 데이터 로드
# original_df = pd.read_csv(INPUT_PATH)
# print(f"🧪 원본 데이터 행 수: {len(original_df)}")

# # Time 정수 timestamp로 변환
# original_df["Time"] = pd.to_datetime(original_df["Time"])
# original_df["Time"] = original_df["Time"].astype(np.int64) // 10**9

# # 시작 타임 설정 (원본 max Time 다음부터)
# new_time_start = original_df["Time"].max() + 1

# # 증강 데이터 생성
# augmented_list = []
# for i in range(AUGMENT_COUNT):
#     session = original_df.copy()
#     session["RSSI"] = session["RSSI"] + np.random.normal(0, NOISE_STD, size=len(session))
#     session["Time"] = new_time_start + i  # 증강본은 같은 시간값을 공유
#     augmented_list.append(session)

# # 병합 및 저장
# augmented_df = pd.concat(augmented_list, ignore_index=True)
# final_df = pd.concat([original_df, augmented_df], ignore_index=True)
# print(f"✨ 최종 데이터 행 수: {len(final_df)}")

# # 저장
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = os.path.join(OUTPUT_DIR, f"wifi_rssi_augmented_{timestamp}.csv")
# final_df.to_csv(output_path, index=False, encoding="utf-8")
# print(f"📁 저장 완료: {output_path}")

import os
import pandas as pd
import numpy as np
from datetime import datetime

# 설정값
INPUT_PATH = "./finger_printing/datasets/train_dataset.csv"
OUTPUT_DIR = "./finger_printing/datasets/"
AUGMENT_COUNT = 2  # 원본 대비 추가로 몇 배 생성할지
NOISE_STD = 3.0
DROP_RATIO = 0.1  # 각 세션별 10% dropout
SEED = 42

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
for i in range(AUGMENT_COUNT):
    session = original_df.copy()

    # 노이즈 추가
    session["RSSI"] = session["RSSI"] + np.random.normal(0, NOISE_STD, size=len(session))

    # 10% 무작위 dropout
    drop_indices = np.random.choice(session.index, size=int(len(session) * DROP_RATIO), replace=False)
    session = session.drop(index=drop_indices).reset_index(drop=True)

    # 새로운 timestamp 부여
    session["Time"] = new_time_start + i

    augmented_list.append(session)

# 병합 및 저장
augmented_df = pd.concat(augmented_list, ignore_index=True)
final_df = pd.concat([original_df, augmented_df], ignore_index=True)
print(f"✨ 최종 데이터 행 수: {len(final_df)}")

# 저장
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(OUTPUT_DIR, f"train_dataset_augmented.csv")
final_df.to_csv(output_path, index=False)
print(f"📁 저장 완료: {output_path}")