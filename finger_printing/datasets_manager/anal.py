# import pandas as pd

# # CSV 파일 불러오기
# file_path = './finger_printing/datasets/train_dataset.csv'  # 여기에 CSV 파일 경로를 입력하세요
# merged_df = pd.read_csv(file_path)

# # 데이터 확인 (첫 5줄 출력)
# # print(merged_df.head())

# # 'location'과 'mac'을 기준으로 그룹화하고, RSSI 값에 대한 통계 계산
# rssi_stats = merged_df.groupby(['Location', 'MAC'])['RSSI'].agg(
#     avg_rssi='mean',         # 평균값
#     # var_rssi='var',          # 분산
#     std_rssi='std',          # 표준편차
#     max_rssi='max',          # 최대값
#     min_rssi='min',           # 최소값
#     count='size'             # 그룹 내 요소 갯수
# ).reset_index()

# # 결과 출력
# print("\n[RSSI 통계] 위치와 MAC 주소별 RSSI 통계량:")
# # print(rssi_stats)
# rssi_stats_sorted = rssi_stats.dropna().sort_values(by='std_rssi', ascending=False)

# # pandas 출력 옵션 설정: 최대 100행까지 모두 출력
# pd.set_option('display.max_rows', 100)
# print(rssi_stats_sorted.head(100))

# # 표준편차들의 평균 계산
# average_std_rssi = rssi_stats['std_rssi'].mean()
# min_rssi = rssi_stats['std_rssi'].min()
# max_std_rssi = rssi_stats['std_rssi'].max()
# print(f"\n[전체 표준편차 평균]: {average_std_rssi:.2f} [전체 표준편차 최소: {min_rssi:.2f} [전체 표준편차 최대]: {max_std_rssi:.2f}")

# # 결과를 CSV로 저장 (선택사항)
# rssi_stats.to_csv('rssi_statistics.csv', index=False)

# # [전체 표준편차 평균]: 1.46 [전체 표준편차 최소: 0.00 [전체 표준편차 최대]: 17.68

import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter

# 칼만 필터 함수 정의
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

# CSV 파일 불러오기
file_path = './finger_printing/datasets/train_dataset.csv'  # CSV 파일 경로 입력
df = pd.read_csv(file_path)

# 칼만 필터 적용 (MAC별로 그룹화해서 RSSI에 필터 적용)
df['rssi_filtered'] = df.groupby('MAC')['RSSI'].transform(apply_kalman_filter)

# 칼만 필터 적용 전 RSSI 통계량 계산
rssi_stats_before = df.groupby(['Location', 'MAC'])['RSSI'].agg(
    avg_rssi='mean',
    std_rssi='std',
    max_rssi='max',
    min_rssi='min',
    count='size'
).reset_index()

# 칼만 필터 적용 후 rssi_filtered 통계량 계산
rssi_stats_after = df.groupby(['Location', 'MAC'])['rssi_filtered'].agg(
    avg_rssi_filtered='mean',
    std_rssi_filtered='std',
    max_rssi_filtered='max',
    min_rssi_filtered='min',
    count_filtered='size'
).reset_index()

# pandas 출력 옵션 설정 (최대 100행 출력)
pd.set_option('display.max_rows', 100)

print("\n[칼만 필터 적용 전 RSSI 통계] 위치와 MAC 주소별 통계량:")
print(rssi_stats_before.dropna().sort_values(by='std_rssi', ascending=False).head(100))

print("\n[칼만 필터 적용 후 RSSI 통계] 위치와 MAC 주소별 통계량:")
print(rssi_stats_after.dropna().sort_values(by='std_rssi_filtered', ascending=False).head(100))

# 전체 표준편차 통계 출력 (전, 후)
print(f"\n[전체 RSSI 표준편차 평균 (필터 전)]: {rssi_stats_before['std_rssi'].mean():.2f}")
print(f"[전체 RSSI 표준편차 최소 (필터 전)]: {rssi_stats_before['std_rssi'].min():.2f}")
print(f"[전체 RSSI 표준편차 최대 (필터 전)]: {rssi_stats_before['std_rssi'].max():.2f}\n")

print(f"[전체 RSSI 표준편차 평균 (필터 후)]: {rssi_stats_after['std_rssi_filtered'].mean():.2f}")
print(f"[전체 RSSI 표준편차 최소 (필터 후)]: {rssi_stats_after['std_rssi_filtered'].min():.2f}")
print(f"[전체 RSSI 표준편차 최대 (필터 후)]: {rssi_stats_after['std_rssi_filtered'].max():.2f}\n")

# 필요하다면 CSV로 저장
rssi_stats_before.to_csv('rssi_stats_before.csv', index=False)
rssi_stats_after.to_csv('rssi_stats_after.csv', index=False)