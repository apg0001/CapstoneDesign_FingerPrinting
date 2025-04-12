import pandas as pd

# CSV 파일 불러오기
file_path = './finger_printing/datasets/merged/wifi_rssi_log_merged_20250412_224616_new.csv'  # 여기에 CSV 파일 경로를 입력하세요
merged_df = pd.read_csv(file_path)

# 데이터 확인 (첫 5줄 출력)
print(merged_df.head())

# 'location'과 'mac'을 기준으로 그룹화하고, RSSI 값에 대한 통계 계산
rssi_stats = merged_df.groupby(['Location', 'MAC'])['RSSI'].agg(
    avg_rssi='mean',         # 평균값
    var_rssi='var',          # 분산
    std_rssi='std',          # 표준편차
    max_rssi='max',          # 최대값
    min_rssi='min',           # 최소값
    count='size'             # 그룹 내 요소 갯수
).reset_index()

# 결과 출력
print("\n[RSSI 통계] 위치와 MAC 주소별 RSSI 통계량:")
print(rssi_stats)


# 표준편차들의 평균 계산
average_std_rssi = rssi_stats['std_rssi'].mean()
min_rssi = rssi_stats['std_rssi'].min()
max_std_rssi = rssi_stats['std_rssi'].max()
print(f"\n[전체 표준편차 평균]: {average_std_rssi:.2f} [전체 표준편차 최소: {min_rssi:.2f} [전체 표준편차 최대]: {max_std_rssi:.2f}")

# 결과를 CSV로 저장 (선택사항)
rssi_stats.to_csv('rssi_statistics.csv', index=False)