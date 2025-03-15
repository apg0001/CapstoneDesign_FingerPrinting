import pandas as pd
from datetime import datetime

# 현재 날짜 및 시간 가져오기
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV 파일 경로
file_path = "./datasets/wifi_rssi_log_modified_20250315_232818.csv"  # 원본 CSV 파일
output_path = f"./datasets/wifi_rssi_log_filtered_{current_time}.csv"  # 저장할 CSV 파일

# CSV 파일 읽기
df = pd.read_csv(file_path)

# "Location" 열에서 'hall' 또는 'room'으로 시작하는 값만 남기고 나머지 삭제
df_filtered = df[df["Location"].str.startswith(("hall", "room"), na=False)]

# 변경된 CSV 저장
df_filtered.to_csv(output_path, index=False, encoding="utf-8")

print(f"'hall' 또는 'room'으로 시작하지 않는 행 삭제 완료!")
print(f"저장된 파일: {output_path}")