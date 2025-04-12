import pandas as pd
from datetime import datetime
import re

# 현재 날짜 및 시간 가져오기
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV 파일 경로
file_path = f"./finger_printing/datasets/merged/wifi_rssi_log_merged_20250412_224616_new.csv"  # 수정할 CSV 파일
output_path = f"./finger_printing/datasets/merged/wifi_rssi_log_merged_20250412_224616_new2.csv"  # 저장할 파일

# 변경할 열과 값 지정
column_name = "Location"  # 변경할 열 이름

# 여러 값 변경 (딕셔너리 사용)
# replace_dict = {
#     "5143-1": "hall_5143_8",
#     "5143-2": "hall_5143_7",
#     "5143-3": "hall_5143_6",
#     "5143-4": "hall_5143_5",
#     "5143-5": "hall_5143_4",
#     "5143-6": "hall_5143_3",
# }
# replace_dict = {
#     "hall_5128_9_1": "hall_5128_4_1",
#     "hall_5130_2_1": "hall_5130_1_2",
#     "hall_5103_3_1": "hall_5103_3_2",
# }
replace_dict = {
    "hall_5128_9_1n": "hall_5128_9_1",
    "hall_5130_2_1n": "hall_5130_2_1",
    "hall_5103_3_1(찐)": "hall_5103_3_1",
}

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 특정 열에서 값 변경
df[column_name] = df[column_name].replace(replace_dict)
# df[column_name] = df[column_name].apply(lambda x: re.sub('-', '_', x))

# 변경된 CSV 저장
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"저장된 파일: {output_path}")