import pandas as pd
from datetime import datetime
import re

# 현재 날짜 및 시간 가져오기
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV 파일 경로
file_path = f"./finger_printing/datasets/train_dataset.csv"  # 수정할 CSV 파일
output_path = f"./finger_printing/datasets/train_dataset_new.csv"  # 저장할 파일

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
# replace_dict = {
#     "hall_5128_9_1n": "hall_5128_9_1",
#     "hall_5130_2_1n": "hall_5130_2_1",
#     "hall_5103_3_1(찐)": "hall_5103_3_1",
# }

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 특정 열에서 값 변경
# df[column_name] = df[column_name].replace(replace_dict)
# df[column_name] = df[column_name].apply(lambda x: re.sub('hall_toilet_5', 'hall_toilet5', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_7', 'toilet_5_7_0', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_6', 'toilet_5_6_0', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_5', 'toilet_5_5_0', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_4', 'toilet_5_4_0', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_3', 'toilet_5_3_0', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_2', 'toilet_5_2_0', x))
# df[column_name] = df[column_name].apply(lambda x: re.sub('toilet_5_1', 'toilet_5_1_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_l_0', 'stair_5l_0_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_l_1', 'stair_5l_1_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_l_2', 'stair_5l_2_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_l_3', 'stair_5l_3_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_6_l_0', 'stair_5l_4_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_r_0', 'stair_5r_0_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_r_1', 'stair_5r_1_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_r_2', 'stair_5r_2_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_5_r_3', 'stair_5r_3_0', x))
df[column_name] = df[column_name].apply(lambda x: re.sub('stair_6_r_0', 'stair_5r_4_0', x))

# 변경된 CSV 저장
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"저장된 파일: {output_path}")