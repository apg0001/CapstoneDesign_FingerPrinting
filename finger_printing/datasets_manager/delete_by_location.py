import pandas as pd

# CSV 불러오기
df = pd.read_csv("/Users/gichanpark/Desktop/castone_design/finger_printing/datasets/merged/wifi_rssi_log_merged_20250508_225643_new.csv")  # '파일이름.csv'를 실제 파일명으로 바꿔주세요

# 지울 location 리스트
locations_to_remove = ['hall_', 'hall_5103_4', 'hall_5143_', 'hall_6103_', 'stair_5', 'stair_5_l_', 'stair_5r', 'stair_5r_0']  # 여기에 지우고 싶은 Location 값들을 넣으세요

# 해당 location이 아닌 행만 남기기
filtered_df = df[~df['Location'].isin(locations_to_remove)]

# 결과 저장 (선택)
filtered_df.to_csv('filtered_output.csv', index=False)