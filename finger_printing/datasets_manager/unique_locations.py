import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("/Users/gichanpark/Desktop/castone_design/finger_printing/datasets/train_dataset_renamed.csv")  # '파일이름.csv'를 실제 파일명으로 바꿔주세요

# Location 열의 고유값을 오름차순으로 정렬
# unique_locations = sorted(df['Location'].dropna().unique())
unique_locations = sorted(df['Location'].dropna().unique(), reverse=True)

for location in unique_locations:
    print(location)