import pandas as pd

# CSV 파일 불러오기
# df = pd.read_csv("/Users/gichanpark/Desktop/castone_design/finger_printing/datasets/train_dataset.csv")
df = pd.read_csv("/Users/gichanpark/Desktop/castone_design/finger_printing/datasets/filtered_dataset2.csv")

# Location 열의 고유값을 오름차순으로 정렬
# unique_locations = sorted(df['Location'].dropna().unique())
unique_locations = sorted(df['Location'].dropna().unique(), reverse=True)

for i, location in enumerate(unique_locations):
    # print(i, location)
    print(location)