import os
import pandas as pd
from datetime import datetime

# 병합할 CSV 파일이 있는 디렉토리 경로
directory = "./finger_printing/datasets/raw"  # 원하는 디렉토리로 변경

# 현재 날짜 및 시간 가져오기
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV 파일 리스트 가져오기
csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

print(f"{len(csv_files)}개의 CSV 파일을 찾았습니다.")

# 모든 DataFrame을 저장할 리스트
df_list = []

# CSV 파일을 하나씩 읽기
for file in csv_files:
    file_path = os.path.join(directory, file)
    try:
        df = pd.read_csv(file_path)
        df_list.append(df)
        print(f"{file} - {df.shape[0]}개 행 로드 완료")
    except Exception as e:
        print(f"{file} 읽기 중 오류 발생: {e}")

# 모든 DataFrame 병합
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # location 컬럼 이름 찾기 (대소문자 구분 없이)
    location_col = None
    for col in merged_df.columns:
        if col.lower() == 'location':
            location_col = col
            break

    if location_col:
        # location 값이 room, hall, toilet로 시작하는 행만 필터링
        merged_df = merged_df[merged_df[location_col].str.lower().str.startswith(("room", "hall", "toilet"))]

        unique_locations = merged_df[location_col].unique()
        print(f"\n[✔] '{location_col}' 컬럼에 존재하는 고유 Location 값들 (필터링 후):")
        for loc in unique_locations:
            print(f"- {loc}")
    else:
        print("\n[⚠] 'location' 컬럼을 찾을 수 없습니다.")

    # 중복 행 제거
    before_dedup = merged_df.shape[0]
    merged_df.drop_duplicates(inplace=True)
    after_dedup = merged_df.shape[0]

    # 병합된 CSV 저장 (파일명에 날짜 및 시간 추가)
    merged_directory = "./finger_printing/datasets/merged"
    os.makedirs(merged_directory, exist_ok=True)
    output_file = os.path.join(merged_directory, f"wifi_rssi_log_merged_{current_time}.csv")
    merged_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n병합 완료: {output_file}")
    print(f"총 {before_dedup}개 → 중복 제거 후 {after_dedup}개 행")
else:
    print("병합할 데이터가 없습니다.")