    # 🔍 location 컬럼에 존재하는 고유 값 출력
    location_col = None
    for col in merged_df.columns:
        if col.lower() == 'location':
            location_col = col
            break

    if location_col:
        unique_locations = merged_df[location_col].unique()
        print(f"\n[✔] '{location_col}' 컬럼에 존재하는 고유 Location 값들:")
        for loc in unique_locations:
            print(f"- {loc}")
    else:
        print("\n[⚠] 'location' 컬럼을 찾을 수 없습니다.")