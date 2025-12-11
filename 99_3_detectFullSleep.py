import os
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

base_folder = 

all_files = glob.glob(os.path.join(base_folder, '**', '*.csv'), recursive=True)

if not all_files:
    print(f"❌ [오류] '{base_folder}' 폴더에 CSV 파일이 없습니다.")
else:
    print(f"✅ 총 {len(all_files)}개의 병합된 파일을 찾았습니다.")
    
len(all_files) # 3399개의 csv 파일 존재

# 통잠잔 날 추리기 -------------------------------------------------
# 1. 개별 csv에서 수면 시간(21:00~08:59)에 대해 분단위 값이 180 연속적으로 0인 구간이 없다면, 통잠
# 2. 통잠 잔 날의 csv 파일명 리스트로 저장
# 3. 통잠 잔 날의 데이터만 폴더로 따로 저장(유저별/날짜별)
def has_consecutive_non_zeros(series, min_consecutive):
    is_not_zero = (series != 0)
    group_ids = is_not_zero.diff().ne(0).cumsum()
    sleep_groups = group_ids[is_not_zero]

    if sleep_groups.empty: # 만약 수면(0이 아님)한 적이 아예 없다면
            return False # 통잠 실패
    
    run_lengths = sleep_groups.groupby(sleep_groups).size()
    return (run_lengths >= min_consecutive).any()

MIN_CONSECUTIVE_SLEEP = 180 # 180분 (3시간) 연속 '수면'
full_sleep_days = []
total_files_checked = 0

for dirpath, dirnames, filenames in os.walk(base_folder): # 모든 하위 폴더 순회
    for filename in filenames: # 모든 파일 순회
        if not filename.endswith('.csv'): # CSV가 아니면 건너뜀
            continue
            
        file_path = os.path.join(dirpath, filename) # 파일 전체 경로
        total_files_checked += 1 # 검사한 파일 수 +1
        
        try:
            df = pd.read_csv(file_path) # 파일 읽기
            
            # 수면 시간 (21:00 ~ 08:59) 데이터 필터링
            df_sleep_time = df[(df['time_hour'] >= 21) | (df['time_hour'] < 9)]
            
            sleep_data_series = df_sleep_time['sleep_depth'] # sleep_depth 컬럼만 추출
            
            # [수정] 180분 연속 '수면'(0 아님)이 있었는지 검사
            # is_full_sleep = True이면 "통잠 성공", False이면 "통잠 실패"
            is_full_sleep = has_consecutive_non_zeros(sleep_data_series, MIN_CONSECUTIVE_SLEEP)
            
            # [수정] 통잠에 성공한 경우(True) 리스트에 추가
            if is_full_sleep: 
                full_sleep_days.append(file_path)
                
        except Exception as e:
            print(f"  [오류] 파일 처리 실패: {file_path}, {e}")

print(f"\n총 {total_files_checked}개의 파일 중 {len(full_sleep_days)}일이 '통잠' 조건에 해당합니다.")

# '통잠' 데이터 저장 -------------------------------------------------
output_folder_full_sleep = 'full_sleep_days_data'
os.makedirs(output_folder_full_sleep, exist_ok=True)

copied_count = 0
error_count = 0

for source_file_path in full_sleep_days:
    try:
        filename = os.path.basename(source_file_path)
        user_id = os.path.basename(os.path.dirname(source_file_path))
    
        dest_user_folder = os.path.join(output_folder_full_sleep, user_id)
        os.makedirs(dest_user_folder, exist_ok=True)

        dest_file_path = os.path.join(dest_user_folder, filename)
        
        shutil.copy2(source_file_path, dest_file_path)
        copied_count += 1
        
    except FileNotFoundError:
        print(f"  [오류] 원본 파일을 찾을 수 없습니다: {source_file_path}")
        error_count += 1
    except Exception as e:
        print(f"  [오류] 파일 복사 중 문제 발생: {source_file_path}, {e}")
        error_count += 1

print("\n--- 11. '통잠' 데이터 저장 완료 ---")
print(f"✅ 총 {copied_count}개의 '통잠' 파일을 '{output_folder_full_sleep}' 폴더로 복사했습니다.")
if error_count > 0:
    print(f"❌ {error_count}개의 파일 복사에 실패했습니다.")
    
# 통잠 잔 날의 시간 추리기 -----------------------------------------------
# - 통잠 잔 날짜 데이터에 대해 잠에 든 시간 부터 깬 시간 데이터베이스화
# - 유저/날짜/온도/습도/조도/심장박동수/호흡량/수면깊이/수면시작시간/수면끝시간
# - 수면시작시간: 0이 아닌 값이 연속적으로 180분이 넘는 기간의 시작 시간
# - 수면끝시간: 0이 아닌 값이 연속적으로 180분이 넘는 기간의 끝 시간
base_folder_full_sleep = 'full_sleep_days_data'
sleep_records = []
MIN_CONSECUTIVE_SLEEP = 180

for dirpath, dirnames, filenames in os.walk(base_folder_full_sleep):
    for filename in filenames:
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(dirpath, filename)
        
        try:
            df = pd.read_csv(file_path)
            
            df_sleep_time = df[(df['time_hour'] >= 21) | (df['time_hour'] < 9)]
            
            # 180분 이상 연속 수면(0 아님) 구간 찾기
            series = df_sleep_time['sleep_depth']
            is_not_zero = (series != 0)
            group_ids = is_not_zero.diff().ne(0).cumsum()
            sleep_groups = group_ids[is_not_zero]
            run_lengths = sleep_groups.groupby(sleep_groups).size()
            
            # 180분 이상인 그룹(통잠)들의 ID 찾기
            full_sleep_group_ids = run_lengths[run_lengths >= MIN_CONSECUTIVE_SLEEP].index
            
            if len(full_sleep_group_ids) == 0:
                continue
                
            for group_id in full_sleep_group_ids:
                group_indices = sleep_groups[sleep_groups == group_id].index
                sleep_session_data = df_sleep_time.loc[group_indices]
                
                start_time = sleep_session_data.iloc[0]['datetime']
                end_time = sleep_session_data.iloc[-1]['datetime']
                
                sleep_records.append({
                    'csv_file_path': file_path,
                    'sleep_start_time': start_time, # 수면 시작 시간
                    'sleep_end_time': end_time, # 수면 끝 시간
                    'duration_minutes': len(sleep_session_data) # 수면 지속 시간 (분)
                })

        except Exception as e:
            print(f"  [오류] 파일 처리 실패: {file_path}, {e}")

df_full_sleep_records = pd.DataFrame(sleep_records)
print(f"총 {len(df_full_sleep_records)}개의 '통잠' 세션(180분 이상 연속 수면)이 발견되었습니다.")
# df_full_sleep_records.to_csv('C:\\Users\\user\\Desktop\\HRI\\full_sleep_time_record.csv', index=False)
