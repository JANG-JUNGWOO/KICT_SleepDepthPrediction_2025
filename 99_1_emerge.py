import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

folder_name = 

df_env = pd.read_csv(f"{folder_name}/env_raw_100.csv", encoding='latin1')
df_radar = pd.read_csv(f"{folder_name}/radar_hb_rp_raw_100.csv", encoding='latin1')
df_lifelog = pd.read_csv(f"{folder_name}/lifelog_raw_100.csv", encoding='latin1')

dataframes = {'df_env': df_env, 'df_radar': df_radar, 'df_lifelog': df_lifelog }

date_columns_to_convert = {}
for name, df in dataframes.items():
    for col in df.columns:
        if df[col].dtype == 'object' and ('date' in col or 'created_at' in col):
            try:
                pd.to_datetime(df[col], errors='coerce')
                date_columns_to_convert.setdefault(name, []).append(col)
            except:
                pass

if date_columns_to_convert:
    print("Columns to Convert to Datetime:")
    for name, cols in date_columns_to_convert.items():
        pass

for name, cols in date_columns_to_convert.items():
    for col in cols:
        dataframes[name][col] = pd.to_datetime(dataframes[name][col], errors='coerce')
        
for name, df in dataframes.items():
    print("\nVisualizing Missing Values:")
    plt.figure(figsize=(5, 3))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(f'Missing Values Heatmap for {name}')
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()
    
# env 원본 파일을 변환 evn_v2------------------------------------------------
df_env_v2 = df_env.copy(deep=True) # 원본 env.csv 복사

temp_cols = [col for col in df_env_v2.columns if col.startswith('temp_')] # temp 00~23 열들 temp_col로 리스트화
df_temp_melted = pd.melt(df_env_v2, id_vars=['user_id', 'time_ymd'], value_vars=temp_cols, var_name='time_hour', value_name='temp') # temp 00~23 값 melt 함수로 재구조화

humi_cols = [col for col in df_env_v2.columns if col.startswith('humi_')]
df_humi_melted = pd.melt(df_env_v2, id_vars=['user_id', 'time_ymd'], value_vars=humi_cols, var_name='time_hour', value_name='humi')

illu_cols = [col for col in df_env_v2.columns if col.startswith('illu_')]
df_illu_melted = pd.melt(df_env_v2, id_vars=['user_id', 'time_ymd'], value_vars=illu_cols, var_name='time_hour', value_name='illu')

df_env_v2 = pd.merge(df_temp_melted, df_humi_melted, on=['user_id', 'time_ymd', 'time_hour'])
df_env_v2 = pd.merge(df_env_v2, df_illu_melted, on=['user_id', 'time_ymd', 'time_hour'])

df_env_v2['time_hour'] = df_env_v2['time_hour'].str.split('_').str[1].astype(int)

df_temp_melted['time_hour'] = df_temp_melted['time_hour'].str.replace('temp_', '')
df_humi_melted['time_hour'] = df_humi_melted['time_hour'].str.replace('humi_', '')
df_illu_melted['time_hour'] = df_illu_melted['time_hour'].str.replace('illu_', '')

df_env_v2 = pd.merge(df_temp_melted, df_humi_melted, on=['user_id', 'time_ymd', 'time_hour'])
df_env_v2 = pd.merge(df_env_v2, df_illu_melted, on=['user_id', 'time_ymd', 'time_hour'])

df_env_v2['time_hour'] = df_env_v2['time_hour'].astype(int)

df_env_v2 = df_env_v2.sort_values(by=['user_id', 'time_ymd', 'time_hour']).reset_index(drop=True)

print("Original env DataFrame:")
print(df_env.shape) # (4286, 74)
print(df_env.head(3))
print("Transformed env_v2 DataFrame:")
print(df_env_v2.shape) # (151248, 6)
print(df_env_v2.head(3)) # 시간단위로 나뉘어진 env파일 -> env_v2

# df_env_v2의 유저별 날짜 데이터 0값 확인 -----------------------------------------
zero_counts = (df_env_v2[['temp', 'humi', 'illu']] == 0).sum()

print("Number of zero values in selected columns of df_env_v2:")
print(zero_counts)

# env_v2 보간 -------------------------------------------------------
# 1. 날짜별로 시간 데이터가 잘 들어가 있는지 확인 (0~23시가 모두 있는지, 누락된 시간 확인)
# 2. 전체 데이터에서 0값 보간
#  - 연속 4개 이상 0값: Null 처리
#  - 연속 3개 이하 0값: 앞뒤 값으로 선형 보간
# 3. 시간단위 데이터를 분단위로 확장
# 4. 보간된 데이터를 유저별/날짜별로 저장

# 날짜별로 시간데이터(0~23시)가 모두 있는지 확인 -----------------------------
missing_hours_info = []
for user_id in df_env_v2['user_id'].unique():
    user_df = df_env_v2[df_env_v2['user_id'] == user_id]
    for date in user_df['time_ymd'].unique():
        day_df = user_df[user_df['time_ymd'] == date]
        hours = set(day_df['time_hour'])
        missing_hours = set(range(24)) - hours
        if missing_hours:
            missing_hours_info.append((user_id, date, sorted(list(missing_hours))))

if missing_hours_info:
    print("[경고] 일부 날짜에 누락된 시간(hour)이 있습니다:")
    for info in missing_hours_info:
        print(f"유저 {info[0]}, 날짜 {info[1]}: 누락된 시간 {info[2]}")
else:
    print("모든 유저/날짜에 대해 0~23시 데이터가 존재합니다.")
    
# 전체 데이터에서 temp, humi만 0값 보간, 인덱스 오류 수정 (연속 4개 이상 Null 처리, 3개 이하는 선형 보간)
def interpolate_env_data(df, value_cols=['temp', 'humi'], max_linear=3):
    df_result = []
    for user_id in df['user_id'].unique(): # 유저별로 반복
        user_df = df[df['user_id'] == user_id] 
        for date in user_df['time_ymd'].unique(): # 날짜별로 반복
            day_df = user_df[user_df['time_ymd'] == date].sort_values('time_hour').copy()
            for col in value_cols: # temp, humi 열에 대해
                vals = day_df[col].values
                mask = (vals == 0) # 0값 마스크
                i = 0
                while i < len(vals):
                    if mask[i]:
                        start = i
                        while i < len(vals) and mask[i]:
                            i += 1
                        end = i
                        run_length = end - start
                        idx_range = day_df.index[start:end]
                        if run_length >= 4: # 4개 이상 연속 0값은 Null 처리
                            day_df.loc[idx_range, col] = np.nan
                        elif run_length > 0: # 1~3개 연속 0값은 선형 보간
                            left = start - 1
                            right = end
                            if left >= 0 and right < len(vals): # 양쪽에 값이 있을 때만 보간
                                interp_vals = np.linspace(day_df.iloc[left][col], day_df.iloc[right][col], run_length+2)[1:-1]
                                day_df.loc[idx_range, col] = interp_vals
                            else:
                                day_df.loc[idx_range, col] = np.nan
                    else:
                        i += 1
            df_result.append(day_df)
    return pd.concat(df_result, ignore_index=True)

df_env_v2_interp = interpolate_env_data(df_env_v2)

null_counts = df_env_v2_interp[['temp', 'humi', 'illu']].isnull().sum()
print("Null 값 개수 (temp, humi, illu):")
print(null_counts)

print(df_env_v2_interp.shape)

# 시간단위 데이터를 분단위로 확장 ---------------------------------------------
# illu열은 9:00의 데이터로 9:00~9:59 데이터를 동일하게 보간
# temp, humi열은 9:00와 10:00의 앞뒤 데이터로 9:00~9:59 데이터를 선형 보간
# 만약, 9:00이 Null 값이라면, 확장되는 열들도 Null 처리하기
def expand_and_interpolate_group(group):
    group['datetime'] = pd.to_datetime(
        group['time_ymd'].astype(str) + ' ' + group['time_hour'].astype(str) + ':00:00', # time_ymd를 str로 변환
        errors='coerce'
    )
    group = group.dropna(subset=['datetime']) # NaT (잘못된 날짜 변환) 행 제거

    value_cols = ['temp', 'humi', 'illu']

    if group.duplicated(subset=['datetime']).any():
        agg_rules = {
            'temp': 'mean', 'humi': 'mean', 'illu': 'mean'
        }
        group = group.groupby('datetime').agg(agg_rules)
        group = group.sort_index()
    else:
        group = group.set_index('datetime')
        group = group[value_cols]
        group = group.sort_index()

    if group.empty:
        return pd.DataFrame()

    nan_times_temp = group[group['temp'].isnull()].index
    nan_times_humi = group[group['humi'].isnull()].index

    last_time = group.index.max()
    dummy_time = last_time + pd.DateOffset(hours=1)
    group.loc[dummy_time] = np.nan 

    group_resampled = group.resample('min').asfreq()

    group_resampled = group_resampled.iloc[:-1]

    group_resampled['illu'] = group_resampled.groupby(
        [group_resampled.index.date, group_resampled.index.hour]
    )['illu'].transform('ffill')

    group_resampled['temp'] = group_resampled['temp'].interpolate(method='linear')
    group_resampled['humi'] = group_resampled['humi'].interpolate(method='linear')

    for nan_time in nan_times_temp:
        hour_end = nan_time + pd.Timedelta(minutes=59)
        hour_end = min(hour_end, group_resampled.index.max()) 
        group_resampled.loc[nan_time:hour_end, 'temp'] = np.nan
        
    for nan_time in nan_times_humi:
        hour_end = nan_time + pd.Timedelta(minutes=59)
        hour_end = min(hour_end, group_resampled.index.max())
        group_resampled.loc[nan_time:hour_end, 'humi'] = np.nan

    return group_resampled

df_env_v3_list = df_env_v2_interp.groupby(['user_id', 'time_ymd']).apply(expand_and_interpolate_group)

df_env_v3 = df_env_v3_list
df_env_v3 = df_env_v3.reset_index()

if 'datetime' not in df_env_v3.columns:
    date_col_name = df_env_v3.columns[2] 
    print(f"   ... 'datetime' 컬럼명을 '{date_col_name}'에서 'datetime'으로 변경 ...")
    df_env_v3 = df_env_v3.rename(columns={date_col_name: 'datetime'})
    
df_env_v3['time_hour'] = df_env_v3['datetime'].dt.hour # 'datetime'에서 'time_hour' 추출
df_env_v3['time_minute'] = df_env_v3['datetime'].dt.minute # 'datetime'에서 'time_minute' 추출
df_env_v3['time_ymd'] = df_env_v3['time_ymd'].astype(str) # 타입을 str로 통일

final_cols = [ # 최종 사용할 컬럼 리스트 정의
    'user_id', 'time_ymd', 'time_hour', 'time_minute', 
    'temp', 'humi', 'illu', 'datetime'
]

df_env_v3 = df_env_v3[final_cols].copy()
df_env_v3['user_id'] = df_env_v3['user_id'].astype(int)

print(f"df_env_v3 Shape: {df_env_v3.shape}")
print(df_env_v3.head())

null_counts_minute = df_env_v3[['temp', 'humi','illu']].isnull().sum()
print("분 단위 DataFrame의 Null 값 개수 (temp, humi, illu):")
print(null_counts_minute)
print(df_env_v3.info()) # 메모리 사용량 확인
print(df_env_v3.shape)

# 보간된 데이터를 저장하기 전에 df_env_minute 검증 ------------------------------------------------
unique_users = df_env_v3['user_id'].unique()
num_unique_users = len(unique_users)
print(f"\n[검토 1] 총 {num_unique_users}명의 유저 데이터가 있습니다.")

day_counts_per_user = df_env_v3.groupby('user_id')['time_ymd'].nunique()
print("\n[검토 2] 유저별 보유 날짜 데이터 개수:")
print(day_counts_per_user.describe()) # 일수(day)의 (평균, 최소, 최대) 통계

df_review = pd.DataFrame(index=unique_users)
df_review.index.name = 'user_id'
df_review['day_count'] = day_counts_per_user # 2번 결과 저장

missing_days_info = []
for user_id in unique_users:
    # 해당 유저의 고유한 날짜(str) 목록을 가져와 datetime으로 변환 후 정렬
    user_dates = pd.to_datetime(df_env_v3[df_env_v3['user_id'] == user_id]['time_ymd'].unique()).sort_values()
    
    if len(user_dates) > 0:
        # 유저의 첫 날짜와 마지막 날짜
        first_date = user_dates.min()
        last_date = user_dates.max()
        
        # 총 기간 (일수)
        total_span_days = (last_date - first_date).days + 1
        
        # 실제 보유한 날짜 수
        actual_days = len(user_dates)
        
        # 누락된 날짜 수
        missing_days = total_span_days - actual_days
        
        # 리뷰 DataFrame에 저장
        df_review.loc[user_id, 'first_date'] = first_date
        df_review.loc[user_id, 'last_date'] = last_date
        df_review.loc[user_id, 'total_span_days'] = total_span_days
        df_review.loc[user_id, 'missing_days_count'] = missing_days
        df_review.loc[user_id, 'is_continuous'] = (missing_days == 0) # 누락일 = 0 이면 연속
    else:
        df_review.loc[user_id, 'missing_days_count'] = 0
        df_review.loc[user_id, 'is_continuous'] = False

continuous_users_count = df_review['is_continuous'].sum()
print(f"-> 총 {num_unique_users}명 중 {continuous_users_count}명이 데이터 누락일(gap)이 없습니다.")

key_cols = ['user_id', 'time_ymd', 'time_hour', 'time_minute']
duplicates_count = df_env_v3.duplicated(subset=key_cols).sum()
print(f"  - 분 단위 중복 행 개수: {duplicates_count} (0이 아니면 3단계 코드 오류)")

key_null_counts = df_env_v3[['user_id', 'time_ymd', 'time_hour', 'time_minute', 'datetime']].isnull().sum()
print(f"  - 주요 키 컬럼 Null 값 합계: {key_null_counts.sum()} (0이 아니면 3단계 코드 오류)")

data_null_counts = df_env_v3[['temp', 'humi', 'illu']].isnull().sum()
print("  - 데이터 컬럼 Null 값 (보간 결과):")
print(data_null_counts)
print(f"    -> (temp/humi/illu의 Null 개수가 60의 배수가 아니면 3단계 코드 오류)")

df_users_with_gaps = df_review[df_review['missing_days_count'] > 0]
print("\n[검토 5] 데이터 누락일(Gap)이 있는 유저 목록:")
print(df_users_with_gaps.sort_values(by='missing_days_count', ascending=False))

user_ids_with_gaps = df_users_with_gaps.index.tolist()
print(f"\n-> 누락일이 있는 유저 ID (총 {len(user_ids_with_gaps)}명):") # 10074 유저가 문제
# df_review.to_csv(f"{folder_name}/env_review.csv", encoding='latin1')

# 보간된 데이터를 유저별/날짜별로 저장 ---------------------------------------------
# 폴더 구조: env_by_user_daily / {user_id} / {user_id}_{prev_mmdd}_{next_mmdd}.csv
# 각 파일 내용: 전날 9:00 ~ 다음날 8:59 데이터
# 맨 앞 날짜: 당일 9:00 ~ 다음날 8:59 (9시 이전 제외)
# 맨 뒤 날짜: 전날 9:00 ~ 당일 8:59 (9시 이후 제외)
output_folder = os.path.join(folder_name, 'env_by_user_daily')
os.makedirs(output_folder, exist_ok=True)

try:
    df_env_v3_indexed = df_env_v3.set_index(
        pd.to_datetime(df_env_v3['datetime'])
    ).sort_index()
except Exception as e:
    print(f"오류: datetime 인덱스 설정 실패 - {e}")
    raise

total_files_saved = 0
total_files_skipped = 0

for user_id in df_env_v3_indexed['user_id'].unique():
    user_df = df_env_v3_indexed[df_env_v3_indexed['user_id'] == user_id]
    
    user_folder = os.path.join(output_folder, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    
    all_dates = pd.Series(pd.to_datetime(user_df['time_ymd']).unique()).sort_values()
    
    if len(all_dates) < 2:
        print(f"  [Skip] 유저 {user_id}: 파일 생성을 위한 최소 2일치 데이터가 없습니다.")
        continue

    start_date = all_dates.min() + pd.Timedelta(hours=9)
    end_date = all_dates.max() - pd.Timedelta(days=1)
    
    date_range_to_save = pd.date_range(start=start_date, end=end_date)

    for prev_day_dt in date_range_to_save:
        next_day_dt = prev_day_dt + pd.Timedelta(days=1)
        
        start_time_str = f"{prev_day_dt.strftime('%Y-%m-%d')} 09:00:00"
        end_time_str = f"{next_day_dt.strftime('%Y-%m-%d')} 08:59:59"

        prev_mmdd = prev_day_dt.strftime('%m%d')
        next_mmdd = next_day_dt.strftime('%m%d')
        filename = f"{user_id}_{prev_mmdd}_{next_mmdd}.csv"
        filepath = os.path.join(user_folder, filename)
        
        daily_data = user_df.loc[start_time_str:end_time_str]
        
        if len(daily_data) == 1440:
            daily_data.to_csv(filepath, index=False, encoding='utf-8-sig')
            total_files_saved += 1
        else:
            total_files_skipped += 1

print(f"총 {total_files_saved}개의 파일이 성공적으로 저장되었습니다.")
print(f"총 {total_files_skipped}개의 파일이 (데이터 누락으로) 건너뛰었습니다.")

# 저장된 각 유저 폴더의 모든 파일에 대해 데이터 길이 확인 -----------------------------
problem_files = [] # (파일명, 실제 행 수)를 저장
total_files_checked = 0

for dirpath, dirnames, filenames in os.walk(output_folder):
    
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                total_files_checked += 1
                
                if row_count != 1440:
                    problem_files.append((file_path, row_count))
                    
            except pd.errors.EmptyDataError:

                print(f"  [오류] 파일이 비어있습니다: {file_path}")
                problem_files.append((file_path, 0))
            except Exception as e:
                print(f"  [오류] 파일을 읽는 중 문제 발생 ({file_path}): {e}")
                problem_files.append((file_path, -1))

print("\n--- 6. 파일 무결성 검사 완료 ---")
print(f"총 {total_files_checked}개의 .csv 파일을 검사했습니다.")

if not problem_files:
    print("✅ [성공] 모든 파일이 정확히 1440개의 데이터 행을 가지고 있습니다.")
else:
    print(f"❌ [실패] {len(problem_files)}개의 파일에서 문제가 발견되었습니다.")
    print("--------------------------------------------------")
    for f_path, count in problem_files:
        print(f"  - 파일: {f_path}")
        print(f"    -> 행 수: {count} (1440이 아님)")
    print("--------------------------------------------------")

# 모든 저장된 파일에서 temp, humi의 0값이 없는지 확인하고, null 값이 포함된 파일 리스트 출력
files_with_zeros = [] # 0값이 발견된 파일 (있으면 안 됨)
files_with_nulls = [] # Null값이 발견된 파일 (있을 수 있음)
total_files_checked = 0

for dirpath, dirnames, filenames in os.walk(output_folder):
    
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            total_files_checked += 1
            
            try:
                df = pd.read_csv(file_path)
                
                if (df['temp'] == 0).any() or (df['humi'] == 0).any():
                    files_with_zeros.append(file_path)
                    
                if df['temp'].isnull().any() or \
                   df['humi'].isnull().any() or \
                   df['illu'].isnull().any():
                    files_with_nulls.append(file_path)
                        
            except Exception as e:
                print(f"  [오류] 파일 읽기 실패: {file_path}, {e}")

print(f"총 {total_files_checked}개의 .csv 파일을 검사했습니다.")

if not files_with_zeros:
    print("✅ [성공] 모든 파일의 'temp', 'humi' 컬럼에서 0값이 발견되지 않았습니다.")
else:
    print(f"❌ [오류] {len(files_with_zeros)}개의 파일에서 'temp' 또는 'humi'에 0값이 발견되었습니다!")
    print("  (2단계 0값 보간 로직이 잘못되었을 수 있습니다.)")
    print("  --- (0값 포함 파일 리스트) ---")
    for f_path in files_with_zeros:
        print(f"  - {f_path}")
            
# 2. Null 값 결과 보고
if not files_with_nulls:
    print("ℹ️ [정보] Null 값을 포함한 파일이 없습니다.")
else:
    print(f"ℹ️ [정보] 총 {len(files_with_nulls)}개의 파일이 Null 값을 포함하고 있습니다.")
    print("  (3단계 보간 시 원본 Null 또는 4개 이상 연속 0값으로 인해 정상적으로 생성된 Null입니다.)")
    
    # Null 리스트가 너무 길면 터미널이 멈출 수 있으므로, 상위 10개만 예시로 출력
    print("  --- (Null 포함 파일 리스트) ---")
    for f_path in files_with_nulls[:10]:
        print(f"  - {f_path}")
    if len(files_with_nulls) > 10:
        print(f"  ... 외 {len(files_with_nulls) - 10}개의 파일에 Null이 더 있습니다.")
        
# radar 원본 파일을 변환 radar_v2------------------------------------------------
# 1. 가로축으로 되어 있는 hb1,2,3 이런 원본 데이터를 새로로 배열
# 2. Null 값이 있는지 확인
# 3. 띄엄띄엄 있는 행들 사이에 연속되는 분단위 데이터를 Null 값으로 모두 보간
# 4. 앞서 env_by_user_daily 처럼 저장
print(df_radar)

df_radar_transformed = df_radar.copy()

df_radar_transformed['created_at'] = pd.to_datetime(df_radar_transformed['created_at']).dt.floor('min')

df_radar_transformed = df_radar_transformed.drop(columns=['time_ymd', 'time_ym', 'id'])

df_hb_melted = pd.melt(df_radar_transformed,
                       id_vars=['user_id', 'created_at'],
                       value_vars=[f'hb{i}' for i in range(1, 6)],
                       var_name='hb_minute', value_name='hb')

df_rp_melted = pd.melt(df_radar_transformed,
                       id_vars=['user_id', 'created_at'],
                       value_vars=[f'rp{i}' for i in range(1, 6)],
                       var_name='rp_minute', value_name='rp')

all_melted = [df_hb_melted, df_rp_melted]

for df in all_melted:
    type_and_minute = df.columns[-2] # (e.g., 'hb_minute')
    df_type = type_and_minute.split('_')[0] # (e.g., 'hb')
    df['minute'] = df[type_and_minute].str.replace(df_type, '').astype(int)
    df['datetime'] = df['created_at'] - pd.to_timedelta(5 - df['minute'], unit='m')
    df.rename(columns={df.columns[-3]: 'value'}, inplace=True)
    df['type'] = df_type
    df.drop(columns=[type_and_minute, 'minute', 'created_at'], inplace=True)

df_concat = pd.concat(all_melted)

df_radar_v2 = df_concat.pivot_table(index=['user_id', 'datetime'],
                                  columns='type',
                                  values='value').reset_index()

df_radar_v2.columns.name = None

df_radar_v2 = df_radar_v2.sort_values(by=['user_id', 'datetime']).reset_index(drop=True)
df_radar_v2['date'] = df_radar_v2['datetime'].dt.date
df_radar_v2['time'] = df_radar_v2['datetime'].dt.strftime('%H:%M')

df_radar_v2 = df_radar_v2[['user_id', 'date', 'time', 'hb', 'rp', 'datetime']]

print(df_radar_v2.head(50))
print(df_radar_v2.shape)

null_counts = df_radar_v2.isnull().sum()
print("컬럼별 Null 값 개수:")
print(null_counts)

print("\n데이터 타입 (info):")
df_radar_v2.info()

zero_counts = (df_radar_v2[['hb', 'rp']] == 0).sum()
print("\n데이터 0값 개수 (참고):")
print(zero_counts)

# 띄엄띄엄 있는 분 단위 Null 값으로 채우기 ---
def fill_missing_minutes(group):
    """
    유저별/날짜별(group)로 00:00~23:59 (1440분) 템플릿을 만들어
    띄엄띄엄 있는 데이터를 재배치(reindex)합니다.
    빈 시간은 자동으로 NaN(Null)으로 채워집니다.
    """
    current_date = pd.to_datetime(group['date'].iloc[0])
    full_day_index = pd.date_range(start=current_date, periods=1440, freq='min')
    group = group.set_index('datetime')
    group_resampled = group.reindex(full_day_index)
    
    group_resampled['user_id'] = group['user_id'].iloc[0] # 그룹의 user_id로 채움
    group_resampled['date'] = current_date.date()
    group_resampled['time'] = group_resampled.index.strftime('%H:%M')
    
    group_resampled['datetime'] = group_resampled.index
    
    return group_resampled

df_radar_v3_list = df_radar_v2.groupby(['user_id', 'date']).apply(fill_missing_minutes)
df_radar_v3 = df_radar_v3_list.reset_index(drop=True)
df_radar_v3 = df_radar_v3.sort_values(by=['user_id', 'datetime']).reset_index(drop=True)

print(f"df_radar_v3 Shape: {df_radar_v3.shape}")

print("\n[보간 결과 확인 (Head 10)]")
print(df_radar_v3.head(10))

print("\n[보간 결과 확인 (Null 값 개수)]")
print(df_radar_v3.isnull().sum()) # hb, rp, mi의 Null 값 개수가 크게 증가해야 정상

print("\n[보간 결과 확인 (Info)]")
df_radar_v3.info()

# df_radar_v3 데이터 검토 ---------------------------------------------
unique_users_radar = df_radar_v3['user_id'].unique()
print(f"[검토 1] 총 {len(unique_users_radar)}명의 유저 데이터가 있습니다.")

user_day_count = df_radar_v3.groupby(['user_id', 'date']).ngroups
expected_rows = user_day_count * 1440 # 1440분
print(f"[검토 2] (유저*날짜) 고유 쌍: {user_day_count}개")
print(f"[검토 2] 예상 행 수: {user_day_count} * 1440 = {expected_rows}")
print(f"[검토 2] 실제 행 수: {len(df_radar_v3)}")
if expected_rows == len(df_radar_v3):
    print("  -> ✅ 행 수 일치. 모든 (유저*날짜) 쌍이 1440분으로 확장되었습니다.")
else:
    print("  -> ❌ 오류: 행 수가 1440의 배수가 아닙니다! (2단계 코드 오류)")

key_null_counts = df_radar_v3[['user_id', 'date', 'time', 'datetime']].isnull().sum().sum()
print(f"[검토 3] 주요 키 컬럼 Null 값 합계: {key_null_counts} (0이어야 함)")

data_null_counts = df_radar_v3[['hb', 'rp']].isnull().sum()
print("[검토 3] 데이터 컬럼 Null 값 (예상대로 많아야 함):")
print(data_null_counts)

duplicates_count = df_radar_v3.duplicated(subset=['user_id', 'datetime']).sum()
print(f"[검토 4] 분 단위 중복 행 개수: {duplicates_count} (0이어야 함)")

# radar_v3 보간된 데이터를 유저별/날짜별로 저장 ---------------------------------------------
output_folder_radar = 'radar_by_user_daily'
os.makedirs(output_folder_radar, exist_ok=True)

try:
    df_radar_v3_indexed = df_radar_v3.set_index(
        pd.to_datetime(df_radar_v3['datetime'])
    ).sort_index()
except Exception as e:
    print(f"오류: datetime 인덱스 설정 실패 - {e}")
    raise

total_files_saved_radar = 0
total_files_skipped_radar = 0

for user_id in df_radar_v3_indexed['user_id'].unique():
    user_df = df_radar_v3_indexed[df_radar_v3_indexed['user_id'] == user_id]

    user_folder = os.path.join(output_folder_radar, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    
    all_dates = pd.Series(pd.to_datetime(user_df['date']).unique()).sort_values()
    
    if len(all_dates) < 2:
        continue
        
    start_date = all_dates.min() 
    end_date = all_dates.max() - pd.Timedelta(days=1)
    
    date_range_to_save = pd.date_range(start=start_date, end=end_date)
    
    for prev_day_dt in date_range_to_save:
        next_day_dt = prev_day_dt + pd.Timedelta(days=1)

        start_time_str = f"{prev_day_dt.strftime('%Y-%m-%d')} 09:00:00"
        end_time_str = f"{next_day_dt.strftime('%Y-%m-%d')} 08:59:59"
        
        prev_mmdd = prev_day_dt.strftime('%m%d')
        next_mmdd = next_day_dt.strftime('%m%d')
        filename = f"{user_id}_{prev_mmdd}_{next_mmdd}.csv"
        filepath = os.path.join(user_folder, filename)
        
        daily_data = user_df.loc[start_time_str:end_time_str]
        
        if len(daily_data) == 1440:
            daily_data.to_csv(filepath, index=False, encoding='utf-8-sig')
            total_files_saved_radar += 1
        else:
            total_files_skipped_radar += 1

print(f"총 {total_files_saved_radar}개의 (Radar) 파일이 성공적으로 저장되었습니다.")
print(f"총 {total_files_skipped_radar}개의 (Radar) 파일이 (데이터 누락으로) 건너뛰었습니다.")

# 각 파일의 행 길이가 1440인지 확인 (radar) -----------------------------
problem_files_radar = [] # (파일명, 실제 행 수)를 저장 
total_files_checked_radar = 0

for dirpath, dirnames, filenames in os.walk(output_folder_radar):
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                total_files_checked_radar += 1
                
                if row_count != 1440:
                    problem_files_radar.append((file_path, row_count))
                    
            except pd.errors.EmptyDataError:

                print(f"  [오류] 파일이 비어있습니다: {file_path}")
                problem_files_radar.append((file_path, 0))
            except Exception as e:
                print(f"  [오류] 파일을 읽는 중 문제 발생 ({file_path}): {e}")
                problem_files_radar.append((file_path, -1))
print("\n--- (Radar) 파일 무결성 검사 완료 ---")
print(f"총 {total_files_checked_radar}개의 .csv 파일을 검사했습니다.")

# [Env + Radar] 데이터 병합 및 저장 ---------------------------------------------
folder_env = 'env_by_user_daily'
folder_radar = 'radar_by_user_daily'
output_folder_merged = 'env_radar_merged'

os.makedirs(output_folder_merged, exist_ok=True)

total_files_merged = 0
total_files_skipped = 0

final_columns_order = [
    # --- 1. 키 컬럼 ---
    'user_id', 
    'datetime', 
    'time_ymd', 
    'time_hour', 
    'time_minute',
    # --- 2. Env 데이터 ---
    'temp', 
    'humi', 
    'illu',
    # --- 3. Radar 데이터 ---
    'hb', 
    'rp'
]

for dirpath, dirnames, filenames in os.walk(folder_env):
    if dirpath == folder_env:
        continue
    user_id = os.path.basename(dirpath)
    
    output_user_folder = os.path.join(output_folder_merged, user_id)
    os.makedirs(output_user_folder, exist_ok=True)

    for filename in filenames:
        if not filename.endswith('.csv'):
            continue
        env_file_path = os.path.join(dirpath, filename)
        radar_file_path = os.path.join(folder_radar, user_id, filename)
        
        if os.path.exists(radar_file_path):
            try:
                df_env = pd.read_csv(env_file_path)
                df_radar = pd.read_csv(radar_file_path)
                df_merged = pd.merge(
                    df_env,
                    df_radar[['user_id', 'datetime', 'hb', 'rp']],
                    on=['user_id', 'datetime'],
                    how='inner' # 두 파일에 모두 존재하는 행만 병합 (1440행)
                )
                df_merged = df_merged[final_columns_order]
                
                output_file_path = os.path.join(output_user_folder, filename)
                df_merged.to_csv(output_file_path, index=False, encoding='utf-8-sig')
                
                total_files_merged += 1
                
            except Exception as e:
                print(f"  [오류] 파일 병합/저장 실패: {filename}, {e}")
        else:
            total_files_skipped += 1

print("\n--- 8. 병합 완료 ---")
print(f"✅ 총 {total_files_merged}개의 [Env+Radar] 파일이 성공적으로 병합/저장되었습니다.")
print(f"ℹ️ {total_files_skipped}개의 Env 파일이 (짝이 되는 Radar 파일이 없어) 건너뛰었습니다.")

# 각 파일의 행 길이가 1440인지 확인 + 열 개수 가 같은지도 확인 (env_radar_merged) -----------------------------
problem_files_radar = [] # (파일명, 실제 행 수)를 저장 
total_files_checked_radar = 0

for dirpath, dirnames, filenames in os.walk(output_folder_merged):
    if dirpath == output_folder_merged:
        continue
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                total_files_checked_radar += 1
                
                if row_count != 1440:
                    problem_files_radar.append((file_path, row_count))
                    
            except pd.errors.EmptyDataError:

                print(f"  [오류] 파일이 비어있습니다: {file_path}")
                problem_files_radar.append((file_path, 0))
            except Exception as e:
                print(f"  [오류] 파일을 읽는 중 문제 발생 ({file_path}): {e}")
                problem_files_radar.append((file_path, -1))
print("\n--- (Merged) 파일 무결성 검사 완료 ---")
print(f"총 {total_files_checked_radar}개의 .csv 파일을 검사했습니다.")

# lifelog 원본 파일을 변환 lifelog_v2------------------------------------------------
print(df_lifelog.head(3))
len(df_lifelog['sleep_depth_1_list'][10]) # 수면 깊이 # 04-가장 숙면

df_lifelog = df_lifelog[['care_recipient_id', 'sleep_depth_1_list', 'lifelog_date']].copy()
df_lifelog.rename(columns={'care_recipient_id': 'user_id'}, inplace=True)
df_lifelog['lifelog_date'] = pd.to_datetime(df_lifelog['lifelog_date'], format='%Y%m%d')
df_lifelog['lifelog_date'] = df_lifelog['lifelog_date'].dt.date
print(df_lifelog.head()) # 쓸것만 남김

# df_lifelog의 sleep_depth_1_list 열에 Null 값이 있는지 확인 후 Null 값 행 제거
df_lifelog['sleep_depth_1_list'].isnull().sum() # 31개

df_lifelog = df_lifelog.dropna(subset=['sleep_depth_1_list']).reset_index(drop=True)
df_lifelog['sleep_depth_1_list'].isnull().sum() # 0개

# df_lifelog['sleep_depth_1_list'] 의 값 중 길이가 다른 행 표기
lengths = df_lifelog['sleep_depth_1_list'].astype(str).str.len()
unique_lengths = lengths.unique()
print(f"sleep_depth_1_list 열의 고유한 문자열 길이: {unique_lengths}")
# 길이가 2880이 아닌 행 7개 검거
for length in unique_lengths:
    if length != 2880:
        count = (lengths == length).sum()
        print(f"길이 {length}인 행 개수: {count}")
        
# 길이가 2880이 아닌 행 제거
df_lifelog = df_lifelog[lengths == 2880].reset_index(drop=True)
print(f"길이 2880인 행만 남긴 후 데이터프레임 크기: {df_lifelog.shape}")

def split_sleep_depth(df, column_name):
    new_df = df.copy()
    new_df[column_name] = new_df[column_name].astype(str)

    string_length = new_df[column_name].str.len().max()
    num_columns = string_length // 2

    for i in range(num_columns):
        new_col_name = f'sleep_{i+1:02d}'
        new_df[new_col_name] = new_df[column_name].str[i*2:(i+1)*2]

    for i in range(num_columns):
        new_col_name = f'sleep_{i+1:02d}'
        new_df[new_col_name] = pd.to_numeric(new_df[new_col_name], errors='coerce')

    new_df = new_df.drop(columns=[column_name])
    return new_df

df_lifelog_split = split_sleep_depth(df_lifelog, 'sleep_depth_1_list')
print(df_lifelog_split.head())
df_lifelog_split.info()

sleep_cols = [col for col in df_lifelog_split.columns if col.startswith('sleep_')]

df_lifelog_melted = pd.melt(df_lifelog_split,
                            id_vars=['user_id', 'lifelog_date'],
                            value_vars=sleep_cols,
                            var_name='sleep_interval',
                            value_name='sleep_depth')

df_lifelog_melted['interval'] = df_lifelog_melted['sleep_interval'].str.replace('sleep_', '').astype(int)
df_lifelog_melted['minute'] = df_lifelog_melted['interval'] - 1
df_lifelog_melted['hour'] = (df_lifelog_melted['minute'] // 60) % 24

df_lifelog_v2 = df_lifelog_melted[['user_id', 'lifelog_date', 'minute', 'hour', 'sleep_depth']].copy()
df_lifelog_v2 = df_lifelog_v2.sort_values(by=['user_id', 'lifelog_date', 'minute']).reset_index(drop=True)

print(df_lifelog_v2.head())
df_lifelog_v2.shape

print("컬럼별 Null 값 개수:")
null_counts = df_lifelog_v2.isnull().sum()
print(null_counts) # Null 값 없음

print(df_lifelog_v2)
# lifelog 보간된 데이터를 유저별/날짜별로 저장 ---------------------------------------------
df_lifelog_v3 = df_lifelog_v2.copy()
df_lifelog_v3['datetime'] = pd.to_datetime(df_lifelog_v3['lifelog_date']) + \
                            pd.to_timedelta(df_lifelog_v3['minute'], unit='m')

try:
    df_lifelog_v3_indexed = df_lifelog_v3.set_index('datetime').sort_index()
except Exception as e:
    print(f"오류: datetime 인덱스 설정 실패 - {e}")
    raise

output_folder_lifelog = 'lifelog_by_user_daily' 
os.makedirs(output_folder_lifelog, exist_ok=True) 

total_files_saved_lifelog = 0
total_files_skipped_lifelog = 0

final_cols = ['user_id', 'date', 'datetime', 'hour', 'minute', 'sleep_depth']

for user_id in df_lifelog_v3_indexed['user_id'].unique():
    user_df = df_lifelog_v3_indexed[df_lifelog_v3_indexed['user_id'] == user_id]
    
    user_folder = os.path.join(output_folder_lifelog, str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    
    # 'lifelog_date' (date 객체)을 'date'로 이름을 변경하여 사용
    user_df = user_df.rename(columns={'lifelog_date': 'date'})
    all_dates = pd.Series(pd.to_datetime(user_df['date']).unique()).sort_values()
    
    if len(all_dates) < 2:
        continue
        
    start_date = all_dates.min() 
    end_date = all_dates.max() - pd.Timedelta(days=1)
    
    date_range_to_save = pd.date_range(start=start_date, end=end_date)
    
    for prev_day_dt in date_range_to_save:
        next_day_dt = prev_day_dt + pd.Timedelta(days=1)
        
        start_time_str = f"{prev_day_dt.strftime('%Y-%m-%d')} 09:00:00"
        end_time_str = f"{next_day_dt.strftime('%Y-%m-%d')} 08:59:59"
        
        prev_mmdd = prev_day_dt.strftime('%m%d')
        next_mmdd = next_day_dt.strftime('%m%d')
        filename = f"{user_id}_{prev_mmdd}_{next_mmdd}.csv"
        filepath = os.path.join(user_folder, filename)
        
        daily_data = user_df.loc[start_time_str:end_time_str]
        
        if len(daily_data) == 1440:
            daily_data = daily_data.reset_index() 

            daily_data = daily_data[final_cols]
            daily_data.to_csv(filepath, index=False, encoding='utf-8-sig')
            total_files_saved_lifelog += 1
        else:
            total_files_skipped_lifelog += 1

print(f"총 {total_files_saved_lifelog}개의 (Lifelog) 파일이 성공적으로 저장되었습니다.")
print(f"총 {total_files_skipped_lifelog}개의 (Lifelog) 파일이 (데이터 누락으로) 건너뛰었습니다.")

# 각 파일의 행 길이가 1440인지 확인 (lifelog) -----------------------------
problem_files_lifelog = [] # (파일명, 실제 행 수)를 저장
total_files_checked_lifelog = 0

for dirpath, dirnames, filenames in os.walk(output_folder_lifelog):
    if dirpath == output_folder_lifelog:
        continue
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                total_files_checked_lifelog += 1
                
                if row_count != 1440:
                    problem_files_lifelog.append((file_path, row_count))
                    
            except pd.errors.EmptyDataError:

                print(f"  [오류] 파일이 비어있습니다: {file_path}")
                problem_files_lifelog.append((file_path, 0))
            except Exception as e:
                print(f"  [오류] 파일을 읽는 중 문제 발생 ({file_path}): {e}")
                problem_files_lifelog.append((file_path, -1))
                
print("\n--- (Lifelog) 파일 무결성 검사 완료 ---")
print(f"총 {total_files_checked_lifelog}개의 .csv 파일을 검사했습니다.")
if not problem_files_lifelog:
    print("✅ [성공] 모든 파일이 정확히 1440개의 데이터 행을 가지고 있습니다.")
else:
    print(f"❌ [실패] {len(problem_files_lifelog)}개의 파일에서 문제가 발견되었습니다.")
    print("--------------------------------------------------")
    for f_path, count in problem_files_lifelog:
        print(f"  - 파일: {f_path}")
        print(f"    -> 행 수: {count} (1440이 아님)")
    print("--------------------------------------------------")

# env+radar+lifelog 병합 및 저장 ---------------------------------------------
folder_env_radar = 'env_radar_merged'
folder_lifelog = 'lifelog_by_user_daily'
env_radar_lifelog_merged = 'env_radar_lifelog_merged'
os.makedirs(env_radar_lifelog_merged, exist_ok=True)
total_files_merged_all = 0

env_radar_lifelog_columns_order = [
    # --- 1. 키 컬럼 ---
    'user_id', 
    'datetime', 
    'time_ymd', 
    'time_hour', 
    'time_minute',
    # --- 2. Env 데이터 ---
    'temp', 
    'humi', 
    'illu',
    # --- 3. Radar 데이터 ---
    'hb', 
    'rp',
    # --- 4. Lifelog 데이터 ---
    'sleep_depth'
]

for dirpath, dirnames, filenames in os.walk(folder_env_radar):
    if dirpath == folder_env_radar:
        continue
    user_id = os.path.basename(dirpath)
    
    output_user_folder = os.path.join(env_radar_lifelog_merged, user_id)
    os.makedirs(output_user_folder, exist_ok=True)

    for filename in filenames:
        if not filename.endswith('.csv'):
            continue
        env_radar_file_path = os.path.join(dirpath, filename)
        lifelog_file_path = os.path.join(folder_lifelog, user_id, filename)
        
        if os.path.exists(lifelog_file_path):
            try:
                df_env_radar = pd.read_csv(env_radar_file_path)
                df_lifelog = pd.read_csv(lifelog_file_path)

                df_env_radar['datetime'] = pd.to_datetime(df_env_radar['datetime'])
                df_lifelog['datetime'] = pd.to_datetime(df_lifelog['datetime'])
                
                df_merged_all = pd.merge(
                    df_env_radar,
                    df_lifelog[['user_id', 'datetime', 'sleep_depth']],
                    on=['user_id', 'datetime'],
                    how='inner' 
                )

                df_merged_all = df_merged_all[env_radar_lifelog_columns_order]
                
                output_file_path = os.path.join(output_user_folder, filename)
                df_merged_all.to_csv(output_file_path, index=False, encoding='utf-8-sig')
                
                total_files_merged_all += 1
                
            except Exception as e:
                print(f"  [오류] 파일 병합/저장 실패: {filename}, {e}")
        else:
            continue

print(f"✅ 총 {total_files_merged_all}개의 [Env+Radar+Lifelog] 파일이 성공적으로 병합/저장되었습니다.")

# 각 파일의 행 길이가 1440인지 확인 (env_radar_lifelog_merged) -----------------------------
problem_files_env_radar_lifelog = [] # (파일명, 실제 행 수)를 저장
total_files_checked_env_radar_lifelog = 0   

for dirpath, dirnames, filenames in os.walk(env_radar_lifelog_merged):
    if dirpath == env_radar_lifelog_merged:
        continue
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)
            
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                total_files_checked_env_radar_lifelog += 1
                
                if row_count != 1440:
                    problem_files_env_radar_lifelog.append((file_path, row_count))
                    
            except pd.errors.EmptyDataError:

                print(f"  [오류] 파일이 비어있습니다: {file_path}")
                problem_files_env_radar_lifelog.append((file_path, 0))
            except Exception as e:
                print(f"  [오류] 파일을 읽는 중 문제 발생 ({file_path}): {e}")
                problem_files_env_radar_lifelog.append((file_path, -1))
print("\n--- (env_radar_lifelog Merged) 파일 무결성 검사 완료 ---")
print(f"총 {total_files_checked_env_radar_lifelog}개의 .csv 파일을 검사했습니다.")
if not problem_files_env_radar_lifelog:
    print("✅ [성공] 모든 파일이 정확히 1440개의 데이터 행을 가지고 있습니다.")
