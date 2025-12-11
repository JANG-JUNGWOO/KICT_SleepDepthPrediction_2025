import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv('all_full_sleep_data_v3.csv', encoding = 'utf-8')
print(f'Shape of data: {data.shape}') # (3915360, 11)
print(data.info())
print('\nDescriptive Statistics:')
print(data.describe())

# HR, RP 보간(현재 Null값 처리)
zero_counts = (data[['hb', 'rp']] == 0).sum()
print("\n[hb, rp 컬럼 0 값 현황]")
print(zero_counts)

null_counts = data[['hb', 'rp']].isnull().sum()
print("\n[hb, rp 컬럼 Null 값 현황]")
print(null_counts)

total_rows = len(data)
null_percentages = (null_counts / total_rows) * 100
print("\n[hb, rp 컬럼 Null 값 비율]")
print(null_percentages.round(2).astype(str) + ' %')

# 0 값 Null로 보간 -----------------------------------------------------
zero_counts = (data[['hb', 'rp']]==0).sum()
print("\n[hb, rp 컬럼 0 값 현황]")
print(zero_counts)

data[['hb', 'rp']] = data[['hb', 'rp']].replace(0, np.nan)
'''
[hb, rp 컬럼 0 값 현황]
hb    603313      1843538
rp    722540      2274792
'''
data.shape
data.to_csv('all_full_sleep_data_0toNull_v3.csv', index=False, encoding='utf-8-sig')

# Null 값 분포 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(data[['hb', 'rp']].head(100000).isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values in hb & rp (Before Interpolation)')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()

# Null 값 행들 선형 보간 -----------------------------------------------------
data[['hb', 'rp']] = data[['hb', 'rp']].interpolate(method='linear', limit_direction='both')
print("\n[보간 후 Null 값 현황]")
final_null_counts = data[['hb', 'rp']].isnull().sum()
print(final_null_counts)

if final_null_counts.sum() == 0:
    print("✅ 'hb', 'rp' 컬럼의 Null 값을 모두 제거했습니다.")
else:
    print("⚠️ 일부 Null 값이 남아있습니다. 전체 컬럼이 Null인 경우일 수 있습니다.")

# data.to_csv('all_full_sleep_data_itp_linear.csv', index=False, encoding='utf-8-sig')


# Null 값 행들 이전값 보간 -----------------------------------------------------
data[['hb', 'rp']] = data[['hb', 'rp']].ffill()
final_null_counts = data[['hb', 'rp']].isnull().sum()
print(final_null_counts)

if final_null_counts.sum() == 0:
    print("✅ 'hb', 'rp' 컬럼의 Null 값을 모두 제거했습니다.")
else:
    print("⚠️ 일부 Null 값이 남아있습니다. 데이터 시작 부분에 Null이 있는 경우일 수 있습니다.")
    
# data.to_csv('all_full_sleep_data_itp_former.csv', index=False, encoding='utf-8-sig')

