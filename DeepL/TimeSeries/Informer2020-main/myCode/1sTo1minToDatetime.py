import pandas as pd
import matplotlib.pyplot as plt

# 读取csv文件
df = pd.read_csv(r"E:\Document\CodeSpace\Study\DeepL\TimeSeries\Informer2020-main\data\mydata\POWER0.3-0.4_LOOP.csv")
print(df.head(20))
# # 确保日期列是datetime格式
df['date'] = pd.to_datetime('2023-11-03') + pd.to_timedelta(df['date'], unit='s')
# # 将日期列设置为索引
df.set_index('date',inplace=True)
print(df.head(100))
df.to_csv('toDatatime.csv')
# # 重采样为1分钟一次，并使用均值聚合
# df_resampled = df.resample('60s').max()
# print(df_resampled.head(20))
# # 保存到新的csv文件
# # df_resampled.to_csv('resampled_file_60s_max.csv')
# print(df_resampled.describe())

# 查看变化曲线

# 使用matplotlib进行可视化
df = pd.read_csv('resampled_file_60s_max.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['vol_170101010.tempf'], label='vol_17010101010.tempf')
plt.xlabel('Datetime')
plt.ylabel('vol_17010101010.tempf Value')
plt.title('Value Trend Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()