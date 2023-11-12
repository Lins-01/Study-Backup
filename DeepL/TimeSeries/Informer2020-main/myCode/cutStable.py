import pandas as pd
import matplotlib.pyplot as plt
# # 读取数据
# df = pd.read_csv(r"toDatatime.csv")
#
# # 计算vol_17010101010.tempf列的变化率（使用diff()函数计算与上一个数据点的差）
# df['change'] = df['vol_170101060.tempf'].diff().abs()
#
# # 定义稳态条件参数
# epsilon = 0.00001  # 定义变化率的阈值
# N = 10           # 定义连续多少个数据点低于阈值可以认为是稳态
#
# # 找到稳态开始的点
# # 我们假设最初的几个点不可能直接处于稳态
# for i in range(N, len(df)):
#     # 如果连续N个点的变化率都低于epsilon，我们认为找到了稳态开始的点
#     if all(df['change'][i-N:i] < epsilon):
#         break_point = i - N
#         break
# else:
#     break_point = len(df)  # 如果没找到稳态，保留全部数据
#
# # 截断数据至稳态之前的部分
# df_stable_before = df[:break_point]
#
# # 去掉 change列
# # 假设我们要删除名为'change'的列
# df_stable_before = df_stable_before.drop('change', axis=1)
#
#
# # 将截断后的数据保存到新的CSV文件
# df_stable_before.to_csv('truncated_dataset.csv', index=False)
# print(df.head(5))
# print('数据已截断!')

# 使用matplotlib进行可视化
df = pd.read_csv('truncated_dataset.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['vol_170101060.tempf'], label='vol_170101060.tempf')
plt.xlabel('Datetime')
plt.ylabel('vol_17010101010.tempf Value')
plt.title('Value Trend Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

temp_17006 = df['vol_170101060.tempf']
print(temp_17006.describe())