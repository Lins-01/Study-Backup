import pandas as pd

# 读取CSV文件
csv_file_path = './origin.csv'
data = pd.read_csv(csv_file_path)

# 每隔4个数据采样一次
sampled_values = data['value'][::4]


sampled_values.to_csv('value.csv', index=False) 
# 打印采样结果
print("Sampled Values:")
print(sampled_values)

