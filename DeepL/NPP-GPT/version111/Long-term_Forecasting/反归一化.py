import pandas as pd
import numpy as np
import pickle

# 加载scaler对象
with open('scaler_test.pkl', 'rb') as f:
    scaler = pickle.load(f)

file = r"min_true.csv"
data = pd.read_csv(file)
data = np.array(data)
# 创建一个与scaler.scale_形状相同的全零数组，并将data[:, 0]赋值给该数组的第一个元素
data_expanded = np.zeros((data.shape[0], scaler.scale_.shape[0]))
data_expanded[:, 0] = data[:, 0]

# 进行反归一化
data_inverse = scaler.inverse_transform(data_expanded)

# 将反归一化后的结果转换为Pandas DataFrame
data_df = pd.DataFrame(data_inverse[:, 0])

# 将DataFrame保存为CSV文件
file_name = file+'_inverse.csv'
data_df.to_csv(file_name, index=False)

print("inverse is done")
