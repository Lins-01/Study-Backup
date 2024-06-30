import pandas as pd
import os
from scipy.ndimage import uniform_filter1d  # 用于平滑的函数


window_size = 15
# 定义平滑函数
def smooth_column(data, window_size=window_size):
    # 数据量不变，只是平滑处理，用周围数据的平均值代替原数据
    return uniform_filter1d(data, size=window_size)

# 读取并平滑处理目录中的所有CSV文件
def process_csv_files(directory,window_size=window_size):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 处理每一列除了'date'
            for column in df.columns:
                # if column != 'date':
                #     df[column] = smooth_column(df[column], window_size)
                df[column] = smooth_column(df[column], window_size)

            # 保存平滑后的数据到新文件
            smoothed_file_path = file_path.replace('.csv', '_smoothed.csv')
            df.to_csv(smoothed_file_path, index=False)
    return smoothed_file_path

# 调用函数处理指定目录
# 替换此处的'data_directory'为包含你的CSV文件的目录的路径
process_csv_files('./datasets/lins')

