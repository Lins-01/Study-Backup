import numpy as np
import pandas as pd
from utils.timefeatures import time_features

df_raw = pd.read_csv('./data/ETT/ETTh1.csv')
cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]
df_stamp = df_raw[['date']][0:100]

print(cols_data)
print(df_data)
print(df_data[0:255])
print(df_stamp)

