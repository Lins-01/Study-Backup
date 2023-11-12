import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r'E:\Document\CodeSpace\Data_set\CASHIPS-data\r51\Trend\POWER0.3-0.4_LOOP1.csv')

selected_columns = df[
    ['vol_170101060.tempf', 'junc_170103064.mflowj']]
print(selected_columns.describe())
print(selected_columns.head())


df2 = pd.read_csv(r"E:\Document\CodeSpace\Study\DeepL\TimeSeries\Informer2020-main\data\ETT\ETTh1.csv")
s2 = df2[['HUFL','LUFL','OT']]
print(s2.describe())
print(s2.head())
