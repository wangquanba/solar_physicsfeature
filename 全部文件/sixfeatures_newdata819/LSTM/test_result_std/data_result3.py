import random
import numpy as np
import pandas as pd

data = pd.read_csv(fr'../test_result/LSTM_feature3.csv', header=0)
data_array = np.array(data)
# print(data_array)
# print(data_array.shape)
data_T = data_array.transpose()
num = data_T[1]
feature = data_T[2]
TSS = data_T[3]
HSS = data_T[4]
Accuracy = data_T[5]
# print(num)
# print(feature)
# print(TSS)
# print(HSS)
# print(Accuracy)
feature_set = feature[0:20]

# 先找出数据特征，在找到第几个数据集，然后找到所有的TSS\HSS\Accuracy
TSS_data = []
HSS_data = []
Accuracy_data = []
# TSS_total = 0
# HSS_total = 0
# Accuracy_total = 0
# 按照数据特征进行排列
for m in feature_set:
    for i in range(len(num)):
        for x in range(0, 10):
            if num[i] == x and feature[i] == m:
                TSS_data.append(TSS[i])
                HSS_data.append(HSS[i])
                Accuracy_data.append(Accuracy[i])

TSS_data = np.array(TSS_data)
TSS_data = TSS_data.reshape(20, 10)
HSS_data = np.array(HSS_data)
HSS_data = HSS_data.reshape(20, 10)
Accuracy_data = np.array(Accuracy_data)
Accuracy_data = Accuracy_data.reshape(20, 10)
# print(Accuracy_mean)
# print(TSS_data[0])第一个数据特征的十次结果
TSS_mean = []
TSS_std = []
HSS_mean = []
HSS_std = []
Accuracy_mean = []
Accuracy_std = []
for i in range(0, 20):
    # TSS_data=TSS_data[i]
    TSS_mean.append(np.mean(TSS_data[i]))
    TSS_std.append(np.std(TSS_data[i]))
    HSS_mean.append(np.mean(HSS_data[i]))
    HSS_std.append(np.std(HSS_data[i]))
    Accuracy_mean.append(np.mean(Accuracy_data[i]))
    Accuracy_std.append(np.std(Accuracy_data[i]))
# print(TSS_mean)
# print(TSS_std)
# print(HSS_mean)
# print(HSS_std)
# print(Accuracy_mean)
# print(Accuracy_std)
# 组成一个表格的形式
result_finish = []
for i in range(0, 20):
    result_finish.append([feature_set[i],TSS_mean[i], TSS_std[i], HSS_mean[i], HSS_std[i], Accuracy_mean[i], Accuracy_std[i]])

# print(feature_set[i])
save_result = np.array(result_finish)
# print(save_result)
save_result_df = pd.DataFrame(save_result, index=None,columns = ['feature','TSS_mean', 'TSS_std', 'HSS_mean', 'HSS_std', 'Accuracy_mean',
           'Accuracy_std'])
save_result_df.to_csv('./result_std/LSTM3.csv')
