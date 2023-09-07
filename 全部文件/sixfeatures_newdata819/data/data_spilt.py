import random

import numpy
import numpy as np
import pandas as pd

# 读取数据
# data = pd.read_csv('./data_reduced_standard.csv', index_col=0, header=0)
data = pd.read_csv('./data_filled_standard.csv', index_col=0, header=0)

# 转置，将列转成行
data = np.array(data)
# print(data)
# print(len(data))
seed_list = [22,31,45,49,67,180,186,187,201,220]
# 将数据集的测试集、验证集、训练集划分成十个不一样的，进行之后进行交叉验证
for x in range(0,10):
        # 使用set存放不重复的活动区【set中重复值只存一次】
        random.seed(seed_list[x])  # 随机数种子
        AR_C_set = set()
        AR_M_set = set()


        # 开始向set中添加活动区，方便直接删除重复的活动区
        for i in range(len(data)):
            if data[i][1] == 0:
                AR_C_set.add(data[i][0])
            if data[i][1] == 1:
                AR_M_set.add(data[i][0])
        # print(len(AR_C_set))
        # print(len(AR_M_set))
        #C 189
        #M 99
        # 将set转为list方便后续操作
        AR_C_list = list(AR_C_set)
        AR_M_list = list(AR_M_set)
        # np.savetxt("AR_C_list.csv",   AR_C_list, delimiter=",", fmt="%s")
        # np.savetxt("AR_M_list.csv", AR_M_list, delimiter=",", fmt="%s")

        # ==============开始分割C级数据===================
        # 存储分割的C级数据集
        train_C_AR_list = []
        valid_C_AR_list = []
        test_C_AR_list = None
        # 分割C级耀斑活动区训练集
        # 189*0.6 = 113
        # 189*0.2 = 38
        # 分成 113 + 38 + 38 =189
        # 113 + 38 = 151
        random.shuffle(AR_C_list)
        train_C_AR_list = AR_C_list[:113]
        valid_C_AR_list = AR_C_list[113:151]
        test_C_AR_list = AR_C_list[151:]
        # print(train_C_AR_list)
        # print(valid_C_AR_list)
        # print(test_C_AR_list)
        # print(len(train_C_AR_list))
        # print(len(valid_C_AR_list))
        # print(len(test_C_AR_list))

        # ==============C级数据分割 结束===================

        # # ==============开始分割M级数据===================
        train_M_AR_list = []
        valid_M_AR_list = []
        test_M_AR_list = None

        # 分割M级耀斑活动区训练集
        # 99*0.6 = 60
        # 99*0.2 = 20
        # 分成 60 + 20 + 19 =99
        # 分成 60 + 20 = 80
        random.shuffle(AR_M_list)
        train_M_AR_list = AR_M_list[:60]
        valid_M_AR_list = AR_M_list[60:80]
        test_M_AR_list = AR_M_list[80:]
        # print(train_M_AR_list)
        # print(valid_M_AR_list)
        # print(test_M_AR_list)
        # print(len(train_M_AR_list))
        # print(len(valid_M_AR_list))
        # print(len(test_M_AR_list))

        train_total = []
        train_total.extend(train_C_AR_list)
        train_total.extend(train_M_AR_list)
        # print(train_total)
        # print(len(train_total))
        # 173
        valid_total = []
        valid_total.extend(valid_C_AR_list)
        valid_total.extend(valid_M_AR_list)
        # print(valid_total)
        # print(len(valid_total))
        # 58
        test_total = []
        test_total.extend(test_C_AR_list)
        test_total.extend(test_M_AR_list)
        # print(test_total)
        # print(len(test_total))
        train_all = []
        for item in train_total:
                for m in data:
                        if m[0] == item:
                                train_all.append(m)
        train_all = np.array(train_all)
        # print(train_all)
        # print("train_all shape:", train_all.shape)
        int_columns = [0,1,2]
        format_str = ["%d","%d","%d"] + ["%.20f"]*(train_all.shape[1]-len(int_columns))


        # numpy.savetxt(f"../data_spilt/data_reduced/train_data_reduced_{x}.csv", train_all, delimiter=",", fmt=format_str)
        np.savetxt(f"../data_spilt/data_filled/train_data_filled_{x}.csv", train_all, delimiter = ",", fmt = format_str, comments = "")

        valid_all = []
        for item in valid_total:
                for m in data:
                        if m[0] == item:
                                valid_all.append(m)
        valid_all = np.array(valid_all)
        # print(valid_all)
        int_columns = [0, 1, 2]
        format_str = ["%d", "%d", "%d"] + ["%.20f"] * (valid_all.shape[1] - len(int_columns))
        # numpy.savetxt(f"../data_spilt/data_reduced/valid_data_reduced_{x}.csv", valid_all, delimiter=",", fmt=format_str)
        numpy.savetxt(f"../data_spilt/data_filled/valid_data_filled_{x}.csv", valid_all, delimiter=",", fmt=format_str)

        test_all = []
        for item in test_total:
                for m in data:
                        if m[0] == item:
                                test_all.append(m)
        test_all = np.array(test_all)
        # print(test_all)
        int_columns = [0, 1, 2]
        format_str = ["%d", "%d", "%d"] + ["%.20f"] * (test_all.shape[1] - len(int_columns))
        # numpy.savetxt(f"../data_spilt/data_reduced/test_data_reduced_{x}.csv", test_all, delimiter=",", fmt=format_str)
        numpy.savetxt(f"../data_spilt/data_filled/test_data_filled_{x}.csv", test_all, delimiter=",", fmt=format_str)

