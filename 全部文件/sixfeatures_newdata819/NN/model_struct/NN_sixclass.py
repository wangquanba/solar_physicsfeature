import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from scoreClass import Metric
import warnings
from tensorflow.keras.models import load_model


# Set GPU memory allocation to prevent overflow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

warnings.filterwarnings("ignore")
BATCH_SIZE=32
LEARN_RATE=0.0009996131311254346
EPOCH = 120
DROPOUT_RATE = 0.758473836125765
l1_l2_rate = 0.0003810196375778589
size = 6

def set_seed():
    # Solve the problem of reproducibility of training results by setting random seed
    # Solve solution：https://zhuanlan.zhihu.com/p/95416326
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(11)
    np.random.seed(11)
    random.seed(11)
    # tf.set_random_seed(12)
    tf.random.set_seed(12)

def load_NN_model():
    model = Sequential([
        Dense(32,input_shape=(size,),kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate)),
        Activation('relu'),
        Dropout(DROPOUT_RATE),
        Dense(128,kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate)),
        Activation('relu'),
        Dropout(DROPOUT_RATE),
        Dense(64,kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate)),
        Activation('relu'),
        Dropout(DROPOUT_RATE),
        Dense(16,kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate)),
        Activation('relu'),
        Dropout(DROPOUT_RATE),
        Dense(2,kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate)),
        Activation('softmax')
    ])
    return model

def data_process(data_df, feature_index: list):
    """
    :param data_df:
    :param feature_index:  3 4 5 6 7 8 9
    :return:
    """
    data_T = np.array(data_df).transpose()
    label = data_T[1]
    y_data = to_categorical(label, 2)
    x_data_T = []
    # print(feature_index)
    for i in feature_index:
        # print(i)
        x_T = data_T[i]
        x_data_T.append(x_T)
    x_data = np.array(x_data_T).transpose()
    # print(x_data.shape)
    # print(x_data)
    return x_data, y_data

set_seed()
def get_weight_dir(every_class_num_list: list):
    """
   Calculate class weights based on the number of samples in each class
   :param every_class_num_list: Number of samples for each class
   :return: A dictionary containing weights for each class
   """
    all_samples = 0
    num_classes = len(every_class_num_list)
    for i in every_class_num_list:
        all_samples += i
    weight_dir = {}
    index = 0
    for class_number in every_class_num_list:  # class_number 代表每个类别的样本数
        weight_dir[index] = all_samples / (class_number * num_classes)
        index += 1
    return weight_dir

if __name__ == '__main__':
    data = []
    all_loss = []

    for x in range(0, 10):
        set_seed()
        train_df = pd.read_csv(f'../../data_spilt/data_filled/train_data_filled_{x}.csv', header=None)
        test_df = pd.read_csv(f'../../data_spilt/data_filled/test_data_filled_{x}.csv', header=None)
        valid_df = pd.read_csv(f'../../data_spilt/data_filled/valid_data_filled_{x}.csv', header=None)
        feature_name = ['energy', 'shear_mean', 'uj_mean', 'uhc_mean', 'GBH_MEAN', 'ALP_MEAN']

        x_train, y_train = data_process(train_df, [3, 4, 5, 6, 7, 8])
        x_valid, y_valid = data_process(valid_df, [3, 4, 5, 6, 7, 8])
        x_test, y_test = data_process(test_df, [3, 4, 5, 6, 7, 8])
        num_0 = 0
        num_1 = 0
        for l in y_train:
            # argmax是输出的标签的最大值处在哪个位置
            if 0 == np.argmax(l):
                num_0 += 1
            else:
                num_1 += 1

        every_class_num_list = [num_0, num_1]
        class_weight = get_weight_dir(every_class_num_list)
        # print(class_weight)
        model = load_NN_model()

        model.summary()
        adam = Adam(LEARN_RATE)
        model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',  # 'categorical_crossentropy',  # 交叉熵
            metrics=['acc']  # [Accuracy, TSS, FAR, HSS, Precision, Recall]  # 准确率,
        )
        best_valid_TSS = float('-inf')
        best_test_TSS = float('-inf')
        best_model = "./save_model/model_sixclass" + str(
            x) + str(2) + ".h5"

        # 开始训练
        my_history = {'loss':[],'val_loss':[]}
        for i in range(EPOCH):
            print('训练到第' + str(i + 1) + '代，共' + str(EPOCH) + '代')
            history_per_epoch = model.fit(
                x_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=1,
                class_weight=class_weight,
                validation_data=(x_valid, y_valid),
                verbose=1,
                shuffle=True,
            )
            my_history['loss'].append(history_per_epoch.history['loss'][0])
            my_history['val_loss'].append(history_per_epoch.history['val_loss'][0])

            y_pred1 = model.predict(x_valid).argmax(axis=1)
            y_true1 = y_valid.argmax(axis=1)
            va = Metric(y_true1, y_pred1)
            if va.TSS()[1] >= best_valid_TSS:
                best_valid_TSS = va.TSS()[1]
                model.save(best_model)
        # 单个cv训练结束
        all_loss.append(my_history)
        model = load_model(best_model)
        y_pred = model.predict(x_test).argmax(axis=1)
        y_true = y_test.argmax(axis=1)
        m = Metric(y_true, y_pred)
        TSS = m.TSS()
        HSS = m.HSS()
        Accuracy = m.Accuracy()
        print(
            f"使用的数据集为data{x}，数据特征个数为{6},数据特征为{feature_name[0]+feature_name[1]+feature_name[2]+feature_name[3]+feature_name[4]+feature_name[5]},TSS最大值：{m.TSS()},HSS最大值：{m.HSS()},"
            f"Accuracy最大值:{m.Accuracy()}")
        data.append([x, feature_name[0]+feature_name[1]+feature_name[2]+feature_name[3]+feature_name[4]+feature_name[5], TSS[0], HSS[0], Accuracy[0]])

    # 绘制训练loss
    plt.figure(dpi=600)
    for i,history in enumerate(all_loss):
        plt.plot(np.arange(EPOCH)+1,history['loss'],label=f'train loss CV{i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../figure/train_loss_figure/NN_6train.png')  # 保存位置
    # plt.show()
    # 绘制验证loss
    plt.figure(dpi=600)
    for i,history in enumerate(all_loss):
        plt.plot(np.arange(EPOCH)+1,history['val_loss'],label=f'valid loss CV{i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../figure/valid_loss_figure/NN_6valid.png')  # 保存位置
    # plt.show()

    save_data = np.array(data)
    save_data = pd.DataFrame(save_data, index=None, columns=['data', 'feature', 'TSS', 'HSS', 'Accuracy'])
    save_data.to_csv('../test_result/NN_feature6.csv')