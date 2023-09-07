from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from scoreClass import Metric
from tensorflow.keras.layers import Flatten, Input,Dense, Dropout, BatchNormalization,LSTM,Bidirectional,multiply
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import warnings
import tensorflow as tf
import os
import random


# Set GPU memory allocation to prevent overflow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
warnings.filterwarnings("ignore")

BATCH_SIZE = 8
LEARN_RATE = 8.912601046788379e-05
EPOCH = 150
DROPOUT_RATE = 0.6451370887503093
timestep = 40
INPUT_SIZE = 5
l1_l2_rate = 0.0004239952143301659
def set_seed():
    # Solve the problem of reproducibility of training results by setting random seed
    # Solve solution：https://zhuanlan.zhihu.com/p/95416326
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(11)
    np.random.seed(11)
    random.seed(11)
    tf.set_random_seed(12)
    # tf.random.set_seed(12)


# Attention
def attention_3d_block(inputs):
    input_dim=int(inputs.shape[2])
    a=inputs
    a_probs=Dense(input_dim,activation="softmax")(a)
    output_attention_mul=multiply([inputs,a_probs])
    return output_attention_mul

set_seed()
# Calculate class weights based on the number of samples in each class
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
        for class_number in every_class_num_list:  # class_number represents the number of samples for each category
            weight_dir[index] = all_samples / (class_number * num_classes)
            index += 1
        return weight_dir

def data_process(data_df, feature_index: list):
            """
            :param data_df:
            :param feature_index:  3 4 5 6 7 8
            :return:
            """
            data_T = np.array(data_df).transpose()
            label = data_T[1]
            new_label=[]
            count = 0
            for i in range(0, len(label)):
                # print(i)
                count += 1
                if count % timestep == 0:
                    new_label.append(label[i])
                    count = 0
            new_label = np.array(new_label).astype(int)
            # print(len(new_label))
            # print(new_label)
            # Change to one-hot mode
            y_data = to_categorical(new_label, 2)
            # print(len(y_data))
            # print(y_data.shape)
            # print(y_data)
            x_data_T = []
            # print(feature_index)
            for i in feature_index:
                # print(i)
                x_T = data_T[i]
                x_data_T.append(x_T)
            # print(x_data_T)
            x_data = np.array(x_data_T).transpose()
            x_data=x_data.reshape((-1, timestep, INPUT_SIZE))
            # print(x_data.shape)
            # print(x_data)
            return x_data, y_data

# BiLSTM moudle
class ConstructModel(object):
    def __init__(self, configure_maps):
        self.__configure_maps = configure_maps


    def first(self):
        inputs = Input(shape=(timestep, INPUT_SIZE))

        BiLstm_out = Bidirectional(LSTM(32, input_shape=(timestep, INPUT_SIZE), return_sequences=True))(inputs)
        attention_mul = attention_3d_block(BiLstm_out)
        attention_mul = BatchNormalization()(attention_mul)
        attention_mul = Dropout(DROPOUT_RATE)(attention_mul)

        attention_mul = Flatten()(attention_mul)

        attention_mul = Dense(128, kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate))(attention_mul)
        attention_mul = BatchNormalization()(attention_mul)
        attention_mul = Dropout(DROPOUT_RATE)(attention_mul)

        attention_mul = Dense(32, kernel_regularizer=l1_l2(l1_l2_rate, l1_l2_rate), bias_regularizer=l1_l2(l1_l2_rate, l1_l2_rate))(attention_mul)
        attention_mul = BatchNormalization()(attention_mul)
        attention_mul = Dropout(DROPOUT_RATE)(attention_mul)

        out_put = Dense(self.__configure_maps["num_of_classes"], activation="softmax")(attention_mul)
        model = Model(inputs=[inputs], outputs=out_put)
        model.compile(optimizer=Adam(learning_rate=self.__configure_maps["1"]["learning_rate"]),
                      loss='categorical_crossentropy')
        return model


if __name__ == '__main__':
            data = []

            for x in range(0, 10):
                set_seed()
                train_df = pd.read_csv(f'../../data_spilt/data_filled/train_data_filled_{x}.csv',header = None)
                test_df = pd.read_csv(f'../../data_spilt/data_filled/test_data_filled_{x}.csv',header = None)
                valid_df = pd.read_csv(f'../../data_spilt/data_filled/valid_data_filled_{x}.csv',header = None)
                feature_name = ['energy', 'shear_mean', 'uj_mean', 'uhc_mean', 'GBH_MEAN', 'ALP_MEAN']
                # print(train_df)
                for i in range(3, 9):
                    set_seed()
                    num1 = i
                    # print(num1)
                    for p in range(4, 9):
                        if p <= 8 and p > i:
                            num2 = p
                            # print(num2)
                            for z in range(5, 9):
                                if z <= 8 and z > p:
                                    num3 = z
                                    for v in range(6, 9):
                                        if v <= 8 and v > z:
                                            num4 = v
                                            for n in range(7, 9):
                                                if n <= 8 and n > v:
                                                    num5 = n
                                                    x_train, y_train = data_process(train_df, [num1, num2, num3, num4, num5])
                                                    x_valid, y_valid = data_process(valid_df, [num1, num2, num3, num4, num5])
                                                    x_test, y_test = data_process(test_df, [num1, num2, num3, num4, num5])
                                                    # print(x_train)
                                                    # print(x_valid)
                                                    # print(y_valid)
                                                    # print(x_test)
                                                    # print(y_test)
                                                    num_0 = 0
                                                    num_1 = 0
                                                    for l in y_train:
                                                        if 0 == np.argmax(l):
                                                            num_0 += 1
                                                        elif 1 == np.argmax(l):
                                                            num_1 += 1
                                                    every_class_num_list = [num_0, num_1]
                                                    print(every_class_num_list)
                                                    class_weight = get_weight_dir(every_class_num_list)
                                                    print(class_weight)

                                                    conf = {"num_of_classes": 2, "nesterov": True, "verbose": 2,
                                                            "1": {
                                                                "batch_size": BATCH_SIZE, "epochs": EPOCH, "learning_rate": LEARN_RATE,
                                                                "input_shape": (timestep, INPUT_SIZE),  # 修改
                                                            }
                                                                }

                                                    constructor = ConstructModel(conf)
                                                    model = constructor.first()
                                                    model.summary()

                                                    best_valid_TSS = float('-inf')
                                                    best_test_TSS = float('-inf')
                                                    best_model = "../save_model/model_fiveclass" + str(
                                                        x) + ".h5"

                                                    # 开始训练
                                                    for j in range(EPOCH):
                                                        print('训练到第' + str(j + 1) + '代，共' + str(EPOCH) + '代')
                                                        history_per_epoch = model.fit(
                                                            x_train, y_train,
                                                            batch_size=BATCH_SIZE,
                                                            epochs=1,
                                                            class_weight=class_weight,
                                                            validation_data=(x_valid, y_valid),
                                                            shuffle=True,
                                                        )

                                                        y_pred1 = model.predict(x_valid).argmax(axis=1)
                                                        y_true1 = y_valid.argmax(axis=1)
                                                        va = Metric(y_true1, y_pred1)
                                                        if va.TSS()[1] >= best_valid_TSS:
                                                            best_valid_TSS = va.TSS()[1]
                                                            model.save(best_model)

                                                        print(f"验证集，当前TSS：{va.TSS()},当前HSS：{va.HSS()},当前Accuracy{va.Accuracy()}")

                                                    # 单个cv训练结束
                                                    model = load_model(best_model)
                                                    y_pred = model.predict(x_test).argmax(axis=1)
                                                    y_true = y_test.argmax(axis=1)
                                                    m = Metric(y_true, y_pred)
                                                    TSS = m.TSS()
                                                    HSS = m.HSS()
                                                    Accuracy = m.Accuracy()
                                                    print(
                                                        f"{x},数据特征个数为{5}，数据特征为{feature_name[num1 - 3]}+{feature_name[num2 - 3]}+{feature_name[num3 - 3]}+{feature_name[num4 - 3]}+{feature_name[num5 - 3]}，"
                                                        f"TSS最大值：{m.TSS()}，HSS最大值：{m.HSS()}，Accuracy最大值:{m.Accuracy()}")
                                                    data.append(
                                                        [x, feature_name[num1 - 3]+feature_name[num2 - 3]+feature_name[num3 - 3]+feature_name[num4 - 3]+feature_name[num5 - 3], TSS[0], HSS[0],
                                                         Accuracy[0]])
                                                else:
                                                    n = 0
                                        else:
                                            v = 0
                                else:
                                    z = 0
                        else:
                            p = 0

            save_data = np.array(data)
            # print(save_data)
            save_data = pd.DataFrame(save_data, index=None, columns=['data', 'feature', 'TSS', 'HSS', 'Accuracy'])
            save_data.to_csv('../test_result/LSTM_feature5.csv')


