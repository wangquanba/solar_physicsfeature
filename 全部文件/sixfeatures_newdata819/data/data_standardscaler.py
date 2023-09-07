import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

filled_data = pd.read_csv(r'./filled_data.csv')
filled_data = np.array(filled_data)
filled_data_T = filled_data.transpose()
# print(filled_data_T)

AR = filled_data_T[0]
# print(AR)
# print(AR[1])
label = filled_data_T[1]
time = filled_data_T[2]
AR_label_time = filled_data_T[:3]
AR_label_time_T = AR_label_time.transpose()
# print(AR_label_time_T)
data = filled_data_T[3:]
data = np.array(data, dtype='float')
data_T = data.transpose()
scaler = StandardScaler()
# print(data_T.transpose().shape)
# print(time.transpose().shape)
# print(data_T.shape)
# print(data_T.transpose())
# print(data_T[0])
# print(data_T.transpose()[0])

# Standardize by column
scalers = []
standardized_features = []
for col in range (data_T.shape[1]):
    scaler = StandardScaler()
    standardized_column = scaler.fit_transform(data_T[:,col].reshape(-1,1))
    scalers.append(scaler)
    standardized_features.append(standardized_column)
data_standardized = np.concatenate(standardized_features,axis=1)
# print(data_standardized)
data_new = np.concatenate((AR_label_time_T,data_standardized),axis=1)


for i in range(len(data_new)):
    if(data_new[i,1]=='M'):
        data_new[i,1]=1
    else:
        data_new[i,1]=0
# print(data_new)


save_data_df = pd.DataFrame(data_new,index=None,columns=['AR','KEY','time','energy','shear_mean','uj_mean','uhc_mean','GBH_MEAN','ALP_MEAN'])
save_data_df.to_csv('./data_filled_standard.csv')



