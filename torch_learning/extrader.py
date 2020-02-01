import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### function of data normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def min_max_normalize(my_matrix):
    scaler = MinMaxScaler()
    scaler.fit(my_matrix)
    my_matrix_normorlize=scaler.transform(my_matrix)
    return my_matrix_normorlize

def normalize(my_matrix):
    scaler = preprocessing.StandardScaler().fit(my_matrix)
    my_matrix_normorlize=scaler.transform(my_matrix)
    return my_matrix_normorlize

#####################################

def get_onehot_list(train_list):
    tr_one = {}
#     tr_two = {}
#     tr_three = {}
    for item in train_list:
        tr_one[item] = 1
    i = 0
    for item in tr_one.keys():
        tmp = np.zeros(len(tr_one))
        tmp[i] = 1
        tr_one[item] = tmp
        i += 1
    return tr_one

def text_to_onehot(item):
#     tmp = item.split('-')
    onehot_data = text_onehot_list[item]
    return onehot_data    
################################
columns = ['Rotation_speed','Total_rate','one','two','three','RTD','Temperature']
train =pd.read_csv('C:\\.ipynb_checkpoints\\training_data700.csv', names=columns)
print(len(np.array(train)))
train.head()
train.Temperature=pd.to_numeric(train.Temperature)
######################################
### normalize speed
speed = np.array([train.Rotation_speed.values],dtype = np.float32).transpose(1,0)
print(speed.shape)
speed_min = min_max_normalize(speed)
print(speed_min[:4])
speed_nor = normalize(speed)
# print(speed_nor[:4])

### normalize total rate
Total_rate = np.array([train.Total_rate.values],dtype = np.float32).transpose(1,0)
print(Total_rate.shape)
Total_rate_min = min_max_normalize(Total_rate)
print(Total_rate_min[:4])
Total_rate_nor = normalize(Total_rate)
# print(speed_nor[:4])

################################
