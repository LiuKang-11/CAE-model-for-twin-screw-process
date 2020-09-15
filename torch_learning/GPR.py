import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
# 创建数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


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

#Part2
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

#Part3
columns = ['Rotation_speed','Total_rate','one','two','three','RTD','Temperature']
train =pd.read_csv('D:/PYTHON/train_data.csv', names=columns)
print(len(np.array(train)))
train.head()
train.Temperature=pd.to_numeric(train.Temperature)

#Part4
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

#Part5
### get the text_onehot_list for transfer string to one-hot vector
train_one = np.concatenate((train.one.values,train.two.values,train.three.values))
# test_one = np.concatenate((test.one.values,test.two.values,test.three.values))
text_onehot_list = get_onehot_list(train_one)
print(text_onehot_list)
print(text_onehot_list[train_one[20]])
print(type(text_to_onehot(train_one[20])))

#Part6
### prepare the training data
quantity = np.concatenate((speed_min,Total_rate_min),1)
#print(quantity.shape)
# (3140,2)
train_y = np.concatenate((np.array([train.RTD.values]).transpose(1,0),
                         np.array([train.Temperature.values]).transpose(1,0)),axis = 1)

#print(train_y.shape)
# (3140,2)
train_one = np.array(train.one.values)
train_two = np.array(train.two.values)
train_three = np.array(train.three.values)

test_yR = train_y[2800:3140,0:1]
test_yT = train_y[2800:3140,1:2]
#print(len(test_y),test_y.shape)
# (340,2)

test_quantity = quantity[2800:3140,:]
#print(test_quantity.shape)
# (340,2)

test_one = train_one[2800:3140]
test_two = train_two[2800:3140]
test_three = train_three[2800:3140]
#print(test_one.shape)
# (340)

train_yR = train_y[0:2800,0:1]
train_yT = train_y[0:2800,1:2]
#print(len(train_y))
#2800
quantity = quantity[0:2800,:]
'''
train_one = train_one[0:2800]
train_two = train_two[0:2800]
train_three = train_three[0:2800]
'''
training_data = np.concatenate((np.array([train.RTD.values]).transpose(1,0),
                         np.array([train.Temperature.values]).transpose(1,0),
                        np.array([train.one.values]).transpose(1,0),np.array([train.two.values]).transpose(1,0),
                        np.array([train.three.values]).transpose(1,0),speed_min,Total_rate_min),axis = 1)
print(training_data.shape)
print(type(train_y))


train_yR = train_y[0:2800,0:1]
train_yT = train_y[0:2800,1:2]
std_train_yT=min_max_normalize(train_yT)
std2_train_yT=normalize(train_yT)
std2_train_yR=normalize(train_yR)
std2_train_2target= np.concatenate((train_yR,std2_train_yT),axis = 1)

#print(len(train_y))
#2800
train_quantity = quantity[0:2800,:]

train_one1 = train_one[0:2800]
train_two1 = train_two[0:2800]
train_three1 = train_three[0:2800]


'''
for i in range(len(train_one)):
    train.one[i] = np.array(text_to_onehot(train_one[i]))
    train.two[i] = np.array(text_to_onehot(train_two[i]))
    train.three[i] = np.array(text_to_onehot(train_three[i]))
'''

RTD = np.array([train.RTD.values],dtype = np.float32).transpose(1,0)
RTD_min = min_max_normalize(RTD)
RTD_nor = normalize(RTD)
train.RTD=RTD_min


### normalize Temp
Temperature = np.array([train.Temperature.values],dtype = np.float32).transpose(1,0)
Temperature_min = min_max_normalize(Temperature)
Temperature_nor = normalize(Temperature)
train.Temperature=Temperature_min

### normalize speed
speed = np.array([train.Rotation_speed.values],dtype = np.float32).transpose(1,0)
#print(speed.shape)
speed_min = min_max_normalize(speed)
#print(speed_min[:4])
speed_nor = normalize(speed)
train.Rotation_speed=speed_min
# print(speed_nor[:4])

### normalize total rate
Total_rate = np.array([train.Total_rate.values],dtype = np.float32).transpose(1,0)
#print(Total_rate.shape)
Total_rate_min = min_max_normalize(Total_rate)
#print(Total_rate_min[:4])
Total_rate_nor = normalize(Total_rate)
train.Total_rate=Total_rate_min






training_data = np.concatenate((np.array([train.RTD.values]).transpose(1,0),
                         np.array([train.Temperature.values]).transpose(1,0),
                        np.array([train.one.values]).transpose(1,0),np.array([train.two.values]).transpose(1,0),
                        np.array([train.three.values]).transpose(1,0),speed_min,Total_rate_min),axis = 1)

#print(training_data)
#print('train_x:',training_data[0:2800, 2:7])
#print('train_rtd:',training_data[0:2800,0])
trainX=torch.empty(2800,92)

for i in range(len(train_yR)):

    en_input1 = torch.FloatTensor(text_to_onehot(train_one1[i])).unsqueeze(0)
    en_input2 = torch.FloatTensor(text_to_onehot(train_two1[i])).unsqueeze(0)
    en_input3 = torch.FloatTensor(text_to_onehot(train_three1[i])).unsqueeze(0)
    q1=torch.FloatTensor(quantity[i]).view(-1,2)
    
    trainX[i] = torch.cat((en_input1,en_input2,en_input3,q1),1)



#trainX=training_data[0:2800, 2:7]
trainRTD=training_data[0:2800,0]
testRTD=training_data[2800:3140,0]
trainTEMP=training_data[0:2800,1]
testTEMP=training_data[2800:3140,1]
train2=training_data[0:2800,0:2]
test2=training_data[2800:3140,0:2]
trainTemp=training_data[0:2800,1]
trainX=trainX.detach().numpy()
#print('train:',trainX)

# 核函数的取值
 
kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
 
# 创建高斯过程回归,并训练
 
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
 
reg.fit(trainX, train2)

print('reg:',reg)
 
###MSE R2
testX=torch.empty(340,92)

for i in range(len(test_yR)):
    en_input1 = torch.FloatTensor(text_to_onehot(test_one[i])).unsqueeze(0)
    en_input2 = torch.FloatTensor(text_to_onehot(test_two[i])).unsqueeze(0)
    en_input3 = torch.FloatTensor(text_to_onehot(test_three[i])).unsqueeze(0)
    q1=torch.FloatTensor(test_quantity[i]).view(-1,2)
    
    testX[i] = torch.cat((en_input1,en_input2,en_input3,q1),1)

testX=testX.detach().numpy()

y_pred = reg.predict(testX, return_std=False)
print(type(y_pred))
print(type(test2))
print('predict:',y_pred.reshape(-1,2))
print('real:',testTEMP.reshape(-1,2))

 
