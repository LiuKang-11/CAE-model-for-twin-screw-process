#Part1
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
train =pd.read_csv('D:/PYTHON/training_data700-1.csv', names=columns)
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
print(text_onehot_list[train_one[20]])
print(text_to_onehot(train_one[20]).shape)

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

test_yR = train_y[2800:3201,0:1]
test_yT = train_y[2800:3201,1:2]
#print(len(test_y),test_y.shape)
# (340,2)

test_quantity = quantity[2800:3201,:]
#print(test_quantity.shape)
# (340,2)

test_one = train_one[2800:3201]
test_two = train_two[2800:3201]
test_three = train_three[2800:3201]
#print(test_one.shape)
# (340)

train_yR = train_y[0:2800,0:1]
train_yT = train_y[0:2800,1:2]
#print(len(train_y))
#2800
quantity = quantity[0:2800,:]

train_one = train_one[0:2800]
train_two = train_two[0:2800]
train_three = train_three[0:2800]

training_data = np.concatenate((np.array([train.RTD.values]).transpose(1,0),
                         np.array([train.Temperature.values]).transpose(1,0),
                        np.array([train.one.values]).transpose(1,0),np.array([train.two.values]).transpose(1,0),
                        np.array([train.three.values]).transpose(1,0),speed_min,Total_rate_min),axis = 1)
print(training_data.shape)
print(type(train_y))

#Part7
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision


# hyperparameters
EPOCH = 70
BATCH_SIZE = 16
LR = 0.002

class Extrader(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Extrader, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        # 压缩
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
#             nn.Linear(64, 16),
            nn.Tanh(),
#             nn.Linear(64, 16),
#             nn.Tanh(),
            nn.Linear(hidden_size, output_size),   # 压缩成3个特征, 进行 3D 图像可视化
        )
        
    def forward(self, x):
        return(self.model(x))

#Part8
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        # 压缩
        self.encoder = nn.Sequential(
#             nn.Linear(5, 32),
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
#             nn.Linear(15, 15),
#             nn.Tanh(),
#             nn.Linear(15, 15),
#             nn.Tanh(),
            nn.Linear(hidden_size, 1),   # 压缩成3个特征, 进行 3D 图像可视化
#             nn.Tanh(),
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
#             nn.Linear(15, 15),
#             nn.Tanh(),
#             nn.Linear(15, 15),
#             nn.Tanh(),
            nn.Linear(hidden_size,output_size),
#             nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#Part9
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        self.AE = AutoEncoder(30,15,30)
        self.extrater = Extrader(5,64,1)
        
    def forward(self, quality, quantity):
        encoded_1,decoded_1 = self.AE(quality[0].unsqueeze(0))
#         print(encoded_1.shape)
        encoded_2,decoded_2 = self.AE(quality[1].unsqueeze(0))
        encoded_3,decoded_3 = self.AE(quality[2].unsqueeze(0))
#         print(quantity[0,0].shape)
        input_value = torch.cat((encoded_1, encoded_2,
                                 encoded_3,quantity),1)
#         print(input_value.shape)
        out = self.extrater(input_value)
        return decoded_1,decoded_2,decoded_3,out

#Part10
### training the integrated model
my_model = IntegratedModel()
EPOCH = 100
BATCH_SIZE = 16
LR = 0.002
optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)
# EX_optimizer = torch.optim.Adam(my_model.extrater.parameters(), lr=LR)
loss_func = nn.MSELoss()

iteration = 0
AE_loss = []
MSE_loss = []
embeddingvalue = []
writer = SummaryWriter('runs/NN_result')

#RTD
for epoch in range(EPOCH):
#     total_loss1 = 0
#     total_loss2 = 0
    for i in range(len(train_yR)):
        ### train AE
        iteration += 1
        targetR = torch.FloatTensor(train_yR[i]).view(-1,1).detach()
        en_input1 = torch.FloatTensor(text_to_onehot(train_one[i])).unsqueeze(0)
        en_input2 = torch.FloatTensor(text_to_onehot(train_two[i])).unsqueeze(0)
        en_input3 = torch.FloatTensor(text_to_onehot(train_three[i])).unsqueeze(0)
        x = torch.cat((en_input1,en_input2,en_input3),0)
        y = torch.FloatTensor(quantity[i]).unsqueeze(0)
        d1,d2,d3,outR = my_model(x,y)
        
        ae_loss = loss_func(d1, en_input1) + loss_func(d2, en_input2) + loss_func(d3, en_input3)
        optimizer.zero_grad()               # clear gradients for this training step
        ae_loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
#         total_loss1 += loss.item()
#         loss1 = total_loss1/2800
        if (iteration % 50 == 0):
            AE_loss.append(ae_loss.item())

        ### train extrater
        targetR = torch.FloatTensor(train_yR[i]).view(-1,1).detach()
        en_input1 = torch.FloatTensor(text_to_onehot(train_one[i])).unsqueeze(0)
        en_input2 = torch.FloatTensor(text_to_onehot(train_two[i])).unsqueeze(0)
        en_input3 = torch.FloatTensor(text_to_onehot(train_three[i])).unsqueeze(0)
        x = torch.cat((en_input1,en_input2,en_input3),0)
        y = torch.FloatTensor(quantity[i]).unsqueeze(0)
        d1,d2,d3,outR = my_model(x,y)
        
        mse_loss = loss_func(outR, targetR)
        optimizer.zero_grad()               # clear gradients for this training step
        mse_loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
#         total_loss2 += loss.item()
#         loss2 = total_loss2/2800
#         print('epoch',epoch,loss.item())
        if (iteration % 50 == 0):
            MSE_loss.append(mse_loss.item())

        if(iteration % 50 == 0):
            for item in text_onehot_list.keys():
                input_ = torch.FloatTensor(text_onehot_list[item]).unsqueeze(0)
                e1,d1 = my_model.AE(input_)
                embeddingvalue.append(e1)
                
    writer.add_scalar('AE_loss', ae_loss, epoch)
    writer.add_scalar('MSE_loss', mse_loss, epoch)
    if(epoch%10 == 0):
        print('epoch [{}/{}], ae_loss:{:.4f}'
          .format(epoch+1, EPOCH, ae_loss.item()))
        print('epoch [{}/{}], mse_loss:{:.4f}'
          .format(epoch+1, EPOCH, mse_loss.item()))



writer.close()