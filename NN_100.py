import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader


###quantity data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
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



###encode list
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


classes=[
    'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 'kR1-50-4', 'kR2-50-4', 'kR3-50-4',\
    'mL1c-50-15','mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',\
    'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',\
    'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',\
    'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',\
    'sL3-50-33','sR1-50-33', 'sR2-50-33', 'sR3-50-33'
]
text_onehot_list = get_onehot_list(classes)



columns = ['Rotation_speed','Total_rate','one','two','three','RTD','Temperature']
train =pd.read_csv('D:/PYTHON/training_data700-1.csv', names=columns)
#train=train.sample(frac=1)
train.head()
train.Temperature=pd.to_numeric(train.Temperature)
train.RTD=pd.to_numeric(train.RTD)


#Part4
### normalize speed
speed = np.array([train.Rotation_speed.values],dtype = np.float32).transpose(1,0)
print(speed.shape)
speed_min = min_max_normalize(speed)
print(speed_min[:4])
speed_nor = normalize(speed)
train.Rotation_speed=speed_nor
# print(speed_nor[:4])

### normalize total rate
Total_rate = np.array([train.Total_rate.values],dtype = np.float32).transpose(1,0)
print(Total_rate.shape)
Total_rate_min = min_max_normalize(Total_rate)
print(Total_rate_min[:4])
Total_rate_nor = normalize(Total_rate)
train.Total_rate=Total_rate_nor


### normalize RTD

RTD = np.array([train.RTD.values],dtype = np.float32).transpose(1,0)
RTD_min = min_max_normalize(RTD)
RTD_nor = normalize(RTD)
train.RTD=RTD_nor


### normalize Temp
Temperature = np.array([train.Temperature.values],dtype = np.float32).transpose(1,0)
Temperature_min = min_max_normalize(Temperature)
Temperature_nor = normalize(Temperature)
train.Temperature=Temperature_nor

train_one = np.array(train.one.values)
train_two = np.array(train.two.values)
train_three = np.array(train.three.values)

# test_one = np.concatenate((test.one.values,test.two.values,test.three.values))

#Part6
### prepare the training data
#quantity = np.concatenate((speed_nor,Total_rate_nor),1)
#print(quantity.shape)
# (3140,2)
train_y = np.concatenate((np.array([train.RTD.values]).transpose(1,0),
                         np.array([train.Temperature.values]).transpose(1,0)),axis = 1)

#print(train_y.shape)
# (3140,2)
def split_dataframe(df, chunk_size = 400): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

b=split_dataframe(train2)  #string
#print(str(b[0].one))
#print(type(str(b[0].one)))


'''
#meta_data = [] #存放标签
label1=b[0].one.values.tolist()
print(len(label1))
label2=b[0].two.values.tolist()
label3=b[0].three.values.tolist()
label1=label1+label2
label1=label1+label3   
print('meta3:',len(label1))
'''


for i in range(len(train_one)):
    train.one[i] = np.array(text_to_onehot(train_one[i]))
    train.two[i] = np.array(text_to_onehot(train_two[i]))
    train.three[i] = np.array(text_to_onehot(train_three[i]))



a=split_dataframe(train)  #img
print(type(a[0]))
print(a[0].one.array)





### NN model
class Extrader(nn.Module):
    def __init__(self):
        super(Extrader, self).__init__()
        # 压缩
        self.model = nn.Sequential(
            nn.Linear(92, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 64),#
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 64),#
            nn.Dropout(0.5),
            nn.Tanh(),#
            nn.Linear(64, 1),   # 压缩成3个特征, 进行 3D 图像可视化
        )
        
    def forward(self, x):
        return(self.model(x))


###Combine Model
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        #self.AE = AE()
        #self.CNNAE.load_state_dict(torch.load('D:/PYTHON/  CNNstd_rtd.pth'))
        self.extrater = Extrader()
        
    def forward(self, input_value):

        out = self.extrater(input_value)
        return out




###Training Process
for j in range(99):

    model = IntegratedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    EPOCH = 101
    loss_func = nn.MSELoss()
    writer = SummaryWriter('tempNN_100_{}/NN_result'.format(j+1))

    for epoch in range(EPOCH):
        meta_data = [] #存放标签
        features = torch.zeros(0)  #PCA用
        model.train()
        for i in range(6):
            targetR = torch.FloatTensor(a[i].RTD.array).view(-1,1).detach()
            targetT = torch.FloatTensor(a[i].Temperature.array).view(-1,1).detach()
            target= torch.cat([targetR,targetT],axis=1)
            m1 = torch.FloatTensor(a[i].one.array)
            m2 = torch.FloatTensor(a[i].two.array)
            m3 = torch.FloatTensor(a[i].three.array)
    
            q1=torch.FloatTensor(a[i].Rotation_speed.array).view(-1,1)
            q2=torch.FloatTensor(a[i].Total_rate.array).view(-1,1)
            quality=torch.cat([m1,m2,m3],axis=1)
            quantity=torch.cat([q1,q2],axis=1)
            input_values=torch.cat([quality,quantity],axis=1)
            #input_value = torch.cat((e11, e22, e33,quantity),1)
            #print(input_value.shape)


            out= model(input_values)
            #e11,e21,e31,d11,d21,d31,out1= model(n1,n2,n3,quantity)
            #e12,e22,e32,d12,d22,d32,out2= model(n11,n21,n31,quantity)
            #e13,e23,e33,d13,d23,d33,out3= model(n12,n22,n32,quantity)

            #loss1 = loss_func(d1, m1) + loss_func(d2, m2) + loss_func(d3, m3)#+\
            #loss_func(d11, m1) + loss_func(d21, m2) + loss_func(d31, m3)+\
            #loss_func(d12, m1) + loss_func(d22, m2) + loss_func(d32, m3)+\
            #loss_func(d13, m1) + loss_func(d23, m2) + loss_func(d33, m3)
            loss2= loss_func(out,targetT)#+ loss_func(out1,targetT)+ loss_func(out2,targetT)+ loss_func(out3,targetT)
            loss= loss2
            # apply gradients

            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()     

            print('epoch [{}/{}],batch[{}/{}], cnn_loss:{:.4f}, nn_loss:{:.4f}'
            .format(epoch, EPOCH, i+1 , 8, loss1.item(),loss2.item()))


            #print(len(meta_data))

        

        


        writer.add_embedding(features, metadata=meta_data, global_step=epoch)
        writer.add_scalar('MSE_loss', loss2, epoch)




        model.eval()

        targetR = torch.FloatTensor(a[7].RTD.array).view(-1,1).detach()
        targetT = torch.FloatTensor(a[7].Temperature.array).view(-1,1).detach()
        target= torch.cat([targetR,targetT],axis=1)


        target= torch.cat([targetR,targetT],axis=1)
        m1 = torch.FloatTensor(a[7].one.array)
        m2 = torch.FloatTensor(a[7].two.array)
        m3 = torch.FloatTensor(a[7].three.array)
    
        q1=torch.FloatTensor(a[7].Rotation_speed.array).view(-1,1)
        q2=torch.FloatTensor(a[7].Total_rate.array).view(-1,1)
        quality=torch.cat([m1,m2,m3],axis=1)
        quantity=torch.cat([q1,q2],axis=1)
        input_values=torch.cat([quality,quantity],axis=1)
        loss2= loss_func(out,targetT)

        loss= loss2
            # apply gradients

        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()     
        print('epoch [{}/{}],batch[{}/{}], val_cnn_loss:{:.4f}, val_nn_loss:{:.4f}'
            .format(epoch, EPOCH, 8 , 8, loss1.item(),loss2.item()))


        writer.add_scalar('val_MSE_loss', loss2, epoch)
    writer.close()

    torch.save(model.state_dict(), './tempNN_100_{}.pth'.format(j+1))
#tensorboard --logdir=runs
#http://localhost:6006/