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
from sklearn.metrics import r2_score
 
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
train2=pd.DataFrame(train, copy=True)


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





###CNN model
class AE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AE, self).__init__()
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
#            nn.Linear(hidden_size, 1),   # 压缩成3个特征, 进行 3D 图像可视化
#             nn.Tanh(),
        )
        self.encoder2 = nn.Linear(hidden_size, 1)

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
        feature = self.encoder(x)
        encoded = self.encoder2(feature)
        decoded = self.decoder(encoded)
        return feature, encoded, decoded

### NN model
class Extrader(nn.Module):
    def __init__(self):
        super(Extrader, self).__init__()
        # 压缩
        self.model = nn.Sequential(
            nn.Linear(5, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 64),#
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 64),#
            nn.Dropout(0.5),
            nn.Tanh(),#
            nn.Linear(64, 2),   # 压缩成3个特征, 进行 3D 图像可视化
        )
        
    def forward(self, x):
        return(self.model(x))


###Combine Model
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        self.AE = AE(30,15,30)
        #self.CNNAE.load_state_dict(torch.load('D:/PYTHON/  CNNstd_rtd.pth'))
        self.extrater = Extrader()
        
    def forward(self, m1,m2,m3, quantity):
        feature1,encoded_1,decoded_1 = self.AE(m1)
#         print(encoded_1.shape)
        feature2,encoded_2,decoded_2 = self.AE(m2)
        feature3,encoded_3,decoded_3 = self.AE(m3)
#         print(quantity[0,0].shape)
        e1=torch.reshape(encoded_1,(-1,1))
        e2=torch.reshape(encoded_2,(-1,1))
        e3=torch.reshape(encoded_3,(-1,1))

        input_value = torch.cat((e1, e2, e3,quantity),1)
#         print(input_value.shape)
        out = self.extrater(input_value)
        return e1,e2,e3,decoded_1,decoded_2,decoded_3,out




###Training Process
for j in range(99):

    model = IntegratedModel()
    model.load_state_dict(torch.load('D:/PYTHON/150_2tAE_100_weight/150_2tAE_100_{}.pth'.format(j+1)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    EPOCH = 1
    loss_func = nn.MSELoss()
    #writer = SummaryWriter('tempAE_100_{}/NN_result'.format(j+1))

    for epoch in range(EPOCH):
        meta_data = [] #存放标签
        features = torch.zeros(0)  #PCA用
        model.train()
        '''
        for i in range(6):
            targetR = torch.FloatTensor(a[i].RTD.array).view(-1,1).detach()
            targetT = torch.FloatTensor(a[i].Temperature.array).view(-1,1).detach()
            target= torch.cat([targetR,targetT],axis=1)
            m1 = torch.FloatTensor(a[i].one.array)
            m2 = torch.FloatTensor(a[i].two.array)
            m3 = torch.FloatTensor(a[i].three.array)
    
            q1=torch.FloatTensor(a[i].Rotation_speed.array).view(-1,1)
            q2=torch.FloatTensor(a[i].Total_rate.array).view(-1,1)
            quantity=torch.cat([q1,q2],axis=1)

            #input_value = torch.cat((e11, e22, e33,quantity),1)
            #print(input_value.shape)


            e1,e2,e3,d1,d2,d3,out= model(m1,m2,m3,quantity)
            #e11,e21,e31,d11,d21,d31,out1= model(n1,n2,n3,quantity)
            #e12,e22,e32,d12,d22,d32,out2= model(n11,n21,n31,quantity)
            #e13,e23,e33,d13,d23,d33,out3= model(n12,n22,n32,quantity)

            loss1 = loss_func(d1, m1) + loss_func(d2, m2) + loss_func(d3, m3)#+\
            #loss_func(d11, m1) + loss_func(d21, m2) + loss_func(d31, m3)+\
            #loss_func(d12, m1) + loss_func(d22, m2) + loss_func(d32, m3)+\
            #loss_func(d13, m1) + loss_func(d23, m2) + loss_func(d33, m3)
            loss2= loss_func(out,targetR)#+ loss_func(out1,targetT)+ loss_func(out2,targetT)+ loss_func(out3,targetT)
            loss= 1*loss1+ 1*loss2
            # apply gradients

            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()     

            print('epoch [{}/{}],batch[{}/{}], cnn_loss:{:.4f}, nn_loss:{:.4f}'
            .format(epoch, EPOCH, i+1 , 8, loss1.item(),loss2.item()))

            features = torch.cat((features, e1))
            features = torch.cat((features, e2))
            features = torch.cat((features, e3))
            label1=b[i].one.values.tolist()
            label2=b[i].two.values.tolist()
            label3=b[i].three.values.tolist()
            meta_data = meta_data + label1 +label2 + label3
            #print(len(meta_data))

        

        


        writer.add_embedding(features, metadata=meta_data, global_step=epoch)
        writer.add_scalar('AE_loss', loss1, epoch)
        writer.add_scalar('MSE_loss', loss2, epoch)


        '''

        model.eval()

        targetR = torch.FloatTensor(a[7].RTD.array).view(-1,1).detach()
        targetT = torch.FloatTensor(a[7].Temperature.array).view(-1,1).detach()
        target= torch.cat([targetR,targetT],axis=1)

        m1 = torch.FloatTensor(a[7].one.array)
        m2 = torch.FloatTensor(a[7].two.array)
        m3 = torch.FloatTensor(a[7].three.array)

        q1=torch.FloatTensor(a[7].Rotation_speed.array).view(-1,1)
        q2=torch.FloatTensor(a[7].Total_rate.array).view(-1,1)
        quantity=torch.cat([q1,q2],axis=1)

            #input_value = torch.cat((e11, e22, e33,quantity),1)
            #print(input_value.shape)


        e1,e2,e3,d1,d2,d3,out= model(m1,m2,m3,quantity)
        
        out0=out[:,0]
        out0=torch.reshape(out0, (340, 1))
        out1=out[:,1]
        out1=torch.reshape(out1, (340, 1))
        
        loss1 = loss_func(d1, m1) + loss_func(d2, m2) + loss_func(d3, m3)
        loss2= loss_func(out1,targetT)*2.987

        loss= 1*loss1+ 1*loss2
            # apply gradients

        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()     
        print(loss2)

        r2=r2_score(targetT.detach().numpy(), out1.detach().numpy())
        #print('r2:',r2)




        #writer.add_scalar('val_MSE_loss', loss2, epoch)
    #writer.close()

    #torch.save(model.state_dict(), './tempAE_100_{}.pth'.format(j+1))
#tensorboard --logdir=runs
#http://localhost:6006/