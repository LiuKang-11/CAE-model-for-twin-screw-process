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

path='D:/PYTHON/螺桿單元'
# normalize or not     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
img_data = torchvision.datasets.ImageFolder(path,
                                            transform=transforms.Compose([
                                                transforms.Scale(32),
                                                transforms.CenterCrop(28),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])
                                            )

print(img_data.imgs)
#print(len(img_data))
#print(len(data_loader))
print(img_data.class_to_idx)

img_loader = torch.utils.data.DataLoader(img_data, batch_size=30, num_workers=0,shuffle=False)

images,labels=next(iter(img_loader))
print(images.shape)
print(labels.shape)


classes=[
    'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 'kR1-50-4', 'kR2-50-4', 'kR3-50-4',\
    'mL1c-50-15', 'mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',\
    'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',\
    'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',\
    'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',\
    'sL3-50-33', 'sR1-50-33', 'sR2-50-33', 'sR3-50-33'
]

def imshow(img):
    img = img.permute(1,2,0)
    img = torch.clamp(img,0,1)
    plt.imshow(img)
   
dataiter = iter(img_loader)
images, labels = dataiter.next()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 30))

for idx in np.arange(30):
    ax = fig.add_subplot(6, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ".format( classes[labels[idx]]),fontsize=10)
plt.show()

#CNN Model
class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  #(16,10,10)
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2)
        )  # 16,5,5
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 4, 3, stride=2, padding=1),  # 
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=1)  #(3,2,2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, stride=2),  # 
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # 
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # 
            nn.BatchNorm2d(3),
            nn.Tanh()
            
          
        )

    def forward(self, x):
        feature_map= self.conv1(x)
        encode = self.conv2(feature_map)
        decode = self.decoder(encode)
        return feature_map, encode, decode

#################################################################
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def to_img(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 28, 28)
    return x
def to_img2(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0)*16, 1, 5, 5)
    return x

def to_img3(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0)*4, 1, 2, 2)
    return x





print(len(img_loader))
model = CNNAE()
#print(model)
criterion = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=0.003,weight_decay=1e-5)


# number of epochs to train the model
n_epochs =1010
writer = SummaryWriter('runs/CNNAE_result')
#tensorboard --logdir=runs

for epoch in range(n_epochs):
    for i, (img, labels) in enumerate(img_loader):
    # ===================forward=====================
        feature, encode , output = model(img)
        loss = criterion(output, img)
    # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    writer.add_scalar('loss', loss, epoch)

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, n_epochs, loss.item()))
    if epoch % 100 == 0:
        
        pic = to_img(output.data)
        save_image(pic, './img_tanh/image_{}.png'.format(epoch))
        pic2 = to_img2(feature.data)
        save_image(pic2,'./feature_tanh/image_{}.png'.format(epoch))
        pic3 = to_img3(encode.data)
        save_image(pic3,'./encode_tanh/image_{}.png'.format(epoch))
        
        #data_relu  data_tanh   data_sigmoid
print(labels)
a=encode.view(-1,16)
b=a.detach().numpy()
#print(b)
#print(b[20])

encode_list={}
i=0
for item in classes:
    encode_list[item]=b[i]
    i += 1

print(encode_list)

torch.save(model.state_dict(), './  CNNAE_tanh.pth')


img_grid = torchvision.utils.make_grid(images)
# show images
matplotlib_imshow(img_grid, one_channel=False)

# write to tensorboard
writer.add_image('Origin_img', img_grid)

#tensorboard --logdir=runs
writer.add_graph(model, images)
writer.close()


### NN PART####

def text_to_encode(item):
#     tmp = item.split('-')
    onehot_data = encode_list[item]
    return onehot_data

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


#Part9
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        self.extrater = Extrader(50,64,1)
        
    def forward(self, input_value):
        
#         print(input_value.shape)
        out = self.extrater(input_value)
        return out

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

        ### train extrater
        targetR = torch.FloatTensor(train_yR[i]).view(-1,1).detach()
        en_input1 = torch.FloatTensor(text_to_encode(train_one[i])).unsqueeze(0)
        en_input2 = torch.FloatTensor(text_to_encode(train_two[i])).unsqueeze(0)
        en_input3 = torch.FloatTensor(text_to_encode(train_three[i])).unsqueeze(0)
        x = torch.cat((en_input1,en_input2,en_input3),1)
        #print(x.shape)
        y = torch.FloatTensor(quantity[i]).unsqueeze(0)
        #print(y.shape)
        input_value=torch.cat((x,y),1)
        #print(input_value.shape)#  1,50
        outR = my_model(input_value)
        
        mse_loss = loss_func(outR, targetR)
        optimizer.zero_grad()               # clear gradients for this training step
        mse_loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
#         total_loss2 += loss.item()
#         loss2 = total_loss2/2800
#         print('epoch',epoch,loss.item())
        if (iteration % 50 == 0):
            MSE_loss.append(mse_loss.item())

                

    writer.add_scalar('MSE_loss', mse_loss, epoch)
    if(epoch%10 == 0):
        print('epoch [{}/{}], mse_loss:{:.4f}'
          .format(epoch, EPOCH, mse_loss.item()))



writer.close()
