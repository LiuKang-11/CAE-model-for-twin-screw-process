import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
###quantity data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
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



#Img data preprocess
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

###encode list
path='D:/Python/螺桿單元'
img_data = torchvision.datasets.ImageFolder(path,
                                            transform=transforms.Compose([
                                                transforms.Scale(32),
                                                transforms.CenterCrop(28),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])
                                            )



#print(img_data.imgs)
#print(len(img_data))
#print(len(data_loader))
#print(img_data.class_to_idx)

img_loader = torch.utils.data.DataLoader(img_data, batch_size=30, num_workers=0,shuffle=False)

images,labels=next(iter(img_loader))

#print(images.shape)
#print(labels.shape)


classes=[
    'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 'kR1-50-4', 'kR2-50-4', 'kR3-50-4',\
    'mL1c-50-15','mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',\
    'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',\
    'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',\
    'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',\
    'sL3-50-33', 'sR1-50-33','sR2-50-33', 'sR3-50-33'
]

def imshow(img):
    img = img.permute(1,2,0)
    img = torch.clamp(img,0,1)
    plt.imshow(img)
   
dataiter = iter(img_loader)
images, labels = dataiter.next()


fig = plt.figure(figsize=(25, 30))
'''
for idx in np.arange(30):
    ax = fig.add_subplot(6, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ".format( classes[labels[idx]]),fontsize=10)
plt.show()
'''
encode_list={}
i=0
for item in classes:
    encode_list[item]=images.numpy()[i]
    i += 1

#print(encode_list)
def text_to_encode(item):
#     tmp = item.split('-')
    onehot_data = encode_list[item]
    return onehot_data


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

# ML Parameters
lr = 1e-3
epochs = 50
batch_size = 400

# Input Dataset
input_datase_file = 'D:/Python/training_data700-1.csv'

# Normalization
scaler = MinMaxScaler(feature_range=(-1, 1))

# Load Dataset 
columns = ['Rotation_speed','Total_rate','one','two','three','RTD','Temperature']
df = pd.read_csv(input_datase_file,names=columns)
df.head()




class ExtruderDataset(data.Dataset):
    
    def __init__(self):
        
        # read CSV
        self.df = pd.read_csv(input_datase_file, names = columns) # read columns
        self.orig_speed = df.Rotation_speed.to_numpy()
        self.orig_rate = df.Total_rate.to_numpy()
        self.orig_rtd = df.RTD.to_numpy()
        self.orig_temp = df.Temperature.to_numpy()
        #name of screw elements
        self.train_one_name = np.array(df.one.values)
        self.train_two_name = np.array(df.two.values)
        self.train_three_name = np.array(df.three.values)
        #Transfer name to img
        self.train_one = np.array(df.one.values)
        self.train_two = np.array(df.two.values)
        self.train_three = np.array(df.three.values)
        for i in range(len(self.train_one)):
            self.train_one[i] = np.array(text_to_encode(self.train_one[i]))
            self.train_two[i] = np.array(text_to_encode(self.train_two[i]))
            self.train_three[i] = np.array(text_to_encode(self.train_three[i]))  
     
        # store another normalized dataset
        self.normalized_speed = np.copy(self.orig_speed)
        self.normalized_speed = self.normalized_speed.reshape(-1, 1)
        self.normalized_rate = np.copy(self.orig_rate)
        self.normalized_rate = self.normalized_rate.reshape(-1, 1)
        self.normalized_rtd = np.copy(self.orig_rtd)
        self.normalized_rtd = self.normalized_rtd.reshape(-1, 1)
        self.normalized_temp = np.copy(self.orig_temp)
        self.normalized_temp = self.normalized_temp.reshape(-1, 1)
        # calculate normalization
        self.normalized_speed = scaler.fit_transform(self.normalized_speed)
        self.normalized_speed = self.normalized_speed.reshape(-1,1)
        self.normalized_rate = scaler.fit_transform(self.normalized_rate)
        self.normalized_rate = self.normalized_rate.reshape(-1,1)
        self.normalized_rtd = scaler.fit_transform(self.normalized_rtd)
        self.normalized_rtd = self.normalized_rtd.reshape(-1,1)
        self.normalized_temp = scaler.fit_transform(self.normalized_temp)
        self.normalized_temp = self.normalized_temp.reshape(-1,1)       
        #Train X (speed and rate)
        self.train_x = np.concatenate((self.normalized_speed,
                         self.normalized_rate),axis = 1)           
        #Train Y
        self.train_y = np.concatenate((self.normalized_rtd,
                         self.normalized_temp),axis = 1)         
        # use X history data generate one target 
        #self.sample_len = 18

    def __len__(self):
        
        return len(self.orig_speed)

    def __getitem__(self, idx):
        
        x = self.train_x[idx].astype(np.float32)
        one = self.train_one[idx].astype(np.float32)
        two = self.train_two[idx].astype(np.float32)
        three = self.train_three[idx].astype(np.float32)
        y = self.train_y[idx].astype(np.float32)
        name1 = self.train_one_name[idx]
        name2 = self.train_two_name[idx]
        name3 = self.train_three_name[idx]

        return x,one,two,three,y,name1,name2,name3
        #return x (speed rate) img(one two three) y(rtd temp)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# Load dataset
dataset = ExtruderDataset()
    #print(dataset.train_y[1,0])
# Split training and validation set 70%train 30%test
train_len = int(0.7*len(dataset))
valid_len = len(dataset) - train_len
TrainData, ValidationData = random_split(dataset,[train_len, valid_len])
# Load into Iterator (each time get one batch)
train_loader = data.DataLoader(TrainData, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = data.DataLoader(ValidationData, batch_size=batch_size, shuffle=True, num_workers=0)
'''
print("Total: ", len(dataset))
print("Training Set: ", len(TrainData))
print("Validation Set: ", len(ValidationData))

b=[]

for _, (x,one,two,three,y,name1,name2,name3) in enumerate(train_loader):
    #print(y[:,:]) #2 targets
    #print(y[:,0]) #only rtd
    #print(y[:,1]) #only temp
    #a = list(name1[:])
    #b += a
print(b)
'''

###CNN model
class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  #(16,10,10)
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2)
        )  # (16,5,5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1),  # (3,3,3)
            nn.BatchNorm2d(3),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=1)  #(3,2,2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 3, stride=2),  # (16,5,5)
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (8,15,15)
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # (3,28,38)
            nn.BatchNorm2d(3),
            nn.Tanh()
            
          
        )

    def forward(self, x):
        feature_map= self.conv1(x)
        encode = self.conv2(feature_map)
        encode2=torch.reshape(encode,(-1,4))
        decode = self.decoder(encode)
        return feature_map, encode, decode

### NN model
class Extrader(nn.Module):
    def __init__(self):
        super(Extrader, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(38, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 64),#
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 64),#
            nn.Dropout(0.5),
            nn.Tanh(),#
            nn.Linear(64, 1),   # 
        )
        
    def forward(self, x):
        return(self.model(x))


###Combine Model
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        self.CNNAE = CNNAE()
        #self.CNNAE.load_state_dict(torch.load('D:/PYTHON/  CNNstd_rtd.pth'))
        self.extrater = Extrader()
        
    def forward(self, m1,m2,m3, quantity):
        feature1,encoded_1,decoded_1 = self.CNNAE(m1)
#         print(encoded_1.shape)
        feature2,encoded_2,decoded_2 = self.CNNAE(m2)
        feature3,encoded_3,decoded_3 = self.CNNAE(m3)
#         print(quantity[0,0].shape)
        e1=torch.reshape(encoded_1,(-1,12))
        e2=torch.reshape(encoded_2,(-1,12))
        e3=torch.reshape(encoded_3,(-1,12))

        input_value = torch.cat((e1, e2, e3,quantity),1)
#         print(input_value.shape)
        out = self.extrater(input_value)
        return e1,e2,e3,decoded_1,decoded_2,decoded_3,out


# Define model
model = IntegratedModel()
    #print(model)

# Load into GPU if necessary
model = model.to(device)

# Define loss function
criterion = nn.MSELoss()
#criterion = nn.MSELoss(reduction='sum')

# Define optimization strategy
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)
writer = SummaryWriter('rtd_CNNAE/result')

###########################
# Train with training set #
###########################


def train(model, iterator, optimizer, criterion, device): 
    model.train()     # Enter Train Mode
    train_loss = 0
    cae_loss = 0
    nn_loss = 0
    meta_data = [] 
    features = torch.zeros(0) #save embedding
    for _, (x,one,two,three,y,name1,name2,name3) in enumerate(iterator):
        
        # move to GPU if necessary
        x,one,two,three,y = x.to(device), one.to(device), two.to(device), three.to(device),\
            y.to(device)
        
        # generate prediction
        optimizer.zero_grad()
        #load img and quantity value 
        m1 = one
        m2 = two
        m3 = three
        quantity = x
        # put data into model
        e1,e2,e3,d1,d2,d3,preds= model(m1,m2,m3,quantity)
        preds = preds.view(-1,1)
        # rtd or temp
        target = y[:,0].view(-1,1)
        # calculate loss
        loss1 = criterion(d1, m1) + criterion(d2, m2) + criterion(d3, m3)#+\
        loss2 = criterion(preds, target)
        loss = loss1 + loss2
        # compute gradients and update weights
        loss.backward()
        optimizer.step()
        
        # record training losses
        cae_loss += loss1.item()
        nn_loss += loss2.item()
        train_loss += loss.item()

        # save embedding value
        features = torch.cat((features, e1))
        features = torch.cat((features, e2))
        features = torch.cat((features, e3))
        label1 = list(name1[:])
        label2 = list(name2[:])
        label3 = list(name3[:])
        meta_data = meta_data + label1 +label2 + label3
    # print completed result
   
        print('batch[{}/{}], train_cae_loss:{:.4f}, train_nn_loss:{:.4f}'
          .format(_+1 , 6, loss1.item(),loss2.item()))
    print('train_cae_loss: %f' % (cae_loss))
    print('train_nn_loss: %f' % (nn_loss))    
    print('train_loss: %f' % (train_loss))
    return cae_loss, nn_loss, train_loss, features, meta_data



def test(model, iterator, criterion, device): 
    model.eval()     # Enter Train Mode
    test_loss = 0
    cae_loss = 0
    nn_loss = 0
    for _, (x,one,two,three,y,name1,name2,name3) in enumerate(iterator):
        
        # move to GPU if necessary
        x,one,two,three,y = x.to(device), one.to(device), two.to(device), three.to(device),\
            y.to(device)
        
        # generate prediction
        optimizer.zero_grad()
        #load img and quantity value 
        m1 = one
        m2 = two
        m3 = three
        quantity = x
        # put data into model
        e1,e2,e3,d1,d2,d3,preds= model(m1,m2,m3,quantity)
        preds = preds.view(-1,1)
        # rtd or temp
        target = y[:,0].view(-1,1)
        # calculate loss
        loss1 = criterion(d1, m1) + criterion(d2, m2) + criterion(d3, m3)#+\
        loss2 = criterion(preds, target)
        loss = loss1 + loss2
        
        # record training losses
        cae_loss += loss1.item()
        nn_loss += loss2.item()
        test_loss += loss.item()

    # print completed result
   
        print('batch[{}/{}], train_cae_loss:{:.4f}, train_nn_loss:{:.4f}'
          .format(_+1 , 3, loss1.item(),loss2.item()))
    print('test_cae_loss: %f' % (cae_loss))
    print('test_nn_loss: %f' % (nn_loss))    
    print('test_loss: %f' % (test_loss))
    return cae_loss, nn_loss, test_loss









# Running
for epoch in range(epochs):
    print("===== Epoch %i =====" % epoch)
    ###Training Part
    loss1, loss2, total_loss, features, meta_data = train(model, train_loader, optimizer, criterion, device)
    # decoder result
    if epoch % 10 == 0:
        for i in range(1):
            for i, (img, labels) in enumerate(img_loader):
    # ===================forward=====================
                feature,encoded,decoded = model.CNNAE(img)

            pic = to_img(decoded.data)
            save_image(pic, './img_decoder/image_{}.png'.format(epoch))
    #save embedding value
    writer.add_embedding(features, metadata=meta_data, global_step=epoch)
    writer.add_scalar('CNNAE_loss', loss1, epoch)
    writer.add_scalar('MSE_loss', loss2, epoch)

    ###Test Part
    test(model, test_loader, criterion, device)


#tensorboard --logdir=runs
#tensorboard --logdir=rtd_CNNAE
#http://localhost:6006/  