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

path='D:/PYTHON/螺桿單元-外插2'
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


classes=['sL3-50-33']
def imshow(img):
    img = img.permute(1,2,0)
    img = torch.clamp(img,0,1)
    plt.imshow(img)
   
dataiter = iter(img_loader)
images, labels = dataiter.next()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 30))

for idx in np.arange(1):
    ax = fig.add_subplot(6, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ".format( classes[labels[idx]]),fontsize=10)
plt.show()

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
            nn.MaxPool2d(2, stride=1)  #(2,2,2)
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

#### Def some fxn for diagram       
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
model2 = CNNAE()
model2.load_state_dict(torch.load('D:/PYTHON/  CNNAE_tanh2.pth'))
criterion = nn.MSELoss()

for i, (img, labels) in enumerate(img_loader):
    feature, encode , output = model2(img)
    loss = criterion(output, img)

print(loss)
pic = to_img(output.data)
save_image(pic, './img_test/image_{}.png'.format(2))
