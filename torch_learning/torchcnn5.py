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
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2)
        )  # 16,5,5
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # 
            nn.Tanh(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=1)  #(8,2,2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # 
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # 
            nn.Tanh(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # 
            nn.Tanh(),
            nn.BatchNorm2d(3)
          
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
    x = x.view(x.size(0)*8, 1, 2, 2)
    return x





print(len(img_loader))
model = CNNAE()
#print(model)
criterion = nn.MSELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)


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

torch.save(model.state_dict(), './conv_autoencoder.pth')
 
img_grid = torchvision.utils.make_grid(images)
# show images
matplotlib_imshow(img_grid, one_channel=False)

# write to tensorboard
writer.add_image('Origin_img', img_grid)

#tensorboard --logdir=runs
writer.add_graph(model, images)
writer.close()

    