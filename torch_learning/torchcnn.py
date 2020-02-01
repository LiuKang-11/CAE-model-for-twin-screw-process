import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.optim as optim


path='D:/PYTHON/螺桿單元'
# normalize or not     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
img_data = torchvision.datasets.ImageFolder(path,
                                            transform=transforms.Compose([
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ])
                                            )

#print(len(img_data))


#print(len(data_loader))
print(img_data.class_to_idx)

img_loader = torch.utils.data.DataLoader(img_data, batch_size=30, num_workers=0,shuffle=True)

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

list(enumerate(img_loader))
#CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,110,110) #(220/2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,106,106)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,51,51)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0) #output_shape=(8,23,23)
        self.relu4 = nn.ReLU() # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512) 
        self.relu5 = nn.ReLU() # activation
        self.fc2 = nn.Linear(512, 256) 
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(256, 100) 
        self.relu7 = nn.ReLU()
        self.fc4 = nn.Linear(100, 30) 

        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        out = self.cnn1(x) # Convolution 1
        out = self.relu1(out)
        out = self.maxpool1(out)# Max pool 1
        out = self.cnn2(out) # Convolution 2
        out = self.relu2(out) 
        out = self.maxpool2(out) # Max pool 2
        out = self.cnn3(out) # Convolution 3
        out = self.relu3(out)
        out = self.maxpool3(out) # Max pool 3
        out = self.cnn4(out) # Convolution 4
        out = self.relu4(out)
        out = self.maxpool4(out) # Max pool 4
        out = out.view(out.size(0), -1) # last CNN faltten con. Linear NN
        out = self.fc1(out) # Linear function (readout)
        out = self.relu5(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)
        out = self.relu7(out)
        out = self.fc4(out)
        out = self.output(out)

        return out

model = CNN_Model()
#print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9)

# number of epochs to train the model
n_epochs =5

valid_loss_min = np.Inf # track change in validation loss

#train_losses,valid_losses=[],[]

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
 
    print('running epoch: {}'.format(epoch))

    model.train()
    for i, data in enumerate(img_loader, 0):
        images, labels = data
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the batch loss
        loss = criterion(outputs, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()   
        if i % 2 == 1:     
            print('[%d, %5d] loss: %.3f' %
                    (epoch , i + 1, train_loss / 2))
            train_loss = 0.0

print('Finished Training')
        
   
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('log1/img_data')
