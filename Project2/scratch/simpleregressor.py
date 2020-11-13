#dependencies
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D as Conv2D, MaxPooling2D
from keras.utils import np_utils
# import keras
# from keras.models import Sequential
from keras import backend as K
from keras import losses
# from keras.layers.normalization import BatchNormalization

def read_images(path,num_images,xdim,ydim):
    images = []
    all_paths = os.listdir(path)
    mini_set = all_paths[:num_images]
    for i in mini_set:
        # print(i)
        try:
          file = path+"/"+i
          image = cv2.imread(file)
          image = cv2.resize(image,(xdim,ydim))
          images.append(image)
        except Exception:
          pass

    return images


#function to convert rgb to lab 
def rgb_to_lab(images):
    lab_images = []
    for i in images:
        lab_image= cv2.cvtColor(i, cv2.COLOR_RGB2LAB)
        lab_images.append(lab_image)

    return lab_images

def lab_to_rgb(images):
    rgb_images = []
    for i in images:
        lab_image= cv2.cvtColor(i, cv2.COLOR_LAB2RGB)
        lab_images.append(lab_image)

#function to extract the l channels and ab for training
def extract_channels(lab_images):
    l_channels = []
    a_channels = []
    b_channels = []
    for i in lab_images:
        l,a,b = cv2.split(i)
        l_channels.append(l)
        a_channels.append(a)
        b_channels.append(b)

    return np.array(l_channels), np.array(a_channels), np.array(b_channels)

#function to create train and test data
def create_train_data(l,a,b):
    train_data = []
    for i in l:
        train_data.append(np.array(i,dtype= 'float32'))
    train_labels_a = []
    train_labels_b = []
    for i in a:
        # train_labels_a.append(np.array(i.flatten(),dtype='float32'))
        train_labels_a.append(np.mean(np.array(i.flatten(),dtype='float32')))
    for i in b:
        # train_labels_b.append(np.array(i.flatten(),dtype='float32'))
        train_labels_b.append(np.mean(np.array(i.flatten(),dtype='float32')))
    train_labels = []
    for i,j in zip(train_labels_a,train_labels_b):
        # train_labels.append(np.concatenate((i,j),axis = 0))
        train_labels.append((i,j))
    
    return train_data, train_labels

path = "/content/Augmented"  #path needs to be changed
num_images = 8250   #this number can be changed according to number of image data in train folder
#converting all images to 128*128
x_dim = 128
y_dim = 128
images = read_images(path,num_images,x_dim,y_dim) #read images
num_images = len(images)
lab_images = rgb_to_lab(images) #convert rgb to lab space images
l,a,b = extract_channels(lab_images) #separate l, a and b channels
train_data, train_labels = create_train_data(l,a,b)


train_data = np.reshape(train_data,(num_images,x_dim,y_dim,1))
train_data.shape[3]
# train_labels = np.reshape(train_labels,(num_images,128*128*2))
train_labels = np.reshape(train_labels,(num_images,2))
img_cols = 128
img_rows = 128
if K.image_data_format() == 'channels_first':
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# print(input_shape)
# input_shape = (128, 128, 1)

# Pytorch
class NeuralNet(nn.Module):
    def __init__(self, op_dim):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,1) # ip channels, op channels, kernel, stride
        self.conv2 = nn.Conv2d(16,16,4,1)
        self.conv3 = nn.Conv2d(16,32,4,1)
        self.conv3_1 = nn.Conv2d(32,32,4,1)
        self.conv4 = nn.Conv2d(32,64,3,1)
        self.conv4_1 = nn.Conv2d(64,64,3,1)
        self.linear1 = nn.Linear(train_labels.shape[1], 128) # ip features, op features
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear2 = nn.Linear(64, op_dim)
    def forward(self,x):

        # print(x.shape)
        x=self.conv1(x)
        # print(x.shape)
        x=F.relu(x)
        x=F.max_pool2d(x,2)

        x=self.conv2(x)
        # print(x.shape)
        # x=F.relu(x)
        # x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x=F.max_pool2d(x,2)
        
        x=self.conv3(x)
        # print(x.shape)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.conv3_1(x)
        x=F.relu(x)
        x=self.bn2(x)
        x=F.max_pool2d(x,2)
        # print(x.shape)
        

        x=self.conv4(x)
        # print(x.shape)
        x=F.relu(x)
        x=self.conv4_1(x)
        x=F.relu(x)
        # print(x.shape)
        # x = self.fc1(x)
        # x = self.bn3(x)
        # print(x.shape)

        x = torch.flatten(x, 1)
        # print(x.shape)
        # x = self.linear1(x)
        # x = F.relu(x)
        x = self.linear2(x)
        # print(x.shape)
        output = F.log_softmax(x, dim=1)
        # print(output)
        return output

# train_loader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)
model = NeuralNet(2)
optimizer = optim.Adam(model.parameters())

print(model)

import torch
from torchvision import transforms
from skimage.color import rgb2lab, rgb2gray
class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        # print(path,target)
        img = self.loader(path)
        # print(img)
        # print(self.transform)
        # print(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original,dtype='float32')
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255.0
            # print(img_lab)
            img_ab = []
            img_ab.append((np.mean(img_lab[:, :, 1]), np.mean(img_lab[:, :, 2])))
            img_ab = np.reshape(img_ab, (1,2))
            # print(img_ab)
            # print(img_ab.shape)
            # img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original)
            # img_original = np.reshape(img_original,(1,128,128))
            #print(img_original.shape)
            target = img_ab.flatten()
            #print(target.shape)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img_original, img_ab),target

original_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])


from torch.autograd import Variable

# criterion = nn.L1Loss()
criterion = nn.MSELoss()
# losses=AverageMeter()

train_set = TrainImageFolder('/content/', original_transform)
train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=3)
def epoch(epid):
    model.train()
    avgloss = 0.0
    for batch_idx,(data,classes) in enumerate(train_loader):
        original_img = data[0].unsqueeze(1).float()
        img_ab = data[1].float()
        # print(img_ab.dtype)
        classes = Variable(classes)
        optimizer.zero_grad()
        class_output = model(original_img)
        loss = criterion(class_output, classes.float()) 
        # Compute gradient and optimize
        optimizer.zero_grad()
        # print(loss.dtype)
        loss.backward(retain_graph=True)
        optimizer.step()
        # print(batch_idx, loss)
        avgloss += loss.data.tolist()
        if batch_idx%10 ==0 :
            rint("Epoch: "+str(epid)+" Batch "+str(batch_idx)+" Done, Loss: "+str(avgloss))
    print("Epoch: "+str(epid)+"  -> loss "+avgloss)

for i in range(1,6):
  epoch(i)

# model.compile(loss='mean_squared_error',
#               optimizer='rmsprop',
#               metrics=['accuracy'])