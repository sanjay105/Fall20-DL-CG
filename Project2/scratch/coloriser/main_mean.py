import torch
import torch.nn as nn
from torchvision import datasets, transforms

from ColorizationNet import ColorizationNet
from ImageLoaderMean import ImageLoaderMean,ImageLoaderABchannel
from utils import *

train_dir = '/blue/cis6930/sbanda/images/'
use_gpu = torch.cuda.is_available()

def initModel(is_mean):
    model = ColorizationNet(is_mean=is_mean)
    # print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    return model,criterion,optimizer

def loadTrainData(train_dir,is_mean):
    trainTransform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    
    if is_mean:
        trainImageData = ImageLoaderMean(train_dir, trainTransform)
    else:
        trainImageData = ImageLoaderABchannel(train_dir, trainTransform)
    
    trainData = torch.utils.data.DataLoader(trainImageData, batch_size=64, shuffle=True)
    return trainData


def main():
    is_mean = True
    model,criterion,optimizer = initModel(is_mean)
    print("Model Init Done")
    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()
    epochs = 100
    trainData = loadTrainData(train_dir,is_mean)
    print("Train Loader Done")
    for i in range(epochs):
        train(trainData,model,criterion,optimizer,i,is_mean)
    torch.save(model.state_dict(), '/models/model_mean.pkl')


if __name__=='__main__':
    main()