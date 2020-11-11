import torch
import torch.nn as nn
from torchvision import datasets, transforms

from ColorizationNet import ColorizationNet
from ImageLoaderMean import ImageLoaderMean,ImageLoaderABchannel
from utils import *
from predict import *


# Training data set directory
train_dir = '/blue/cis6930/sbanda/images/'
# Raw dataset directory
raw_dir = "/blue/cis6930/sbanda/images/face_images/"
# Augemented dataset directory
aug_data_dir = "/blue/cis6930/sbanda/images/face_images/Augumented/"
# Test dataset to validate
test_channel_path = "/blue/cis6930/sbanda/gray/"


# Use GPU, when available
use_gpu = torch.cuda.is_available()


def initModel(is_mean):
    """
    Initializing model, computing MSE loss, and optimizing model parameters
    :param is_mean: boolean flag for mean of (a,b) channels
    :return: model, criterion for losses, and optimizer
    """
    # Sets the model
    model = ColorizationNet(is_mean=is_mean)
    # print(model)

    # Sets the criteria for Mean Squared Error
    criterion = nn.MSELoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # returns respective objects
    return model, criterion, optimizer


def loadTrainData(train_dir, is_mean):
    """
    Loads the training data
    :param train_dir: directory path for training set
    :param is_mean: boolean flag for mean of (a,b) channels
    :return: loaded data set (training)
    """
    # Applies transformations on training set
    trainTransform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])

    # If mean, loads mean train data set, else loads (a, b) channel train set
    if is_mean:
        trainImageData = ImageLoaderMean(train_dir, trainTransform)
    else:
        trainImageData = ImageLoaderABchannel(train_dir, trainTransform)

    # loads data with batch size of 64 and shuffles it
    trainData = torch.utils.data.DataLoader(trainImageData, batch_size=64, shuffle=True)

    # returns loaded data
    return trainData


def main():
    """
    Entry point of the program/project
    :return: void
    """
    """Augument the given data to 10x"""
    #augment_data(raw_dir,aug_data_dir)
    
    
    # Channel does not compute mean values
    is_mean = False

    # Initializes model, criterion, and optimizer
    model, criterion, optimizer = initModel(is_mean)
    print("Model Init Done")

    # If GPU available, uses it
    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    # Sets number of epochs
    epochs = 20

    # Loads train data
    trainData = loadTrainData(train_dir, is_mean)
    print("Train Loader Done")

    # Iterates through epochs and trains the model
    for i in range(epochs):
        train(trainData, model, criterion, optimizer, i, is_mean)
    
    # Saves the model to a file
    torch.save(model.state_dict(), '/home/sbanda/Fall20-DL-CG/Project2/Finale/models/model_channel.pkl')
    
    colorizes the grayscale image in test set with the model
    colorize('/home/sbanda/Fall20-DL-CG/Project2/Finale/models/model_channel.pkl',test_channel_path)


if __name__ == '__main__':
    main()
