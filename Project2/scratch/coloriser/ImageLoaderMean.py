import numpy as np
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import torch
from torchvision import datasets, transforms
# For utilities
import os, shutil, time

class ImageLoaderMean(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        #print(path)
        if self.transform is not None:
            originalImage = self.transform(img)
            originalImage = np.asarray(originalImage)
            labImage = rgb2lab(originalImage)
            labImage = (labImage + 128) / 255
            channelAB = np.asarray([np.mean(labImage[:, :, 1]),np.mean(labImage[:, :, 2])],dtype='float32')
            channelAB = torch.from_numpy(channelAB)

            originalImage = rgb2gray(originalImage)
            originalImage = torch.from_numpy(originalImage).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return originalImage, channelAB, target
    
class ImageLoaderABchannel(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        #print(path)
        if self.transform is not None:
            originalImage = self.transform(img)
            originalImage = np.asarray(originalImage)
            labImage = rgb2lab(originalImage)
            labImage = (labImage + 128) / 255
            channelAB = labImage[:, :, 1:3]
            channelAB = torch.from_numpy(channelAB.transpose((2, 0, 1))).float()

            originalImage = rgb2gray(originalImage)
            originalImage = torch.from_numpy(originalImage).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return originalImage, channelAB, target