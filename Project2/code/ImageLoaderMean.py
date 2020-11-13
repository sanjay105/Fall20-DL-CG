import numpy as np
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import torch
from torchvision import datasets, transforms


class ImageLoaderMean(datasets.ImageFolder):
    """
    Custom Image Loader for Mean of (a,b) channels that converts image from RGB to gray prior to loading
    """
    def __getitem__(self, index):
        """
        Supports indexing for data-sets
        :param index:
        :return: original image, LAB image, and target class
        """
        # loads path & class
        path, target = self.imgs[index]

        # loads image from path
        img = self.loader(path)

        if self.transform is not None:
            # applies transforms on image and converts to numpy array
            originalImage = self.transform(img)
            originalImage = np.asarray(originalImage)

            # Gets the LAB image & normalizes it
            labImage = rgb2lab(originalImage)
            labImage = (labImage + 128) / 255

            # Extracts a, b channels from LAB image, finds mean and converts to a tensor
            channelAB = np.asarray([np.mean(labImage[:, :, 1]),np.mean(labImage[:, :, 2])], dtype='float32')
            channelAB = torch.from_numpy(channelAB)

            # Extracts gray image from original & converts as a tensor
            originalImage = rgb2gray(originalImage)
            originalImage = torch.from_numpy(originalImage).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        # returns original image, ab-image, and target class
        return originalImage, channelAB, target

    
class ImageLoaderABchannel(datasets.ImageFolder):
    """
    Custom Image Loader for (a,b) channels that converts image from RGB to gray prior to loading
    """
    def __getitem__(self, index):
        """
        Supports indexing for data-sets
        :param index:
        :return: original image, LAB image, and target class
        """
        # loads path & class
        path, target = self.imgs[index]

        # loads image from path
        img = self.loader(path)

        if self.transform is not None:
            # applies transforms on image and converts to numpy array
            originalImage = self.transform(img)
            originalImage = np.asarray(originalImage)

            # Gets the LAB image & normalizes it
            labImage = rgb2lab(originalImage)
            labImage = (labImage + 128) / 255

            # Extracts a, b channels from LAB image and converts to a tensor
            channelAB = labImage[:, :, 1:3]
            channelAB = torch.from_numpy(channelAB.transpose((2, 0, 1))).float()

            # Extracts gray image from original & converts as a tensor
            originalImage = rgb2gray(originalImage)
            originalImage = torch.from_numpy(originalImage).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        # returns original image, ab-image, and target class
        return originalImage, channelAB, target