import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import torch
import os, shutil, time

use_gpu = torch.cuda.is_available()

class Aggregator(object):
    '''A handy class from the PyTorch ImageNet tutorial'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.value, self.average, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def convertToRGB(grayImage, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
       Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf()  # clear matplotlib
    originalImage = torch.cat((grayImage, ab_input), 0).numpy()  # combine channels
    originalImage = originalImage.transpose((1, 2, 0))  # rescale for matplotlib
    originalImage[:, :, 0:1] = originalImage[:, :, 0:1] * 100
    originalImage[:, :, 1:3] = originalImage[:, :, 1:3] * 255 - 128
    originalImage = lab2rgb(originalImage.astype(np.float64))
    grayImage = grayImage.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayImage, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=originalImage, fname='{}{}'.format(save_path['colorized'], save_name))


def validateMean(testData, model, criterion):
    model.eval()

    # Prepare value counters and timers
    run_time, data_time, losses = Aggregator(), Aggregator(), Aggregator()

    end = time.time()
    already_saved_images = False
    for i, (X, y, target) in enumerate(testData):
        data_time.update(time.time() - end)

        # Use GPU
        if use_gpu: 
            X, y, target = X.cuda(), y.cuda(), target.cuda()

        # Run model and record loss
        pred_ab = model(X)  # throw away class predictions
        # print(pred_ab.shape)
        loss = criterion(pred_ab, y)
        losses.update(loss.item(), X.size(0))
        pred_ab = (pred_ab+128)/255
        print(pred_ab, y)

        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.average:.4f})\t'.format(
                i, len(testData), batch_time=run_time, loss=losses))

    print('Finished validation.')
    return losses.average


def train(trainData, model, criterion, optimizer, epoch, is_mean):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    run_time, data_time, losses = Aggregator(), Aggregator(), Aggregator()

    end = time.time()
    for i, (X, y, target) in enumerate(trainData):

        # Use GPU if available
        if use_gpu:
            X, y, target = X.cuda(), y.cuda(), target.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        pred_ab = model(X)
        #     print(pred_ab.shape)
        loss = criterion(pred_ab, y)
        losses.update(loss.item(), X.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        run_time.update(time.time() - end)
        end = time.time()
        #     print(i)
        # Print model accuracy -- in the code below, value refers to value, not validation
        if i % 25 == 0:
            if is_mean:
                logfile = open('/home/sbanda/Fall20-DL-CG/Project2/Final_code/logs/mean.txt', 'a')
            else:
                logfile = open('/home/sbanda/Fall20-DL-CG/Project2/Final_code/logs/channel.txt', 'a')
            logfile.write('Epoch: [{0}][{1}/{2}]\t'
                          'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                          'Data {data_time.value:.3f} ({data_time.average:.3f})\t'
                          'Loss {loss.value:.4f} ({loss.average:.4f})\t\n'.format(
                epoch, i, len(trainData), batch_time=run_time,
                data_time=data_time, loss=losses))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                  'Data {data_time.value:.3f} ({data_time.average:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.average:.4f})\t\n'.format(
                epoch, i, len(trainData), batch_time=run_time,
                data_time=data_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))

def validateChannel(testData, model, criterion):
    model.eval()

    # Prepare value counters and timers
    run_time, data_time, losses = Aggregator(), Aggregator(), Aggregator()

    end = time.time()
    already_saved_images = False
    for i, (X, y, target) in enumerate(testData):
        data_time.update(time.time() - end)

        # Use GPU
        if use_gpu: 
            X, y, target = X.cuda(), y.cuda(), target.cuda()

        # Run model and record loss
        pred_ab = model(X)  # throw away class predictions
        # print(pred_ab.shape)
        # print(y.shape)
        loss = criterion(pred_ab, y)
        losses.update(loss.item(), X.size(0))
        for j in range(min(len(pred_ab), 10)): # save at most 5 images
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
            save_name = 'img-{}.jpg'.format(i * testData.batch_size + j)
            convertToRGB(X[j].cpu(), ab_input=pred_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
        # Record time to do forward passes and save images
        run_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, value refers to both value and validation
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.average:.4f})\t'.format(
                i, len(testData), batch_time=run_time, loss=losses))

    print('Finished validation.')
    return losses.average