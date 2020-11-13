import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import torch,cv2
import time

# If GPU available, use it
use_gpu = torch.cuda.is_available()


class Aggregator(object):
    """
    Aggregator class is used for applying aggregate functions
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets aggregator object's value, average, sum, and count
        :return: object
        """
        self.value, self.average, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        """
        Updates aggregator object's value, average, sum, and count
        :param val: updating value
        :param n: number of iterations
        :return: object
        """
        self.value = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def convertToRGB(grayImage, ab_input, save_path=None, save_name=None):
    """
    Converts L (Gray Image) to RGB channeled image
    :param grayImage: gray scale image
    :param ab_input: a & b channels of the LAB input image
    :param save_path: boolean value to either store the output or not
    :param save_name: name of the output image file
    :return: void
    """
    # clears plots of MATPLOTLIB
    plt.clf()

    # concatenates the gray image(L channel) with a & b channels
    originalImage = torch.cat((grayImage, ab_input), 0).numpy()

    # transposes the image
    originalImage = originalImage.transpose((1, 2, 0))

    # Normalises on different channel images
    originalImage[:, :, 0:1] = originalImage[:, :, 0:1] * 100
    originalImage[:, :, 1:3] = originalImage[:, :, 1:3] * 255 - 128

    # Converts LAB image to RGB image
    originalImage = lab2rgb(originalImage.astype(np.float64))
    
    # Converts grayscale to numpy array
    grayImage = grayImage.squeeze().numpy()

    # Saves the X and y_hat images to the respective files
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayImage, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=originalImage, fname='{}{}'.format(save_path['colorized'], save_name))
    return grayImage,originalImage


def train(trainData, model, criterion, optimizer, epoch, is_mean):
    """
    Trains the defined model
    :param trainData: data set for training
    :param model: defined model
    :param criterion: criterion for losses
    :param optimizer: used for optimizing the model
    :param epoch: epoch ID
    :param is_mean: used for mean value regression
    :return: void
    """
    # Start status of each epoch
    print('Starting training epoch {}'.format(epoch))
    # Starts to train
    model.train()

    # Defining counters and timers
    run_time, data_time, losses = Aggregator(), Aggregator(), Aggregator()
    # Starts time unit
    end = time.time()

    # Iterates on train data set for losses
    for i, (X, y, target) in enumerate(trainData):
        # Updates time unit
        data_time.update(time.time() - end)

        # Uses GPU, if available
        if use_gpu:
            X, y, target = X.cuda(), y.cuda(), target.cuda()

        # Runs model and stores losses
        pred_ab = model(X)
        # print(pred_ab.shape)
        loss = criterion(pred_ab, y)
        losses.update(loss.item(), X.size(0))

        # Computes zero gradient and optimizes the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stores & updates the time for all passes
        run_time.update(time.time() - end)
        end = time.time()

        # Printing model accuracy and losses and writes it to a log file
        if i % 25 == 0:
            if is_mean:
                logfile = open('/home/sbanda/Fall20-DL-CG/Project2/Finale/logs/mean.txt', 'a')
            else:
                logfile = open('/home/sbanda/Fall20-DL-CG/Project2/Finale/logs/channel.txt', 'a')
            logfile.write('Epoch: [{0}][{1}/{2}]\t'
                          'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                          'Data {data_time.value:.3f} ({data_time.average:.3f})\t'
                          'Loss {loss.value:.9f} ({loss.average:.9f})\t\n'.format(
                epoch, i, len(trainData), run_time=run_time,
                data_time=data_time, loss=losses))

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                  'Data {data_time.value:.3f} ({data_time.average:.3f})\t'
                  'Loss {loss.value:.9f} ({loss.average:.9f})\t\n'.format(
                epoch, i, len(trainData), run_time=run_time,
                data_time=data_time, loss=losses))

    # prints when finished
    print('Finished training epoch {}'.format(epoch))


def validateMean(testData, model, criterion):
    """
    Validates/Tests the trained model of mean of (a, b) channels on test data to predict outputs (y_hats)
    :param testData: data set used for testing/validating
    :param model: trained model
    :param criterion: loss criterion
    :return: average of losses
    """
    # Evaluates model
    model.eval()

    # Defining counters and timers
    run_time, data_time, losses = Aggregator(), Aggregator(), Aggregator()

    # starting time unit
    end = time.time()

    # already_saved_images = False
    # Iterates on test data set for predictions
    for i, (X, y, target) in enumerate(testData):
        # Updates time unit
        data_time.update(time.time() - end)

        # Uses GPU, if available
        if use_gpu:
            X, y, target = X.cuda(), y.cuda(), target.cuda()

        # Runs model and stores losses
        pred_ab = model(X)
        # print(pred_ab.shape)
        loss = criterion(pred_ab, y)
        losses.update(loss.item(), X.size(0))

        # Normalize predictions to compare with original (y_hat with y)
        pred_ab = (pred_ab + 128) / 255
        print(pred_ab, y)

        # Printing status to console
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                  'Loss {loss.value:.9f} ({loss.average:.9f})\t'.format(
                i, len(testData), run_time=run_time, loss=losses))

    # prints when finished
    print('Finished validation.')

    # returns average of all losses
    return losses.average


def validateChannel(testData, model, criterion,result_path):
    """
    Validates/Tests the trained model of (a, b) channels on test data to predict outputs (y_hats)
    :param testData: data set used for testing/validating
    :param model: trained model
    :param criterion: loss criterion
    :return: average of losses
    """
    # Evaluates model
    model.eval()

    # Defining counters, timers and starts a time unit
    run_time, data_time, losses = Aggregator(), Aggregator(), Aggregator()
    end = time.time()

    # already_saved_images = False
    # Iterates through test data set for predictions
    for i, (X, y, target) in enumerate(testData):
        # updates time unit
        data_time.update(time.time() - end)

        # Use GPU, if available
        if use_gpu: 
            X, y, target = X.cuda(), y.cuda(), target.cuda()

        # Runs model and stores losses
        pred_ab = model(X)
        # print(pred_ab.shape)
        loss = criterion(pred_ab, y)
        losses.update(loss.item(), X.size(0))
        
        # Saves gray scale and predicted images
        for j in range(min(len(pred_ab), 10)):
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
            save_name = 'img-{}.jpg'.format(i * testData.batch_size + j)
            grayImage,colorizedImage = convertToRGB(X[j].cpu(), ab_input=pred_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
            #grayImage,colorizedImage = cv2.resize(grayImage,(128,128)),cv2.resize(colorizedImage,(128,128))
            #result.append(cv2.hconcat([grayImage,colorizedImage]))
            
        #cv2.imwrite(result_path+str(i)+".jpg",cv2.vconcat(result))
        
        # Stores & updates the time for all passes
        run_time.update(time.time() - end)
        end = time.time()

        # Prints model accuracy and losses
        if i % 25 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {run_time.value:.3f} ({run_time.average:.3f})\t'
                  'Loss {loss.value:.9f} ({loss.average:.9f})\t'.format(
                i, len(testData), run_time=run_time, loss=losses))

    # Prints when finished
    print('Finished validation.')

    # returns average of losses
    return losses.average