import os.path
import shutil

import skimage.io as io
import numpy as np
from skimage.color import rgb2lab

import torch.nn as nn
import torch.nn.functional as F
import torch


from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
#import matplotlib.pyplot as plt

import os
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

rootdir = '/blue/cis6930/sbanda/images/Augmented/'  
newrootdir = '/blue/cis6930/sbanda/images/'





for parent, dirnames, filenames in os.walk(rootdir): 
    # for dirname in dirnames:                            
    #     print("parent is:" + parent)
    #     print("dirname is:" + dirname)
    # messagefile = open('./deletemessage.txt', 'a')
    # messagefile.write(os.path.split(parent)[1])
    # messagefile.close()

    for filename in filenames:  
        # print("parent is:" + parent)
        # print("filename is:" + filename)
        # print("the full name of the file is:" + os.path.join(parent,filename)) 
        path = os.path.join(parent, filename)
        img = io.imread(path)

        # remove gray image of 1 channel
        if len(img.shape) == 2:
            newpath = os.path.join(newrootdir, os.path.split(parent)[1])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            shutil.move(path, newpath)

        else:
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            num = r.size
            num1 = img.size

            # remove gray image of 3 channel
            if np.sum(abs(r - g) < 30) / num > 0.9 and np.sum(abs(r - b) < 30) / num > 0.9:
                newpath = os.path.join(newrootdir, os.path.split(parent)[1])
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                shutil.move(path, newpath)

            else:
                # remove image with small color variance
                img = rgb2lab(img)
                ab = [img[:, :, 1], img[:, :, 2]]
                variance = np.sqrt(np.sum(np.power(ab[0] - np.sum(ab[0]) / num, 2)) / num) + \
                           np.sqrt(np.sum(np.power(ab[1] - np.sum(ab[1]) / num, 2)) / num)
                if variance < 6:
                    newpath = os.path.join(newrootdir, os.path.split(parent)[1])
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    shutil.move(path, newpath)



class LowLevelFeatNet(nn.Module):
    def __init__(self):
        super(LowLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = F.relu(self.bn6(self.conv6(x1)))
        if self.training:
            x2 = x1.clone()
        else:
            x2 = F.relu(self.bn1(self.conv1(x2)))
            x2 = F.relu(self.bn2(self.conv2(x2)))
            x2 = F.relu(self.bn3(self.conv3(x2)))
            x2 = F.relu(self.bn4(self.conv4(x2)))
            x2 = F.relu(self.bn5(self.conv5(x2)))
            x2 = F.relu(self.bn6(self.conv6(x2)))
        return x1, x2


class MidLevelFeatNet(nn.Module):
    def __init__(self):
        super(MidLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GlobalFeatNet(nn.Module):
    def __init__(self):
        super(GlobalFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(25088, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 25088)
        x = F.relu(self.bn5(self.fc1(x)))
        output_512 = F.relu(self.bn6(self.fc2(x)))
        output_256 = F.relu(self.bn7(self.fc3(output_512)))
        return output_512, output_256


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 205)
        self.bn2 = nn.BatchNorm1d(205)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.log_softmax(self.bn2(self.fc2(x)))
        return x


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, mid_input, global_input):
        w = mid_input.size()[2]
        h = mid_input.size()[3]
        global_input = global_input.unsqueeze(2).unsqueeze(2).expand_as(mid_input)
        fusion_layer = torch.cat((mid_input, global_input), 1)
        fusion_layer = fusion_layer.permute(2, 3, 0, 1).contiguous()
        fusion_layer = fusion_layer.view(-1, 512)
        fusion_layer = self.bn1(self.fc1(fusion_layer))
        fusion_layer = fusion_layer.view(w, h, -1, 256)

        x = fusion_layer.permute(2, 3, 0, 1).contiguous()
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = self.upsample(x)
        x = F.sigmoid(self.bn5(self.conv4(x)))
        x = self.upsample(self.conv5(x))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.low_lv_feat_net = LowLevelFeatNet()
        self.mid_lv_feat_net = MidLevelFeatNet()
        self.global_feat_net = GlobalFeatNet()
        self.class_net = ClassificationNet()
        self.upsample_col_net = ColorizationNet()

    def forward(self, x1, x2):
        x1, x2 = self.low_lv_feat_net(x1, x2)
        #print('after low_lv, mid_input is:{}, global_input is:{}'.format(x1.size(), x2.size()))
        x1 = self.mid_lv_feat_net(x1)
        #print('after mid_lv, mid2fusion_input is:{}'.format(x1.size()))
        class_input, x2 = self.global_feat_net(x2)
        #print('after global_lv, class_input is:{}, global2fusion_input is:{}'.format(class_input.size(), x2.size()))
        class_output = self.class_net(class_input)
        #print('after class_lv, class_output is:{}'.format(class_output.size()))
        output = self.upsample_col_net(x1, x2)
        #print('after upsample_lv, output is:{}'.format(output.size()))
        return class_output, output



scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    #transforms.ToTensor()
])


class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        # print(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img_original, img_ab), target


class ValImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img_scale = img.copy()
        img_original = img
        img_scale = scale_transform(img_scale)

        img_scale = np.asarray(img_scale)
        img_original = np.asarray(img_original)

        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)
        return (img_original, img_scale), target


original_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

have_cuda = torch.cuda.is_available()
print(have_cuda)
epochs = 10

data_dir = "/blue/cis6930/sbanda/images/"
train_set = TrainImageFolder(data_dir, original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
color_model = ColorNet()
if os.path.exists('./colornet_params.pkl'):
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())


def train(epoch):
    color_model.train()

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            messagefile = open('./message.txt', 'a')
            original_img = data[0].unsqueeze(1).float()
            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()
            class_output, output = color_model(original_img, original_img)
            ems_loss = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod()
            cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)
            loss = ems_loss + cross_entropy_loss
            # lossmsg = 'loss: %.9f\n' % (loss.data[0])
            # messagefile.write(lossmsg)
            # ems_loss.backward(retain_variables=True)
            ems_loss.backward(retain_graph = True)
            cross_entropy_loss.backward()
            optimizer.step()
            if batch_idx % 4 == 0:
                # print(loss)
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.tolist())
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'colornet_params.pkl')
                messagefile.close()
                print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx * len(data)/ len(train_loader), loss.data.tolist()))
    except Exception:
        logfile = open('log.txt', 'a')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params.pkl')


for epoch in range(1, epochs + 1):
    print(epoch)
    train(epoch)