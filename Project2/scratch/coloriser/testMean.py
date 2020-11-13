import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils import *
from ImageLoaderMean import *
from ColorizationNet import *
model = ColorizationNet(True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
    loadModel = torch.load('/home/sbanda/Fall20-DL-CG/Project2/Final_code/model_mean.pkl')
else:
    loadModel = torch.load('/home/sbanda/Fall20-DL-CG/Project2/Final_code/model_mean.pkl', map_location=torch.device('cpu'))

model.load_state_dict(loadModel)

model.eval()
testTransforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
testImageData = ImageLoaderMean('/blue/cis6930/sbanda/color/', testTransforms)
testData = torch.utils.data.DataLoader(testImageData, batch_size=64, shuffle=False)

losses = validateMean(testData, model, criterion)