import torch
import torch.nn as nn
from torchvision import datasets, transforms

from ColorizationNet import ColorizationNet
from ImageLoaderMean import ImageLoaderMean,ImageLoaderABchannel
from utils import *
from dataPrep import *

def predictMean(model_path, test_path):
    # Create Model Object
    model = ColorizationNet(True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    # Use GPU, when available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        # Load data from pickel file
        model_params = torch.load(model_path)
    else:
        # Load data from pickel file
        model_params = torch.load(model_path,map_location=torch.device('cpu'))
    
    # load parameteres to the model
    model.load_state_dict(model_params)
    model.eval()
    
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
#     val_imagefolder = ImageLoader('/blue/cis6930/sbanda/color/' , val_transforms)
    val_imagefolder = ImageLoaderMean(test_path , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

    losses = validateMean(val_loader, model, criterion)
    
    print(losses)

res_path = '/blue/cis6930/sbanda/result/'
def colorize(model_path, test_path):
    # Create Model Object
    model = ColorizationNet(False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    # Use GPU, when available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        # Load data from pickel file
        model_params = torch.load(model_path)
    else:
        # Load data from pickel file
        model_params = torch.load(model_path,map_location=torch.device('cpu'))
    
    # load parameteres to the model
    model.load_state_dict(model_params)
    model.eval()
    
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
#     val_imagefolder = ImageLoader('/blue/cis6930/sbanda/color/' , val_transforms)
    val_imagefolder = ImageLoaderABchannel(test_path , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

    losses = validateChannel(val_loader, model, criterion,res_path)
    
    print(losses)