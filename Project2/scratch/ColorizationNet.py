
import torch.nn as nn                                                                                           # Torch library - For implementing neural network and other image utils
import torchvision.models as models                                                                             # Pre-built models 

class ColorizationNet(nn.Module):
    '''
    ColorizationNet implements neural network model - useful to train and test on the data.
    - Its purpose is to output average a,b values in L*a*b image.
    - It contains multiple convolution layers followed by 4 fully connected layers.
    - Performed batch normalization after every convolution layer to solve exploding and vanishing gradient problem
    '''
    def __init__(self, is_mean,input_size=128):
        super(ColorizationNet, self).__init__()
        # To extract secondary features
        SECONDARY_FEATURE_SIZE = 128
        # useful to separate model from predicting mean values(of size (2,1)) and a,b channels (of size 128*128))
        self.is_Mean = is_mean
        # Utilizing pre-built resnet18 from torch library
        residualNetwork = models.resnet18(num_classes=365)

        residualNetwork.conv1.weight = nn.Parameter(residualNetwork.conv1.weight.sum(dim=1).unsqueeze(1))

        self.secondary_resnet = nn.Sequential(*list(residualNetwork.children())[0:6])

        # Upsampling the image
        self.upsampling = nn.Sequential(
        nn.Conv2d(SECONDARY_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=2)
        )
        # Connecting convolution layers to fully connected layers
        self.fullyConnected1 = nn.Linear(100352, 1024)
        self.fullyConnected2 = nn.Linear(1024, 128)
        self.fullyConnected3 = nn.Linear(128, 16)
        self.fullyConnected4 = nn.Linear(16, 2)

    def forward(self, input):

        # Pass L channel as input into ResidualNet gray
        secondaryFeatures = self.secondary_resnet(input)

        # Upsampling the output to get colors
        output = self.upsampling(secondaryFeatures)
        # If is_mean is true - only outputs mean a,b. Otherwise outputs (128,128) output of a,b channels
        if self.is_Mean:
            output = output.view(output.size(0), -1)
            output = self.fullyConnected1(output)
            output = self.fullyConnected2(output)
            output = self.fullyConnected3(output)
            output = self.fullyConnected4(output)

        return output
