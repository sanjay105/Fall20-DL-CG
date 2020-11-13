
import torch.nn as nn
import torchvision.models as models

class ColorizationNet(nn.Module):
    def __init__(self, is_mean,input_size=128):
        super(ColorizationNet, self).__init__()
        SECONDARY_FEATURE_SIZE = 128
        self.is_Mean = is_mean
        ## First half: ResNet
        residualNetwork = models.resnet18(num_classes=365)

        residualNetwork.conv1.weight = nn.Parameter(residualNetwork.conv1.weight.sum(dim=1).unsqueeze(1))

        self.secondary_resnet = nn.Sequential(*list(residualNetwork.children())[0:6])

        ## Second half: Upsampling
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

        self.fullyConnected1 = nn.Linear(100352, 1024)
        self.fullyConnected2 = nn.Linear(1024, 128)
        self.fullyConnected3 = nn.Linear(128, 16)
        self.fullyConnected4 = nn.Linear(16, 2)

    def forward(self, input):

        # Pass input through ResNet-gray to extract features
        secondaryFeatures = self.secondary_resnet(input)

        # Upsample to get colors
        output = self.upsampling(secondaryFeatures)
        if self.is_Mean:
            output = output.view(output.size(0), -1)
            output = self.fullyConnected1(output)
            output = self.fullyConnected2(output)
            output = self.fullyConnected3(output)
            output = self.fullyConnected4(output)

        return output
