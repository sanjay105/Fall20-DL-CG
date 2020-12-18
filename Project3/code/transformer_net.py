import torch


class TransformerNet(torch.nn.Module):
    """
    Class for transformation network
    """
    def __init__(self, style_number):

        super(TransformerNet, self).__init__()

        self.convolution1 = ConvLayer(3, 32, kernel_size = 9, stride = 1)
        self.bn1 = batch_InstanceNorm2d(style_number, 32)

        self.convolution2 = ConvLayer(32, 64, kernel_size = 3, stride = 2)
        self.bn2 = batch_InstanceNorm2d(style_number, 64)

        self.convolution3 = ConvLayer(64, 128, kernel_size = 3, stride = 2)
        self.bn3 = batch_InstanceNorm2d(style_number, 128)

        self.residual_block1 = ResidualBlock(128)
        self.residual_block2 = ResidualBlock(128)
        self.residual_block3 = ResidualBlock(128)
        self.residual_block4 = ResidualBlock(128)
        self.residual_block5 = ResidualBlock(128)

        self.upsample1 = UpsampleConvLayer(128, 64, kernel_size = 3, stride = 1, upsample = 2)
        self.bn4 = batch_InstanceNorm2d(style_number, 64)

        self.upsample2 = UpsampleConvLayer(64, 32, kernel_size = 3, stride = 1, upsample = 2)
        self.bn5 = batch_InstanceNorm2d(style_number, 32)

        self.convolution4 = ConvLayer(32, 3, kernel_size = 9, stride = 1)
        self.relu = torch.nn.ReLU()

    def forward(self, X, style_id):
 
        y = self.relu(self.bn1(self.convolution1(X), style_id))
        # print(y.shape)
        y = self.relu(self.bn2(self.convolution2(y), style_id))
        # print(y.shape)
        y = self.relu(self.bn3(self.convolution3(y), style_id))
        # print(y.shape)
        y = self.residual_block1(y)
        # print(y.shape)
        y = self.residual_block2(y)
        # print(y.shape)
        y = self.residual_block3(y)
        # print(y.shape)
        y = self.residual_block4(y)
        # print(y.shape)
        y = self.residual_block5(y)
        # print(y.shape)
        y = self.relu(self.bn4(self.upsample1(y), style_id))
        # print(y.shape)
        y = self.relu(self.bn5(self.upsample2(y), style_id))
        # print(y.shape)
        y = self.convolution4(y)
        # print(y.shape)
        return y


class batch_InstanceNorm2d(torch.nn.Module):
    """
    Batch Instance Normalization
    introduced in https://arxiv.org/abs/1610.07629
    created and applied based on our limited understanding, there is a room for improvement
    """
    def __init__(self, style_num, input_channels):
        super(batch_InstanceNorm2d, self).__init__()
        self.inns = torch.nn.ModuleList([torch.nn.InstanceNorm2d(input_channels, affine=True) for i in range(style_num)])

    def forward(self, x, style_id):
        output = torch.stack([self.inns[style_id[i]](x[i].unsqueeze(0)).squeeze_(0) for i in range(len(style_id))])
        return output


class ConvLayer(torch.nn.Module):
    """
    Class for Convolutional Network Layer
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        # Similar dimension after padding
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride)

    def forward(self, x):
        output = self.reflection_pad(x)
        output = self.conv2d(output)
        return output


class ResidualBlock(torch.nn.Module):
    """
    Class for ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.convolution1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.convolution2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.in1(self.convolution1(x)))
        output = self.in2(self.convolution2(output))
        output = output + residual # need relu right after
        return output


class UpsampleConvLayer(torch.nn.Module):
    """
    Upsampling Convolution Layer Class
    Input is upsampled and convolution is applied on it.
    Better results are produced compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        output = self.reflection_pad(x_in)
        output = self.conv2d(output)
        return output