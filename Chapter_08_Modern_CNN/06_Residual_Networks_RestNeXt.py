"""
only if larger function classes contain the smaller ones are we guaranteed that increasing them strictly increases the
expressive power of the network. For deep neural networks, if we can train the newly-added layer into an identity
function f(x) = x, the new model will be as effective as the original model. As the new model may get a better solution
to fit the training dataset, the added layer might make it easier to reduce training errors.

****  Residual Blocks ****

Normal:
input x ----> block ----> f(x) ---> activation function

Residual:
input x ----> block -----> f(x) - x + x ----> activation function

Block:

3x3 Convd -> Batchnorm -> Relu -> Convd -> Batchnorm

"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


blk = Residual(3)
X = torch.randn(4, 3, 6, 6)
print(">>> Without 1x1 conv")
print(blk(X).shape)

blk2 = Residual(3, use_1x1conv=True)
print(">>> With 1x1 conv")
print(blk2(X).shape)

"""
************ ResNet Model ***********
The first two layers of ResNet are the same as those of the GoogLeNet
The difference is the batch normalization layer added after each convolutional layer in ResNet.
"""


class ResNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


"""
GoogLeNet uses four modules made up of Inception blocks. However, ResNet uses four modules made up of residual blocks, 
each of which uses several residual blocks with the same number of output channels. The number of channels in the first 
module is the same as the number of input channels. Since a max-pooling layer with a stride of 2 has already been used, 
it is not necessary to reduce the height and width. In the first residual block for each of the subsequent modules, the 
number of channels is doubled compared with that of the previous module, and the height and width are halved.
"""


@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)


@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.b1())
    for i, b in enumerate(arch):
        self.net.add_module(f'b{i + 2}', self.block(*b, first_block=(i == 0)))
    self.net.add_module('Last', nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.LazyLinear(num_classes)
    ))
    self.net.apply(d2l.init_cnn)

"""
****************** ResNet 18 *****************
"""

class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, num_classes)


ResNet18().layer_summary((1, 1, 96, 96))

"""
Sequential output shape:	 torch.Size([1, 64, 24, 24])
Sequential output shape:	 torch.Size([1, 64, 24, 24])
Sequential output shape:	 torch.Size([1, 128, 12, 12])
Sequential output shape:	 torch.Size([1, 256, 6, 6])
Sequential output shape:	 torch.Size([1, 512, 3, 3])
Sequential output shape:	 torch.Size([1, 10])


***************  ResNeXt ***************************
that each ResNet block simply stacks layers between residual connections. This design can be varied by replacing stacked 
layers with concatenated parallel transformations, leading to ResNeXt 
“bottlenecked” 
"""

class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1,
                               stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                               stride=strides, padding=1,
                               groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                               stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

