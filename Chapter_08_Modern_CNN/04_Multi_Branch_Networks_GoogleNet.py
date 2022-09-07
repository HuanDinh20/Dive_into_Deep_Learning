"""
In 2014, GoogLeNet won the ImageNet Challenge using a structure that combined the strengths of NiN repeated blocks,
and a cocktail of convolution kernels.
It is arguably also the first network that exhibits a clear distinction among the stem, body, and head in a CNN.
This design pattern has persisted ever since in the design of deep networks:

1.  The stem is given by the first 2-3 convolutions that operate on the image. They extract low-level features from the
underlying images.
2.  This is followed by a body of convolutional blocks.
3.  The head maps the features obtained so far to the required classification, segmentation, detection, or tracking
problem at hand.

The key contribution in GoogLeNet was the design of the network body. It solved the problem of selecting convolution
kernels in an ingenious way. it simply concatenated multi-branch convolutions.

*********** Inception Blocks **************

The basic convolutional block in GoogLeNet is called an Inception block, stemming from the meme “we need to go deeper”
of the movie Inception.



"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each branch
    """To gain some intuition for why this network works so well, consider the combination of the filters. They explore
    the image in a variety of filter sizes. This means that details at different extents can be recognized efficiently
    by filters of different sizes. At the same time, we can allocate different amounts of parameters for different
    filters."""
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(F.relu(self.b4_1(x))))
        return torch.cat((b1, b2, b3, b4), dim=1)


"""
***************** GoogLeNet Model ***************
GoogLeNet uses a stack of a total of 9 inception blocks, arranged into 3 groups with max-pooling in between, and global 
average pooling in its head to generate its estimates. Max-pooling between inception blocks reduces the dimensionality. 
At its stem, the first module is similar to AlexNet and LeNet.

"""


class GoogLeNet(d2l.Classifier):

    def b1(self):
        """ The first module uses a 64-channel 7 x 7 convolutional layer"""
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


@d2l.add_to_class(GoogLeNet)
def b2(self):
    """The second module uses two convolutional layers:
    1.  first, a 64-channel 1 x 1  convolutional layer
    2.  3 x 3convolutional layer that triples the number of channels.
    This corresponds to the second branch in the Inception block and concludes the design of the body.
    At this point we have 192 channels.
    """
    return nn.Sequential(
        nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(192, kernel_size=3), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


@d2l.add_to_class(GoogLeNet)
def b3(self):
    """
    The third module connects two complete Inception blocks in series.
    The number of output channels of the first Inception block is 64 + 128 +32 + 32 = 256
    This amounts to a ratio of the number of output channels among the four branches of  2 : 4 : 1 : 1
    The number of output channels of the second Inception block is increased to 128 + 192 + 96 64 = 480
    yielding a ratio of 4 : 6 : 3 : 2
    As before, we need to reduce the number of intermediate dimensions in the second and third channel
    """
    return nn.Sequential(
        Inception(64 , (96, 128), (16, 32), 32),
        Inception(128, (238, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

@d2l.add_to_class(GoogLeNet)
def b4(self):
    """
    The fourth module is more complicated. It connects five Inception blocks in series

    """
    return nn.Sequential(
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

@d2l.add_to_class(GoogLeNet)
def b5(self):
    return nn.Sequential(
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    )

@d2l.add_to_class(GoogLeNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogLeNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(
        self.b1(), self.b2(), self.b3(), self.b4(),
        self.b5(), nn.LazyLinear(num_classes))
    self.net.apply(d2l.init_cnn)


model = GoogLeNet().layer_summary((1, 1, 96, 96))
"""
Sequential output shape:	 torch.Size([1, 64, 24, 24])
Sequential output shape:	 torch.Size([1, 192, 11, 11])
Sequential output shape:	 torch.Size([1, 480, 6, 6])
Sequential output shape:	 torch.Size([1, 832, 3, 3])
Sequential output shape:	 torch.Size([1, 1024])
Linear output shape:	 torch.Size([1, 10])
"""