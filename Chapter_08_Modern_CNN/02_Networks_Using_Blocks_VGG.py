"""
While AlexNet offered empirical evidence that deep CNNs can achieve good results, it did not provide a general template
to guide subsequent researchers in designing new networks. In the following sections, we will introduce several
heuristic concepts commonly used to design deep networks.

Progress in this field mirrors that of VLSI (very large scale integration) in chip design where engineers moved from
placing transistors to logical elements to logic block. Similarly, the design of neural network architectures has grown
progressively more abstract, with researchers moving from thinking in terms of individual neurons to whole layers, and
now to blocks, repeating patterns of layers.

The idea of using blocks first emerged from the Visual Geometry Group (VGG) at Oxford University.
It is easy to implement these repeated structures in code with any modern deep learning framework by using loops and
subroutines.

****** VGG Blocks
The basic building block of CNNs is a sequence of the following:
1. a convolutional layer with padding to maintain the resolution
2. a non-linearity such as a ReLU
3.  a pooling layer such as max-pooling to reduce the resolution
One of the problems with this approach is that the spatial resolution decreases quite rapidly
. In particular, this imposes a hard limit of log( d ) convolutional layers on the network before all dimensions (d) are
used up. For instance, in the case of ImageNet, it would be impossible to have more than 8 convolutional layers in this
way.

The key idea by Simonyan and Zisserman was to use multiple convolutions in between downsampling via max-pooling in the
form of a block. They were primarily interested in whether deep or wide networks perform better.

A VGG block consists of a sequence of convolutions with 3x3 kernels with padding of 1 (keeping height and width)
followed by a 2 x 2 max-pooling layer with stride of 2 (halving height and width after each block).


 """

import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


"""
******* VGG Network
Like AlexNet and LeNet, the VGG Network can be partitioned into two parts: 
1. the first consisting mostly of convolutional and pooling layers
2. the second consisting of fully connected layers that are identical to those in AlexNet

The key difference is that the convolutional layers are grouped in nonlinear transformations that leave the 
dimensonality unchanged, followed by a resolution-reduction step. 

"""

class VGG(d2l.Classifier):
    def __init__(self, arch, lr = 0.1, num_classes = 10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = []
        in_chanels = 1
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)


"""
The original VGG network had 5 convolutional blocks, among which the first two have one convolutional layer each and the 
latter three contain two convolutional layers each. The first block has 64 output channels and each subsequent block 
doubles the number of output channels, until that number reaches 512. Since this network uses 8 convolutional layers and
3 fully connected layers, it is often called VGG-11.
"""

VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))

"""
One might argue that VGG is the first truly modern convolutional neural network. While AlexNet introduced many of the 
components of what make deep learning effective at scale, it is VGG that arguably introduced key properties such as 
blocks of multiple convolutions and a preference for deep and narrow networks. It is also the first network that is 
actually an entire family of similarly parametrized models, giving the practitioner ample trade-off between complexity 
and speed. This is also the place where modern deep learning frameworks shine. It is no longer necessary to generate XML 
config files to specify a network but rather, to assmple said networks through simple Python code.
"""