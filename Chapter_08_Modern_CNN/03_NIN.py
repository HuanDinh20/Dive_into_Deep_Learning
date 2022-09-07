"""
LeNet, AlexNet, and VGG all share a common design pattern: extract features exploiting spatial structure via a sequence of convolutions and pooling layers and post-process the representations via fully connected layers. The improvements upon LeNet by AlexNet and VGG mainly lie in how these later networks widen and deepen these two modules.

This design poses two major challenges:
 1. First, the fully connected layers at the end of the architecture consume tremendous numbers of parameters.
 2.  Second, it is equally impossible to add fully connected layers earlier in the network to increase the degree of
 non-linearity: doing so would destroy the spatial structure and require potentially even more memory.
The network in network (NiN) blocks offer an alternative, capable of solving both problems in one simple strategy.
They were proposed based on a very simple insight:

1.  use 1 x 1 convolutions to add local nonlinearities across the channel activations
2. use global average pooling to integrate across all locations in the last representation layer

***************  NiN Blocks  ********************
 the inputs and outputs of convolutional layers consist of four-dimensional tensors with axes corresponding to
 the example, channel, height, and width. Also recall that the inputs and outputs of fully connected layers are
 typically two-dimensional tensors corresponding to the example and feature. The idea behind NiN is to apply a fully
 connected layer at each pixel location (for each height and width). The resulting 1 x 1 convolution can be thought as a
 fully connected layer acting independently on each pixel location.

"""

import torch
from torch import nn
from d2l import torch as d2l


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())


"""
******************** NiN Model ************************
    
    NiN uses the same initial convolution sizes as AlexNet. The kernel sizes are 11 x 11, 5 x 5, and 3 x 3, respectively,
and the numbers of output channels match those of AlexNet. Each NiN block is followed by a max-pooling layer with a 
stride of 2 and a window shape of 3 x 3.

    The second significant difference between NiN and both AlexNet and VGG is that NiN avoids fully connected layers 
altogether. Instead, NiN uses a NiN block with a number of output channels equal to the number of label classes, 
followed by a global average pooling layer, yielding a vector of logits.This design significantly reduces the number 
of required model parameters, albeit at the expense of a potential increase in training time.

"""


class Nin(d2l.Classifier):
    def __init__(self, lr = 0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.ner = nn.Sequential(
            nin_block(), nn.MaxPool2d(),
            nin_block(), nn.MaxPool2d(),
            nin_block(), nn.MaxPool2d(),
            nn.Dropout2d(),
            nin_block(), nn.AdaptiveAvgPool2d(),
            nn.Flatten())
        self.net.apply(d2l.init_cnn)


model = Nin()
X = torch.rand(1, 1, 227, 224)
for layer in model.net:
    X =layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

"""
**********************  Summary
NiN has dramatically fewer parameters than AlexNet and VGG. This stems from the fact that it needs no giant fully 
connected layers and fewer convolutions with wide kernels. Instead, it uses local 1 x 1  convolutions and global average 
pooling. These design choices influenced many subsequent CNN designs.
******************** Questions **************
1. Why are there two 1 × 1 convolutional layers per NiN block? What happens if you add one?
What happens if you reduce this to one?
    1.1.    The depth of the input or number of filters used in convolutional layers often increases with the depth of 
    the network, resulting in an increase in the number of resulting feature maps. It is a common model design pattern.
    A large number of feature maps in a convolutional neural network can cause a problem as a convolutional operation 
    must be performed down through the depth of the input, as it can result in considerably more parameters (weights) 
    and, in turn, computation to perform the convolutional operations (large space and time complexity).
    Pooling layers are designed to downscale feature maps and systematically halve the width and height of feature maps
    in the network. Nevertheless, pooling layers do not change the number of filters in the model, the depth, or number 
    of channels.Deep convolutional neural networks require a corresponding pooling type of layer that can downsample or 
    reduce the depth or number of feature maps.
        1.1.1: 
            * 1x1 -> control the number of feature maps
            * A 1×1 filter will only have a single parameter or weight for each channel in the input -> results in a 
            single output value -> resulting in a feature map with the same width and height as the input
            * it is a linear weighting or projection of the input. a nonlinearity is used as with other convolutional 
            layers, allowing the projection to perform non-trivial computation on the input feature maps.
            * a way to usefully summarize the input feature maps, allows the tuning of the number of summaries of the 
            input feature maps to create, effectively allowing the depth of the feature maps to be increased or 
            decreased as needed.
            * be used at any point in a convolutional neural network to control the number of feature maps. As such, 
            it is often referred to as a projection operation or projection layer, or even a feature map or channel 
            pooling layer.
            * 

"""

