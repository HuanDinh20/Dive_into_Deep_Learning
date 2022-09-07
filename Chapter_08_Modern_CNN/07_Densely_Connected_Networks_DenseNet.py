"""
ResNet significantly changed the view of how to parametrize the functions in deep networks.
DenseNet is characterized by both the connectivity pattern where each layer connects to all the preceding layers and
the concatenation operation (rather than the addition operator in ResNet) to preserve and reuse features from earlier
layers.
**************** From ResNet to DenseNet ********************
ResNet decomposes f into a simple linear term and a more complex nonlinear one.
What if we want to capture (not necessarily add) information beyond two terms? One solution was DenseNet
x -> [x, f1(x), f2(x, f1(x)), f3([x, f1(x), f2(x, f1(x))])....)
In the end, all these functions are combined in MLP to reduce the number of features again. In terms of implementation
this is quite simple: rather than adding terms, we concatenate them. The name DenseNet arises from the fact that the
dependency graph between variables becomes quite dense. The last layer of such a chain is densely connected to all
previous layers.

The main components that compose a DenseNet are :
1.  dense blocks
2.  transition layers

****************** Dense Blocks ****************
DenseNet uses the modified “batch normalization, activation, and convolution” structure of ResNet

"""
import torch
from torch import nn
from d2l import torch as d2l


def conv_block(num_channels):
    """Block structure"""
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    """A dense block consists of multiple convolution blocks, each using the same number of output channels. In the
    forward propagation, however, we concatenate the input and output of each convolution block on the channel dimension."""
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X

blk = DenseBlock(2, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape

"""
Since each dense block will increase the number of channels, adding too many of them will lead to an excessively complex 
model. A transition layer is used to control the complexity of the model. It reduces the number of channels by using 
the 1 x 1 convolutional layer and halves the height and width of the average pooling layer with a stride of 2, further 
reducing the complexity of the model.
"""

def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

blk = transition_block(10)
blk(Y).shape


class DenseNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.b1())
    for i, num_convs in enumerate(arch):
        self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                          growth_rate))
        # The number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(arch) - 1:
            num_channels //= 2
            self.net.add_module(f'tran_blk{i+1}', transition_block(
                num_channels))
    self.net.add_module('last', nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.LazyLinear(num_classes)))
    self.net.apply(d2l.init_cnn)


model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)


"""
The main components that compose DenseNet are dense blocks and transition layers. For the latter, we need to keep the 
dimensionality under control when composing the network by adding transition layers that shrink the number of channels 
again. In terms of cross-layer connections, unlike ResNet, where inputs and outputs are added together, DenseNet 
concatenates inputs and outputs on the channel dimension. Although these concatenation operations reuse features to 
achieve computational efficiency, unfortunately they lead to heavy GPU memory consumption. As a result, applying 
DenseNet may require more complex memory-efficient implementations that may increase training time
"""