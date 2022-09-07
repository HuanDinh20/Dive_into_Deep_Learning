"""
The 2010s has witnessed shift from feature engineering to network engineering in computer vision.
Specifically, neural architecture search (NAS) is the process of automating neural network architectures. Given a fixed
search space, NAS uses a search strategy to automatically select an architecture within the search space based on the
returned performance estimation. The outcome of NAS is a single network instance.
Instead of focusing on designing such individual instances, an alternative approach is to design network design spaces
that characterize populations of networks

*************** The AnyNet Design Space *************
The initial design space is called AnyNet, a relatively unconstrained design space, where we can focus on exploring
network structure assuming standard, fixed blocks such as ResNeXt. Specifically, the network structure includes elements
such as the number of blocks and the number of output channels in each stage, and the number of groups (group width) and
bottleneck ratio within each ResNeXt block.


"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())


@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)


@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.stem(stem_channels))
    for i, s in enumerate(arch):
        self.net.add_module(f'stage{i+1}', self.stage(*s))
    self.net.add_module('head', nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.LazyLinear(num_classes)))
    self.net.apply(d2l.init_cnn)


"""
*************** Constraining Design Spaces with Lower Error Distributions   *************

"""
