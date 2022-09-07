"""
a single neuron :
    1.  takes some set of inputs
    2.  generates a corresponding scalar output
    3.  has a set of associated parameters that can be updated to optimize some objective function of interest

layers:
    1. Take a set of inputs
    2. Generates a corresponding output
    3. Described by a set of tuable parameters

for MLPs, both the entire model and its constituent layers share this structure. The entire model takes in raw inputs
(the features), generates outputs (the predictions), and possesses parameters (the combined parameters from all
constituent layers)

To implement these complex networks, we introduce the concept of a neural network module.
A module could describe a single layer, a component consisting of multiple layers, or the entire model itself! One
benefit of working with the module abstraction is that they can be combined into larger artifacts, often recursively.

From a programming standpoint, a module is represented by a class. Any subclass of it must define a forward propagation
method that transforms its input into output and must store any necessary parameters. Note that some modules do not
require any parameters at all. Finally a module must possess a backpropagation method, for purposes of calculating
gradients. Fortunately, due to some behind-the-scenes magic supplied by the auto differentiation when defining our own
module, we only need to worry about parameters and the forward propagation method.

 The forward propagation (forward) method is also remarkably simple: it chains each module in the list together,
 passing the output of each as input to the next.
"""

import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

print("Torch Seuqtial shape: ")
X = torch.rand(2, 20)
print(net(X).shape)


"""
*********************** A Custom Module **************************

Perhaps the easiest way to develop intuition about how a module works is to implement one ourselves. Before we implement 
our own custom module, we briefly summarize the basic functionality that each module must provide:

1. Ingest input data as arguments to its forward propagation method
2. Generate an output by having the forward propagation method return a value. Note that the output may have a different 
shape from the input. 
3. Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation method. 
Typically this happens automatically.
4. Store and provide access to those parameters necessary to execute the forward propagation computation.
5. Initialize model parameters as needed.


"""

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        """Note that it takes X as input, calculates the hidden representation
        with the activation function applied, and outputs its logits."""
        return self.out(F.relu(self.hidden(X)))


"""
*********************** The Sequential Module **********************
Sequential was designed to daisy-chain other modules together.
define two key methods:
1. A method to append modules one by one to a list
2. A forward propagation method to pass an input through the chain of modules, in the same order as they were appended.


"""


class MySequential(nn.Module):
    def __init__(self, *arg):
        """A forward propagation method to pass an input through the chain of modules,
        in the same order as they were appended.
        These modules can be accessed by the children method later. In this way the system knows the added modules,
        and it will properly initialize each module’s parameters."""
        super(MySequential, self).__init__()
        for idx, module in enumerate(arg):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X


net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print("shape of sequential")
print(net(X).shape)

"""
************ Executing Code in the Forward Propagation Method **************
The Sequential class makes model construction easy, allowing us to assemble new architectures without having to define 
our own class. However, not all architectures are simple daisy chains. When greater flexibility is required, we will 
want to define our own blocks. For example, we might want to execute Python’s control flow within the forward 
propagation method. Moreover, we might want to perform arbitrary mathematical operations, not simply relying on 
predefined neural network layers.

You might have noticed that until now, all of the operations in our networks have acted upon our network’s activations 
and its parameters. Sometimes, however, we might want to incorporate terms that are neither the result of previous 
layers nor updatable parameters. We call these constant parameters.
"""


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


"""
In this FixedHiddenMLP model, we implement a hidden layer whose weights (self.rand_weight) are initialized randomly at 
instantiation and are thereafter constant. This weight is not a model parameter and thus it is never updated by 
backpropagation. The network then passes the output of this “fixed” layer through a fully connected layer.

We can mix and match various ways of assembling modules together. In the following example, we nest modules in some creative ways.
"""


class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
print("Chimera Park: ")
print(chimera(X))

"""
************ Summary *******
1. Layers are modules.
2. Many layers can comprise a module.
3. Many modules can comprise a module.
4. A module can contain code.
5. Modules take care of lots of housekeeping, including parameter initialization and backpropagation.
6. Sequential concatenations of layers and modules are handled by the Sequential module.
"""