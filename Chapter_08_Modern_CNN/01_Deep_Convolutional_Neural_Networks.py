"""
***************** AlexNet *********************
input: image 3 * 224 * 224
-> Conv -> MaxPool - > Conv -> MaxPool -> Conv -> Conv -> Conv -> MaxPool -> Fc -> Fc -> FC

********** Architecture
In AlexNet’s first layer, the convolution window shape is 11x11. Since the images in ImageNet are eight times higher
and wider than the MNIST images, objects in ImageNet data tend to occupy more pixels with more visual detail.
Consequently, a larger convolution window is needed to capture the object. The convolution window shape in the second
layer is reduced to 5x5 followed by 3x3. In addition, after the first, second, and fifth convolutional layers,
the network adds max-pooling layers with a window shape of 3x3 and a stride of 2. Moreover, AlexNet has ten times more
convolution channels than LeNet.
After the last convolutional layer there are two fully connected layers with 4096 outputs. These two huge fully
connected layers produce model parameters of nearly 1 GB. Due to the limited memory in early GPUs, the original AlexNet
used a dual data stream design, so that each of their two GPUs could be responsible for storing and computing only its
half of the model. Fortunately, GPU memory is comparatively abundant now, so we rarely need to break up models across
GPUs these days (our version of the AlexNet model deviates from the original paper in this aspect).
*** Activation Funciton
Besides, AlexNet changed the sigmoid activation function to a simpler ReLU activation function. On the one hand, the
computation of the ReLU activation function is simpler. For example, it does not have the exponentiation operation
found in the sigmoid activation function. On the other hand, the ReLU activation function makes model training easier
when using different parameter initialization methods. This is because, when the output of the sigmoid activation
function is very close to 0 or 1, the gradient of these regions is almost 0, so that backpropagation cannot continue to
update some of the model parameters. In contrast, the gradient of the ReLU activation function in the positive interval
is always 1. Therefore, if the model parameters are not properly initialized, the sigmoid function may obtain a gradient \
of almost 0 in the positive interval, so that the model cannot be effectively trained.

*******  Capacity Control and Preprocessing

AlexNet controls the model complexity of the fully connected layer by dropout, while LeNet only uses weight decay. To
augment the data even further, the training loop of AlexNet added a great deal of image augmentation, such as flipping,
clipping, and color changes.This makes the model more robust and the larger sample size effectively reduces over-fitting

"""

import torch
from torch import nn
from d2l import torch as d2l

class AlexNet(d2l.Classifier):
    def __init__(self, lr = 0.1, num_classes = 10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),

            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )
        self.net.apply(d2l.init_cnn)


model = AlexNet()
model.layer_summary((1, 1, 224, 224))
"""
Conv2d output shape:	 torch.Size([1, 96, 54, 54])
ReLU output shape:	 torch.Size([1, 96, 54, 54])
MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
Conv2d output shape:	 torch.Size([1, 256, 26, 26])
ReLU output shape:	 torch.Size([1, 256, 26, 26])
MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
Conv2d output shape:	 torch.Size([1, 384, 12, 12])
ReLU output shape:	 torch.Size([1, 384, 12, 12])
Conv2d output shape:	 torch.Size([1, 384, 12, 12])
ReLU output shape:	 torch.Size([1, 384, 12, 12])
Conv2d output shape:	 torch.Size([1, 256, 12, 12])
ReLU output shape:	 torch.Size([1, 256, 12, 12])
MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
Flatten output shape:	 torch.Size([1, 6400])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 4096])
ReLU output shape:	 torch.Size([1, 4096])
Dropout output shape:	 torch.Size([1, 4096])
Linear output shape:	 torch.Size([1, 10])

"""
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)

"""
*** Discussion
AlexNet’s structure bears a striking resemblance to LeNet, with a number of critical improvements, both for accuracy 
(dropout) and for ease of training (ReLU). What is equally striking is the amount of progress that has been made in 
terms of deep learning tooling. What was several months of work in 2012 can now be accomplished in a dozen lines of code 
using any modern framework.

Reviewing the architecture, we see that AlexNet has an Achilles heel when it comes to efficiency: the last two hidden 
layers require matrices of size 6400 x 4096 and 4096 x 4096, respectively. This corresponds to 164 MB of memory and 81 
MFLOPs of computation, both of which are a nontrivial outlay, especially on smaller devices, such as mobile phones. 
Although it seems that there are only a few more lines in AlexNet’s implementation than in LeNet’s, it took the academic 
community many years to embrace this conceptual change and take advantage of its excellent experimental results. This 
was also due to the lack of efficient computational tools.
"""