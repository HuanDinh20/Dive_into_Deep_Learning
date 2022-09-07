"""
    We mentioned that large datasets are a prerequisite for the success of deep neural networks in various applications.
mage augmentation generates similar but distinct training examples after a series of random changes to the training
mages, thereby expanding the size of the training set. Alternatively, image augmentation can be motivated by the fact
that random tweaks of training examples allow models to less rely on certain attributes, thereby improving their
generalization ability. For example, we can crop an image in different ways to make the object of interest appear
in different positions, thereby reducing the dependence of a model on the position of the object. We can also adjust
factors such as brightness and color to reduce a model’s sensitivity to color. It is probably true that image
augmentation was indispensable for the success of AlexNet at that time. In this section we will discuss this widely
used technique in computer vision.

**************** Common Image Augmentation Methods ******************

In our investigation of common image augmentation methods, we will use the following  image an 400 x 500 example.
Most image augmentation methods have a certain degree of randomness. To make it easier for us to observe the effect of
image augmentation, next we define an auxiliary function apply. This function runs the image augmentation method aug
multiple times on the input image img and shows all the results.


"""

import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('cat1.jpg')
d2l.plt.imshow(img)