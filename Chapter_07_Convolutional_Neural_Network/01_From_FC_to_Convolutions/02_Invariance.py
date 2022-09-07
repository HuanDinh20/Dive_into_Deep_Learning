"""
Imagine that you want to detect an object in an image. It seems reasonable that whatever method we use to recognize
objects should not be overly concerned with the precise location of the object in the image. Ideally, our system should
exploit this knowledge. Pigs usually do not fly and planes usually do not swim. Nonetheless, we should still recognize
a pig were one to appear at the top of the image. CNNs systematize this idea of spatial invariance, exploiting it to
learn useful representations with fewer parameters.
We can now make these intuitions more concrete by enumerating a few desiderata to guide our design of a neural
network architecture suitable for computer vision:

1. In the earliest layers, our network should respond similarly to the same patch, regardless of where it appears in the
 image. This principle is called translation invariance.
2. The earliest layers of the network should focus on local regions, without regard for the contents of the image
in distant regions. This is the locality principle. Eventually, these local representations can be aggregated to make
predictions at the whole image level.

************************ Translation Invariance **********************
Translation invariance in images implies that all patches of an image will be treated in the same manner.

Translation invariance in images implies that all patches of an image will be treated in the same manner.
Translation invariance in images implies that all patches of an image will be treated in the same manner.
CNNS are a special family of neural networks that contain convolutional layers.
Channels on input and output allow our model to capture multiple aspects of an image at each spatial location.

"""