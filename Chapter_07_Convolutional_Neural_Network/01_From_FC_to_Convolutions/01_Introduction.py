"""To this day, the models that we have discussed so far remain appropriate options when we are dealing with
tabular data. By tabular, we mean that the data consist of rows corresponding to examples and columns corresponding to
features. With tabular data, we might anticipate that the patterns we seek could involve interactions among the
features, but we do not assume any structure a priori concerning how the features interact.
Sometimes, we truly lack knowledge to guide the construction of craftier architectures. In these cases,
an MLP may be the best that we can do. However, for high-dimensional perceptual data,
such structure-less networks can grow unwieldy.
For instance, let us return to our running example of distinguishing cats from dogs. Say that we do a thorough job in
data collection, collecting an annotated dataset of one-megapixel photographs.
This means that each input to the network has one million dimensions,
  even an aggressive reduction to one thousand hidden dimensions would require a fully-connected layer characterized by
 10^6 * 10^3 =   10 ^ 9 parameters. Unless we have lots of GPUs, a talent for distributed optimization,
 and an extraordinary amount of patience, learning the parameters of this network may turn out to be infeasible.
 A careful reader might object to this argument on the basis that one megapixel resolution may not be necessary.
 However, while we might be able to get away with one hundred thousand pixels, our hidden layer of size 1000
 grossly underestimates the number of hidden units that it takes to learn good representations of images,
 so a practical system will still require billions of parameters. Moreover, learning a classifier by fitting so many
 parameters might require collecting an enormous dataset. And yet today both humans and computers are able to
 distinguish cats from dogs quite well, seemingly contradicting these intuitions. That is because images exhibit rich
 structure that can be exploited by humans and machine learning models alike. Convolutional neural networks (CNNs) are
 one creative way that machine learning has embraced for exploiting some of the known structure in natural images.
"""
