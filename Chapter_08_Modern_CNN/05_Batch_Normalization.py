"""
    Training deep neural networks is difficult. Getting them to converge in a reasonable amount of time can be tricky. In
this section, we describe batch normalization, a popular and effective technique that consistently accelerates the
convergence of deep networks. Together with residual blocks batch normalization has made it possible for practitioners
to routinely train networks with over 100 layers. A secondary (serendipitous) benefit of batch normalization is its
inherent regularization.

***************** Training Deep Networks *********************
1. Intuitively, this standardization plays nicely with our optimizers since it puts the parameters a priori at a
similar scale.
2. Second, for a typical MLP or CNN, as we train, the variables (e.g., affine transformation outputs in MLP) in
intermediate layers may take values with widely varying magnitudes: both along the layers from the input to the output,
across units in the same layer, and over time due to our updates to the model parameters. The inventors of batch
normalization postulated informally that this drift in the distribution of such variables could hamper the convergence
of the network.
3. Third, deeper networks are complex and tend to be more easily capable of overfitting. This means that regularization
becomes more critical. A common technique for regularization is noise injection.

***** Summary:
1. Preprocessing
2. Numerical stability
3. Regularization

****
Batch normalization is applied to individual layers, or optionally, to all of them: In each training iteration, we first
normalize the inputs (of batch normalization) by subtracting their mean and dividing by their standard deviation, where
both are estimated based on the statistics of the current minibatch. Next, we apply a scale coefficient and an offset to
recover the lost degrees of freedom. It is precisely due to this normalization based on batch statistics that batch
normalization derives its name.

***

BN(x) = np.dot(Gamma, (x - mean_B)/ STD_B) + beta
x: belong to B
gamma: Scale Parameters
B: Minibatch
beta: Shift parameters

***

This turns out to be a recurring theme in deep learning. For reasons that are not yet well-characterized theoretically,
various sources of noise in optimization often lead to faster training and less over-fitting:
this variation appears to act as a form of regularization and relate the properties of batch normalization to Bayesian
priors and penalties respectively.
    In particular, this sheds some light on the puzzle of why batch normalization works best for moderate
minibatches sizes in the 50 ~ 100 range. This particular size of minibatch seems to inject just the “right amount” of
noise per layer:
 * a larger minibatch regularizes less due to the more stable estimates,
 whereas tiny minibatches destroy useful signal due to high variance.


********************** Batch Normalization Layers **********************

*** Batch Normalization Layers
h = phi( BN( Wx + b ))

Recall that mean and variance are computed on the same minibatch on which the transformation is applied.

*** Convolutional Layers
Similarly, with convolutional layers, we can apply batch normalization after the convolution and before the nonlinear
activation function.

The key difference from batch normalization in fully connected layers is that we apply the operation on a per-channel
basis across all locations. This is compatible with our assumption of translation invariance that led to convolutions:
we assumed that the specific location of a pattern within an image was not critical for the purpose of understanding.

*
 minibatches contain m examples
 for each channel, the output of the convolution :
    height p
    width q
convolutional layers: m x p x q elements per output channel simultaneously
Thus, we collect the values over all spatial locations when computing the mean and variance and consequently apply the
same mean and variance within a given channel to normalize the value at each spatial location. Each channel has its own
scale and shift parameters, both of which are scalars.

Note that in the context of convolutions the batch normalization is well-defined even for minibatches of size 1: after
all, we have all the locations across an image to average. Consequently, mean and variance are well defined, even if
it’s just within a single observation.
layer norm. It works just like a batch norm, just that it is applied one image at a time. There are cases where layer
normalization improves the accuracy of a model. We skip further details and recommend the interested reader to consult
the original paper.

**************  Batch Normalization During Prediction *******************
 Batch Normalization During Prediction:
 1. the noise in the sample mean and the sample variance arising from estimating each on minibatches are no longer
 desirable once we have trained the model
2. we might not have the luxury of computing per-batch normalization statistics

Typically, after training, we use the entire dataset to compute stable estimates of the variable statistics and then fix
them at prediction time. Consequently, batch normalization behaves differently during training and at test time.
Recall that dropout also exhibits this characteristic.


*** Implementation from Scratch *********************

"""

import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enable to determine whether the current mode is training mode or prediction mode
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 2D-CNN Layers
            # calculate the mean and variance on the channel dimension (axis=1)
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update mean and variance using moving average
        moving_mean = momentum * moving_mean + (0.1 - momentum) * mean
        moving_var = momentum * moving_var + (0.1 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data


"""
    We can now create a proper BatchNorm layer. Our layer will maintain proper parameters for scale gamma and shift beta, 
both of which will be updated in the course of training. Additionally, our layer will maintain moving averages of the 
means and variances for subsequent use during model prediction.
    Putting aside the algorithmic details, note the design pattern underlying our implementation of the layer. Typically, we 
define the mathematics in a separate function, say batch_norm. We then integrate this functionality into a custom layer, 
whose code mostly addresses bookkeeping matters, such as moving data to the right device context, allocating and 
initializing any required variables, keeping track of moving averages (here for mean and variance), and so on. This 
pattern enables a clean separation of mathematics from boilerplate code. Also note that for the sake of convenience we 
did not worry about automatically inferring the input shape here, thus we need to specify the number of features 
throughout. By now all modern deep learning frameworks offer automatic detection of size and shape in the high-level 
batch normalization APIs (in practice we will use this instead).

"""

class BatchNorm(nn.Module):
    # num_features: number of output for a fully connected layer or a convolutional layer.
    # num_dims:  2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean  = nn.Parameter(torch.zeros(shape))
        self.moving_var = nn.Parameter(torch.ones(shape))

    def forward(self, X):
        # if X is not in the main memory, coppy moving_mean and moving_var to X.device
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, eps=1e-5, momentum=0.9)
        return Y

"""
Note that we used the variable momentum to govern the aggregation over past mean and variance estimates. 
This is somewhat of a misnomer as it has nothing whatsoever to do with the momentum term of optimization
Nonetheless, it is the commonly adopted name for this term and in deference to API naming convention we use the same 
variable name in our code, too.

*************** Apply to Lenet **********
"""

class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(num_classes))


trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)

""""
In the original paper proposing batch normalization, the authors, in addition to introducing a powerful and useful tool, 
offered an explanation for why it works: by reducing internal covariate shift. Presumably by internal covariate shift 
the authors meant something like the intuition expressed above—the notion that the distribution of variable values 
changes over the course of training. However, there were two problems with this explanation: i) This drift is very 
different from covariate shift, rendering the name a misnomer. ii) The explanation offers an under-specified intuition 
but leaves the question of why precisely this technique works an open question wanting for a rigorous explanation. 
Throughout this book, we aim to convey the intuitions that practitioners use to guide their development of deep neural 
networks. However, we believe that it is important to separate these guiding intuitions from established scientific fact 
Eventually, when you master this material and start writing your own research papers you will want to be clear to 
delineate between technical claims and hunches.


***** Summary
On a more practical note, there are a number of aspects worth remembering about batch normalization:
1. During model training, batch normalization continuously adjusts the intermediate output of the network by utilizing 
the mean and standard deviation of the minibatch, so that the values of the intermediate output in each layer 
throughout the neural network are more stable
2.  Batch normalization for fully connected layers and convolutional layers are slightly different


 In fact, for convolutional layers, layer normalization can sometimes be used as an alternative
1. Like a dropout layer, batch normalization layers have different behaviors in training mode and prediction mode
2. Batch normalization is useful for regularization and improving convergence in optimization. On the other hand, the 
original motivation of reducing internal covariate shift seems not to be a valid explanation.

"""
