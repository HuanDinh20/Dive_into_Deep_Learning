"""
For a deep learning problem, we will usually define a loss function first.
Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss.
In optimization, a loss function is often referred to as the objective function of the optimization problem.
y tradition and convention most optimization algorithms are concerned with minimization. If we ever need to maximize an
objective there is a simple solution: just flip the sign on the objective.

******************** Goal of Optimization *************************
Although optimization provides a way to minimize the loss function for deep learning, in essence, the goals of
optimization and deep learning are fundamentally different.
The former is primarily concerned with minimizing an objective
whereas the latter is concerned with finding a suitable model, given a finite amount of data.


training error and generalization error generally differ: since the objective function of the optimization algorithm is
usually a loss function based on the training dataset, the goal of optimization is to reduce the training error.
However, the goal of deep learning (or more broadly, statistical inference) is to reduce the generalization error.

"""


import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l


def f(x):
    """risk"""
    return x * torch.cos(np.pi * x)


def g(x):
    """empirical"""
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

"""
o illustrate the aforementioned different goals, let’s consider the empirical risk and the risk. 
the empirical risk is an average loss on the training dataset 
while the risk is the expected loss on the entire population of data. 
As a result, here g is less smooth than f
"""

def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))


"""
**************** Optimization Challenges in Deep Learning *****
                        * Local Minima *
                        
The objective function of deep learning models usually has many local optima. When the numerical solution of an 
optimization problem is near the local optimum, the numerical solution obtained by the final iteration may only minimize 
the objective function locally, rather than globally, as the gradient of the objective function’s solutions approaches 
or becomes zero. Only some degree of noise might knock the parameter out of the local minimum. In fact, this is one of 
the beneficial properties of minibatch stochastic gradient descent where the natural variation of gradients over
 mini-batches is able to dislodge the parameters from local minima.
 
 
************** Saddle Points ********************
Besides local minima, saddle points are another reason for gradients to vanish. A saddle point is any location where all 
gradients of a function vanish but which is neither a global nor a local minimum. 

The solution of the function could be a local minimum, a local maximum, or a saddle point at a position where the 
function gradient is zero:
1. When the eigenvalues of the function’s Hessian matrix at the zero-gradient position are all positive, 
we have a local minimum for the function.
2. When the eigenvalues of the function’s Hessian matrix at the zero-gradient position are all negative, 
we have a local maximum for the function.
3. When the eigenvalues of the function’s Hessian matrix at the zero-gradient position are negative and positive, 
we have a saddle point for the function.


For high-dimensional problems the likelihood that at least some of the eigenvalues are negative is quite high. This 
makes saddle points more likely than local minima. We will discuss some exceptions to this situation in the next section 
when introducing convexity. In short, convex functions are those where the eigenvalues of the Hessian are never negative
. Sadly, though, most deep learning problems do not fall into this category. Nonetheless it is a great tool to study 
optimization algorithms.

*********** Vanishing Gradients 
Probably the most insidious problem to encounter is the vanishing gradient

Gradients of unpredictable magnitude also threaten the stability of our optimization algorithms. We may be facing 
parameter updates that are either 
(i) excessively large, destroying our model (the exploding gradient problem); or 
(ii) excessively small (the vanishing gradient problem), rendering learning impossible as parameters hardly move on each 
update.


As we saw, optimization for deep learning is full of challenges. Fortunately there exists a robust range of algorithms 
that perform well and that are easy to use even for beginners. Furthermore, it is not really necessary to find the best 
solution. Local optima or even approximate solutions thereof are still very useful.

************* Summary *********************
* Minimizing the training error does not guarantee that we find the best set of parameters 
to minimize the generalization error.
* The optimization problems may have many local minima.
* The problem may have even more saddle points, as generally the problems are not convex.
* Vanishing gradients can cause optimization to stall. Often a reparameterization of the problem helps. 
Good initialization of the parameters can be beneficial, too.


"""