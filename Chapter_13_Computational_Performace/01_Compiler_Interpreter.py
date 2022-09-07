"""
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
    Python is an interpreted language. When evaluating the above fancy_func function it performs the operations making up
the function’s body in sequence. That is, it will evaluate e = add(a, b) and store the results as variable e, thereby
changing the program’s state. The next two statements f = add(c, d) and g = add(e, f) will be executed similarly,
performing additions and storing the results as variables.
    Although imperative programming is convenient, it may be inefficient. On the one hand, even if the add function is
repeatedly called throughout fancy_func, Python will execute the three function calls individually. If these are
executed, say, on a GPU (or even on multiple GPUs), the overhead arising from the Python interpreter can become
overwhelming. Moreover, it will need to save the variable values of e and f until all the statements in fancy_func have
been executed. This is because we do not know whether the variables e and f will be used by other parts of the program
after the statements e = add(a, b) and f = add(c, d) are executed.

********* Symbolic Programming ************
Consider the alternative, symbolic programming, where computation is usually performed only once the process has been
fully defined. This strategy is used by multiple deep learning frameworks, including Theano and TensorFlow (the latter
has acquired imperative extensions). It usually involves the following steps:
1. Define the operations to be executed.
2. Compile the operations into an executable program.
3. Provide the required inputs and call the compiled program for execution.

This allows for a significant amount of optimization:
1.  we can skip the Python interpreter in many cases, thus removing a performance bottleneck that can become significant
on multiple fast GPUs paired with a single Python thread on a CPU
2. a compiler might optimize and rewrite the above code into print((1 + 2) + (3 + 4)) or even print(10).
This is possible since a compiler gets to see the full code before turning it into machine instructions.


"""


def add_():
    return '''
def add(a, b):
    return a + b
'''


def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''


def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)

"""
The differences between imperative (interpreted) programming and symbolic programming are as follows:

1. Imperative programming is easier. When imperative programming is used in Python, the majority of the code is 
straightforward and easy to write. It is also easier to debug imperative programming code. This is because it is easier 
to obtain and print all relevant intermediate variable values, or use Python’s built-in debugging tools.

2. Symbolic programming is more efficient and easier to port. Symbolic programming makes it easier to optimize the code 
during compilation, while also having the ability to port the program into a format independent of Python. This allows 
the program to be run in a non-Python environment, thus avoiding any potential performance issues related to the Python 
interpreter.


************ Hybrid Programming **************
Historically most deep learning frameworks choose between an imperative or a symbolic approach. For example, Theano, 
TensorFlow (inspired by the former), Keras, and CNTK formulate models symbolically. Conversely, Chainer and PyTorch 
take an imperative approach. An imperative mode was added to TensorFlow 2.0 and Keras in later revisions.

 ** Pytorch 
As mentioned above, PyTorch is based on imperative programming and uses dynamic computation graphs. In an effort to 
leverage the portability and efficiency of symbolic programming, developers considered whether it would be possible to 
combine the benefits of both programming models. This led to a torchscript that lets users develop and debug using pure 
imperative programming, while having the ability to convert most programs into symbolic programs to be run when 
product-level computing performance and deployment are required.
 ** TensorFlow
The imperative programming paradigm is now the default in Tensorflow 2, a welcoming change for those new to the 
language. However, the same symbolic programming techniques and subsequent computational graphs still exist in 
TensorFlow, and can be accessed by the easy-to-use tf.function decorator. This brought the imperative programming 
paradigm to TensorFlow, allowed users to define more intuitive functions, then wrap them and compile them into 
computational graphs automatically using a feature the TensorFlow team refers to as autograph.

******* Hybridizing the Sequential Class ************
The easiest way to get a feel for how hybridization works is to consider deep networks with multiple layers. 
Conventionally the Python interpreter will need to execute the code for all layers to generate an instruction that can 
then be forwarded to a CPU or a GPU. For a single (fast) computing device this does not cause any major issues. On the 
other hand, if we use an advanced 8-GPU server such as an AWS P3dn.24xlarge instance Python will struggle to keep all 
GPUs busy. The single-threaded Python interpreter becomes the bottleneck here. Let’s see how we can address this for 
significant parts of the code by replacing Sequential with HybridSequential. We begin by defining a simple MLP.


"""

import torch
from torch import nn
from d2l import torch as d2l


# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
print('net x :', net(x))

"""By converting the model using torch.jit.script function, we are able to compile and optimize the computation in the 
MLP. The model’s computation result remains unchanged."""

net = torch.jit.script(net)
print("torch script new: ", net(x))

"""
******************************* Acceleration by Hybridization ****************************
To demonstrate the performance improvement gained by compilation we compare the time needed to evaluate net(x) before 
and after hybridization.


"""


class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')


net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)

"""
As is observed in the above results, after an nn.Sequential instance is scripted using the torch.jit.script function, 
computing performance is improved through the use of symbolic programming.

*   *   *   *   *   *   * Serialization ***************
One of the benefits of compiling the models is that we can serialize (save) the model and its parameters to disk. 
This allows us to store a model in a manner that is independent of the front-end language of choice. This allows us to 
deploy trained models to other devices and easily use other front-end programming languages. At the same time the code 
is often faster than what can be achieved in imperative programming. Let’s see the save function in action.
"""
net.save('my_mlp')

"""
** Summary ****************
1. Imperative programming makes it easy to design new models since it is possible to write code with control flow and 
the ability to use a large amount of the Python software ecosystem.

2. Symbolic programming requires that we specify the program and compile it before executing it. 
The benefit is improved performance.

                            ** MXNET **
                            
MXNet is able to combine the advantages of both approaches as needed.
Models constructed by the HybridSequential and HybridBlock classes are able to convert imperative programs
into symbolic programs by calling the hybridize function.
"""
