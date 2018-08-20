
import numpy
numpy.seterr(all='ignore')


def sigmoid(x, grad=False):
    if grad:
        return dsigmoid(x)
    return 1. / (1 + numpy.exp(-x))
'''
优点：
1.Sigmoid函数的输出映射在(0,1)之间，单调连续，输出范围有限，优化稳定，可以用作输出层。
2.求导容易。

缺点：
1.由于其软饱和性，容易产生梯度消失，导致训练出现问题。
2.其输出并不是以0为中心的。
'''

def dsigmoid(x):
    return x * (1. - x)

def tanh(x, grad=False):
    if grad:
        return dtanh(x)
    return numpy.tanh(x)
'''
优点：
1.比Sigmoid函数收敛速度更快。
2.相比Sigmoid函数，其输出以0为中心。
缺点：
还是没有改变Sigmoid函数的最大问题——由于饱和性产生的梯度消失。
'''
def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x, grad=False):
    if grad:
        return dReLU(x)
    return x * (x > 0)
'''
优点：
1.相比起Sigmoid和tanh，ReLU(e.g. a factor of 6 in Krizhevsky et al.)在SGD中能够快速收敛。例如在下图的实验中，在一个四层的卷积神经网络中，实线代表了ReLU，虚线代表了tanh，ReLU比起tanh更快地到达了错误率0.25处。据称，这是因为它线性、非饱和的形式。
2.Sigmoid和tanh涉及了很多很expensive的操作（比如指数），ReLU可以更加简单的实现。
3.有效缓解了梯度消失的问题。
4.在没有无监督预训练的时候也能有较好的表现。
5.提供了神经网络的稀疏表达能力。

缺点：
随着训练的进行，可能会出现神经元死亡，权重无法更新的情况。如果发生这种情况，那么流经神经元的梯度从这一点开始将永远是0。也就是说，ReLU神经元在训练中不可逆地死亡了。
'''
def dReLU(x):
    return 1. * (x > 0)

def getNBatch(len, batch_size=None):
    if batch_size is None or batch_size<1:
        batch_size = len
    return  len // batch_size + (0 if len % batch_size == 0 else 1), batch_size

# # probability density for the Gaussian dist
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * numpy.power(scale, 2)
#     e = numpy.exp( - numpy.power((x - mean), 2) / s )

#     return e / numpy.square(numpy.pi * s)

