import numpy as np
import matplotlib.pyplot as plt

from MDLearn.DL.RBM import RBM
from MDLearn.DL.DBN import DBN
from MDLearn.DL.dA import dA
from MDLearn.DL.SdA import SdA
from MDLearn.DL.NN.BP import BP

import sys
sys.path.append(".")
import input_data

def Test_rbm():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_images = mnist.train.images

    print(mnist_images.shape, mnist_images)

    rbm = RBM( n_visible=784, n_hidden=500)
    # train
    # errs = rbm.train(input=mnist_images,lr=1.0,epochs=1, batch_size=100, show=True)
    errs = rbm.train(input=mnist_images,lr=1.0,epochs=1, batch_size=100, show=True,gaus=True)
    plt.plot(errs)
    plt.show()

def Test_dbn():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_images = mnist.train.images

    print(mnist_images.shape, mnist_images)

    dbn = DBN( n_ins=784, hidden_layer_sizes=[400,100], n_outs=10)
    # pre-training (TrainUnsupervisedDBN)
    errs = dbn.pretrain(input=mnist_images, lr=1.0, k=1, epochs=10, batch_size=100)

    plt.plot(errs)
    plt.show()
    '''0.32'''

def Test_bp():
    '''
    说明，不好用。需要再调试下
    :return:
    '''
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_images = mnist.train.images
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    # print(mnist_images.shape, mnist_images)
    print(trY.shape)
    return
    bp = BP( n_ins=784, hidden_layer_sizes=[500,300], n_outs=10)
    # bp = NeuralNetWork( [784, 500,300,10])
    # pre-training (TrainUnsupervisedDBN)
    bp.fitBatch(trX, trY, learning_rate=1.0, epochs=10)


    print ( np.mean(np.argmax(teY, axis=1) == bp.predict(teX)) )

    '''
        def __init__(self, epoches, learning_rate, batchsize, momentum, penaltyL2,
                 dropoutProb):
    opts = DLOption(10, 1., 100, 0.0, 0., 0.)

    nn = NN([500, 300], opts, trX, trY)

    
    0.852163636364
    0.884563636364
    0.897909090909
    0.905745454545
    0.912272727273
    0.917181818182
    0.920872727273
    0.925145454545
    0.927963636364
    0.930763636364
    0.9297'''

if __name__ == "__main__":
    Test_dbn()
    # Test_bp()
    # Test_rbm()
