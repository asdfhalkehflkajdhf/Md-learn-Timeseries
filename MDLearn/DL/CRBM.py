# -*- coding: utf-8 -*-

import sys
import numpy
from MDLearn.DL.RBM import RBM
from MDLearn.DL.utils import *


class CRBM(RBM):
    def propdown(self, h):
        pre_activation = numpy.dot(h, self.W.T) + self.vbias
        return pre_activation
        


    def sample_v_given_h(self, h0_sample):
        a_h = self.propdown(h0_sample)
        en = numpy.exp(-a_h)
        ep = numpy.exp(a_h)

        v1_mean = 1 / (1 - en) - 1 / a_h
        '''uniform样本均匀分布在半开区间'''
        U = numpy.array(self.rng.uniform(
            low=0,
            high=1,
            size=v1_mean.shape))

        v1_sample = numpy.log((1 - U * (1 - ep))) / a_h

        return [v1_mean, v1_sample]



def test_crbm(learning_rate=0.1, k=1, training_epochs=2000):
    data = numpy.array([[0.4, 0.5, 0.5, 0.,  0.,  0.],
                        [0.5, 0.3,  0.5, 0.,  0.,  0.],
                        [0.4, 0.5, 0.5, 0.,  0.,  0.],
                        [0.,  0.,  0.5, 0.3, 0.5, 0.],
                        [0.,  0.,  0.5, 0.4, 0.5, 0.],
                        [0.,  0.,  0.5, 0.5, 0.5, 0.]])


    rng = numpy.random.RandomState(123)

    # construct CRBM
    rbm = CRBM(input=data, n_visible=6, n_hidden=5, rng=rng)

    # train
    for epoch in range(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        # cost = rbm.get_reconstruction_cross_entropy()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
    # v = numpy.array([[0.5, 0.5, 0., 0., 0., 0.],
    #                  [0., 0., 0., 0.5, 0.5, 0.]])

    print (rbm.reconstruct(data))

def aaa1():
    import numpy as np
    import matplotlib.pyplot as plt

    #画线
    x1, y1 = 0, 30
    x2, y2 = 100, 200
    x3, y3 = 200, -10

    x1, y1 = 0, 0.5
    x2, y2 = 0, 1
    x3, y3 = 0.75, 1


    # x1, y1 = 0.5, 0.1
    # x2, y2 = 0.8, 0.5
    # x3, y3 = 1, 0
    #个数
    sample_size = 200


    #画图边线
    # theta = np.arange(0, 1, 0.001)
    # x = theta * x1 + (1 - theta) * x2
    # y = theta * y1 + (1 - theta) * y2
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x1 + (1 - theta) * x3
    # y = theta * y1 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x2 + (1 - theta) * x3
    # y = theta * y2 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)

    #生成点
    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3

    return x, y
    #画点
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(x, y, 'ro')
    plt.grid(True)
    # plt.savefig('demo.png')
    plt.show()

def aaa2():
    import numpy as np
    import matplotlib.pyplot as plt

    #画线
    x1, y1 = 0, 30
    x2, y2 = 100, 200
    x3, y3 = 200, -10

    x1, y1 = 0, 0.5
    x2, y2 = 0, 1
    x3, y3 = 0.75, 1


    x1, y1 = 0.5, 0.1
    x2, y2 = 0.8, 0.5
    x3, y3 = 1, 0
    #个数
    sample_size = 200


    #画图边线
    # theta = np.arange(0, 1, 0.001)
    # x = theta * x1 + (1 - theta) * x2
    # y = theta * y1 + (1 - theta) * y2
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x1 + (1 - theta) * x3
    # y = theta * y1 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x2 + (1 - theta) * x3
    # y = theta * y2 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)

    #生成点
    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3

    return x,y
    #画点
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(x, y, 'ro')
    plt.grid(True)
    # plt.savefig('demo.png')
    plt.show()

def Test_crbm():
    import numpy as np
    import matplotlib.pyplot as plt

    x,y = aaa1()
    x1,y1=aaa2()

    x = np.array([x, x1]).ravel()
    y= np.array([y, y1]).ravel()
    # x=x+x1
    # y=y+y1

    #显示画点
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.plot(x, y, 'ro')
    # plt.grid(True)
    # # plt.savefig('demo.png')
    # plt.show()

    data=[]
    data.append(x)
    data.append(y)
    data = np.array(data).T

    # print(data.shape, data)
    # plt.plot(data[:,0],data[:,1], 'ro')
    # plt.show()

    rbm = CRBM(n_visible=2, n_hidden=4)
    rbm.train(data, lr=2, epochs=1,batch_size=500)
    rData = rbm.reconstruct(data)

    print(rData.shape, rData)

    plt.plot(rData[:,0],rData[:,1], 'ro')
    plt.show()
    #
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.plot(rData[0], rData[1], 'ro')
    # plt.grid(True)
    # # plt.savefig('demo.png')
    # plt.show()
if __name__ == "__main__":
    test_crbm()
    # Test_crbm()
