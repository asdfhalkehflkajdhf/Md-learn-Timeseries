# -*- coding: utf-8 -*-

import sys
import numpy as np
from cDBN_M.BP_SELF import BP
from cDBN_M.cRBM_SELF import RBM
from utils import *


class DBN(object):
    def __init__(self, layer_sizes=[3, 3], rng=None ,bp_layer=[]):
        '''

        :param input:
        :param label:
        :param n_ins:
        :param hidden_layer_sizes:
        :param n_outs:
        :param rng: 随机数发生器
        '''

        self.rbm_layers = []
        self.n_layers = len(layer_sizes)-1  # = len(self.rbm_layers)

        if rng is None:
            rng = np.random.RandomState(1234)

        '''最少为一层'''
        assert self.n_layers > 1

        # construct multi-layer
        for i in range(self.n_layers):
            # layer_size
            # construct rbm_layer
            rbm_layer = RBM(
                            n_visible=layer_sizes[i],
                            n_hidden=layer_sizes[i+1],
                            rng = rng)
            self.rbm_layers.append(rbm_layer)


        self.bp_layers=None
        self._n_bp_layers = len(bp_layer)
        if self._n_bp_layers>0:
            para = [layer_sizes[-1]]+bp_layer
            self.bp_layers = BP(para)


    '''对ＤＢＮ进行拟合'''
    def pretrain(self, input, lr=0.1, k=1, epochs=100, batch_size=1000, residual_error=1e-3, gaus=False, show=False):
        '''

        :param input:
        :param lr:
        :param k:
        :param epochs:
        :param residual_error: RBM重构误差
        :param gaus: RBM是否使用高斯分布
        :return:
        '''
        import MDLearn.utils.Draw as draw
        err=None
        layer_input = input
        for i in range(self.n_layers):
            rbm = self.rbm_layers[i]
            err = rbm.train(lr=lr, k=k, epochs=epochs, batch_size=batch_size,input=layer_input, residual_error=residual_error, gaus=gaus)
            # draw.plot_all_point(err)
            err = rbm.get_errs(layer_input)
            if show : print("rbm errs %d:"%(i),err)
            layer_input = rbm.forward(layer_input)
            # print(np.mean(err))

        return err



    def forward(self, x):
        layer_input = x

        for i in range(self.n_layers):
            layer_input = self.rbm_layers[i].forward(layer_input)
        return layer_input

    def getHyperParameter(self):
        W_list =[]
        b_list=[]
        for i in range(self.n_layers):
            W_list.append(self.rbm_layers[i].W)
            b_list.append(self.rbm_layers[i].b)
        return W_list, b_list

    '''对ＤＢＮ进行微调'''
    def train(self,input,label, lr=0.1, k=1, epochs=100, batch_size=10, residual_error=1e-3, gaus=False, show=False):
        err=None
        layer_input = input
        for i in range(self.n_layers):
            rbm = self.rbm_layers[i]
            err = rbm.train(lr=lr, k=k, epochs=epochs, batch_size=batch_size,input=layer_input, residual_error=residual_error, gaus=gaus)
            # draw.plot_all_point(err)
            err = rbm.get_errs(layer_input)
            if show : print("rbm errs %d:"%(i),err)
            layer_input = rbm.forward(layer_input)
            # print(np.mean(err))

        assert self.bp_layers is not None
        self.bp_layers.train(layer_input,label,lr=lr,epochs=epochs)

        return err

    def predict(self, xTest, epochs=1, batch=1000):
        layer_input = xTest

        for rbm in self.rbm_layers:
            layer_input = rbm.forward(layer_input)

        assert self.bp_layers is not None
        out = self.bp_layers.predict(layer_input)
        return out



def dbn_test():
    pass

def Test_dbn_bp2():
    import input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    dbn = DBN( layer_sizes=[784,400,100],bp_layer=[10])
    # pre-training (TrainUnsupervisedDBN)
    errs = dbn.train(input=trX,label=trY, lr=1.0, k=1, epochs=10, batch_size=100)
    print("dbn :",errs)

    res = dbn.predict(teX)

    print(np.mean(np.argmax(teY, axis=1) == np.argmax(res, 1) ))


    # plt.plot(errs)
    # plt.show()
    '''0.32'''

if __name__ == "__main__":
    # dbn_test()
    Test_dbn_bp2()
