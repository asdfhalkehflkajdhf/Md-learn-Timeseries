# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from RBM_SELF import RBM
from utils import *


class DBN(object):
    def __init__(self, input=None, label=None, \
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2, \
                 rng=None,
                 outputMap="sigmoid"  # "tanh"  # "identity" #"sigmoid"
                 ):
        '''

        :param input:
        :param label:
        :param n_ins:
        :param hidden_layer_sizes:
        :param n_outs:
        :param rng: 随机数发生器
        '''
        self.x = input
        self.y = label

        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if rng is None:
            rng = numpy.random.RandomState(1234)

        assert self.n_layers > 0

        # construct multi-layer
        for i in range(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]
            # construct rbm_layer
            rbm_layer = RBM(
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            )
            self.rbm_layers.append(rbm_layer)

        # layer for output using Logistic Regression
        # layer_input

        # self.log_layer = LogisticRegression(
        #
        #                                     n_in=hidden_layer_sizes[-1],
        #                                     n_out=n_outs,
        #                                     outputMap = outputMap
        #                                     )


    def pretrain(self, input, lr=0.1, k=1, epochs=100, batch_size=10, residual_error=1e-3, gaus=False):
        '''

        :param input:
        :param lr:
        :param k:
        :param epochs:
        :param residual_error: RBM重构误差
        :param gaus: RBM是否使用高斯分布
        :return:
        '''
        # pre-train layer-wise

        if input is not None:
            self.x = input

        import MDLearn.utils.Draw as draw

        layer_input = self.x
        for i in range(self.n_layers):
            rbm = self.rbm_layers[i]
            err = rbm.train(lr=lr, k=k, epochs=epochs, batch_size=batch_size,input=layer_input, residual_error=residual_error, gaus=gaus)
            layer_input = self.rbm_layers[i - 1].forward(layer_input)
            draw.plot_all_point(err)
            # print(numpy.mean(err))
        return self.rbm_layers[-1].errs


    def finetune(self, lable=None, lr=0.1, epochs=100,  residual_error=1e-3):
        layer_input = self.sigmoid_layers[-1].forward()
        # layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        if lable is not None:
            self.y = lable
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            re = self.log_layer.train(input=layer_input, lable=self.y, lr=lr)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost

            lr *= 0.95
            epoch += 1

            if abs(numpy.mean(re) ) < residual_error:
                return

    def predict(self, x):
        layer_input = x

        for i in range(self.n_layers):
            sigmoid_layer = self.rbm_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        return layer_input
        out = self.log_layer.predict(layer_input)
        return out


# def dbn_test(pretrain_lr=0.1, pretraining_epochs=100, k=1, \
#              finetune_lr=0.1, finetune_epochs=200):
#     x = numpy.array([[1, 1, 1, 0, 0, 0],
#                      [1, 0, 1, 0, 0, 0],
#                      [1, 1, 1, 0, 0, 0],
#                      [0, 0, 1, 1, 1, 0],
#                      [0, 0, 1, 1, 0, 0],
#                      [0, 0, 1, 1, 1, 0]])
#     y = numpy.array([[1, 0],
#                      [1, 0],
#                      [1, 0],
#                      [0, 1],
#                      [0, 1],
#                      [0, 1]])
#
#     rng = numpy.random.RandomState(123)
#
#     # construct DBN
#     dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=2, rng=rng)
#
#     # pre-training (TrainUnsupervisedDBN)
#     dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
#
#     # fine-tuning (DBNSupervisedFineTuning)
#     dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
#
#     # test
#     x = numpy.array([[1, 1, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 1, 0],
#                      [1, 1, 1, 1, 1, 0]])
#
#     print(dbn.predict(x))


# if __name__ == "__main__":
#     dbn_test()
