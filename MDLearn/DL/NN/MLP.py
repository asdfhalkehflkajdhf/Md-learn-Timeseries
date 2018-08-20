# -*- coding: utf-8 -*-

import sys
import numpy
from MDLearn.DL.HiddenLayer import HiddenLayer
from MDLearn.DL.LogisticRegression import LogisticRegression
from MDLearn.DL.utils import *


class MLP(object):
    def __init__(self, input, label, n_in, n_hidden, n_out,
                        rng=None,
                        outputMap="softmax" #"softmax"  # "tanh"  # "identity" #"sigmoid"
                 ):

        self.x = input
        self.y = label

        if rng is None:
            rng = numpy.random.RandomState(1234)

        # construct hidden_layer
        self.hidden_layer = HiddenLayer(input=self.x,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        rng=rng,
                                        activation=tanh)

        # construct log_layer
        self.log_layer = LogisticRegression(input=self.hidden_layer.output,
                                            label=self.y,
                                            n_in=n_hidden,
                                            n_out=n_out,
                                            outputMap=outputMap
                                            )

    def train(self, lr=0.1, n_epochs=100):

        for epoch in range(n_epochs):
            # forward hidden_layer
            layer_input = self.hidden_layer.forward()

            # forward & backward log_layer
            # self.log_layer.forward(input=layer_input)
            self.log_layer.train(input=layer_input)

            # backward hidden_layer
            self.hidden_layer.backward(lr=lr,inputs=self.log_layer.x, bp_err= self.log_layer.bp_err, W=self.log_layer.W)

            # backward log_layer
            # self.log_layer.backward()


    def predict(self, x):
        x = self.hidden_layer.output(input=x)
        return self.log_layer.predict(x)


def test_mlp(n_epochs=500):

    x = numpy.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = numpy.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])

    # y = numpy.array([[0],                     [0],                     [0],                     [1]])
    rng = numpy.random.RandomState(123)


    # construct MLP
    classifier = MLP(input=x, label=y, n_in=2, n_hidden=3, n_out=2, rng=rng)
    # classifier = MLP(input=x, label=y, n_in=2, n_hidden=3, n_out=1, rng=rng)

    # train
    for epoch in range(n_epochs):
        classifier.train()


    # test
    y = classifier.predict(x)
    # y[y>0.9]=1
    # y[y<0.9]=0
    print (y)


def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):
    x = numpy.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])
    y = numpy.array([[1],
                     [1],
                     [1],
                     [0],
                     [0],
                     [0]])
    rng = numpy.random.RandomState(123)

    # construct DBN
    # dbn = MLP(input=x, label=y, n_ins=6, hidden_layer_sizes=[4, 4], n_outs=1, rng=rng)
    classifier = MLP(input=x, label=y, n_in=6, n_hidden=8, n_out=1, rng=rng)

    # train
    # for epoch in range(100):
    classifier.train()


    # test
    print (classifier.predict(x))


if __name__ == "__main__":
    # test_dbn()
    test_mlp()
