#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
Test some function.
"""
import sys
import numpy as np
sys.path.append(".")
import input_data
from opts import DLOption
from dbn_tf import DBN
from nn_tf import NN

def Test_nn():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    '''def __init__(self, epoches, learning_rate, batchsize, momentum, penaltyL2,
                 dropoutProb):'''
    opts = DLOption(10, 1., 100, 0.0, 0., 0.)

    nn = NN([500, 300], opts, trX, trY)
    nn.train()
    print ( np.mean(np.argmax(teY, axis=1) == nn.predict(teX)) )

def test_bp2():
    print("test neural network")

    x = np.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = np.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])

    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.RandomState(123)

    # network = BP(n_ins=2, hidden_layer_sizes=[3], n_outs=2, rng=rng)
    # network.fitBatch(x, y, learning_rate=0.1, epochs=500)
    opts = DLOption(10, 1., 100, 0.0, 0., 0.)

    nn = NN([3], opts, x, y)
    nn.train()
    print(nn.predict(x))
    print ( np.mean(np.argmax(y, axis=1) == nn.predict(x)) )


    return

if __name__ == "__main__":
    # Test_nn()
    test_bp2()
