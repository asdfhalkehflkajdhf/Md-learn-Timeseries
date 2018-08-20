from __future__ import absolute_import, print_function, division
import sys

import numpy

sys.path.append(".")

import convolutional_mlp
import dA
import DBN
import logistic_cg
import logistic_sgd

def logistic_sgd_test():
    logistic_sgd.sgd_optimization_mnist(n_epochs=10)

if __name__ == "__main__":
    logistic_sgd_test()
