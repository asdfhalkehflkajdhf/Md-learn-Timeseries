# -*- coding: utf-8 -*-

import sys
import numpy
from MDLearn.DL.utils import *
from numpy import newaxis
import numpy as np

class LogisticRegression(object):
    def __init__(self, n_in, n_out,input=None, label=None,W=None, b=None, rng=None,outputMap="softmax"):
        self.x = input
        self.y = label

        if rng is None:
            rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_in
            W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))

        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0
        self.W = W
        self.b = b

        # self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        # self.b = numpy.zeros(n_out)  # initialize bias 0
        self.outputMap = outputMap

    def train(self, lr=0.1, input=None, lable=None, L2_reg=0.00):
        '''
        这个函数训练反回误差
        :param lr:
        :param input:
        :param L2_reg:
        :return:
        '''
        if input is not None:
            self.x = input
        if lable is not None:
            self.y = lable
        #获取最后一层输出
        p_y_given_x = self.output(self.x)
        #计算输出误差
        if self.y.ndim == 1:
            self.y=self.y[:newaxis]
        d_y = self.y - p_y_given_x
        #计算反向误差
        '''
        注意：需要判断下,如果是一维数据，需要加维，否则报错
        '''
        if numpy.array(self.x).ndim == 1:
            self.x=self.x[newaxis]
        if d_y.ndim==1:
            d_y=d_y[newaxis]

        t1 = numpy.dot(self.x.T, d_y)
        t2 = lr*t1

        # 添加冲量,动量因子
        t3 =  lr*L2_reg*self.W
        t4 = t2 - t3
        t5 = self.W + t4
        self.W =  t5
        # self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        # print(self.b,self.b.shape, d_y, d_y.shape)
        t1 =  lr * numpy.mean(d_y, axis=0)
        t2 = self.b + t1
        self.b = t2
        # self.b += lr * numpy.mean(d_y, axis=0)

        #反向的下层传播误差
        self.bp_err = d_y
        return d_y

    # def train(self, lr=0.1, input=None, L2_reg=0.00):
    #     self.forward(input)
    #     self.backward(lr, L2_reg)

    # def forward(self, input=None):
    #     if input is not None:
    #         self.x = input

    #     p_y_given_x = self.output(self.x)
    #     self.d_y = self.y - p_y_given_x
        
    # def backward(self, lr=0.1, L2_reg=0.00):
    #     self.W += lr * numpy.dot(self.x.T, self.d_y) - lr * L2_reg * self.W
    #     self.b += lr * numpy.mean(self.d_y, axis=0)

    def output(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        if self.outputMap=="softmax":
            return softmax(numpy.dot(x, self.W) + self.b)
        elif self.outputMap=="identity":
            return numpy.dot(x, self.W) + self.b
        elif self.outputMap=="sigmoid":
            return sigmoid(numpy.dot(x, self.W) + self.b)
        elif self.outputMap=="tanh":
            return tanh(numpy.dot(x, self.W) + self.b)


    def predict(self, x):
        return self.output(x)


    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


def test_lr(learning_rate=0.1, n_epochs=500):

    rng = numpy.random.RandomState(123)

    # training data
    d = 2
    N = 10
    x1 = rng.randn(N, d) + numpy.array([0, 0])
    x2 = rng.randn(N, d) + numpy.array([20, 10])
    y1 = [[1, 0] for i in range(N)]
    y2 = [[0, 1] for i in range(N)]

    x = numpy.r_[x1.astype(int), x2.astype(int)]
    # y = numpy.r_[y1, y2]
    y = numpy.r_[y1, y2]

    print(x, y)
    x=numpy.array([[1,2,3],
       [2,3,4],
       [1,2,2],
       [6,7,8],
       [8,9,9]
    ])

    y=numpy.array([
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ])
    y=numpy.array([
        [1],
        [1],
        [1],
        [0],
        [0]
    ])

    print(x, y)
    # construct LogisticRegression
    # classifier = LogisticRegression(input=x, label=y, n_in=d, n_out=2)
    classifier = LogisticRegression(input=x, label=y, n_in=3, n_out=1)

    # train
    for epoch in range(n_epochs):
        classifier.train(lr=learning_rate)
        # cost = classifier.negative_log_likelihood()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.995

    # test
    result = classifier.predict(x)
    print(result)
    return
    for i in range(N):
        print (result[i])
    print ()
    for i in range(N):
        print (result[N+i])



if __name__ == "__main__":
    test_lr()
