# -*- coding: utf-8 -*-

import sys
import numpy
from MDLearn.DL.utils import *
from numpy import newaxis


class HiddenLayer(object):
    def __init__(self,  n_in, n_out, input=None,\
                 W=None, b=None, rng=None, activation=tanh):
        '''

        :param n_in:
        :param n_out:
        :param input:初始化，输入数据
        :param W:
        :param b:
        :param rng:
        :param activation:
        '''
        if rng is None:
            rng = numpy.random.RandomState(1234)

        '''
        说明：Ｗ的矩阵是按，输入个数为矩阵行，　输出个数为矩阵列。则
        
        列如：３－２

            则Ｗ＝     [１，１]
                       [１，１]
                       [１，１]
            input = [1,2,2]
                    [2,2,2]
                    [3,3,3]
            B = [0, 0] 
        
        在计算　y=Wx + b 时，要改为　y=X W +b
        '''
        if W is None:
            a = 1. / n_in
            W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))

        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0

        self.rng = rng
        #保存上一层的输入
        self.x = input
        self.W = W
        self.b = b

        if activation == tanh:
            self.dactivation = dtanh

        elif activation == sigmoid:
            self.dactivation = dsigmoid

        elif activation == ReLU:
            self.dactivation = dReLU
        #
        # elif activation ==softmax:
        #     self.dactivation=None
        else:
            raise ValueError('activation function not supported.')

        self.activation = activation

        ###########################################################################
        #以下内容为反向更新用到的参数
        #保存第一层的输出值
        self.y = None

        #反向的下层传播误差
        self.bp_err = None

        #上一次的
        self.pre_impulse_W=0



    def output(self, input=None):
        '''
        :param input:  #上一层的输出内容，
        :return:
        '''
        if input is not None:
            self.x = input
        #在进行矩阵乘法时，每个神经元的权值是已经累加过的。
        # print("W=",self.W,"b=", self.b)
        linear_output = numpy.dot(self.x, self.W) + self.b
        self.y = self.activation(linear_output)
        return self.y


    def forward(self, input=None):
        '''
        前向输出
        :param input:
        :return:
        '''
        return self.output(input=input)


    def backward(self, inputs,bp_err,lr=0.1, W=None, dropout=False, mask=None, impulse=None):
        '''
        以从输入到输出，为向前
        从输出到输出，为向后

        :param inputs:反向前一层输出结果
        :param bp_err:反向后一层的传播误差
        :param W:默认是Ｎone，使用当前层的权重
        :param lr:
        :param dropout:
        :param mask:
        :param impulse: 冲量项系数
        :return:
        '''

        '''
        z1 -> z2 -> z3
        w1    w2    w3
        e1    e2    e3

        # 当前层为z3 则　当前层误差为e3=输出误差　Ｗ３　　　　　z2层的误差计算为e2=grad(z2)* e3 * W3
        #这里必须为３个参数，第一个为反向前一层一输出值，第二个为反向后一层的误差，第三个为
        '''

        # bp_err = self.dactivation(prev_layer.x) * numpy.dot(prev_layer.bp_err, prev_layer.W.T)
        if numpy.array(inputs).ndim==1:
            inputs=inputs[newaxis]
        if W is not None:
            bp_err = self.dactivation(inputs) * numpy.dot(bp_err, W.T)
        else:
            bp_err = self.dactivation(inputs) * numpy.dot(bp_err, self.W.T)

        if dropout == True:
            bp_err *= mask

        # 添加冲量,动量因子
        if self.x.ndim==1:
            self.x=self.x[newaxis]

        if impulse is not None:
            self.pre_impulse_W = lr * numpy.dot(self.x.T, bp_err) + impulse * self.pre_impulse_W
        else:
            self.pre_impulse_W = lr * numpy.dot(self.x.T, bp_err)

        self.W += self.pre_impulse_W
        # self.W += lr * numpy.dot(self.x.T, bp_err)

        '''
        参数axis的问题
        a = np.array([1, 5, 5, 2])
        print(np.sum(a, axis=0), "\n")
        #对于一维数组，不能进行列处理
        
        #以下是矩阵
        b = np.array([[1],[2],[3]])
        print(np.sum(b, axis=1), "\n")
        a = np.array([[1, 5, 5, 2],
                      [9, 6, 2, 8],
                      [3, 7, 9, 1]])
        print(np.sum(a, axis=0), "\n")
        print(np.sum(a, axis=1), "\n")
        结果：
        13 
        
        [1 2 3] 
        
        [13 18 16 11] 
        
        [13 25 20] 
        
        '''
        self.b += lr * numpy.mean(bp_err, axis=0)

        self.bp_err = bp_err
        return bp_err


    def dropout(self, input, p, rng=None):
        if rng is None:
            rng = numpy.random.RandomState(123)

        mask = rng.binomial(size=input.shape,
                            n=1,
                            p=1-p)  # p is the prob of dropping

        return mask
                     

    def sample_h_given_v(self, input=None):
        if input is not None:
            self.x = input

        v_mean = self.output()
        h_sample = self.rng.binomial(size=v_mean.shape,
                                           n=1,
                                           p=v_mean)
        return h_sample


