# -*- coding: utf-8 -*-
import numpy as np
import numpy
from MDLearn.DL.utils import *

from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression


#ＢＰ网络模型结构
class BP(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[], n_outs=2,\
                 rng=None, W=None, b=None):
        '''

        :param input:前两个参数，最好不要用，只有在ＦＩＴ的时候才需要数据，
        :param label:
        :param n_ins:
        :param hidden_layer_sizes:
        :param n_outs:
        :param rng:
        :param W:
        :param b:
        '''

        self.x = input
        self.y = label

        self.sigmoid_layers = []
        #这个是隐层数
        self.hidden_n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)
        # print("hidden_n_layers=", self.hidden_n_layers)
        if rng is None:
            rng = numpy.random.RandomState(1234)

        assert self.hidden_n_layers >= 0

        # construct multi-layer
        # layer_input=None
        for i in range(self.hidden_n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(
                                        # input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        rng=rng,
                                        W=W,
                                        b=b,
                                        activation=tanh
                                        )
            self.sigmoid_layers.append(sigmoid_layer)

        # 添加输出层
        input_size=None
        if self.hidden_n_layers == 0:
            input_size = n_ins
        else:
            input_size = hidden_layer_sizes[ - 1]

        self.log_layer = LogisticRegression(#input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=input_size,
                                            n_out=n_outs,
                                            W = W,
                                            b = b,
                                            outputMap="sigmoid"#"softmax" #"sigmoid" #"tanh" #"identity" #"sigmoid"
                                            )

    def fit(self, x_train=None, y_train=None, learning_rate=0.1, epochs=10000, shuffle=False,  residual_error=1e-3):
        '''''
        训练网络, 默认按顺序来训练
        方法 1：按训练数据顺序来训练
        方法 2: 随机选择测试
        :param x_train: 输入数据
        :param y_train: 输出数据
        :param learning_rate: 学习率
        :param epochs:权重更新次数
        :param shuffle:数据序列打乱
        '''
        if x_train is not None:
            self.x = x_train
        if y_train is not None:
            self.y = y_train

        indices = np.arange(len(x_train))
        for _ in range(epochs):
            if shuffle:
                #用于将数据序列打乱
                np.random.shuffle(indices)

            #这是数据乱序的
            for i in indices:
                #先进行前向计算
                res = x_train[i]
                for j in range(self.hidden_n_layers):
                    res = self.sigmoid_layers[j].forward(res)

                #计算输出误差
                bp_err = self.log_layer.train(input=res, lable=self.y[i])
                W=self.log_layer.W
                x=self.log_layer.x
                #反向传播
                '''
                反向传播时，三个必先参数，使用的是上一层的返回的结果，上层的输入，上层的误差，上层的权重（重要）
                '''
                if abs(np.mean(bp_err))<residual_error:
                    return
                for j in range(self.hidden_n_layers-1, -1, -1):
                    bp_err = self.sigmoid_layers[j].backward(inputs=x, bp_err = bp_err, lr=learning_rate, W=W)
                    W=  self.sigmoid_layers[j].W
                    x=self.sigmoid_layers[j].x

        pass

    def fitBatch(self, x_train=None, y_train=None, learning_rate=0.1, epochs=1000, Batch=100, shuffle=False, residual_error=1e-3):
        '''''
        训练网络, 默认按顺序来训练
        方法 1：按训练数据顺序来训练
        方法 2: 随机选择测试
        :param x_train: 输入数据
        :param y_train: 输出数据
        :param learning_rate: 学习率
        :param epochs:权重更新次数
        :param shuffle:数据序列打乱
        '''
        if x_train is not None:
            self.x = x_train
        if y_train is not None:
            self.y = y_train

        indices = np.arange(len(x_train))
        for t in range(epochs):
            if shuffle:
                # 用于将数据序列打乱
                np.random.shuffle(indices)

            setp = len(x_train)//Batch
            start=None
            end = None
            # 这是数据乱序的
            for i in range(setp+1):
                start = i*Batch
                end = start+Batch
                # 先进行前向计算
                res = x_train[start:end]
                for j in range(self.hidden_n_layers):
                    res = self.sigmoid_layers[j].forward(res)

                # 计算输出误差
                bp_err = self.log_layer.train(input=res, lable=self.y[start:end])
                W = self.log_layer.W
                x = self.log_layer.x
                # 反向传播
                '''
                反向传播时，三个必先参数，使用的是上一层的返回的结果，上层的输入，上层的误差，上层的权重（重要）
              '''
                # if abs(np.mean(bp_err))<residual_error:
                #     return
                for j in range(self.hidden_n_layers - 1, -1, -1):
                    print(bp_err.shape, W.shape, x.shape)
                    bp_err = self.sigmoid_layers[j].backward(inputs=x, bp_err=bp_err, lr=learning_rate, W=W)
                    W = self.sigmoid_layers[j].W
                    x = self.sigmoid_layers[j].x
            # print( numpy.mean(self.sigmoid_layers[-1].bp_err))
        pass

    def predict(self, x):
        # n_times = np.arange(len(x))
        res = x
        for j in range(self.hidden_n_layers):
            res = self.sigmoid_layers[j].forward(res)

        res = self.log_layer.predict(res)
        return np.array(res)
        # return self.layers[0].calc_output(x)
    def predictBatch(self, x, Batch=100):
        # n_times = np.arange(len(x))
        indices = len(x)
        step = indices//Batch

        resAll=None
        for i in range(step+1):
            start = i*Batch
            end = start+Batch
            res = x[start:end]
            for j in range(self.hidden_n_layers):
                res = self.sigmoid_layers[j].forward(res)
            res = self.log_layer.predict(res)
            if resAll is None:
                resAll=res
            else:
                resAll=np.vstack((resAll, res))

        return np.array(resAll)

    def init_para(self, w_index, w_list,  b_list):

        for index, hidden_i in enumerate(w_index):
            #不能最大层数
            assert hidden_i<=self.hidden_n_layers
            if hidden_i == self.hidden_n_layers:
                assert self.log_layer.W.shape == w_list[index].shape
                self.log_layer.W = w_list[index].copy()
                self.log_layer.b = b_list[index].copy()
            else:
                assert self.sigmoid_layers[hidden_i].W.shape == w_list[index].shape
                self.sigmoid_layers[hidden_i].W=w_list[index].copy()
                self.sigmoid_layers[hidden_i].b=b_list[index].copy()
##############################################################################
#加载数据


def Test_bp2():
    print("test neural network")
    '''说明：隐层用tanh　拟合时使用全部，不能一个一个来fitBatch'''
    x = numpy.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = numpy.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])
    y = numpy.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])
    # x = numpy.array([[0,  0]
    #
    #                  ])
    #
    # y = numpy.array([[0, 1]
    #
    #                  ])
    np.set_printoptions(precision=3, suppress=True)
    rng = numpy.random.RandomState(1234)

    network = BP(n_ins=2, hidden_layer_sizes=[3], n_outs=2, rng=rng)
    network.fitBatch(x, y, learning_rate=0.1, epochs=500)


    # for item in test_data:
    res = network.predictBatch(x)
    print(res)
    return

'''ＲＢＭ　ＢＰ分离，进行ＲＢＭ微调'''
def Test_bp_MNIST():
    import input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    trX = trX[:5000]
    trY = trY[:5000]
    network = BP(n_ins=784, hidden_layer_sizes=[400,100], n_outs=10)
    network.fitBatch(trX, trY, learning_rate=0.1, epochs=1,Batch=100)
    res = network.predict(teX)
    print(np.mean(np.argmax(teY, axis=1) == np.argmax(res, 1) ))

if __name__ == "__main__":
    # Test_bp2()
    Test_bp_MNIST()