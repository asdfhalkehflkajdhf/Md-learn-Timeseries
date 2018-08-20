# -*- coding: utf-8 -*-

import sys
import numpy

from MDLearn.DL.HiddenLayer import HiddenLayer
from MDLearn.DL.LogisticRegression import LogisticRegression
from MDLearn.DL.RBM import RBM
from MDLearn.DL.CRBM import CRBM
from MDLearn.DL.DBN import DBN
from MDLearn.DL.utils import *
from MDLearn.utils.EvalueationIndex import evalueationIndex
import numpy as np

class CDBN(DBN):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 rng=None):
        
        self.x = input
        self.y = label

        self.sigmoid_layers = []
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

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        rng=rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            # construct rbm_layer
            if i == 0:
                rbm_layer = CRBM(input=layer_input,     # continuous-valued inputs
                                 n_visible=input_size,
                                 n_hidden=hidden_layer_sizes[i],
                                 W=sigmoid_layer.W,     # W, b are shared
                                 hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layer_sizes[i],
                                W=sigmoid_layer.W,     # W, b are shared
                                hbias=sigmoid_layer.b)
                
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()


def test_cdbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
              finetune_lr=0.1, finetune_epochs=200):
    x = numpy.array([[0.4, 0.5, 0.5, 0., 0., 0.],
                     [0.5, 0.3, 0.5, 0., 0., 0.],
                     [0.4, 0.5, 0.5, 0., 0., 0.],
                     [0., 0., 0.5, 0.3, 0.5, 0.],
                     [0., 0., 0.5, 0.4, 0.5, 0.],
                     [0., 0., 0.5, 0.5, 0.5, 0.]])

    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])

    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = CDBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[5, 5], n_outs=2, rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)

    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)

    # test
    # x = numpy.array([[0.5, 0.5, 0., 0., 0., 0.],
    #                  [0., 0., 0., 0.5, 0.5, 0.],
    #                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.]])

    print (dbn.predict(x))


##############################################################################
#加载数据
# 标准化
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [( (p-min(window)) / (max(window)-min(window))) for p in window]
        # normalised_window = [p /  window[0] - 1 for p in window]
        # print(normalised_window)
        normalised_data.append(normalised_window)
    return normalised_data
def load_data(filename, seq_len, normalise_window_bool):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    #字符串转数字
    for index in range(len(data)):
        t = float(data[index])
        data[index]=t

    # plt.plot(data, label='data')
    # plt.legend()
    # plt.show()
    sequence_length = seq_len + 1

    result = []

    #对数据进行分块，块大小为seq_len
    for index in range(len(data) - sequence_length):


        result.append((data[index: index + sequence_length]))

    #标准化数据
    if normalise_window_bool:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]
def test_xulie(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):
    import time
    global_start_time = time.time()
    print('> Loading data... ')
    seq_len = 5

    X_train, y_train, X_test, y_test = load_data('SN_m_tot_V2.0_1990.1-2017.8.csv', seq_len, False)
    print('> Data Loaded. Compiling...')
    ###############################################################################
    rng = numpy.random.RandomState(123)
    # construct DBN
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)
    t = y_train[numpy.newaxis].T
    y_train =  y_train[numpy.newaxis].T
    # print(X_train.shape, numpy.shape(y_train))
    dbn = CDBN(input=X_train, label=y_train, n_ins=seq_len, hidden_layer_sizes=[4, 4], n_outs=1, rng=rng)
    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
    y_rbf= dbn.predict(X_test)

    print('Training duration (s) : ', time.time() - global_start_time)
    ###############################################################################
    eI = evalueationIndex(y_rbf, y_test)
    print(y_rbf)
    print(y_test)
    print("MSE:", eI.MSE)
    print("RMSE:", eI.RMSE)


if __name__ == "__main__":
    test_cdbn()
    # test_xulie()