# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from MDLearn.utils.EvalueationIndex import *
from MDLearn.DL.HiddenLayer import HiddenLayer
from MDLearn.DL.LogisticRegression import LogisticRegression
from MDLearn.DL.RBM import RBM
from MDLearn.DL.utils import *


class DBN(object):
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 rng=None):
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
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,  # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # layer for output using Logistic Regression

        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        # self.finetune_cost = self.log_layer.negative_log_likelihood()



    def pretrain(self, lr=0.1, k=1, epochs=100):
        # pre-train layer-wise
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]
            
            for epoch in range(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost


    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost
            
            lr *= 0.95
            epoch += 1


    def predict(self, x):
        layer_input = x
        
        for i in range(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out



def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):

    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])
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
    dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[4, 4], n_outs=1, rng=rng)

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)


    # test
    x = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 0]])
    y_rbf = dbn.predict(x)
    print (y_rbf)
    print(numpy.array(y_rbf).shape)


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
##############################################################################
#加载数据画图

def plot_results_point(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def test_xulie(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):
    import time
    global_start_time = time.time()
    print('> Loading data... ')
    seq_len = 5

    data_src = EvaluationIndex.loadCsvData_Np("SN_m_tot_V2.0_1990.1-2017.8.csv")
    # plt.plot(data_src, label='data')
    # plt.legend()
    # plt.show()

    # 数据还源时使用
    t_mean=np.mean(data_src)
    t_min=np.min(data_src)
    t_max=np.max(data_src)

    #数据预处理
    sequence_length = seq_len + 1
    result = []
    #对数据进行分块，块大小为seq_len
    for index in range(len(data_src) - sequence_length):
        result.append(np.array(data_src[index: index + sequence_length]).ravel())
    # print(result)
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    # np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    #数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_train_normal = (x_train - t_mean) / (t_max - t_min)
    y_train_normal = (y_train - t_mean) / (t_max - t_min)
    x_test_normal = (x_test - t_mean) / (t_max - t_min)
    y_test_normal = (y_test - t_mean) / (t_max - t_min)
    # plt.plot(y_train_normal, label='data')
    # plt.legend()
    # plt.show()

    rng = np.random.RandomState(123)
    #源测试数据
    y_train_normal =  y_train_normal[np.newaxis].T
    y_test_normal = y_test_normal[np.newaxis].T

    # print(X_train, y_train)
    print('> Data Loaded. Compiling...')
    ###############################################################################
    print('> NeuralNetWork. fit - predict...')
    # construct DBN
    dbn = DBN(input=x_train_normal, label=y_train_normal, n_ins=seq_len, hidden_layer_sizes=[4, 4], n_outs=1, rng=rng)
    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
    # fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
    # test
    y_rbf = dbn.predict(x_test_normal)

    print('Training duration (s) : ', time.time() - global_start_time)
    print(y_rbf)
    ###############################################################################
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test_normal)
    plot_results_point(y_rbf, y_test_normal)
    eI.show()



    y_rbf_back = y_rbf*(t_max-t_min)+t_mean
    eI = EvaluationIndex.evalueationIndex(y_rbf_back.ravel(), y_test)
    # plot_results_point(y_rbf_back.ravel(), y_test)
    eI.show()


def test_xulie_duobuyuche(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):
    import time
    global_start_time = time.time()
    print('> Loading data... ')
    seq_len = 5
    f = open("temp-test-print-out.txt", 'w')

    for seq_len in [4,5,6,7,8,9]:
        for nodeNum in [2, 3, 4, 5, 6, 7, 8]:
            X_train, y_train, X_test, y_test = load_data('SN_m_tot_V2.0_1990.1-2017.8.csv', seq_len, True)
            # print('> Data Loaded. Compiling...')
            ###############################################################################
            #数据要注意
            rng = numpy.random.RandomState(123)
            # construct DBN
            X_train = numpy.array(X_train)
            y_train = numpy.array(y_train)
            X_test = numpy.array(X_test)
            y_test = numpy.array(y_test)
            t = y_train[numpy.newaxis].T
            y_train =  y_train[numpy.newaxis].T
            # print(X_train.shape, numpy.shape(y_train))

            dbn = DBN(input=X_train, label=y_train, n_ins=seq_len, hidden_layer_sizes=[nodeNum, 2+nodeNum], n_outs=1, rng=rng)
            # pre-training (TrainUnsupervisedDBN)
            dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
            # fine-tuning (DBNSupervisedFineTuning)
            dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
            y_rbf= dbn.predict(X_test)

            # print('Training duration (s) : ', time.time() - global_start_time)
            ###############################################################################
            eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)
            print(",pretrain_lr:", pretrain_lr, ",finetune_lr:",finetune_lr,",setp:",seq_len, ",nodeNum:", nodeNum, ",MSE:", eI.MSE, ",RMSE:", eI.RMSE, file=f)


if __name__ == "__main__":
    # test_xulie()
    test_dbn()

    # for i in [0.001, 0.01, 0.1, 0.5, 0.9]:
    #     for j in [0.001, 0.01, 0.1, 0.5, 0.9]:
    #         test_xulie_duobuyuche(i,1000, 1, j, 200)
