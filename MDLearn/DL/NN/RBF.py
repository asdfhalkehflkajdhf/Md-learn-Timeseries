# -*- coding: utf-8 -*-

from scipy import *
from scipy.linalg import norm, pinv

from matplotlib import pyplot as plt
# import EvaluationIndex
import numpy as np
import numpy

class RBF:
    def __init__(self, indim, numCenters, outdim):
        '''

        :param indim:       输入维度
        :param numCenters:  中心
        :param outdim:      输出维度
        '''
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        # self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        # xrange()  改名为range(), 要想使用range()       获得一个list, 必须显式调用
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        print ("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        print (G)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

def test_rbf():
    n = 100
    x = mgrid[-1:1:complex(0, n)].reshape(n, 1)
    # set y and add random noise
    y = sin(3 * (x + 0.5) ** 3 - 1)
    # y += random.normal(0, 0.1, y.shape)

    # rbf regression
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)

    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

    for c in rbf.centers:
        # RF prediction lines
        cx = arange(c - 0.7, c + 0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    plt.xlim(-1.2, 1.2)
    plt.show()

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

        # 标准化数据
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

    X_train, y_train, X_test, y_test = load_data('SN_m_tot_V2.0_1990.1-2017.8.csv', seq_len, True)
    print('> Data Loaded. Compiling...')
    ###############################################################################
    rng = numpy.random.RandomState(123)
    # construct DBN
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)
    t = y_train[numpy.newaxis].T
    y_train = y_train[numpy.newaxis].T
    # print(X_train.shape, numpy.shape(y_train))
    rbf = RBF(seq_len, 10, 1)
    rbf.train(X_train, y_train)
    y_rbf = rbf.test(X_test)

    print('Training duration (s) : ', time.time() - global_start_time)
    ###############################################################################
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)

    print("MSE:", eI.MSE)
    print("RMSE:", eI.RMSE)


if __name__ == '__main__':
    test_rbf()
    # test_xulie()