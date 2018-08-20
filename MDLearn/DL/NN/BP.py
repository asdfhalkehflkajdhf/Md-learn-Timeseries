# -*- coding: utf-8 -*-
import numpy as np
import numpy
from MDLearn.DL.utils import  *
from MDLearn.DL.HiddenLayer import HiddenLayer
from MDLearn.DL.LogisticRegression import LogisticRegression

from MDLearn.utils.EvalueationIndex import evalueationIndex
import matplotlib.pyplot as plt

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
                                        activation=tanh,
                                        W=W,
                                        b=b
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
                                            outputMap="softmax" #"softmax" #"tanh" #"identity" #"sigmoid"
                                            )

    def fit(self, x_train=None, y_train=None, learning_rate=0.1, epochs=10000, shuffle=False,  residual_error=1e-10):
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

    def fitBatch(self, x_train=None, y_train=None, learning_rate=0.1, epochs=1000, Batch=100, shuffle=False, residual_error=1e-10, show=False, show_epochs=None):
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
                if abs(np.mean(bp_err))<residual_error:
                    print("训练误差ＯＫ！")
                    return
                for j in range(self.hidden_n_layers - 1, -1, -1):
                    bp_err = self.sigmoid_layers[j].backward(inputs=x, bp_err=bp_err, lr=learning_rate, W=W)
                    W = self.sigmoid_layers[j].W
                    x = self.sigmoid_layers[j].x
            # print( numpy.mean(self.sigmoid_layers[-1].bp_err))
            if show_epochs is not None:
                '''显示内容'''
                if show and t % show_epochs==0:
                    print("epoch[%d]train err:" % (t), bp_err.mean())

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

def plot_results_point(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def sp500_bp_simple_test_所有一起归一化():
    import time
    global_start_time = time.time()
    print('> Loading data... ')
    seq_len = 9

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

    y_train =  np.array(y_train.ravel())[numpy.newaxis].T
    y_test =  np.array(y_test.ravel())[numpy.newaxis].T

    #数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_train_normal = (x_train - t_min) / (t_max - t_min)
    y_train_normal = (y_train - t_min) / (t_max - t_min)
    x_test_normal = (x_test - t_min) / (t_max - t_min)
    y_test_normal = (y_test - t_min) / (t_max - t_min)
    # plt.plot(y_train_normal, label='data')
    # plt.legend()
    # plt.show()
    #源测试数据
    y_train_normal =  np.array(y_train_normal.ravel())[numpy.newaxis].T
    y_test_normal = np.array(y_test_normal.ravel())[numpy.newaxis].T
    # print(x_test_normal, y_test_normal)
    print('> Data Loaded. Compiling...')
    ###############################################################################
    # Fit regression model
    rng = numpy.random.RandomState(123)

    #####################################################################################
    for h in [3]:
        network = BP(n_ins=seq_len, hidden_layer_sizes=[h], n_outs=1, rng=rng)
        for lr in [0.0001]:
            for ep in [15]:
                network.fitBatch(x_train_normal, y_train_normal, learning_rate=lr, epochs=ep)
                y_rbf = network.predict(x_test_normal)
                eI = EvalueationIndex.evalueationIndex(y_rbf, y_test_normal)
                print(h, " ", lr, " ", ep)
                eI.show()
                plot_results_point(y_rbf, y_test_normal)

    return
    #################################################################################
    network = BP(n_ins=seq_len, hidden_layer_sizes=[6], n_outs=1, rng=rng)
    # pre-training (TrainUnsupervisedDBN)
    # print(x_train_normal, y_train_normal)
    network.fitBatch(x_train_normal, y_train_normal, learning_rate=0.001, epochs=6)
    y_rbf= network.predict(x_test_normal)

    print('Training duration (s) : ', time.time() - global_start_time)
    ###############################################################################
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test_normal)
    eI.show()
    # plot_results_point(y_rbf, y_test_normal)

    #还原
    print("所有数据归一化，结果还原对比")
    t = (t_max-t_min)
    t = np.array(y_rbf)*t
    y_rbf_back = t + t_min
    eI = EvaluationIndex.evalueationIndex(y_rbf_back.ravel(), y_test)
    eI.show()
    # plot_results_point(y_rbf_back.ravel(), y_test)

    #测试数据为归一化，lable为未归一化
    print("测试数据为归一化，lable为未归一化")
    network.fitBatch(x_train_normal, y_train, learning_rate=0.001, epochs=6)
    y_rbf= network.predict(x_test_normal)
    ###############################################################################
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)
    eI.show()

def sp500_svr_simple_test_所有一起归一化2():
    '''
    单值预测
    :return:
    '''
    import time
    global_start_time = time.time()
    print('> Loading data... ')

    data_src = EvaluationIndex.loadCsvData_Np("GDP_1981-2016.csv")
    # plt.plot(data_src, label='data')
    # plt.legend()
    # plt.show()

    # 数据还源时使用
    t_mean=np.mean(data_src)
    t_min=np.min(data_src)
    t_max=np.max(data_src)


    #数据预处理
    seq_len = 4
    sequence_length = seq_len + 1
    result = []
    #对数据进行分块，块大小为seq_len
    for index in range(len(data_src) - sequence_length):
        result.append(np.array(data_src[index: index + sequence_length]).ravel())
    # print(result)
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    row = 22
    result_len = len(result)
    # np.random.shuffle(train)
    x_result = result[:, :-1]
    y_result = result[:, -1]

    #数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_result = (x_result - t_min) / (t_max - t_min)
    y_result = (y_result - t_min) / (t_max - t_min)
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)

    y_rbf_all = []
    y_test_all = []
    rng = numpy.random.RandomState(1233)
    for y_i in range(row, result_len):
        x_train = x_result[y_i-row:y_i]
        y_train = y_result[y_i-row:y_i]
        x_test = x_result[y_i:y_i+1]
        y_test = y_result[y_i:y_i+1]

        y_train = y_train[np.newaxis].T

        # y_train =  y_train.ravel()
        # y_test = y_test.ravel()
        network = BP(n_ins=seq_len, hidden_layer_sizes=[8], n_outs=1, rng=rng)

        # pre-training (TrainUnsupervisedDBN)
        # print(x_train_normal, y_train_normal)
        network.fitBatch(x_train, y_train, learning_rate=0.3, residual_error=1e-5)
        y_rbf = network.predict(x_test)
        y_rbf_all.append(y_rbf)
        y_test_all.append(y_test)

    y_rbf_all = np.array(y_rbf_all).ravel()
    y_test_all = np.array(y_test_all).ravel()

    print(y_rbf_all)
    print(y_test_all)

    eI = EvaluationIndex.evalueationIndex(y_rbf_all, y_test_all)
    eI.show()
    plot_results_point(y_rbf_all, y_test_all)

    return


def test_xulie_duobuyuche():
    import time
    global_start_time = time.time()


    f = open("temp-test-print-out.txt", 'w')

    # for seq_len in [3,4,5,6,7,8,9]:
    for seq_len in [9]:

        data_src = EvaluationIndex.loadCsvData_Np("SN_m_tot_V2.0_1990.1-2017.8.csv")
        # plt.plot(data_src, label='data')
        # plt.legend()
        # plt.show()

        # 数据还源时使用
        t_mean = np.mean(data_src)
        t_min = np.min(data_src)
        t_max = np.max(data_src)

        # 数据预处理
        sequence_length = seq_len + 1
        result = []
        # 对数据进行分块，块大小为seq_len
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

        y_train = np.array(y_train.ravel())[numpy.newaxis].T
        y_test = np.array(y_test.ravel())[numpy.newaxis].T

        # 数据归一化
        # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
        x_train_normal = (x_train - t_min) / (t_max - t_min)
        y_train_normal = (y_train - t_min) / (t_max - t_min)
        x_test_normal = (x_test - t_min) / (t_max - t_min)
        y_test_normal = (y_test - t_min) / (t_max - t_min)
        # plt.plot(y_train_normal, label='data')
        # plt.legend()
        # plt.show()
        # 源测试数据
        y_train_normal = np.array(y_train_normal.ravel())[numpy.newaxis].T
        y_test_normal = np.array(y_test_normal.ravel())[numpy.newaxis].T
        ###############################################################################
        rng = numpy.random.RandomState(123)

        # for nodeNum in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for nodeNum in [ 7, 8, 9, 10]:
            network = BP(n_ins=seq_len, hidden_layer_sizes=[nodeNum,nodeNum], n_outs=1, rng=rng)
            # for lr in [0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
            for lr in [ 0.001, 0.002]:
                for ep in range(30):
                    # network.fitBatch(x_train_normal, y_train_normal, learning_rate=lr, epochs=ep)
                    # y_rbf = network.predict(x_test_normal)
                    # eI = EvaluationIndex.evalueationIndex(y_rbf, y_test_normal)
                    network.fitBatch(x_train_normal, y_train, learning_rate=lr, epochs=ep)
                    y_rbf = network.predict(x_test_normal)
                    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)
                    # print(",seq:", seq_len, ",nodeNum:", nodeNum, ",lr:", lr, ",ep:",
                    #       ep, ",MSE:", eI.MSE, ",RMSE:", eI.RMSE, file=f)
                    print(",", seq_len, ",", nodeNum, ",", lr, ",",
                          ep, ",", eI.MSE, ",", eI.RMSE,",",eI.MAPE, file=f)




def Test_bp2():
    print("test neural network")

    x = numpy.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

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
    rng = numpy.random.RandomState(123)

    network = BP(n_ins=2, hidden_layer_sizes=[3], n_outs=2, rng=rng)
    network.fit(x, y, learning_rate=0.1, epochs=500)


    # for item in test_data:
    res = network.predict(x)
    print(res)
    return


if __name__ == "__main__":
    Test_bp2()
    # test_xulie()
    # sp500_svr_simple_test_所有一起归一化2()
    # test_xulie_duobuyuche()