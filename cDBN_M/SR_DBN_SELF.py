# -*- coding: utf-8 -*-

import sys
import numpy as np
from cDBN_M.SR_RBM_SELF import RBM
from cDBN_M.utils import *
from cDBN_M.BP_SELF import BP

class DBN(object):
    ''''''
    '''说明：ＤＢＮ有三种情况：
    第一种，ＤＢＮ训练ＲＢＭ，ＲＢＭ参数给ＢＰ使用这种是最好的。ＢＰ在ＤＢＮ外单独建立
    第二种，ＤＢＮ训练ＲＢＭ，ＲＢＭ自身实现反向计算，过程复杂了，效果没有第一咱好。
        例：
            net = DBN(layer_sizes=[seq_len,20,40], bp_layer=[1])
            net.pretrain(x_train,lr=lr, epochs=200)
            net.fineTune(x_train, y_train,lr=lr, epochs=10000)
    第三种：ＤＢＮ训练ＲＢＭ，ＲＢＭ特征结果，给ＢＰ使用，即，每个数据要经过ＲＢＭ特征提取才进入到ＢＰ处理。
        例：
            net = DBN(layer_sizes=[seq_len,20,40], bp_layer=[1])
            net.train(x_train, y_train, lr=lr, epochs=10000)           
    '''
    def __init__(self, layer_sizes=[3, 3] , bp_layer=[], rng=None):
        '''
        :param input:
        :param label:
        :param n_ins:
        :param hidden_layer_sizes:
        :param n_outs:
        :param rng: 随机数发生器
        '''
        '''说明：layer_sizes的最后一个参数，是bp_layer的第一个输入参数
        例：ＤＢＮ（layer_sizes[10,10,20], bp_layer=[5,1]）
        bp 是[２０，　５，　１]
        '''

        self.rbm_layers = []
        self.n_layers = len(layer_sizes)-1  # = len(self.rbm_layers)

        if rng is None:
            rng = np.random.RandomState(1234)

        '''最少为一层'''
        assert self.n_layers > 0

        # construct multi-layer
        for i in range(self.n_layers):
            # layer_size
            # construct rbm_layer
            rbm_layer = RBM(
                            n_visible=layer_sizes[i],
                            n_hidden=layer_sizes[i+1],
                            rng = rng)
            self.rbm_layers.append(rbm_layer)


        self.bp_layers=None
        self._n_bp_layers = len(bp_layer)
        if self._n_bp_layers>0:
            para = [layer_sizes[-1]]+bp_layer
            self.bp_layers = BP(para)

    def getHyperParameter(self):
        W_list =[]
        b_list=[]
        for i in range(self.n_layers):
            W_list.append(self.rbm_layers[i].W)
            b_list.append(self.rbm_layers[i].b)
        return W_list, b_list

    '''对RBM进行拟合'''
    def pretrain(self, input, lr=0.1, k=1, epochs=100, batch_size=None, residual_error=None, gaus=False, show=False):
        '''

        :param input:
        :param lr:
        :param k:
        :param epochs:
        :param residual_error: RBM重构误差
        :param gaus: RBM是否使用高斯分布
        :return:
        '''
        import MDLearn.utils.Draw as draw
        err = None
        layer_input = input
        for i in range(self.n_layers):
            rbm = self.rbm_layers[i]
            err = rbm.train(lr=lr, k=k, epochs=epochs, batch_size=batch_size, input=layer_input,
                            residual_error=residual_error, gaus=gaus)
            # draw.plot_all_point(err)
            err = rbm.get_errs(layer_input)
            if show: print("rbm %d:" % (i), err)
            layer_input = rbm.forward(layer_input)
            # print(np.mean(err))

        return layer_input

    def rbm_forward(self, x):
        layer_input = x
        for i in range(self.n_layers):
            layer_input = self.rbm_layers[i].forward(layer_input)
        return layer_input
    def rbm_backward(self,x, W, bp_err, lr=0.1):
        layer_input = x
        for i in range(self.n_layers).__reversed__():
            rbm = self.rbm_layers[i]
            bp_err = rbm.backward(layer_input, bp_err, W, lr=lr)
            W = rbm.W
            layer_input = rbm.input
    '''进行反向微调,会传到ＲＭＢ层'''
    def fineTune(self, input, label, lr=0.1, epochs=100, batch_size=None, residual_error=None, gaus=False, show=False):
        ''''''
        '''获取多少个数据'''
        n_dataLen= len(label)
        assert n_dataLen >0
        n_batch, batch_size = getNBatch(n_dataLen, batch_size)

        for _ in range(epochs):
            '''初始化当前批次的errs '''
            cur_epoch_errs = np.zeros((n_batch,))
            cur_epoch_errs_ptr = 0
            '''对当前批数据进行训练'''
            for i in range(n_batch):
                start = i*batch_size
                end = start+batch_size
                # 先进行前向计算
                batch_Xtrain = input[start:end]
                batch_Ytrain = label[start:end]

                '''进行前向传播－－ＲＢＭ部分'''
                layer_input = self.rbm_forward(batch_Xtrain)
                '''进行前向和反向传播－－ＢＰ部分'''
                assert self.bp_layers is not None
                self.bp_layers.train(layer_input, batch_Ytrain, lr=lr, epochs=1)
                '''进行反向传播－－ＲＢＭ部分'''
                bp_err = self.bp_layers.last_bp_err
                W = self.bp_layers.W

                self.rbm_backward(layer_input, W, bp_err, lr)
            pass

            if show:
                print("show info")
            pass


        pass
    def predict(self, xTest, epochs=1, batch=1000):
        layer_input = xTest

        for rbm in self.rbm_layers:
            layer_input = rbm.forward(layer_input)

        assert self.bp_layers is not None
        out = self.bp_layers.predict(layer_input)
        return out

    '''对ＤＢＮ进行微调，不会传到ＲＢＭ层，只在ＢＰ层微调。ＢＰ和ＲＢＭ是分离的，这个是网上找的方法'''
    def train(self,input,label, lr=0.1, k=1, rbmEpochs=100,bpEpochs=1000, batch_size=None, residual_error=None, gaus=False, show=False):
        err=None
        layer_input = input
        for i in range(self.n_layers):
            rbm = self.rbm_layers[i]
            err = rbm.train(lr=lr, k=k, epochs=rbmEpochs, batch_size=batch_size,input=layer_input, residual_error=residual_error, gaus=gaus)
            # draw.plot_all_point(err)
            '''全部进行err计算会内在错误'''
            # err = rbm.get_errs(layer_input)
            if show : print("rbm %d:"%(i),err)
            layer_input = rbm.forward(layer_input)
            # print(np.mean(err))

        assert self.bp_layers is not None
        self.bp_layers.train(layer_input,label,lr=lr,epochs=bpEpochs,batch_size=batch_size)



# def dbn_test(pretrain_lr=0.1, pretraining_epochs=100, k=1, \
#              finetune_lr=0.1, finetune_epochs=200):
#     x = np.array([[1, 1, 1, 0, 0, 0],
#                      [1, 0, 1, 0, 0, 0],
#                      [1, 1, 1, 0, 0, 0],
#                      [0, 0, 1, 1, 1, 0],
#                      [0, 0, 1, 1, 0, 0],
#                      [0, 0, 1, 1, 1, 0]])
#     y = np.array([[1, 0],
#                      [1, 0],
#                      [1, 0],
#                      [0, 1],
#                      [0, 1],
#                      [0, 1]])
#
#     rng = np.random.RandomState(123)
#
#     # construct DBN
#     dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=2, rng=rng)
#
#     # pre-training (TrainUnsupervisedDBN)
#     dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)
#
#     # fine-tuning (DBNSupervisedFineTuning)
#     dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
#
#     # test
#     x = np.array([[1, 1, 0, 0, 0, 0],
#                      [0, 0, 0, 1, 1, 0],
#                      [1, 1, 1, 1, 1, 0]])
#
#     print(dbn.predict(x))


# if __name__ == "__main__":
#     dbn_test()

'''ＲＢＭ　ＢＰ分离，进行ＲＢＭ微调'''
def Test_dbn_MNIST():
    import input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    trX = trX[:5000]
    trY = trY[:5000]
    dbn = DBN( layer_sizes=[784,400,100],bp_layer=[10])
    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(input=trX, lr=1.0, k=1, epochs=1, batch_size=100)
    dbn.fineTune(input=trX, label=trY, lr=1.0, epochs=1, batch_size=100)
    res = dbn.predict(teX)
    print(np.mean(np.argmax(teY, axis=1) == np.argmax(res, 1) ))


'''ＲＢＭ　ＢＰ分离，参数'''
def Test_dbn_bp_MNIST():
    import input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    trX = trX[:5000]
    trY = trY[:5000]
    teX = teX[:1000]
    teY = teY[:1000]
    dbn = DBN( layer_sizes=[784,400,100])
    # pre-training (TrainUnsupervisedDBN)
    errs = dbn.pretrain(input=trX, lr=0.1, k=1, epochs=100, batch_size=100)#, gaus=True)
    print("dbn :",errs)

    network = BP([784,400,100,10])
    network.setActivation([ sigmoid, sigmoid, sigmoid])#softmax
    wList,bList=dbn.getHyperParameter()
    network.setHyperParameter(wList,bList)
    network.train(trX,trY,lr=0.1,epochs=1, batch_size=100)
    # for item in test_data:
    res = network.predict(teX)

    print(np.mean(np.argmax(teY, axis=1) == np.argmax(res, 1) ))
    '''0.098'''

'''ＲＢＭ　ＢＰ分离，结果'''
def Test_dbn_bp2_MNIST():
    import input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    dbn = DBN( layer_sizes=[784,400,100],bp_layer=[10])
    # pre-training (TrainUnsupervisedDBN)
    # errs = dbn.train(input=trX,label=trY, lr=1.0, k=1, rbmEpochs=10, bpEpochs=100, batch_size=100)
    errs = dbn.train(input=trX,label=trY, lr=1.0, k=1, rbmEpochs=1, bpEpochs=1, batch_size=100)
    # print("dbn :",errs)

    res = dbn.predict(teX)

    print(np.mean(np.argmax(teY, axis=1) == np.argmax(res, 1) ))
    '''0.098'''

if __name__ == "__main__":
    # Test_dbn_MNIST()
    Test_dbn_bp_MNIST()
    # Test_dbn_bp2_MNIST()