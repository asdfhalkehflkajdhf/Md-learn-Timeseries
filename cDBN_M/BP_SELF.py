import numpy as np
import time

from cDBN_M.utils import  *


class BP(object):
    def __init__(self, layer_list):
        '''

        :param layer_list:
        例：定义一个一层的网络　bp=BP([2,1]),有两个输入一个输出
             定义一个两层的网络　bp=BP([2,3,1]),有两个输入，一个输出，隐层有3个结点
        '''
        self.n_layer = 0
        '''初始化网络结构'''
        self.setNetStruct(layer_list)

        '''外部使用参数'''
        self.last_bp_err=None
        self.W=None

        #重构误差
        self.re_errs=[]
        #训练误差
        self.train_errs=[]
        #运行时间
        self.use_time=[]

    def setNetStruct(self, layer_list):
        '''初始化网络结构'''
        self._setNetStruct(layer_list)
        '''初始化激活函数'''
        self.setActivation(None)

    def _setNetStruct(self, layer_list):
        self.n_layer = len(layer_list)-1
        '''n_layer 层大小，最少有两个参数，一层'''
        assert self.n_layer>=1


        '''生成随机数,为了不使全部的值为０或是相同的值，避免输出相同的结果'''
        # rng = np.random.RandomState(int(time.time())*100)
        rng = np.random.RandomState(1234)
        '''初始化每一层的计算输出值'''
        self._outValue=[None]*self.n_layer

        self._W=[]
        self._b=[]
        for i in range(self.n_layer):
            a = 1. / layer_list[i]
            row = layer_list[i]
            column = layer_list[i+1]
            w = np.array(rng.uniform(  # initialize W uniformly
                low=-a,high=a,
                size=(int(row), int(column))))
            self._W.append(w)

            '''偏只是一个行向量'''
            b =np.zeros(column)  # initialize bias 0

            self._b.append(b)
        pass

    def setActivation(self,act_list):
        assert self.n_layer >= 1
        '''

        :param act_list:如果为None则所有使用sigmoid,否则必须与层数相同
        :return:
        '''
        self._activation=[]
        if act_list is None:
            for i in range(self.n_layer):
                self._activation.append(sigmoid)
        else:
            '''激活函数个数是少一个，因为第一层没有'''
            assert len(act_list) == self.n_layer
            for i in range( self.n_layer):
                self._activation.append(act_list[i])

        pass

    def setHyperParameter(self, W_list, b_list):
        '''

        :param W_list:
        :param b_list:
        :return:
        '''
        assert self.n_layer >= 1
        '''注意b偏置是一个行向量'''
        '''权重和偏置个数必须要一样'''
        Wlen = len(W_list)
        assert Wlen == len(b_list)

        assert Wlen <= self.n_layer

        for i in range(Wlen):
            assert W_list[i].shape == self._W[i].shape
            self._W[i] =W_list[i].copy()

            assert b_list[i].shape == self._b[i].shape
            self._b[i] =b_list[i].copy()

        pass

    def predict(self, xTest, epochs=1, batch_size=1):
        assert self.n_layer >= 1
        xLen = len(xTest)
        assert xLen > 0
        step, batch_size = getNBatch(xLen, batch_size)
        resAll=None
        for _ in range(epochs):
            for i in range(step):
                start = i*batch_size
                end = start+batch_size
                res = xTest[start:end]
                res = self._forward1(res)
                if resAll is None:
                    resAll=res
                else:
                    resAll=np.vstack((resAll, res))
        return np.array(resAll)

    '''简单前向输出'''
    def _forward1(self, x):
        out_sigmoid = x
        out_state = None
        for i in range(self.n_layer):
            out_state = np.dot(out_sigmoid,self._W[i])+self._b[i]
            out_sigmoid = self._activation[i](out_state)
        return out_sigmoid

    '''训练时向前输出'''
    def _forward2(self, x, y):
        '''

        :param x:说明输出为n行表示n条记录，第一条记录为一处行向量
        :return: 反回最后一层的小批量误差输出
        '''

        out_sigmoid = x
        out_state = None
        for i in range(self.n_layer):
            xinput = out_sigmoid
            out_state = np.dot(out_sigmoid,self._W[i])+self._b[i]
            out_sigmoid = self._activation[i](out_state)
            '''保存每一层的输出，第一个值为状态，第二个值为激活值, 第三个值为输入值'''
            self._outValue[i]=[out_state, out_sigmoid, xinput]

        EL=(y-out_sigmoid)**2
        Etotal = np.mean(EL, axis=1)
        return Etotal

    def _backwardUpdateW(self,xTrain, yLabel,lr=0.1):
        '''

        :param yLabel:
        :return:
        '''

        '''从后向前，'''
        for i in range(self.n_layer).__reversed__():

            '''保存每一层的输出，第一个值为状态，第二个值为激活值, 第三个值为输入值'''
            outValue = self._outValue[i]

            # if i==self.n_layer-1:
            #     '''这个更新是按web文档写的,激活函数求导时参数为zi,对于正常分类不出效果,说明可能是有问题的的'''
            #     dert = -(yLabel - outValue[1]) * self._activation[-1](outValue[0], grad=True)
            # else:
            #     dert = (np.dot( dert, self._W[i+1].T)) * self._activation[i](outValue[0], grad=True)

            if i == self.n_layer - 1:
                '''最后一层单独计算，这个是'''
                '''这个更新是按数据挖掘导论方法写的'''
                dert = (yLabel - outValue[1])
            else:
                dert = (np.dot(dert, self._W[i + 1].T)) * self._activation[i](outValue[1], grad=True)

            '''计算更新量'''
            if i == 0:
                preOutState = xTrain
            else:
                '''小一层的sigmoid输出'''
                preOutState = self._outValue[i - 1][1]
            dert_W=np.dot(preOutState.T, dert)
            dert_b=np.mean(dert, axis=0)

            # print(len(dert_W), dert_W)/
            # print(len(dert_b), dert_b)
            '''进行反向更新'''
            self._W[i]+=lr*dert_W
            self._b[i]+=lr*dert_b

            '''外部使用参数更新'''
            self.W = self._W[0]
        self.last_bp_err = dert
        pass

    def train(self, xTrain, yLabel,lr=0.1, epochs=1, batch_size=None,residual=None, show=False, show_epochs=None):
        assert self.n_layer >= 1

        '''总的errs列表'''
        #重构误差
        self.re_errs=[]
        #训练误差
        self.train_errs=[]
        #运行时间
        self.use_time=[]
        '''开始时间'''
        cur_start_time = time.time()

        xLen = len(xTrain)
        assert xLen > 0

        setp, batch_size = getNBatch(xLen, batch_size)

        for ep in range(epochs):
            start=None
            end = None
            '''初始化当前批次的errs '''
            cur_epoch_errs = np.zeros((setp,))
            cur_epoch_errs_ptr = 0
            '''对当前批数据进行训练'''
            for i in range(setp):
                start = i*batch_size
                end = start+batch_size
                # 先进行前向计算
                batch_Xtrain = xTrain[start:end]
                batch_Ytrain = yLabel[start:end]

                '''反回误差'''
                batch_err = self._forward2(batch_Xtrain, batch_Ytrain)
                if residual is not None and abs(np.mean(batch_err))<residual:
                    print("训练误差ＯＫ！")
                    return
                # print(err)
                self._backwardUpdateW(batch_Xtrain, batch_Ytrain, lr)
                if show_epochs is not None and ep % show_epochs == 0:
                    cur_epoch_errs[cur_epoch_errs_ptr] = batch_err.mean()
                    cur_epoch_errs_ptr += 1

            if show_epochs is not None:
                if ep % show_epochs==0:
                    '''保存当前层的errs'''
                    self.re_errs = np.hstack([self.re_errs, cur_epoch_errs.mean()])
                    self.use_time = np.hstack([self.use_time, time.time()-cur_start_time])
                    self.train_errs = np.hstack([self.train_errs, cur_epoch_errs.mean()])
                '''显示内容'''
                if show and ep % show_epochs==0:
                    print("epoch[%d]train err:" % (ep), self.train_errs[-1])
        pass

    '''以下内容分类时使用'''
    def _softmax(x):
        e = np.exp(x - np.max(x))  # prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

    def classSoftmax(self, x):
        '''
        说明：对最后的结果进行分类
        :return:
        '''
        return self._softmax(x)
        # return self._softmax(self._outValue[-1][0])


def Test_bp():
    print("test neural network")
    '''说明：tanh　softmax'''
    x = np.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = np.array([[0, 1],
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

    network = BP([2,3,2])
    network.setActivation([ tanh, sigmoid])#softmax
    network.train(x,y,lr=0.1,epochs=500)
    # for item in test_data:
    res = network.predict(x)
    print(res)
    return

def Test_bp_MNIST():
    import input_data
    import matplotlib.pyplot as plt

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    bp = BP( [784,400,100,10])
    trX = trX[:5000]
    trY = trY[:5000]

    # pre-training (TrainUnsupervisedDBN)
    bp.train(trX,trY, lr=0.1, epochs=1, batch_size=100)
    res = bp.predict(teX)
    print(np.mean(np.argmax(teY, axis=1) == np.argmax(res, 1) ))

    import MDLearn.utils.Draw as draw
    draw.plot_all_point(bp.re_errs)


if __name__ == "__main__":
    # Test_bp()
    Test_bp_MNIST()
