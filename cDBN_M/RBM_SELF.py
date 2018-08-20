# -*- coding: utf-8 -*-

import sys
import numpy
from numba import jit
import time
from cDBN_M.utils import *

class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, activation=sigmoid, rng=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if rng is None:
            rng = numpy.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0


        self.rng = rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        # 更新梯度
        self.__delta_w = 0
        self.__delta_vb = 0
        self.__delta_hb = 0
        #外部使用
        self.b = self.hbias

        #重构误差
        self.re_errs=[]
        #训练误差
        self.train_errs=[]
        #运行时间
        self.use_time=[]

        self.activation = activation


    def get_delta_fun(self,x_old, x_new, impulses=0, lr=1.0):
        return impulses * x_old + lr * x_new
        return impulses * x_old + lr * x_new * (1 - impulses) / numpy.shape(x_new)[0]


    '''三个采样函数'''
    def contrastive_divergence(self, lr=0.1, k=1, input=None, Gaussian=False, impulses=0):
        '''
        差异对比
        CD-K
        :param lr:
        :param k:
        :param input:
        :return:
        '''
        if input is not None:
            self.input = input
        
        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = ph_sample

        '''以上部分为 derteWij = e<ViHj>d - <ViHj>m 中e<ViHj>d部分数据
        以下部分为采样部分<ViHj>m
        '''
        for step in range(k):
            if step == 0:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(chain_start,Gaussian)
            else:
                nv_means, nv_samples,\
                nh_means, nh_samples = self.gibbs_hvh(nh_samples,Gaussian)

        # chain_end = nv_samples
        #更新参数
        self.__delta_w=self.get_delta_fun(x_old=self.__delta_w,
                                          x_new=(numpy.dot(self.input.T, ph_mean) - numpy.dot(nv_samples.T, nh_means)),
                                          lr=lr,
                                          impulses=impulses
                                          )
        self.__delta_vb=self.get_delta_fun(x_old=self.__delta_vb,
                                          x_new=numpy.mean(self.input - nv_samples, axis=0),
                                          lr=lr,
                                          impulses=impulses
                                          )
        self.__delta_hb=self.get_delta_fun(x_old=self.__delta_hb,
                                          x_new=numpy.mean(ph_mean - nh_means, axis=0),
                                          lr=lr,
                                          impulses=impulses
                                          )

        self.W += self.__delta_w
        self.vbias += self.__delta_vb
        self.hbias += self.__delta_hb

        # cost = self.get_reconstruction_cross_entropy()
        # return cost
    def CD(self, input=None, T=1, lr=0.1, impulses=0, Gaussian=False):
        '''
        cd http://blog.csdn.net/qian2729/article/details/50542764
        cdk http://blog.csdn.net/mytestmy/article/details/9150213
        #输入：一个训练样本x0;隐藏层单元个数m,学习速率alpha,最大训练周期T
        #输出：链接权重矩阵W,可见层的偏置向量a,隐藏层的偏置向量b
        #训练阶段：初始化可见层单元的状态为v1 = x0;W,a,b为随机的较小的数值
        for t = 1:T
            for j = 1:m #对所有隐藏单元
                P(h1j=1|v1)=sigmoid(bj + sum_i(v1i * Wij));
            for i = 1:n#对于所有可见单元
                p(v2i=1|h1)=sigmoid(ai + sum_j(Wij * h1j)
            for j = 1:m #对所有隐藏单元
                P(h2j=1|v2)=sigmoid(bj+sum_j(v2i*Wij))
            W = W + alpha * (P(h1=1|v1)*v1 - P(h2=1|v2)*v2)
            a = a + alpha * (v1 - v2)
            b = b + alpha*(P(h1=1|v1) - P(h2=1|v2))
        :return:
        '''
        if input is not None:
            self.input = input

        for t in range(T):
            hidden_p = self.propup(self.input)
            visible_recon_p = self.propdown(hidden_p)

            if Gaussian:
                '''数据量小，不能使用'''
                visible_recon_p = self.sample_gaussian(visible_recon_p)

            hidden_recon_p = self.propup(visible_recon_p)

            positive_grad = numpy.dot(self.input.T, hidden_p)
            negative_grad = numpy.dot(numpy.array(visible_recon_p).T, hidden_recon_p)

            # 更新参数
            self.__delta_w = self.get_delta_fun(x_old=self.__delta_w,
                                                x_new=( positive_grad - negative_grad),
                                                lr=lr,
                                                impulses=impulses
                                                )
            self.__delta_vb = self.get_delta_fun(x_old=self.__delta_vb,
                                                 x_new=numpy.mean(self.input - visible_recon_p, axis=0),
                                                 lr=lr,
                                                 impulses=impulses
                                                 )
            self.__delta_hb = self.get_delta_fun(x_old=self.__delta_hb,
                                                 x_new=numpy.mean(hidden_p - hidden_recon_p, axis=0),
                                                 lr=lr,
                                                 impulses=impulses
                                                 )

            self.W += self.__delta_w
            self.vbias += self.__delta_vb
            self.hbias += self.__delta_hb
    def CD_K(self, input=None, K=1, lr=0.1, impulses=0, Gaussian=False):
        if input is not None:
            self.input = input

        ''' CD-k 
        http://blog.csdn.net/mytestmy/article/details/9150213
        
        对比散度是英文ContrastiveDivergence（CD）的中文翻译。与Gibbs抽样不同，hinton教授指出当使用训练样本初始化v0的时候，仅需要较少的抽样步数（一般就一步）就可以得到足够好的近似了。
在CD算法一开始，可见单元的状态就被设置为一个训练样本，并用上面的几个条件概率来对隐藏节点的每个单元都从{0,1}中抽取到相应的值，然后再利用 来对可视节点的每个单元都从{0,1}中抽取相应的值，这样就得到了v1了，一般v1就够了，就可以拿来估算梯度了。
        '''
        if input is not None:
            self.input = input

        ''' CD-k '''
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        nh_samples = ph_sample

        for step in range(K):
                nv_means, nv_samples, \
                nh_means, nh_samples = self.gibbs_hvh(nh_samples,Gaussian)

        #更新参数
        self.__delta_w=self.get_delta_fun(x_old=self.__delta_w,
                                          x_new=(numpy.dot(self.input.T, ph_mean) - numpy.dot(nv_samples.T, nh_means)),
                                          lr=lr,
                                          impulses=impulses
                                          )
        self.__delta_vb=self.get_delta_fun(x_old=self.__delta_vb,
                                          x_new=numpy.mean(self.input - nv_samples, axis=0),
                                          lr=lr,
                                          impulses=impulses
                                          )
        self.__delta_hb=self.get_delta_fun(x_old=self.__delta_hb,
                                          x_new=numpy.mean(ph_mean - nh_means, axis=0),
                                          lr=lr,
                                          impulses=impulses
                                          )

        self.W += self.__delta_w
        self.vbias += self.__delta_vb
        self.hbias += self.__delta_hb

        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    '''高斯采样，Gibbs采样相关函数'''
    def sample_gaussian(self, x, stddev=1):
        return x + self.rng.normal(0,stddev,self.input.shape)
    @jit
    def sample_h_given_v(self, v0_sample):
        '''
        给定V 采样H
        :param v0_sample:
        :return:
        '''
        h1_mean = self.propup(v0_sample)
        '''
        		P(N)=(n-N)p^N(1−p)^(n−N)
        二项分布进行采样（size表示采样的次数），参数中的n, p分别对应于公式中的n,p，函数的返回值表示n中成功（success）的次数（也即N）
        '''
        h1_sample = self.rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)

        return [h1_mean, h1_sample]
    @jit
    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)
        
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return self.activation(pre_sigmoid_activation)
    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return self.activation(pre_sigmoid_activation)

    def gibbs_hvh(self, h0_sample, Gaussian=False):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)

        if Gaussian :
            v1_sample = self.sample_gaussian(v1_sample)

        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]
    def gibbs_vhv(self, v0_sample, Gaussian=False):
        h0_mean, h0_sample = self.sample_h_given_v(v0_sample)

        if Gaussian :
            v1_sample = self.sample_gaussian(h0_sample)

        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)

        return [h0_mean, h0_sample,
                v1_mean, v1_sample]

    '''获得重建交叉熵'''
    def get_reconstruction_cross_entropy(self,input, MSE=True):
        '''

        '''
        '''
        重构误差RE = sum(X-aVIS); X表示mini-batch矩阵，aVis为当前预测的可视单元状态值。
        '''
        # print(numpy.sum(  (numpy.array(self.input) - numpy.array(self.vbias).ravel() )**2 ))
        if MSE:
            '''有的是求的开平方根'''
            '''计算误差'''
            hidden = self.activation(numpy.dot(input, self.W)  + self.hbias)
            visible = self.activation(numpy.dot(hidden, self.W.T) + self.vbias)
            err = input - visible
            cross_entropy = numpy.mean(  err**2 )
        else:
            pre_sigmoid_activation_h = numpy.dot(input, self.W) + self.hbias
            sigmoid_activation_h = self.activation(pre_sigmoid_activation_h)

            pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
            sigmoid_activation_v = self.activation(pre_sigmoid_activation_v)

            cross_entropy =  - numpy.mean(
                numpy.sum(input * numpy.log(sigmoid_activation_v) +
                (1 - input) * numpy.log(1 - sigmoid_activation_v),
                          axis=1))
        
        return cross_entropy

    '''计算训练函数'''
    def train(self, input,lr=0.1, k=1, epochs=1000, batch_size=None, residual_error=None, gaus=False, show=False, show_epochs=None):
        '''计算有多少条数据'''

        '''参数检验'''
        if show and (show_epochs is None or show_epochs>epochs):
            print("show 为真时，必须设置show_epochs，具小于epochs。")
            assert False


        n_data = len(input)

        '''开始时间'''
        cur_start_time = time.time()

        '''计算训练批大小'''
        n_batches, batch_size = getNBatch(n_data, batch_size)

        '''总的errs列表'''
        #重构误差
        self.re_errs=[]
        #训练误差
        self.train_errs=[]
        #运行时间
        self.use_time=[]
        for epoch in range(epochs):
            '''初始化当前批次的errs '''
            cur_epoch_errs = numpy.zeros((n_batches,))
            cur_epoch_errs_ptr = 0
            '''对当前批数据进行训练'''
            for b in  range(n_batches):
                batch_x = input[b * batch_size:(b + 1) * batch_size]
                self.CD_K(lr=lr, K=k, input=batch_x, Gaussian=gaus)
                # self.contrastive_divergence(lr=lr, k=k, input=batch_x, Gaussian=gaus)

                if show_epochs is not None and epoch % show_epochs == 0:
                    batch_err = self.get_reconstruction_cross_entropy(batch_x)
                    cur_epoch_errs[cur_epoch_errs_ptr] = batch_err
                    cur_epoch_errs_ptr += 1

            if show_epochs is not None:
                if epoch%show_epochs==0:
                    '''保存当前层的errs'''
                    batch_err = self.get_reconstruction_cross_entropy(input)
                    self.re_errs = numpy.hstack([self.re_errs, batch_err])
                    self.use_time = numpy.hstack([self.use_time, time.time()-cur_start_time])
                    self.train_errs = numpy.hstack([self.train_errs, cur_epoch_errs.mean()])
                '''显示内容'''
                if show and epoch%show_epochs==0:
                    print("epoch[%d]train err:" % (epoch), self.train_errs[-1])

            if residual_error is not None and abs(cur_epoch_errs.mean()) < residual_error:
                print("训练误差ＯＫ！")
                return self.re_errs
        return self.re_errs

    ''' 重构函数 区别于预测函数'''
    def reconstruct(self, v):
        h = self.activation(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = self.activation(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v
#c以下函数不一定会用到，使用ＢＰ反向微调ＲＢＭ时使用，没有把参数直给ＢＰ效果好#####################################################################
    '''前向传播函数，预测函数'''
    def forward(self, v):
        return self.propup(v)

    '''反向传播ＤＢＮ使用'''
    def backward(self, inputs, bp_err, W,lr=0.1,  dropout=False, mask=None, impulse=None):
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

        # bp_err = self.activation(prev_layer.x, grad=True) * numpy.dot(prev_layer.bp_err, prev_layer.W.T)
        if numpy.array(inputs).ndim == 1:
            inputs = inputs[numpy.newaxis]
        if W is not None:
            bp_err = self.activation(inputs, grad=True) * numpy.dot(bp_err, W.T)
        else:
            bp_err = self.activation(inputs, grad=True) * numpy.dot(bp_err, self.W.T)

        if dropout == True:
            assert mask is not None
            bp_err *= mask

        # 添加冲量,动量因子
        if self.input.ndim == 1:
            self.input = self.input[numpy.newaxis]

        if impulse is not None:
            self.pre_impulse_W = lr * numpy.dot(self.input.T, bp_err) + impulse * self.pre_impulse_W
        else:
            self.pre_impulse_W = lr * numpy.dot(self.input.T, bp_err)

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

    '''误差函数'''
    def get_errs(self, v):
        '''单独计算重构误差函数'''
        h_sample = self.sample_h_given_v(v)
        v_sample = self.sample_v_given_h(h_sample)
        err = v - v_sample
        err_mean = numpy.mean(err * err)
        return err_mean


def Test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = numpy.array([
                        [1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 1, 0]
                    ])


    rng = numpy.random.RandomState(123)

    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=3, rng=rng)

    # train
    for epoch in range(training_epochs):
        # rbm.contrastive_divergence(lr=learning_rate, k=k)
        # rbm.CD(lr=learning_rate, T=k)
        rbm.CD(lr=learning_rate, T=k, Gaussian=True)
        cost = rbm.get_reconstruction_cross_entropy()
        # print ('Training epoch %d, cost is ' % epoch, cost)


    # test
    v = numpy.array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
    ])
    res =rbm.reconstruct(v)
    res = numpy.array(res)
    print (res)

def Test_rbmMNIST():
    import sys
    sys.path.append(".")
    import input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    trX = trX[:5000]
    trY = trY[:5000]
    teX = teX[:1000]
    teY = teY[:1000]


    rbm = RBM( n_visible=784, n_hidden=500)
    # train
    # errs = rbm.train(input=mnist_images,lr=1.0,epochs=1, batch_size=100, show=True)
    errs = rbm.train(input=trX,lr=0.1,epochs=100, batch_size=100, show=True,gaus=True, show_epochs=10)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False



    print("重构误差" ,rbm.re_errs[-1])
    plt.plot(rbm.re_errs,"-o")
    plt.title(u"重构误差")
    plt.savefig("重构误差.jpg")

    # plt.show()
    plt.plot(rbm.train_errs,"-*")
    plt.title(u"训练误差")
    plt.savefig("训练误差.jpg")
    # plt.show()
    plt.plot(rbm.use_time,"r-")
    plt.title(u"运行时间")
    plt.savefig("运行时间.jpg")
    # plt.show()
def aaa1():
    import numpy as np
    import matplotlib.pyplot as plt

    #画线
    x1, y1 = 0, 30
    x2, y2 = 100, 200
    x3, y3 = 200, -10

    x1, y1 = 0, 0.5
    x2, y2 = 0, 1
    x3, y3 = 0.75, 1


    # x1, y1 = 0.5, 0.1
    # x2, y2 = 0.8, 0.5
    # x3, y3 = 1, 0
    #个数
    sample_size = 200


    #画图边线
    # theta = np.arange(0, 1, 0.001)
    # x = theta * x1 + (1 - theta) * x2
    # y = theta * y1 + (1 - theta) * y2
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x1 + (1 - theta) * x3
    # y = theta * y1 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x2 + (1 - theta) * x3
    # y = theta * y2 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)

    #生成点
    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3

    return x, y
    #画点
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(x, y, 'ro')
    plt.grid(True)
    # plt.savefig('demo.png')
    plt.show()

def aaa2():
    import numpy as np
    import matplotlib.pyplot as plt

    #画线
    x1, y1 = 0, 30
    x2, y2 = 100, 200
    x3, y3 = 200, -10

    x1, y1 = 0, 0.5
    x2, y2 = 0, 1
    x3, y3 = 0.75, 1


    x1, y1 = 0.5, 0.1
    x2, y2 = 0.8, 0.5
    x3, y3 = 1, 0
    #个数
    sample_size = 200


    #画图边线
    # theta = np.arange(0, 1, 0.001)
    # x = theta * x1 + (1 - theta) * x2
    # y = theta * y1 + (1 - theta) * y2
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x1 + (1 - theta) * x3
    # y = theta * y1 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)
    # x = theta * x2 + (1 - theta) * x3
    # y = theta * y2 + (1 - theta) * y3
    # plt.plot(x, y, 'g--', linewidth=2)

    #生成点
    rnd1 = np.random.random(size=sample_size)
    rnd2 = np.random.random(size=sample_size)
    rnd2 = np.sqrt(rnd2)
    x = rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3
    y = rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3

    return x,y
    #画点
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(x, y, 'ro')
    plt.grid(True)
    # plt.savefig('demo.png')
    plt.show()

def Test_crbm():
    import numpy as np
    import matplotlib.pyplot as plt

    x,y = aaa1()
    x1,y1=aaa2()

    x = np.array([x, x1]).ravel()
    y= np.array([y, y1]).ravel()
    # x=x+x1
    # y=y+y1

    #显示画点
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.plot(x, y, 'ro')
    # plt.grid(True)
    # # plt.savefig('demo.png')
    # plt.show()

    data=[]
    data.append(x)
    data.append(y)
    data = np.array(data).T

    # print(data.shape, data)
    # plt.plot(data[:,0],data[:,1], 'ro')
    # plt.show()

    rbm = RBM(n_visible=2, n_hidden=4)
    rbm.train(data, lr=0.0000005, epochs=4000,batch_size=500)
    rData = rbm.reconstruct(data)

    # print(rData.shape, rData)

    print(numpy.max(rData[:,0]), numpy.min(rData[:,0]))
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.plot(rData[:,0],rData[:,1], 'ro')
    plt.show()
    #
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.plot(rData[0], rData[1], 'ro')
    # plt.grid(True)
    # # plt.savefig('demo.png')
    # plt.show()


if __name__ == "__main__":
    # Test_rbm(learning_rate=0.01)
    Test_rbmMNIST()
    # Test_crbm()