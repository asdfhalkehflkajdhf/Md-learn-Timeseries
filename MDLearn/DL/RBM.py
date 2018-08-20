# -*- coding: utf-8 -*-

import sys
import numpy
from MDLearn.DL.utils import *

class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, rng=None):
        
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

        #外部使用
        self.b = self.hbias
        self.errs=[]

        #更新梯度
        self.__delta_w=0
        self.__delta_vb=0
        self.__delta_hb=0

    def get_delta_fun(self,x_old, x_new, impulses=0, lr=1.0):
        return impulses * x_old + lr * x_new
        return impulses * x_old + lr * x_new * (1 - impulses) / numpy.shape(x_new)[0]



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

    '''高斯采样'''
    def sample_gaussian(self, x, stddev=1):
        return x + self.rng.normal(0,stddev,self.input.shape)

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


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)
        
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)


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

    '''
    获得重建交叉熵
    '''
    def get_reconstruction_cross_entropy(self, MSE=True):
        '''

        '''
        '''
        重构误差RE = sum(X-aVIS); X表示mini-batch矩阵，aVis为当前预测的可视单元状态值。
        '''
        # print(numpy.sum(  (numpy.array(self.input) - numpy.array(self.vbias).ravel() )**2 ))
        if MSE:
            '''有的是求的开平方根'''
            '''计算误差'''
            hidden = sigmoid(numpy.dot(self.input, self.W)  + self.hbias)
            visible = sigmoid(numpy.dot(hidden, self.W.T) + self.vbias)
            err = self.input - visible
            cross_entropy = numpy.mean(  err**2 )
        else:
            pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
            sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

            pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
            sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

            cross_entropy =  - numpy.mean(
                numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
                (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                          axis=1))
        
        return cross_entropy

    def train(self, input=None,lr=0.1, k=1,  epochs=1000, batch_size=10, residual_error=1e-3, gaus=False, show=False):
        '''计算有多少条数据'''
        n_data = len(input)

        '''计算训练批大小'''
        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1
        r_batches = range(n_batches)

        '''总的errs列表'''
        errs=[]
        for epoch in range(epochs):
            '''初始化当前批次的errs '''
            epoch_errs = numpy.zeros((n_batches,))
            epoch_errs_ptr = 0
            '''对当前批数据进行训练'''
            for b in r_batches:
                batch_x = input[b * batch_size:(b + 1) * batch_size]
                self.CD_K(lr=lr, K=k, input=batch_x, Gaussian=gaus)
                # self.contrastive_divergence(lr=lr, k=k, input=batch_x, Gaussian=gaus)
                batch_err = self.get_reconstruction_cross_entropy()
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

                if abs(batch_err) < residual_error:
                    break
            if show:
                print("train err:", epoch_errs.mean())
            errs = numpy.hstack([errs, epoch_errs])
        '''保存当前层的errs'''
        self.errs = errs
        return errs
    '''
    重构
    '''
    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

    def forward(self, v):
        return self.propup(v)



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



if __name__ == "__main__":
    Test_rbm(learning_rate=0.01)
