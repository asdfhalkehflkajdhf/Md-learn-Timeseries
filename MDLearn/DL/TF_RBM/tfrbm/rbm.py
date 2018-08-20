from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from .util import tf_xavier_init


class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None):
        '''

        :param n_visible:
        :param n_hidden:
        :param learning_rate:
        :param momentum:
        :param xavier_const:
        :param err_function:
        :param use_tqdm:
        :param tqdm:

        初始化。
* ` n_visible ` -可见层神经元数目
* ` n_hidden `数隐层神经元
* ` use_tqdm `使用tqdm包进度指示或不
* ` err_function `误差函数（这是* *不*在训练过程中，只是在` get_err `功能），应` MSE `或`余弦`
只有` gbrbm `：
* ` sample_visible `样本数据重建的高斯分布（与重建的价值作为一种手段和`西格玛`参数偏差）或没有（如果没有，每一gaussoid将投影到一个单点）
* `西格玛` -标准偏差的输入数据
*建议*：
*使用bbrbm伯努利分布数据。在这种情况下，××××输入值必须在间隔0到1 ` ` ` `。
*一般分布式数据` 0 `均值和标准差σ` `使用gbrbm。如果不是，就把它标准化。

        '''
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        ##实现进度条的
        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        '''tf.Variable：主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
                声明时，必须提供初始值；
                名称的真实含义，在于变量，也即在真实训练时，其值是会改变的，自然事先需要指定初始值； 
            tf.placeholder：用于得到传递进来的真实的训练样本：
                不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定；
                这也是其命名的原因所在，仅仅作为一种占位符；
        '''
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):
        assert n_epoches > 0

        n_data = data_x.shape[0]

        '''计算训练批大小'''
        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        '''是否进行随机训练'''
        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        '''总的errs列表'''
        errs = []

        '''开始迭代　n_epoches　次'''
        for e in range(n_epoches):
            '''是否进行随机训练'''
            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            '''进行格式化输出，第几次迭代'''
            if verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            '''进行格式化输出，第几次迭代'''
            r_batches = range(n_batches)
            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            '''初始化当前批次的errs '''
            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            '''对当前批数据进行训练'''
            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            '''进行格式化输出，第几次迭代'''
            if verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()
            '''把小批量errs以列的形式添加到总'''
            errs = np.hstack([errs, epoch_errs])

        return errs

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
