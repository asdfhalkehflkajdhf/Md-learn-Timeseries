import tensorflow as tf
import numpy as np
from PIL import Image
#import Image
import sys
import os
sys.path.append(".")
import input_data
from util import tile_raster_images

'''
sample 采样函数
prob一般是概率值,scale一般未是否标准化
'''
def sample_prob(probs):
    '''ReLu(Rectified Linear Units)激活函数'''
    '''
    tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None) 
    tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
    tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None) 
    这几个都是用于生成随机数tensor的。尺寸是shape 
    random_normal: 正太分布随机数，均值mean,标准差stddev 
    truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数 
    random_uniform:均匀分布随机数，范围为[minval,maxval]'''
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(tf.shape(probs))))

alpha = 1.0
batchsize = 100

#读取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

'''placeholder:占位符，同样是一个抽象的概念。用于表示输入输出数据的格式。告诉系统：这里有一个值/向量/矩阵'''
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

rbm_w = tf.placeholder("float", [784, 500])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [500])

'''进行采样'''
h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)

'''更新参数'''
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

'''计算误差'''
h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb))
err = X - v_sample
err_sum = tf.reduce_mean(err * err)

'''初始化 Session'''
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

n_w = np.zeros([784, 500], np.float32)
n_vb = np.zeros([784], np.float32)
n_hb = np.zeros([500], np.float32)
o_w = np.zeros([784, 500], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([500], np.float32)
'''我们都知道feed_dict的作用是给使用placeholder创建出来的tensor赋值
run 说明：https://www.2cto.com/kf/201610/559887.html
'''
print ("err_sum 0:",sess.run( err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb}) )

'''运行结果存放目录'''
RES_PATH="./rbm_test_res"
if os.path.isdir(RES_PATH) == False:
    os.mkdir(RES_PATH)

for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    batch = trX[start:end]
    n_w = sess.run(update_w, feed_dict={ X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    n_vb = sess.run(update_vb, feed_dict={ X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    n_hb = sess.run(update_hb, feed_dict={ X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
    o_w = n_w
    o_vb = n_vb
    o_hb = n_hb
    if start % 10000 == 0:
        print ("err_sum i:", sess.run( err_sum, feed_dict={X: trX, rbm_w: n_w, rbm_vb: n_vb, rbm_hb: n_hb}) )
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save(RES_PATH+"/rbm_%d.png" % (start / 10000))
