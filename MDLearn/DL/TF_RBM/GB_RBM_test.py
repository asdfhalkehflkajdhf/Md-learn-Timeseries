import numpy as np
#import pylab as pl
import sys
sys.path.append(".")

import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
#from tensorflow.examples.tutorials.mnist import input_data
import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

gbrbm = GBRBM(n_visible=784, n_hidden=500, learning_rate=1.0, momentum=0.95, use_tqdm=True, sample_visible=False)
errs = gbrbm.fit(mnist_images, n_epoches=1, batch_size=100)
plt.plot(errs)
plt.show()

#检查一些重建数据：
def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()
IMAGE = 1
image = mnist_images[IMAGE]
image_rec = gbrbm.reconstruct(image.reshape(1,-1))

show_digit(image)
show_digit(image_rec)