import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
#from tensorflow.examples.tutorials.mnist import input_data
import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

bbrbm = BBRBM(n_visible=784, n_hidden=500, learning_rate=1.0, momentum=0.0, use_tqdm=True)
#bbrbm = BBRBM(n_visible=64, n_hidden=32, learning_rate=0.01, momentum=0.95, use_tqdm=True)

errs = bbrbm.fit(mnist_images, n_epoches=1, batch_size=100)
plt.plot(errs)
plt.show()