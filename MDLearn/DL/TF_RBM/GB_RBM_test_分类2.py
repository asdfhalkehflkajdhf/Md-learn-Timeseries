import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt
from tfrbm import BBRBM, GBRBM
#from tensorflow.examples.tutorials.mnist import input_data
import input_data

def Test_gb_rbm():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_images = mnist.train.images

    gbrbm = GBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True, sample_visible=True)
    errs = gbrbm.fit(mnist_images, n_epoches=30, batch_size=10)
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

def Test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = np.array([
                        [1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 1, 0]
                    ])


    rng = np.random.RandomState(123)

    # construct RBM
    # rbm = GBRBM(input=data, n_visible=6, n_hidden=3, momentum=0.95, use_tqdm=True, sample_visible=True)
    gbrbm = GBRBM(n_visible=6, n_hidden=3, learning_rate=0.01, momentum=0.95, use_tqdm=True, sample_visible=True)
    errs = gbrbm.fit(data, n_epoches=1000, batch_size=10)
    plt.plot(errs)



    # test
    v = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
    ])
    res = gbrbm.reconstruct(v)

    res = np.array(res)
    print (res)



if __name__ == "__main__":
    Test_rbm()