# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

def 多图同一张表():
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1,2.5,0.5,2), random_state=2)
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)



    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)


    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(10, 10), facecolor='w')
    plt.subplot(421)
    plt.title(u'原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)
    # plt.show()

    plt.subplot(422)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title(u'旋转后数据')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)
    plt.show()


def 圆():
    samples_num = 200
    t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
    print(t)
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0,samples_num,1)
    for i in i_set:
        len = np.sqrt(np.random.random())
        x[i] = x[i] * len
        y[i] = y[i] * len
    plt.figure(figsize=(10,10.1),dpi=125)
    plt.plot(x,y,'ro')
    # _t = np.arange(0,7,0.1)
    # _x = np.cos(_t)
    # _y = np.sin(_t)
    # plt.plot(_x,_y,'g-')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Random Scatter')
    plt.grid(True)
    # plt.savefig('imag.png')
    plt.show()

def aaa():
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

    print(x, y)
    #画点
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(x, y, 'ro')
    plt.grid(True)
    # plt.savefig('demo.png')
    plt.show()

def 采样():
    ###################################
    #   !/usr/bin/env python
    #   coding=utf-8
    #   __author__ = 'pipi'
    #   ctime 2014.10.11
    #   绘制椭圆和圆形
    ###################################
    from matplotlib.patches import Ellipse, Circle
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ell1 = Ellipse(xy=(0.25, 0.75), width=0.1, height=0.5, angle=-45.0, facecolor='yellow', alpha=0.4)
    cir1 = Circle(xy=(0.75, 0.25), radius=0.2, alpha=0.5)

    print(ell1)
    return

    ax.add_patch(ell1)
    ax.add_patch(cir1)

    x, y = 0.5, 0.5
    ax.plot(x, y, 'ro')

    plt.axis('scaled')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('equal')  # changes limits of x or y axis so that equal increments of x and y have the same length

    plt.show()
if __name__ == "__main__":
    # 采样()

    from os import path, access, R_OK  # W_OK for write permission.

    PATH = 'data/fco2-ppm-mauna-loa-19651980.csv'

    print(path.exists(PATH))
    PATH = 'data'
    print(path.exists(PATH))
    print(path.isfile(PATH))
    print(access(PATH, R_OK))
    # if path.exists(PATH) and  and access(PATH, R_OK):
    #     print (
    #     "File exists and is readable" )
    # else:
    #     print (
    #     "Either file is missing or is not readable" )