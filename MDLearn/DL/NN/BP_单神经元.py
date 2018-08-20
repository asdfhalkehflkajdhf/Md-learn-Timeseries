# -*- coding: utf-8 -*-
# http://blog.csdn.net/net_wolf_007/article/details/52055718
'''
有一个问题，当输入全为０时，没法训练，因为这个没有设置偏置
训练时只能一个一个计算
'''
import numpy as np

# 使用sigmoid函数作为激活函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


# 神经元
class Neuron:
    def __init__(self, len_input):
        # 输入的初始参数, 随机取很小的值(<0.1)
        self.weights = np.random.random(len_input) * 0.1
        # 当前实例的输入
        self.input = np.ones(len_input)
        # 对下一层的输出值
        self.output = 1
        # 误差项
        self.deltas_item = 0
        # 上一次权重增加的量，记录起来方便后面扩展时可考虑增加冲量
        self.last_weight_add = 0

    def calc_output(self, x):
        # 计算输出值
        self.input = x
        self.output = logistic(np.dot(self.weights.T, self.input))
        return self.output

    def get_back_weight(self):
        # 获取反馈差值
        return self.weights * self.deltas_item

    def update_weight(self, target=0, back_weight=0, learning_rate=0.1, layer="OUTPUT"):
        # 更新权传
        if layer == "OUTPUT":
            self.deltas_item = (target - self.output) * logistic_derivative(self.output)
        elif layer == "HIDDEN":
            self.deltas_item = back_weight * logistic_derivative(self.output)

        weight_add = self.input * self.deltas_item * learning_rate + 0.9 * self.last_weight_add  # 添加冲量
        self.weights += weight_add
        self.last_weight_add = weight_add


# 网络层
class NetLayer:
    '''''
    网络层封装
    管理当前网络层的神经元列表
    '''

    def __init__(self, len_node, in_count):
        '''''
        :param len_node: 当前层的神经元数
        :param in_count: 当前层的输入数
        '''
        # 当前层的神经元列表
        self.neurons = [Neuron(in_count) for _ in range(len_node)]
        # 记录下一层的引用，方便递归操作
        self.next_layer = None

    def calc_output(self, x):
        output = np.array([node.calc_output(x) for node in self.neurons])
        if self.next_layer is not None:
            return self.next_layer.calc_output(output)
        return output

    def get_back_weight(self):
        return sum([node.get_back_weight() for node in self.neurons])

    def update_weight(self, learning_rate, target):
        '''''
        更新当前网络层及之后层次的权重
        使用了递归来操作，所以要求外面调用时必须从网络层的第一层（输入层的下一层）来调用
        :param learning_rate: 学习率
        :param target: 输出值
        '''
        layer = "OUTPUT"
        back_weight = np.zeros(len(self.neurons))
        if self.next_layer is not None:
            back_weight = self.next_layer.update_weight(learning_rate, target)
            layer = "HIDDEN"
        for i, node in enumerate(self.neurons):
            target_item = 0 if len(target) <= i else target[i]
            node.update_weight(target=target_item, back_weight=back_weight[i], learning_rate=learning_rate, layer=layer)

        return self.get_back_weight()


class NeuralNetWork:
    def __init__(self, layers):
        self.layers = []
        self.construct_network(layers)
        pass

    def construct_network(self, layers):
        last_layer = None
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            cur_layer = NetLayer(layer, layers[i - 1])
            self.layers.append(cur_layer)
            if last_layer is not None:
                last_layer.next_layer = cur_layer
            last_layer = cur_layer

    def fit(self, x_train, y_train, learning_rate=0.1, epochs=100000, shuffle=False):
        '''''
        训练网络, 默认按顺序来训练
        方法 1：按训练数据顺序来训练
        方法 2: 随机选择测试
        :param x_train: 输入数据
        :param y_train: 输出数据
        :param learning_rate: 学习率
        :param epochs:权重更新次数
        :param shuffle:随机取数据训练
        '''
        n_test = 0
        indices = np.arange(len(x_train))
        for j in range(epochs):
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                self.layers[0].calc_output(x_train[i])
                self.layers[0].update_weight(learning_rate, y_train[i])


    def predict(self, x):
        return self.layers[0].calc_output(x)


def Test_bp():
    print("test neural network")

    data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])
    y_data = np.array([[1, 1],
                       [1, 1],
                       [1, 1],
                       [0, 1],
                       [1, 1],
                       [0, 1],
                       [0, 1],
                       [0, 1]])
    np.set_printoptions(precision=3, suppress=True)

    for item in range(10):
        network = NeuralNetWork([8, 4, 2])
        # 让输入数据与输出数据相等
        network.fit(data, y_data, learning_rate=0.1, epochs=1000)

        print("\n\n", item,  "result")
        for item in data:
            print(item, network.predict(item))

if __name__ == "__main__":
    # Test_dbn()
    Test_bp()
