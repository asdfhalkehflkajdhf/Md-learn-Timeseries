# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

#数据加载
def loadCsvData_Np(fileName, delimiter=',',dtype='float', skiprows=0, ColumnList=None, unpack=False, ndmin=2):
    '''

    :param fileName:
    :param delimiter:
    :param dtype:
    :param skiprows:
    :param ColumnList:
    :param unpack:
    :param ndmin:
    :return:
    '''
    # 使用numpy包
    '''
    注意：一次只能取一同一种数据
    dtype：data-type，可选
    所得数组的数据类型; 默认值：float。
    delimiter : 指定分隔符。
    skiprows：int，可选
    跳过Ｎ行 ; 默认值：0。
    usecols：序列，可选
    要读哪些列，0是第一个。例如， usecols = （1,4,5）将提取第2列，第5列和第6列。默认值“无”将导致所有列都被读取。
    unpack：bool，可选
    如果为True，则返回的数组被转置，因此可以使用x， y， z = loadtxt（...）解压缩参数。当与结构化数据类型一起使用时，将为每个字段返回数组。默认值为False。
    ndmin：int，可选
    返回的数组将具有至少ndmin维度。否则将会挤压一维轴。合法值：0（默认），1或2。
    '''
    from os import path, access, R_OK  # W_OK for write permission.
    if path.isfile(fileName) == False or  access(fileName, R_OK)==False :
        print("文件：[",fileName,"]不存在")
        assert False

    return np.loadtxt(fileName, delimiter=delimiter, dtype=dtype,
                        skiprows=skiprows, usecols=ColumnList,
                        unpack=unpack, ndmin=ndmin)

    # df = lines[1:,:4].astype('float')




    # 使用pandas包
    '''

    sep : str, default ‘,’
    指定分隔符。如果不指定参数，则会尝试使用逗号分隔。
    header : int or list of ints, default ‘infer’
    指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。如果明确设定header=0 就会替换掉原来存在列名。
    usecols : array-like, default None
    返回一个数据子集，该列表中的值必须可以对应到文件中的位置（数字可以对应到指定的列）或者是字符传为文件中的列名。例如：usecols有效参数可能是 [0,1,2]或者是 [‘foo’, ‘bar’, ‘baz’]。使用这个参数可以加快加载速度并降低内存消耗。
    dtype : Type name or dict of column -> type, default None
    每列数据的数据类型。例如 {‘a’: np.float64, ‘b’: np.int32}
    skiprows : list-like or integer, default None
    需要忽略的行数（从文件开始处算起），或需要跳过的行号列表（从0开始）。
    '''
    #
    # df = pd.read_csv('Azhisu_test_data.csv', sep=',', usecols=[2, 4], skiprows=1)
    # # df=df.ix[:,:4]
    # print(df, df.shape)

    return

#数据分割
def dataSplit(x_tarin,y_lable=None,test_size=0.3, random_state=0):
    '''
    train_data：所要划分的样本特征集
    train_target：所要划分的样本结果
    test_size：样本占比，如果是整数的话就是样本的数量
    random_state：是随机数的种子。

    :param x_tarin:
    :param y_lable:
    :param test_size:
    :param random_state:
    :return:
    '''

    '''
    # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)  
    train_X,test_X, train_y, test_y = train_test_split(train,  
                                                       target,  
                                                       test_size = 0.2,  
                                                       random_state = 0) 
    '''
    if y_lable is None:
        x_len = len(x_tarin)
        assert test_size>0
        if test_size<1:
            x_tarin_len = int(x_len*(1-test_size))
        else:
            x_tarin_len = int(x_len - test_size)
        return x_tarin[:x_tarin_len], x_tarin[x_tarin_len:]
    return train_test_split(x_tarin,y_lable,test_size=test_size, random_state=random_state)

#对时间序列数据重组,只处理一列数据
def dataRecombine_Single(x_train, seq_len, delay=0):
    '''

    :param x_train:
    :param seq_len: # 数据预处理
    :param delay: #延迟步数
    :return:
    '''
    assert seq_len>0
    assert delay>-1
    sequence_length = seq_len + 1 +delay
    result = []
    ndata_len = len(x_train)
    assert ndata_len>sequence_length
    # 对数据进行分块，块大小为seq_len
    for index in range(ndata_len - sequence_length + 1):
        temp_res =  np.array(x_train[index: index + seq_len]).ravel()
        temp_res2 = ( np.array(x_train[index + sequence_length-1: index + sequence_length]).ravel())
        result.append(temp_res.tolist()+temp_res2.tolist())
    result = np.array(result)
    # print(result)
    return result,result[:,:-1], result[:,-1:]

class 归一化(object):
    def __init__(self,t_源值):
        self.x = np.array(t_源值)


    def normalization_max_min_0_1(self, x=None):
        if x is not None:
            y = (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            y=(self.x-np.min(self.x))/(np.max(self.x) - np.min(self.x))
        return y


    def normalization_max_min_负1_1(self, x=None):
        if x is not None:
            y = (x - np.mean(x)) / (np.max(x) - np.min(x))
        else:
            y=(self.x-np.mean(self.x))/(np.max(self.x) - np.min(self.x))
        return y

    def normalization_z_score(self, x=None):
        '''
        最常见的标准化方法就是Z标准化，也是SPSS中最为常用的标准化方法，spss默认的标准化方法就是z-score标准化。
也叫标准差标准化，这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
经过处理的数据符合标准正态分布，即均值为0，标准差为1，其转化函数为：
        :param x:
        :return:
        '''
        if x is not None:
            y = (x - np.mean(x)) /np.std(x)
        else:
            y=(self.x-np.mean(self.x))/np.std(self.x)
        return y

    def normalization_log(self, x=None):
        if x is not None:
            y = np.log10(x)
        else:
            y = np.log10(self.x)
        return y

    def normalization_sigmoid_0_1(self, x=None):
        if x is not None:
            y = 1 / (1 + np.exp(-x))
        else:
            y =1 / (1 + np.exp(-self.x))
        return y



def test_归一化():
    X = [2112.40,2215.50,2479.50,2628.50,2857.00,3144.40,3430.70,3841.50,4499.30,5260.10,4584.00,5268.80,5723.80,5820.70,6125.40,6333.90]
    yl = [2074.46,2314.11,2388.03,2697.11,2816.11,3076.24,3394.35,3691.20,4151.94,4896.27,5698.20,4624.40,5778.79,6207.55,6188.02,6581.81]
    a = 归一化(X)

    print(a.normalization_log())
    print(a.normalization_max_min_0_1())
    print(a.normalization_max_min_负1_1())
    print(a.normalization_sigmoid_0_1())
    print(a.normalization_z_score())

def test_dataRecombine_Single():
    X = [2112.40,2215.50,2479.50,2628.50,2857.00,3144.40,3430.70,3841.50,4499.30,5260.10,4584.00,5268.80,5723.80,5820.70,6125.40,6333.90]
    X = range(20)
    _,x, y =dataRecombine_Single(X, 4)
    print(x, y)
#Main Run Thread
if __name__=='__main__':
    # test_归一化()
    test_dataRecombine_Single()

    dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
    6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
    10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
    12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
    13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
    9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
    11999,9390,13481,14795,15845,15271,14686,11054,10395]
    print("fffffffffffffffffff")
    a ,b =dataSplit(dta, test_size=0.1)
    print("adsfasdf=",a, b)