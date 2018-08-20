
import time
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import EvaluationIndex

##############################################################################
#画图
def plot_results_point(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_full(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

##############################################################################
#加载数据
# 标准化
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [( (p-min(window)) / (max(window)-min(window))) for p in window]
        # normalised_window = [p /  window[0] - 1 for p in window]
        # print(normalised_window)
        normalised_data.append(normalised_window)
    return normalised_data
def load_data(filename, seq_len, normalise_window_bool):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    #字符串转数字
    for index in range(len(data)):
        t = float(data[index])
        data[index]=t

    # plt.plot(data, label='data')
    # plt.legend()
    # plt.show()
    sequence_length = seq_len + 1

    result = []

    #对数据进行分块，块大小为seq_len
    for index in range(len(data) - sequence_length):


        result.append((data[index: index + sequence_length]))

    #标准化数据
    if normalise_window_bool:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]
#############################################################
#多步测试
from numpy import newaxis

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    # predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        # predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        predicted.append(model.predict(curr_frame[newaxis]))
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            # predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            predicted.append(model.predict(curr_frame[newaxis]))
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_sequence_full2(model, data, times):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(times):
        print(curr_frame)
        # predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        predicted.append(model.predict(curr_frame[newaxis]))
        curr_frame = curr_frame[1:]
        '''
        insert使用方法：https://docs.scipy.org/doc/numpy/reference/generated/numpy.insert.html
        '''
        curr_frame = np.insert(curr_frame, len(curr_frame), predicted[-1], axis=0)

    return predicted

##############################################################################
#测试数据
def sp500_svr_simple_test_所有一起归一化():
    import time
    global_start_time = time.time()
    print('> Loading data... ')

    data_src = EvaluationIndex.loadCsvData_Np("GDP_1981-2016.csv")
    # plt.plot(data_src, label='data')
    # plt.legend()
    # plt.show()

    # 数据还源时使用
    t_mean=np.mean(data_src)
    t_min=np.min(data_src)
    t_max=np.max(data_src)


    #数据预处理
    seq_len = 4
    sequence_length = seq_len + 1
    result = []
    #对数据进行分块，块大小为seq_len
    for index in range(len(data_src) - sequence_length):
        result.append(np.array(data_src[index: index + sequence_length]).ravel())
    # print(result)
    result = np.array(result)

    print(result)
    return

    row = round(0.9 * result.shape[0])
    row = 22
    train = result[:int(row), :]
    # np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    print("train data")
    print(x_train, y_train)
    print("train data")
    print(x_test, y_test)

    #数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_train_normal = (x_train - t_min) / (t_max - t_min)
    y_train_normal = (y_train - t_min) / (t_max - t_min)
    x_test_normal = (x_test - t_min) / (t_max - t_min)
    y_test_normal = (y_test - t_min) / (t_max - t_min)
    # plt.plot(y_train_normal, label='data')
    # plt.legend()
    # plt.show()
    #源测试数据
    y_train_normal =  y_train_normal.ravel()
    y_test_normal = y_test_normal.ravel()
    print('> Data Loaded. Compiling...')
    ###############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.05)

    print("所有数据归一化，结果不还原")
    svr_rbf.fit(x_train_normal, y_train_normal)

    # y_rbf = svr_rbf.predict(x_test_normal)
    y_rbf = predict_point_by_point(svr_rbf, x_test_normal)

    eI = EvaluationIndex.evalueationIndex(y_rbf*2, y_test_normal)
    eI.show()
    print(y_rbf, y_test_normal)
    plot_results_point(y_rbf*2, y_test_normal)

    # #还原
    # print("所有数据归一化，结果还原对比")
    # t = (t_max-t_min)
    # t = np.array(y_rbf)*t
    # y_rbf_back = t + t_mean
    # # plot_results_point(y_rbf_back, y_test)
    # # plot_results_full(y_rbf_back, y_test)
    # # plot_results_multiple(y_rbf_back, y_test,seq_len)
    # eI = EvaluationIndex.evalueationIndex(y_rbf_back.ravel(), y_test)
    # eI.show()

    # #测试数据为归一化，lable为未归一化
    # print("测试数据为归一化，lable为未归一化")
    # svr_rbf.fit(x_train_normal, y_train)
    # y_rbf = predict_point_by_point(svr_rbf, x_test_normal)
    # ###############################################################################
    # eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)
    # eI.show()
    #
    # #测试数据为归一化，lable为未归一化
    # print("测试数据，lable为未归一化")
    # svr_rbf.fit(x_train, y_train)
    # y_rbf = predict_point_by_point(svr_rbf, x_test)
    # ###############################################################################
    # eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)
    # eI.show()

def sp500_svr_simple_test_所有一起归一化2():
    '''
    单值预测
    :return:
    '''
    import time
    global_start_time = time.time()
    print('> Loading data... ')

    data_src = EvaluationIndex.loadCsvData_Np("GDP_1981-2016.csv")
    # plt.plot(data_src, label='data')
    # plt.legend()
    # plt.show()

    # 数据还源时使用
    t_mean=np.mean(data_src)
    t_min=np.min(data_src)
    t_max=np.max(data_src)


    #数据预处理
    seq_len = 4
    sequence_length = seq_len + 1
    result = []
    #对数据进行分块，块大小为seq_len
    for index in range(len(data_src) - sequence_length):
        result.append(np.array(data_src[index: index + sequence_length]).ravel())
    # print(result)
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    row = 22
    result_len = len(result)
    # np.random.shuffle(train)
    x_result = result[:, :-1]
    y_result = result[:, -1]

    #数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_result = (x_result - t_min) / (t_max - t_min)
    y_result = (y_result - t_min) / (t_max - t_min)
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)

    y_rbf_all = []
    y_test_all = []
    for y_i in range(row, result_len):
        x_train = x_result[y_i-row:y_i]
        y_train = y_result[y_i-row:y_i]
        x_test = x_result[y_i:y_i+1]
        y_test = y_result[y_i:y_i+1]

        y_train =  y_train.ravel()
        y_test = y_test.ravel()
        svr_rbf = SVR(kernel='rbf', C=10000000, gamma=0.1)

        # print("所有数据归一化，结果不还原")
        svr_rbf.fit(x_train, y_train)
        y_rbf = predict_point_by_point(svr_rbf, x_test)
        y_rbf_all.append(y_rbf)
        y_test_all.append(y_test)

    y_rbf_all = np.array(y_rbf_all)
    y_test_all = np.array(y_test_all)

    print(y_rbf_all)
    print(y_test_all)

    eI = EvaluationIndex.evalueationIndex(y_rbf_all, y_test_all)
    eI.show()
    plot_results_point(y_rbf_all, y_test_all)

    return


def sp500_svr_simple_test_每个维归一化():
    import time
    global_start_time = time.time()
    print('> Loading data... ')
    seq_len = 5

    data_src = EvaluationIndex.loadCsvData_Np("SN_m_tot_V2.0_1990.1-2017.8.csv")
    # plt.plot(data_src, label='data')
    # plt.legend()
    # plt.show()

    # 数据还源时使用
    t_mean=np.mean(data_src)
    t_min=np.min(data_src)
    t_max=np.max(data_src)

    #数据预处理
    sequence_length = seq_len + 1
    result = []
    #对数据进行分块，块大小为seq_len
    for index in range(len(data_src) - sequence_length):
        result.append(np.array(data_src[index: index + sequence_length]).ravel())
    # print(result)
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    # np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]


    #数据归一化
    data_normalization = EvaluationIndex.归一化()
    normalised_data = []
    for window in result:
        normalised_window = data_normalization.normalization_max_min_负1_1(window)
        # normalised_window = [p /  window[0] - 1 for p in window]
        # print(normalised_window)
        normalised_data.append(normalised_window.ravel())

    # print(np.array(normalised_data))
    # print(normalised_data)
    normalised_data = np.array(normalised_data)
    row = round(0.9 * normalised_data.shape[0])
    train = normalised_data[:int(row), :]
    # np.random.shuffle(train)
    x_train_normal = train[:, :-1]
    y_train_normal = train[:, -1]
    x_test_normal = normalised_data[int(row):, :-1]
    y_test_normal = normalised_data[int(row):, -1]

    # plt.plot(y_train_normal, label='data')
    # plt.legend()
    # plt.show()
    #源测试数据
    y_train_normal =  y_train_normal.ravel()
    y_test_normal = y_test_normal.ravel()
    print('> Data Loaded. Compiling...')
    ###############################################################################
    # Fit regression model
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=5)
    svr_rbf.fit(x_train_normal, y_train_normal)
    # y_rbf = svr_rbf.predict(x_test_normal)
    y_rbf = predict_point_by_point(svr_rbf, x_test_normal)

    # y_rbf = predict_sequence_full(svr_rbf, x_test_normal, 5) #最大为5

    # y_rbf = predict_sequences_multiple(svr_rbf, x_test_normal, 5, seq_len) #最大为5
    print('Training duration (s) : ', time.time() - global_start_time)
    ###############################################################################
    # look at the results
    # plot_results_point(y_rbf, y_test_normal)
    # plot_results_full(y_rbf, y_test_normal)
    # plot_results_multiple(y_rbf, y_test_normal, seq_len)
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test_normal)
    eI.show()
    #还原
    print("所有数据归一化，结果还原对比")
    t = (t_max-t_min)
    t = np.array(y_rbf)*t
    y_rbf_back = t + t_mean
    # plot_results_point(y_rbf_back, y_test)
    # plot_results_full(y_rbf_back, y_test)
    # plot_results_multiple(y_rbf_back, y_test,seq_len)
    eI = EvaluationIndex.evalueationIndex(y_rbf_back.ravel(), y_test)
    eI.show()

    #测试数据为归一化，lable为未归一化
    print("测试数据为归一化，lable为未归一化")
    svr_rbf.fit(x_train_normal, y_train)
    y_rbf = predict_point_by_point(svr_rbf, x_test_normal)
    ###############################################################################
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)
    eI.show()


def sp500_svr_多次参数测试():

    f = open("temp-test-print-out.txt", 'w')

    print('> Loading data... ')
    data_src = EvaluationIndex.loadCsvData_Np("GDP_1981-2016.csv")
    # plt.plot(data_src, label='data')
    # plt.legend()
    # plt.show()

    # 数据还源时使用
    t_mean=np.mean(data_src)
    t_min=np.min(data_src)
    t_max=np.max(data_src)

    #数据预处理

    seq_len = 5
    for seq_len in range(2, 6):
        sequence_length = seq_len + 1
        result = []
        #对数据进行分块，块大小为seq_len
        for index in range(len(data_src) - sequence_length):
            result.append(np.array(data_src[index: index + sequence_length]).ravel())
        # print(result)
        result = np.array(result)

        row = round(0.9 * result.shape[0])
        row = 22
        train = result[:int(row), :]
        # np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]


        #数据归一化
        # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
        x_train_normal = (x_train - t_min) / (t_max - t_min)
        y_train_normal = (y_train - t_min) / (t_max - t_min)
        x_test_normal = (x_test - t_min) / (t_max - t_min)
        y_test_normal = (y_test - t_min) / (t_max - t_min)
        # plt.plot(y_train_normal, label='data')
        # plt.legend()
        # plt.show()
        #源测试数据
        y_train_normal =  y_train_normal.ravel()
        y_test_normal = y_test_normal.ravel()
        print('> Data Loaded. Compiling...')
    ###############################################################################
        for j in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            # for k in [2, 3, 4, 5]:
            #     svr_rbf = SVR(kernel='poly', C=j, degree=k)
            #     svr_rbf = SVR(kernel='rbf', C=j)
                svr_rbf = SVR(kernel='sigmoid', C=j)

                svr_rbf.fit(x_train_normal, y_train_normal)
                y_rbf = svr_rbf.predict(x_test_normal)
                eI = EvaluationIndex.evalueationIndex(y_rbf, y_test_normal)

                # print("setp=", i, "\tC:",j, "\tdegree:",k, "\tMSE:", eI.MSE, "\tRMSE:", eI.RMSE)
                print(seq_len, ",", j, ",", 1, ",", eI.MSE, ",", eI.RMSE, file=f)

    f.close()
    return

#Main Run Thread
if __name__=='__main__':
    # sp500_svr_多次参数测试()
    sp500_svr_simple_test_所有一起归一化2()
    # sp500_svr_simple_test_每个维归一化()