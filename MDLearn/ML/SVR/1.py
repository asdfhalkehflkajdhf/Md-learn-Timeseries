
import time
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def simple_sin_function_test_svr():

    ###############################################################################
    # Generate sample data

    global_start_time = time.time()

    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    ###############################################################################
    # Add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    ###############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    ###############################################################################
    # look at the results
    plt.scatter(X, y, c='k', label='data')
    plt.plot(X, y_rbf, c='g', label='RBF model')
    plt.plot(X, y_lin, c='r', label='Linear model')
    plt.plot(X, y_poly, c='b', label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
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
#测试数据
def sp500_svr_simple_test():
    import time

    global_start_time = time.time()

    print('> Loading data... ')
    seq_len = 2

    X_train, y_train, X_test, y_test = load_data('SN_m_tot_V2.0_1990.1-2017.8.csv', seq_len, True)

    print('> Data Loaded. Compiling...')
    ###############################################################################
    # Fit regression model
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf = SVR(kernel='rbf', C=1e3)
    svr_rbf.fit(X_train, y_train)
    y_rbf = svr_rbf.predict(X_test)

    print('Training duration (s) : ', time.time() - global_start_time)

    ###############################################################################
    # look at the results
    plot_results_point(y_rbf, y_test)
    eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)

    print("MSE:", eI.MSE)
    print("RMSE:", eI.RMSE)

def sp500_svr_test():
    import time

    global_start_time = time.time()

    print('> Loading data... ')
    seq_len = 5

    print("setp,C,gamma,MSE,RMSE")
    for i in range(2, 6):
        X_train, y_train, X_test, y_test = load_data('SN_m_tot_V2.0_1990.1-2017.8.csv', i, True)
        for j in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            for k in [2, 3, 4, 5]:
                svr_rbf = SVR(kernel='poly', C=j)
                svr_rbf.fit(X_train, y_train)
                y_rbf = svr_rbf.predict(X_test)
                eI = EvaluationIndex.evalueationIndex(y_rbf, y_test)

                # print("setp=", i, "\tC:",j, "\tdegree:",k, "\tMSE:", eI.MSE, "\tRMSE:", eI.RMSE)
                print(i, ",",j, ",",k, ",", eI.MSE, ",", eI.RMSE)

    return

#Main Run Thread
if __name__=='__main__':
    # sp500_svr_test()
    sp500_svr_simple_test()