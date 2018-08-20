
import numpy as np
import MDLearn.utils.LoadData as LD
import MDLearn.DL.SdA as SDA
from sklearn.svm import SVR


data_src = LD.loadCsvData_Np("data/co2-ppm-mauna-loa-19651980.csv", skiprows=1, ColumnList=[1])


# Australia/US	British/US	Canadian/US	Dutch/US	French/US	German/US	Japanese/US	Swiss/US
data_t =data_src[:,0:1].copy()
# 数据还源时使用
t_mean = np.mean(data_t)
t_min = np.min(data_t)
t_max = np.max(data_t)

# 数据预处理
result,x_result,y_result = LD.dataRecombine_Single(data_t, 3)
# print(x_result, y_result)

result_len = len(result)
row = round(0.8 * result.shape[0])
row = result_len - 87
windowSize = row
windowSize = row


# 数据归一化
# data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
x_result = (x_result - t_min) / (t_max - t_min)
y_result = (y_result - t_min) / (t_max - t_min)
# x_result = (x_result - t_mean) / np.std(x_result)
# y_result = (y_result - t_mean) / np.std(x_result)
y_rbf_all = []
y_test_all = []
rng = np.random.RandomState(1233)

for y_i in range(row, result_len):
    if y_i < windowSize:
        continue
    x_train = x_result[y_i - windowSize:y_i]
    y_train = y_result[y_i - windowSize:y_i]
    x_test = x_result[y_i:y_i + 1]
    y_test = y_result[y_i:y_i + 1]

    # y_train = y_train[np.newaxis].T

    svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1)
    y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
    # sda = SDA.SdA(input=x_train, label=y_train, n_ins=seq_len, hidden_layer_sizes=[8], n_outs=1)
    # sda.pretrain(lr=0.5, epochs=1000)
    # sda.finetune(lr=0.5, epochs=100)

    y_rbf = svr_rbf.predict(x_test)
    y_rbf_all.append(y_rbf)
    y_test_all.append(y_test)

# print(np.array(y_rbf_all))#, np.array(y_test_all))

y_rbf_all = np.array(y_rbf_all).ravel()
y_test_all = np.array(y_test_all).ravel()
import MDLearn.utils.EvalueationIndex as EI
ei = EI.evalueationIndex(y_rbf_all, y_test_all)
ei.show()

import MDLearn.utils.Draw as draw
draw.plot_results_point(y_rbf_all, y_test_all)
# print(y_rbf_all)
# print(y_test_all)

