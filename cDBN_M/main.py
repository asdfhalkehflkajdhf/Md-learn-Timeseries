
import numpy as np
import MDLearn.utils.LoadData as LD
from cDBN_M.DBN_SELF import DBN as DBN
from cDBN_M.BP_SELF import BP as BP
import MDLearn.utils.EvalueationIndex as EI

''''''
def DBN_Train(seq_len,win, lr):
    data_src = LD.loadCsvData_Np("../data/co2-ppm-mauna-loa-19651980.csv", skiprows=1, ColumnList=[1])

    # Australia/US	British/US	Canadian/US	Dutch/US	French/US	German/US	Japanese/US	Swiss/US
    data_t = data_src[:, 0:1].copy()
    # 数据还源时使用
    t_mean = np.mean(data_t)
    t_min = np.min(data_t)
    t_max = np.max(data_t)

    # 数据预处理
    result, x_result, y_result = LD.dataRecombine_Single(data_t,seq_len)
    # print(x_result, y_result)

    result_len = len(result)
    row = round(0.8 * result.shape[0])
    row = result_len - 87
    windowSize = row
    windowSize = win

    # 数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_result_GY = ((x_result - t_min) / (t_max - t_min)).copy()
    y_result_GY = ((y_result - t_min) / (t_max - t_min)).copy()
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)
    y_rbf_all = []
    y_test_all = []
    rng = np.random.RandomState(1233)

    for y_i in range(row, row+1):
        if y_i < windowSize:
            continue
        x_train = x_result_GY[y_i - windowSize:y_i]
        y_train = y_result_GY[y_i - windowSize:y_i]
        x_test = x_result[y_i:y_i + 1]
        y_test = y_result[y_i:y_i + 1]

        net = DBN(layer_sizes=[seq_len,20,40], bp_layer=[1])
        net.pretrain(x_train,lr=lr, epochs=200)
        net.fineTune(x_train, y_train,lr=lr, epochs=10000)

        y_rbf = net.predict(x_train)

        import MDLearn.utils.EvalueationIndex as EI
        # ei = EI.evalueationIndex(y_rbf, y_train)
        # print("归一化训练ＲＭＳＥ")
        # ei.show()
        #
        import MDLearn.utils.Draw as draw
        # draw.plot_results_point(y_rbf, y_train)

        y_rbf_haunYuan = y_rbf*(t_max - t_min)+t_min
        y_train_haunYuan = y_result[y_i - windowSize:y_i]

        print("还原训练ＲＭＳＥ")
        ei = EI.evalueationIndex(y_rbf_haunYuan, y_train_haunYuan)
        ei.show()
        draw.plot_results_point(y_rbf_haunYuan, y_train_haunYuan)
    '''DBN_T 效果不好　ＲＭＳＥ　在0.08左右'''
def DBN_Test(seq_len, win, lr):

    data_src = LD.loadCsvData_Np("../data/co2-ppm-mauna-loa-19651980.csv", skiprows=1, ColumnList=[1])


    # Australia/US	British/US	Canadian/US	Dutch/US	French/US	German/US	Japanese/US	Swiss/US
    data_t =data_src[:,0:1].copy()
    # 数据还源时使用
    t_mean = np.mean(data_t)
    t_min = np.min(data_t)
    t_max = np.max(data_t)

    # 数据预处理
    result,x_result,y_result = LD.dataRecombine_Single(data_t, seq_len)
    # print(x_result, y_result)

    result_len = len(result)
    row = round(0.8 * result.shape[0])
    row = result_len - 87
    windowSize = row
    windowSize = win


    # 数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_result_GY = ((x_result - t_min) / (t_max - t_min)).copy()
    y_result_GY = ((y_result - t_min) / (t_max - t_min)).copy()
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)
    y_rbf_all = []
    y_test_all = []
    rng = np.random.RandomState(1233)


    for y_i in range(row, result_len):
        if y_i < windowSize:
            continue
        x_train = x_result_GY[y_i - windowSize:y_i]
        y_train = y_result_GY[y_i - windowSize:y_i]
        x_test = x_result_GY[y_i:y_i + 1]
        y_test = y_result_GY[y_i:y_i + 1]

        # print(x_train, y_train)
        # assert False

        net = DBN(layer_sizes=[seq_len,20,40], bp_layer=[1])
        net.pretrain(x_train,lr=lr, epochs=200)
        net.fineTune(x_train, y_train,lr=lr, epochs=10000)

        y_rbf = net.predict(x_test)

        y_rbf_all.append(y_rbf)
        y_test_all.append(y_test)


    # print("全部预测ＲＭＳＥ")
    # y_rbf_all = np.array(y_rbf_all).ravel()
    # y_test_all = np.array(y_test_all).ravel()
    # ei = EI.evalueationIndex(y_rbf_all, y_test_all)
    # ei.show()
    #
    # import MDLearn.utils.Draw as draw
    # draw.plot_results_point(y_rbf_all, y_test_all)

    '''DBN_Test还原数据'''
    print("DBN_Test还原预测ＲＭＳＥ")
    y_rbf_all = np.array(y_rbf_all)
    y_test_all = np.array(y_test_all)
    y_rbf_haunYuan = y_rbf_all * (t_max - t_min) + t_min
    y_test_haunYuan = y_test_all * (t_max - t_min) + t_min
    ei = EI.evalueationIndex(y_rbf_haunYuan, y_test_haunYuan)
    ei.show()
    # draw.plot_results_point(y_rbf_haunYuan, y_test_haunYuan)

def DBN_BP_Train(seq_len, win,  lr):
    data_src = LD.loadCsvData_Np("../data/co2-ppm-mauna-loa-19651980.csv", skiprows=1, ColumnList=[1])

    # Australia/US	British/US	Canadian/US	Dutch/US	French/US	German/US	Japanese/US	Swiss/US
    data_t = data_src[:, 0:1].copy()
    # 数据还源时使用
    t_mean = np.mean(data_t)
    t_min = np.min(data_t)
    t_max = np.max(data_t)

    # 数据预处理
    result, x_result, y_result = LD.dataRecombine_Single(data_t,seq_len)
    # print(x_result, y_result)

    result_len = len(result)
    row = round(0.8 * result.shape[0])
    row = result_len - 87
    windowSize = row
    windowSize = win
    # 数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_result_GY = ((x_result - t_min) / (t_max - t_min)).copy()
    y_result_GY = ((y_result - t_min) / (t_max - t_min)).copy()
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)
    y_rbf_all = []
    y_test_all = []
    rng = np.random.RandomState(1233)

    for y_i in range(row, row+1):
        if y_i < windowSize:
            continue
        x_train = x_result_GY[y_i - windowSize:y_i]
        y_train = y_result_GY[y_i - windowSize:y_i]
        x_test = x_result[y_i:y_i + 1]
        y_test = y_result[y_i:y_i + 1]

        net = DBN(layer_sizes=[seq_len,20,40], bp_layer=[1])
        net.pretrain(x_train,lr=lr, epochs=200)
        # net.fineTune(x_train, y_train,lr=lr, epochs=10000)
        bp = BP([seq_len,20,40, 1])
        w_list, b_list = net.getHyperParameter()
        bp.setHyperParameter(w_list, b_list)
        bp.train(x_train, y_train, lr=lr, epochs=10000)

        y_rbf = bp.predict(x_train)

        import MDLearn.utils.EvalueationIndex as EI
        # ei = EI.evalueationIndex(y_rbf, y_train)
        # print("归一化训练ＲＭＳＥ")
        # ei.show()

        import MDLearn.utils.Draw as draw
        # draw.plot_results_point(y_rbf, y_train)

        y_rbf_haunYuan = y_rbf*(t_max - t_min)+t_min
        y_train_haunYuan = y_result[y_i - windowSize:y_i]

        print("还原训练ＲＭＳＥ")
        ei = EI.evalueationIndex(y_rbf_haunYuan, y_train_haunYuan)
        ei.show()
        draw.plot_results_point(y_rbf_haunYuan, y_train_haunYuan)

def DBN_BP_Test(seq_len, win, lr):

    data_src = LD.loadCsvData_Np("../data/co2-ppm-mauna-loa-19651980.csv", skiprows=1, ColumnList=[1])


    # Australia/US	British/US	Canadian/US	Dutch/US	French/US	German/US	Japanese/US	Swiss/US
    data_t =data_src[:,0:1].copy()
    # 数据还源时使用
    t_mean = np.mean(data_t)
    t_min = np.min(data_t)
    t_max = np.max(data_t)

    # 数据预处理
    result,x_result,y_result = LD.dataRecombine_Single(data_t, seq_len)
    # print(x_result, y_result)

    result_len = len(result)
    row = round(0.8 * result.shape[0])
    row = result_len - 87
    windowSize = row
    windowSize = win


    # 数据归一化
    # data_normalization = EvaluationIndex.归一化.normalization_max_min_负1_1(data_src)
    x_result_GY = ((x_result - t_min) / (t_max - t_min)).copy()
    y_result_GY = ((y_result - t_min) / (t_max - t_min)).copy()
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)
    y_rbf_all = []
    y_test_all = []
    rng = np.random.RandomState(1233)

    for y_i in range(row, result_len):
        if y_i < windowSize:
            continue
        x_train = x_result_GY[y_i - windowSize:y_i]
        y_train = y_result_GY[y_i - windowSize:y_i]
        x_test = x_result_GY[y_i:y_i + 1]
        y_test = y_result_GY[y_i:y_i + 1]

        # print(x_train, y_train)
        # assert False
        net = DBN(layer_sizes=[seq_len,20,40], bp_layer=[1])
        net.pretrain(x_train,lr=lr, epochs=200)
        # net.fineTune(x_train, y_train,lr=lr, epochs=10000)
        bp = BP([seq_len,20,40, 1])
        w_list, b_list = net.getHyperParameter()
        bp.setHyperParameter(w_list, b_list)
        bp.train(x_test, y_test, lr=lr, epochs=10000)

        y_rbf = bp.predict(x_train)

        y_rbf_all.append(y_rbf)
        y_test_all.append(y_test)


    # print(np.array(y_rbf_all).ravel())
    # print(np.array(y_test_all).ravel())#, np.array(y_test_all))

    # print("全部预测ＲＭＳＥ")
    # y_rbf_all = np.array(y_rbf_all).ravel()
    # y_test_all = np.array(y_test_all).ravel()
    # ei = EI.evalueationIndex(y_rbf_all, y_test_all)
    # ei.show()

    import MDLearn.utils.Draw as draw
    # draw.plot_results_point(y_rbf_all, y_test_all)

    '''还原数据'''
    print("DBN_BP_Test还原预测ＲＭＳＥ")
    y_rbf_all = np.array(y_rbf_all)
    y_test_all = np.array(y_test_all)

    y_rbf_haunYuan = y_rbf_all * (t_max - t_min) + t_min
    y_test_haunYuan = y_test_all * (t_max - t_min) + t_min
    ei = EI.evalueationIndex(y_rbf_haunYuan, y_test_haunYuan)
    ei.show()
    # draw.plot_results_point(y_rbf_haunYuan, y_test_haunYuan)

def test(seq_len, win, lr):
    print(seq_len, win)
    DBN_Test(seq_len, win, lr)
    DBN_BP_Test(seq_len, win, lr)

if __name__ == "__main__":
    # sl, win, hls, lr, ep
    import time
    '''ＤＢＮ目标小于0.045'''

    # DBN_Train(3, 97, 0.1)
    # # DBN_Test(2,3, 0.01)
    # DBN_BP_Train(3, 97, 0.1)
    DBN_BP_Test(2, 20, 0.1)
    test(2, 20, 0.1)
    test(2, 21, 0.1)
    test(2, 49, 0.1)
    test(3, 16, 0.1)
    test(3, 18, 0.1)
    test(3, 20, 0.1)
    test(3, 97, 0.1)
    test(4, 12, 0.1)
    test(4, 24, 0.1)
    test(4, 41, 0.1)
    test(5, 12, 0.1)
    test(5, 21, 0.1)
    test(5, 40, 0.1)
    test(5, 91, 0.1)
    test(5, 93, 0.1)
    test(5, 87, 0.1)
    test(6, 13, 0.1)
    test(6, 25, 0.1)
    test(6, 36, 0.1)
    test(6, 40, 0.1)
    test(6, 94, 0.1)

    test(7, 12, 0.1)
    test(7, 13, 0.1)
    test(7, 18, 0.1)
    test(7, 50, 0.1)

    test(8, 13, 0.1)
    test(8, 14, 0.1)
    test(8, 21, 0.1)
    test(8, 23, 0.1)

    test(9, 12, 0.1)
    test(9, 13, 0.1)
    test(9, 15, 0.1)
    test(9, 16, 0.1)
    test(9, 23, 0.1)

    test(10, 13, 0.1)
    test(10, 16, 0.1)
    test(10, 17, 0.1)
    test(10, 24, 0.1)
    test(10, 29, 0.1)
    test(10, 79, 0.1)

