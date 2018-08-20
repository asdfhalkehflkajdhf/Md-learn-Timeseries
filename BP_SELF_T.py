
import numpy as np
import MDLearn.utils.LoadData as LD
import cDBN_M.BP_SELF as BP

'''训练'''
def BP_Train():

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
    x_result_GY = ((x_result - t_min) / (t_max - t_min)).copy()
    y_result_GY = ((y_result - t_min) / (t_max - t_min)).copy()
    # x_result = (x_result - t_mean) / np.std(x_result)
    # y_result = (y_result - t_mean) / np.std(x_result)

    for y_i in range(row, row+1):
        if y_i < windowSize:
            continue
        x_train = x_result_GY[y_i - windowSize:y_i]
        y_train = y_result_GY[y_i - windowSize:y_i]
        x_test = x_result_GY[y_i:y_i + 1]
        y_test = y_result[y_i:y_i + 1]

        # print(x_train, y_train)
        # assert False

        bp = BP.BP( [len(x_test[0]), 6, 1])
        bp.train(x_train, y_train, lr=0.01, epochs=10000, residual=0.0001, show=True, show_epochs=1000 )


        y_rbf = bp.predict(x_train)


        import MDLearn.utils.EvalueationIndex as EI
        ei = EI.evalueationIndex(y_rbf, y_train)
        ei.show()

        import MDLearn.utils.Draw as draw
        draw.plot_results_point(y_rbf, y_train)

        y_rbf_haunYuan = y_rbf*(t_max - t_min)+t_min
        y_train_haunYuan = y_result[y_i - windowSize:y_i]

        ei = EI.evalueationIndex(y_rbf_haunYuan, y_train_haunYuan)
        ei.show()
        draw.plot_results_point(y_rbf_haunYuan, y_train_haunYuan)


def BP_Test():

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

        bp = BP.BP( [len(x_test[0]), 6, 1])
        bp.train(x_train, y_train, lr=0.01, epochs=10000, residual=0.0001)
        y_rbf = bp.predict(x_test)

        y_rbf_all.append(y_rbf)
        y_test_all.append(y_test)

    # print(np.array(y_rbf_all).ravel())
    # print(np.array(y_test_all).ravel())#, np.array(y_test_all))

    y_rbf_all = np.array(y_rbf_all).ravel()
    y_test_all = np.array(y_test_all).ravel()
    import MDLearn.utils.EvalueationIndex as EI
    ei = EI.evalueationIndex(y_rbf_all, y_test_all)
    ei.show()

    import MDLearn.utils.Draw as draw
    draw.plot_results_point(y_rbf_all, y_test_all)

    '''还原数据'''
    y_rbf_haunYuan = y_rbf_all * (t_max - t_min) + t_min
    y_test_haunYuan = y_test_all * (t_max - t_min) + t_min
    ei = EI.evalueationIndex(y_rbf_haunYuan, y_test_haunYuan)
    ei.show()
    draw.plot_results_point(y_rbf_haunYuan, y_test_haunYuan)

if __name__ == "__main__":
    BP_Test()
    # BP_Train()
    #结果：
    # MAE:0.037941 MPE:0.019193 MAPE:0.056619 MSE:0.002075 RMSE:0.045557