import numpy as np

#预测精度评价指标
class evalueationIndex(object):
    def __init__(self,t_预测值,t_真实值):
        '''
        :param t_预测值:
        :param t_真实值:


        1．预测误差 e
        2．百分误差 PE
        3．平均误差 ME
        4．平均绝对误差 MAE
        5．平均绝对百分对误差 MAPE
        6．均方差 MSE
        7．误差标准差 SDE
        8．均方误差　RMSE

        res：查看是否计算结果成功

        '''

        self.res=False

        真实值 = np.array(t_真实值)
        预测值 = np.array(t_预测值)
        if(真实值.size != 预测值.size):
            return

        obj_size = 真实值.size

        # 预测误差,是个list
        self.e = 真实值-预测值
        # 绝对预测误差，是个list
        self.AE = self.e.__abs__()
        # 平均预测误差，是个值
        self.ME = self.e.sum()/obj_size
        #平均绝对预测误差，是个值
        self.MAE = self.AE.sum()/obj_size

        # 百分误差
        self.PE = self.e/真实值
        # 绝对百分误差
        self.APE = self.PE.__abs__()
        # 平均百分误差
        self.MPE = self.PE.sum() / obj_size
        # 平均绝对百分误差  可以用来衡量一个模型预测结果的好坏，计算公式如下
        self.MAPE = self.APE.sum()/obj_size

        # 均方差
        self.MSE = (self.e**2).sum()/obj_size
        # 均方误差 均方根误差是用来衡量观测值同真值之间的偏差。标准误差 对一组测量中的特大或特小误差反映非常敏感，所以，标准误差能够很好地反映出测量的精密度。
        self.RMSE = np.sqrt(self.MSE)
        # 误差标准差 观测值与其平均数偏差的平方和的平方根。标准差是用来衡量一组数自身的离散程度。
        self.SDE = self.RMSE


        self.res = True
        return
    def show(self):
        print("MAE:%f MPE:%f MAPE:%f MSE:%f RMSE:%f"%(self.MAE,self.MPE,self.MAPE,self.MSE,self.RMSE))

def test_EEvalueationIndex():
    X = [2112.40,2215.50,2479.50,2628.50,2857.00,3144.40,3430.70,3841.50,4499.30,5260.10,4584.00,5268.80,5723.80,5820.70,6125.40,6333.90]
    yl = [2074.46,2314.11,2388.03,2697.11,2816.11,3076.24,3394.35,3691.20,4151.94,4896.27,5698.20,4624.40,5778.79,6207.55,6188.02,6581.81]
    a = EvalueationIndex( yl, X)
    print(a.e)
    print(a.PE)

    print(a.AE)
    print(a.APE)

    print(a.ME, a.MPE, a.MAE, a.MAPE, a.SDE)


#Main Run Thread
if __name__=='__main__':
    test_EEvalueationIndex()
