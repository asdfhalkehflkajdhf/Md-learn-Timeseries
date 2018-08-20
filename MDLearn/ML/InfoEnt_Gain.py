import numpy as np

def __Ent2Dim(D):
    total_res=[]
    for di in D:
        total_res.append(__Ent1Dim(di))
    return np.array(total_res)

def __Ent1Dim(D):
    res=0
    total = np.sum(D)
    for i in D:
        if i == 0 or i == total:
            continue
        res = res+ (-i/total *np. log2(i/ total))
    return res

def ent(D):
    if np.ndim(D)==1: return __Ent1Dim(D )
    elif np.ndim(D)==2:
        return __Ent2Dim (D )
    else:
        print("ndim error")
        return None

def gain(t_属性, t_目标, t_属性分类最大数,t_目标分类最大数):

    '''
        说明：t_属性，　t_目标　list　必须是int型数据，t_属性分类最大数，t_目标分类最大数，分类数的最大
            例１
            data_s = np.array([0,1,1,0,3,
                               0,1,1,1,0,
                               3,3,0,3,1,
                               3,0], dtype=np.int8)
            data_d = np.array([1,1,1,1,1,
                               1,1,1,0,0,
                               0,0,0,0,0,
                               0,0], dtype=np.int8)

            res = gain(data_s, data_d, 4, 2)
            print(res)
            例２
            data_s = np.array([0,1,1,0,2,
                               0,1,1,1,0,
                               2,2,0,2,1,
                               2,0], dtype=np.int8)
            data_d = np.array([1,1,1,1,1,
                               1,1,1,0,0,
                               0,0,0,0,0,
                               0,0], dtype=np.int8)

            res = gain(data_s, data_d, 4, 2)
            print(res)
    '''
    dataS_len=len(t_属性)
    dataD_len=len(t_目标)

    assert dataS_len==dataD_len
    assert np.max(t_属性)<t_属性分类最大数
    assert np.max(t_目标)<t_目标分类最大数



    D = [0 for x in range(0, t_目标分类最大数)]
    SiD = [0 for x in range(0, t_属性分类最大数)]

    #统计具体每个分类有多少个，计算概率
    for i in range(dataD_len):
        ii = int(t_目标[i])
        D[ii]+=1
        ii = int(t_属性[i])
        SiD[ii] += 1

    #统计每个属性类别有多少个目标分类个数
    DI=np.zeros((t_属性分类最大数,t_目标分类最大数), dtype=np.int8)
    for j in range(dataS_len):
        ii = int(t_属性[int(j)])
        jj = int(t_目标[int(j)])
        DI[ii][jj]+=1

    #计算目标熵
    entD= ent(D)
    #计算属性分类熵
    entDI = ent(DI)
    SiD=SiD/np.sum(SiD)
    res = entD - np.sum(SiD*entDI)
    return res

#Main Run Thread
if __name__=='__main__':
    # sp500_svr_simple_test_所有一起归一化2()

    data_s = np.array([1,1,1,1,1,
                       2,2,2,2,0,
                       0,1,2,2,2,
                       1,1], dtype=np.int8)

    data_s = np.array([0,1,1,0,3,
                       0,1,1,1,0,
                       3,3,0,3,1,
                       3,0], dtype=np.int8)
    data_d = np.array([1,1,1,1,1,
                       1,1,1,0,0,
                       0,0,0,0,0,
                       0,0], dtype=np.int8)

    res = gain(data_s, data_d, 4, 2)
    print(res)