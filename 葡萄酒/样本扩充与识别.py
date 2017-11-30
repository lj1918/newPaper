# -*- coding: utf-8 -*-
"""
程序说明：采用蒙特卡洛方法对样本数据进行扩充，然后进行识别，考察大量样本数据下的识别效率和准确率。
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import  io as spi
import time as tm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,train_test_split,validation_curve

from sklearn.preprocessing import StandardScaler  #归一化
from sklearn.decomposition import PCA

filepath = u'D:\\20_同步文件\\newPaper\\葡萄酒\\data\\拉曼光谱\\Raman积分时间2秒'

def load_ranman_data(path=filepath,isJiaozheng=True):
    '''
    加载拉曼光谱数据
    WC-001 – WC-006 王朝干红葡萄酒
    WC-007 – WC-012 王朝海岸解百纳干红葡萄酒
    WC-013 – WC-018 王朝干红葡萄酒（典藏一级）
    CC-001 – CC-006 长城海岸葡萄酒 优级解百纳干红
    CC-007 – CC-012 长城干红葡萄酒
    CC-013 – CC-018 长城海岸葡萄酒
    ZY-001 – ZY-006 张裕干红葡萄酒
    ZY-007 – ZY-012 张裕酿酒师 赤霞珠干红葡萄酒
    ZY-013 – ZY-018 张裕干红葡萄酒 （佐餐级）

    '''
    if isJiaozheng:
        filename = ('%s\\%s'%(path,'CC长城葡萄_jiaozheng.csv'))
        print('加载校正数据文件=',filename)
        cc_data = np.loadtxt(filename,delimiter=',')
        # 扩充一列为标签列
        cc_data = np.hstack((cc_data, np.zeros((cc_data.shape[0] ,1))))
        for item in cc_data:
            if 1 <= item[-2] <= 6:
                item[-1] = 1
            elif 7 <= item[-2] <= 12:
                item[-1] = 2
            elif 13 <= item[-2]:
                item[-1] = 3

        filename = ('%s\\%s' % (path, 'WC王朝葡萄_jiaozheng.csv'))
        print('加载校正数据文件=', filename)
        wc_data = np.loadtxt(filename, delimiter=',')
        # 扩充一列为标签列
        wc_data = np.hstack((wc_data, np.zeros((wc_data.shape[0], 1))))
        for item in wc_data:
            if 1 <= item[-2] <= 6:
                item[-1] = 4
            elif 7 <= item[-2] <= 12:
                item[-1] = 5
            elif 13 <= item[-2]:
                item[-1] = 6

        filename = ('%s\\%s' % (path, 'ZY张裕葡萄_jiaozheng.csv'))
        print('加载校正数据文件=', filename)
        zy_data = np.loadtxt(filename, delimiter=',')
        # 扩充一列为标签列
        zy_data = np.hstack((zy_data, np.zeros((zy_data.shape[0], 1))))
        for item in zy_data:
            if 1 <= item[-2] <= 6:
                item[-1] = 7
            elif 7 <= item[-2] <= 12:
                item[-1] = 8
            elif 13 <= item[-2]:
                item[-1] = 9
    else:
        filename = ('%s\\%s '%(path,'CC长城葡萄.csv'))
        print('加载非校正数据文件=', filename)
        cc_data = np.loadtxt(filename,delimiter=',')
        # 扩充一列为标签列
        cc_data = np.hstack((cc_data, np.zeros((cc_data.shape[0], 1))))
        for item in cc_data:
            if 1 <= item[-2] <= 6:
                item[-1] = 1
            elif 7 <= item[-2] <= 12:
                item[-1] = 2
            elif 13 <= item[-2]:
                item[-1] = 3

        filename = ('%s\\%s ' % (path, 'WC王朝葡萄.csv'))
        print('加载非校正数据文件=', filename)
        wc_data = np.loadtxt(filename, delimiter=',')
        # 扩充一列为标签列
        wc_data = np.hstack((wc_data, np.zeros((wc_data.shape[0], 1))))
        for item in wc_data:
            if 1 <= item[-2] <= 6:
                item[-1] = 4
            elif 7 <= item[-2] <= 12:
                item[-1] = 5
            elif 13 <= item[-2]:
                item[-1] = 6

        filename = ('%s\\%s ' % (path, 'ZY张裕葡萄.csv'))
        print('加载非校正数据文件=', filename)
        zy_data = np.loadtxt(filename, delimiter=',')
        # 扩充一列为标签列
        zy_data = np.hstack((zy_data, np.zeros((zy_data.shape[0], 1))))
        for item in zy_data:
            if 1 <= item[-2] <= 6:
                item[-1] = 7
            elif 7 <= item[-2] <= 12:
                item[-1] = 8
            elif 13 <= item[-2]:
                item[-1] = 9

    # 合并三个数据集
    data = np.vstack( ( cc_data,wc_data,zy_data )  )
    return data

def load_zhiwai_data(path=u'D:\\20_同步文件\\newPaper\\葡萄酒\\data\\紫外光谱'):
    '''
    加载紫外光谱数据
    WC-001 – WC-006 王朝干红葡萄酒
    WC-007 – WC-012 王朝海岸解百纳干红葡萄酒
    WC-013 – WC-018 王朝干红葡萄酒（典藏一级）

    CC-001 – CC-006 长城海岸葡萄酒 优级解百纳干红
    CC-007 – CC-012 长城干红葡萄酒
    CC-013 – CC-018 长城海岸葡萄酒

    ZY-001 – ZY-006 张裕干红葡萄酒
    ZY-007 – ZY-012 张裕酿酒师 赤霞珠干红葡萄酒
    ZY-013 – ZY-018 张裕干红葡萄酒 （佐餐级）

    '''
    filename = ('%s\\%s ' % (path, 'CC长城葡萄.csv'))
    cc_data = np.loadtxt(filename, delimiter=',')
    # 扩充一列为标签列
    cc_data = np.hstack((cc_data, np.zeros((cc_data.shape[0], 1))))
    for item in cc_data:
        if 1 <= item[-2] <= 6:
            item[-1] = 1
        elif 7 <= item[-2] <= 12:
            item[-1] = 2
        elif 13 <= item[-2]:
            item[-1] = 3


    filename = ('%s\\%s ' % (path, 'WC王朝葡萄.csv'))
    wc_data = np.loadtxt(filename, delimiter=',')
    # 扩充一列为标签列
    wc_data = np.hstack((wc_data, np.zeros((wc_data.shape[0], 1))))
    for item in wc_data:
        if 1 <= item[-2] <= 6:
            item[-1] = 4
        elif 7 <= item[-2] <= 12:
            item[-1] = 5
        elif 13 <= item[-2]:
            item[-1] = 6


    filename = ('%s\\%s ' % (path, 'ZY张裕葡萄.csv'))
    zy_data = np.loadtxt(filename, delimiter=',')
    zy_data = np.hstack((zy_data, np.zeros((zy_data.shape[0], 1))))
    for item in zy_data:
        if 1 <= item[-2] <= 6:
            item[-1] = 7
        elif 7 <= item[-2] <= 12:
            item[-1] = 8
        elif 13 <= item[-2]:
            item[-1] = 9


    data  = np.vstack((cc_data, wc_data, zy_data))
    return data

'''
将数据进行扩充：
1）每个维度的均值*0.1*随机数
2）加上原值
'''
def Create_Data(raw_data_x,raw_data_y,muls=10):
    '''每个数据扩展为mul个数据'''
    if muls == 1:
        return raw_data_x,raw_data_y
    mul = muls
    [m,n] = raw_data_x.shape
    # 构造[10*m，n]矩阵
    random_mean = np.ones((m*mul,n)) # 基于均值的随机波动矩阵
    mul_raw_data = np.ones((m*mul,n))
    mul_raw_data_y = np.ones((m*mul,1))
    # raw_data 扩充mul倍
    for i in range(m):
        mul_raw_data[i*mul:(i+1)*mul,: ] = raw_data_x[i, :]
        mul_raw_data_y[i*mul:(i+1)*mul ] = raw_data_y[i]
    # 各维度平均值
    mean_data = np.mean(raw_data_x, axis=0)
    #
    for i in range(m*mul):
        tp =  mean_data * 0.4 * (np.random.ranf(n) - 0.5)
        random_mean[i, :] = tp + random_mean[i, :]
    return (random_mean + mul_raw_data),mul_raw_data_y

if __name__ == '__main__':   
    
    if (1==1):        
        raw_data = load_zhiwai_data()
        xx = raw_data[:,:-2]
        # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
        xx = xx[:,::-1]
        # 紫外光谱测试是测了190到600nm波段 分析的时候 取240-500nm即可
        xx = xx[:,50:311]
        yy = raw_data[:,-1]
    
        
        muls = 100
        # 取第一个样本的数据进行扩成
        [x1,y1] = Create_Data(xx,yy,muls=muls)
        # 保存生成的数据
        np.savetxt('x1.txt',x1)
        np.savetxt('y1.txt',y1)
    
    x = np.loadtxt('x1.txt')
    y = np.loadtxt('y1.txt')
    
    # 归一化处理
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # PCA降维
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)
    
    # svm 识别
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                                        random_state=0)
    # 记录训练模型耗时
    t1 = tm.time()
    # 通过交叉验证获取的超参：C=1.21, kernel='rbf',gamma=0.0316227766
    model = SVC(C=1.21, kernel='rbf',gamma=0.5) 
    model.fit(x_train,y_train)
    t2 = tm.time()
    # 预测测试样本
    y_test_1 = model.predict(x_test)
    # 计算识别比例
    result = np.sum(y_test == y_test_1) / y_test_1.shape[0]
    
    print('训练样本数量:',x_train.shape[0])
    print('测试样本数量:',x_test.shape[0])
    print('训练模型耗时：%f秒'%((t2-t1)))
    print('识别率%.2f %%'%(result *100))
    
    # ===============================
    # 增量学习
    # 3、开始增量学习
    batchs = 20  # 学习批次
    nums = np.floor(x_test.shape[0] / batchs).astype('int')  # 进行切片运算时，必须是整数
    remainder = x_test.shape[0] % batchs
    results = np.zeros((batchs, 4))
    print('=====================================')
    print('学习批次=\t', batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t', remainder)
    print('=====================================')
    print('开始增量学习：')
    batchs_x_train = x_train.copy()
    batchs_y_train = y_train.copy()
    for i in np.arange(0, batchs):
        print('第%d次增量学习' % (i + 1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试
        batchs_x_train = np.vstack((x_train[:,:],x_test[i * nums : ((i + 1) * nums * train_size).astype('int'), :]))
        batchs_y_train = np.concatenate((y_train[:],y_test[i * nums : ((i + 1) * nums * train_size).astype('int')]),axis=0)
        clf = SVC()
        model = clf.fit( batchs_x_train,
                         batchs_y_train
                        )
        # 用剩余的部分进行测试
        result_y = clf.predict(x_test[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        result = result_y - y_test[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        for i in np.arange( clf.n_support_.shape[0]):
            print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[i]))

        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         clf.n_support_[0],
                         clf.n_support_[1],
                         (np.sum(result == 0) / result.shape[0])]

    np.savetxt('results_svm.txt', results)
    # 绘图
    plt.plot(results[:, 0], results[:, -1], 'r-*')
    plt.ylim((0, 1))
    plt.show()
    
    
    # ======================================
    # 非增量学习,多批次增加标签数据
    
    # test_size=0.995,确保train的样本少于54
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.995,
                                                        random_state=0)
    batchs = 20  # 多批次识别
    nums = np.floor(x_test.shape[0] / batchs).astype('int')  # 进行切片运算时，必须是整数
    remainder = x_test.shape[0] % batchs
    results = np.zeros((batchs, 4))
    
    clf = SVC()
    clf.fit(x_train,y_train)
    
    for i in np.arange(0, batchs):
        batchs_x_test = x_test[ i * nums : ((i + 1) * nums), :]
        batchs_y_test = y_test[ i * nums : ((i + 1) * nums)]
        # 用剩余的部分进行测试
        result_y = clf.predict( batchs_x_test )
        result = result_y - batchs_y_test
        print('第 %f 次识别的正确率：%f' % (i,np.sum(result == 0) / result.shape[0]))
 




