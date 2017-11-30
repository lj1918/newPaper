# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
# -*- coding: utf-8 -*-
"""
基于壳向量的线性支持向量机快速增量学习算法
"""
import sys
import os
import time
import numpy as np
# import scipy.io as spi
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Load_data
'''
# from sklearn import cross_validation as cv
This module was deprecated in version 0.18 in favor of the model_selection module 
into which all the refactored classes and functions are moved. Also note that 
the interface of the new CV iterators are different from that of this module. 
This module will be removed in 0.20.
'''
def create_data():
    x,y = make_classification(n_samples=3000,n_classes=2,n_features=50)
    np.savetxt('x.txt',x)
    np.savetxt('y.txt',y)

# -----------------------------------------------------
# 函数区
# -----------------------------------------------------

def normal_svm(Batchs=20,Train_size = 0.05):
    # ======================================================
    # 一、获取数据

    raw_data = Load_data.load_zhiwai_data()
    x = raw_data[:, :-2]
    # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
    x = x[:, ::-1]
    # 紫外光谱测试是测了190到600nm波段 分析的时候 取240-500nm即可
    x = x[:, 50:311]
    y = raw_data[:, -1]
    
    x = np.loadtxt('x1.txt')
    y = np.loadtxt('y1.txt')

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 训练svm模型
    ini_clf = svm.SVC()
    model2 = ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    print('=====================================')
    print('初始化分类器：')
    print('壳向量集作为新的训练样本集,正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('类别：%d 的支持向量数量为%d' % (0, ini_clf.n_support_[0]))
    print('类别：%d 的支持向量数量为%d' % (1, ini_clf.n_support_[1]))

    # 3、开始增量学习
    batchs = Batchs  # 学习批次
    nums = np.floor(test_x.shape[0] / batchs).astype('int')  # 进行切片运算时，必须是整数
    remainder = test_x.shape[0] % batchs
    results = np.zeros((batchs, 7))
    print('=====================================')
    print('学习批次=\t', batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t', remainder)
    print('=====================================')
    print('开始增量学习：')
    batchs_train_x = train_x.copy()
    batchs_train_y = train_y.copy()
    for i in np.arange(0, batchs):
        print('第%d次增量学习' % (i + 1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试

        batchs_train_x = np.vstack((train_x[:, :], test_x[0: ((i + 1) * nums * train_size).astype('int'), :]))
        batchs_train_y = np.concatenate((train_y[:], test_y[0: ((i + 1) * nums * train_size).astype('int')]),
                                        axis=0)
        t1 = time.time()
        clf = svm.SVC()
        model = clf.fit(batchs_train_x,
                        batchs_train_y
                        )
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        t2 = time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))

        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         train_x.shape[0],# 初始训练样本数量
                         nums, #每轮循环新增样本数量
                         clf.n_support_[0],
                         clf.n_support_[1],
                         (np.sum(result == 0) / result.shape[0]),
                         (t2-t1)]

    np.savetxt('results_svm.txt', results)
# 神经网络
def normal_nw(Batchs=20,Train_size = 0.05):
    # ======================================================
    # 一、获取数据

    raw_data = Load_data.load_zhiwai_data()
    x = raw_data[:, :-2]
    # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
    x = x[:, ::-1]
    # 紫外光谱测试是测了190到600nm波段 分析的时候 取240-500nm即可
    x = x[:, 50:311]
    y = raw_data[:, -1]

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 训练svm模型
    ini_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(12), random_state=1)
    model2 = ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    print('=====================================')
    print('初始化分类器：')
    print('壳向量集作为新的训练样本集,正确率：%f' % (np.sum(result == 0) / result.shape[0]))


    # 3、开始增量学习
    batchs = Batchs  # 学习批次
    nums = np.floor(test_x.shape[0] / batchs).astype('int')  # 进行切片运算时，必须是整数
    remainder = test_x.shape[0] % batchs
    results = np.zeros((batchs, 7))
    print('=====================================')
    print('学习批次=\t', batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t', remainder)
    print('=====================================')
    print('开始增量学习：')
    batchs_train_x = train_x.copy()
    batchs_train_y = train_y.copy()
    for i in np.arange(0, batchs):
        print('第%d次增量学习' % (i + 1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试

        batchs_train_x = np.vstack((train_x[:, :], test_x[0: ((i + 1) * nums * train_size).astype('int'), :]))
        batchs_train_y = np.concatenate((train_y[:], test_y[0: ((i + 1) * nums * train_size).astype('int')]),
                                        axis=0)
        t1 = time.time()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(12), random_state=1)
        model = clf.fit(batchs_train_x,
                        batchs_train_y
                        )
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        t2 = time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))


        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         train_x.shape[0],# 初始训练样本数量
                         nums, #每轮循环新增样本数量
                         0,
                         0,
                         (np.sum(result == 0) / result.shape[0]),
                         (t2-t1)]

    np.savetxt('results_nw.txt', results)

# -----------------------------------------------------
# 主程序
# -----------------------------------------------------
if __name__ == '__main__':

    batchs = 50
    normal_nw(Batchs=batchs)
    normal_svm(Batchs=batchs)
    # 绘图
    # os.chdir('D:\\20_同步文件\\pyStudy\\Apple_essence')
    results = np.loadtxt('results_nw.txt')
    results_svm = np.loadtxt('results_svm.txt')

    plt.figure(1)
    # 图1
    plt.subplot(121)
    plt.plot(results[:, 0], results[:, -2] *100, 'r-*',label='MLPNN')
    plt.plot(results_svm[:, 0], results_svm[:, -2]* 100, 'b-+',label='Normal SVM')
    plt.ylim((40, 100))
    plt.title('Classification accuracies of the normal svm and MLPNN')
    plt.xlabel('sample nums')
    plt.ylabel('Classification accuracy(%)')
    plt.legend(loc='down right')
    # 图2
    plt.subplot(122)
    plt.plot(results[:, 0], results[:, -1]*100, 'r-*',label='hull svm')
    plt.plot(results_svm[:, 0], results_svm[:, -1]*100, 'b-+',label='normal svm')
    plt.xlabel('sample nums')
    plt.ylabel('raining time(s)')
    plt.legend(loc='upper left')
    plt.title('training time of the normal svm and hull svm')
    plt.show()
    
    