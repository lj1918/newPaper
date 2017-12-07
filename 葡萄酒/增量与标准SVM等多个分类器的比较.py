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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from matplotlib import pyplot as plt
from matplotlib.font_manager import *
myfont = FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

'''
# from sklearn import cross_validation as cv
This module was deprecated in version 0.18 in favor of the model_selection module 
into which all the refactored classes and functions are moved. Also note that 
the interface of the new CV iterators are different from that of this module. 
This module will be removed in 0.20.
'''

def Normal_svm(Batchs=20,Train_size = 0.05,Do_pca=False):
    # ======================================================
    # 一、获取数据
    # ======================================================
    x = np.loadtxt('x1.txt')
    y = np.loadtxt('y1.txt')

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    if Do_pca == True:
        pca = PCA(n_components=10)
        pca.fit(x)
        x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 训练svm模型
    ini_clf = svm.SVC()
    ini_clf.fit(train_x, train_y)
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
        clf.fit(batchs_train_x,
                        batchs_train_y
                        )
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        t2 = time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))
        print('耗时%f' % ((t2-t1)))
        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         train_x.shape[0],# 初始训练样本数量
                         nums, #每轮循环新增样本数量
                         clf.n_support_[0],
                         clf.n_support_[1],
                         (np.sum(result == 0) / result.shape[0]),
                         (t2-t1)]

    np.savetxt('normal_svm.txt', results)
    return

def Incremental_svm(Batchs=20,ini_train_size = 0.05,Do_pca=False):
    # ======================================================
    # 一、获取数据
    x = np.loadtxt('x1.txt')
    y = np.loadtxt('y1.txt')
    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    if Do_pca == True:
        pca = PCA(n_components=10)
        pca.fit(x)
        x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=ini_train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 1、训练初始svm
    ini_clf = svm.SVC()
    ini_model = ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    sv = ini_model.support_vectors_
    sv_label = train_y[ini_model.support_]
    print('=====================================')
    print('初始化分类器：')
    print('初始化分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('支持向量数量为%d' % (sv.shape[0]))


    # 3、开始增量学习
    batchs = Batchs #学习批次
    nums = np.floor(test_x.shape[0]/batchs).astype('int') # 进行切片运算时，必须是整数
    remainder = test_x.shape[0]%batchs
    results = np.zeros((batchs,7))
    print('=====================================')
    print('学习批次=\t',batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t',remainder)
    print('=====================================')
    print('开始增量学习：')
    for i in np.arange(0, batchs):
        print('第%d次增量学习'%(i+1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试

        train_data = np.concatenate(( test_x[i * nums : ((i + 1) * nums * train_size).astype('int'), :], sv), axis=0 )  # 令A= B∪AHV 作为新训练样本集
        train_data_label = np.concatenate(( test_y[i * nums : ((i + 1) * nums * train_size).astype('int')], sv_label),
                                          axis=0 )
        # 计时开始
        t1 = time.time()
        clf = svm.SVC()
        model = clf.fit(train_data, train_data_label)
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums,:])
        t2= time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))
        print('耗时%f' % ((t2 - t1)))
        sv = model.support_vectors_
        sv_label = train_data_label[model.support_]
        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i,:] = [i,
                        train_x.shape[0],# 初始训练样本数量
                        nums, #每轮循环新增样本数量
                        clf.n_support_[0],
                        clf.n_support_[1],
                        (np.sum(result == 0) / result.shape[0]),
                        (t2-t1)]

    np.savetxt('increment_svm.txt',results)
    return

def Normal_nw(Batchs=20,Train_size = 0.05,Do_pca=False):
    # ======================================================
    # 一、获取数据
    # ======================================================
    x = np.loadtxt('x1.txt')
    y = np.loadtxt('y1.txt')

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    if Do_pca == True:
        pca = PCA(n_components=10)
        pca.fit(x)
        x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 训练svm模型
    ini_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(12), random_state=1)
    ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    print('=====================================')
    print('初始化分类器：')
    print('MLPClassifier,正确率：%f' % (np.sum(result == 0) / result.shape[0]))

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
        clf.fit(batchs_train_x, batchs_train_y )
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        t2 = time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('耗时%f' % ((t2-t1)))
        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         train_x.shape[0],# 初始训练样本数量
                         nums, #每轮循环新增样本数量
                         0,
                         0,
                         (np.sum(result == 0) / result.shape[0]),
                         (t2-t1)]

    np.savetxt('normal_nw.txt', results)
    return

def get_yy(y):
    '''
    返回大类标签
    '''
    yy = np.ones_like( y )
    yy[ np.where(y==1) ]=1
    yy[ np.where(y==2) ]=1
    yy[ np.where(y==3) ]=1
    
    yy[ np.where(y==4) ]=2
    yy[ np.where(y==5) ]=2
    yy[ np.where(y==6) ]=2
    
    yy[ np.where(y==7) ]=3
    yy[ np.where(y==8) ]=3
    yy[ np.where(y==9) ]=3
    return yy

def layer_svm(Batchs=20,Train_size = 0.05,Do_pca=False):
    # ======================================================
    # 一、获取数据
    # ======================================================
    x = np.loadtxt('x1.txt')
    y = np.loadtxt('y1.txt')
    # 大类标签
    yy = get_yy(y)

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    if Do_pca == True:
        pca = PCA(n_components=10)
        pca.fit(x)
        x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)
    # 训练集、测试集的大类标签
    train_yy = get_yy(train_y)
    test_yy = get_yy(test_y)
    
    # =======================================================
    # 三、训练分类模型
    # 1、训练初始svm
    clf = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) # 大类分类器
    model = clf.fit(train_x, train_yy)
    result_yy = clf.predict(test_x)
    result = result_yy - test_yy
    sv = model.support_vectors_
    print('=====================================')
    print('初始化大类分类器：')
    print('初始化大类分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('支持向量数量为%d' % (sv.shape[0]))
    
    clf1 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) # 大类分类器
    model1 = clf1.fit( train_x[ np.where(train_yy==1) ],   train_y[ np.where(train_yy==1)] )
    result_yy = clf.predict(test_x[ np.where(train_yy==1) ])
    result = result_yy - test_yy[ np.where(train_yy==1) ]
    sv1 = model1.support_vectors_
    print('=====================================')
    print('初始化小类1分类器：')
    print('初始化小类1分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('支持向量数量为%d' % (sv1.shape[0]))
    
    clf2 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) # 大类分类器
    model2 = clf2.fit( train_x[ np.where(train_yy==2) ],   train_y[ np.where(train_yy==2)] )
    result_yy = clf2.predict(test_x[ np.where(train_yy==2) ])
    result = result_yy - test_yy[ np.where(train_yy==2) ]
    sv2 = model2.support_vectors_
    print('=====================================')
    print('初始化小类2分类器：')
    print('初始化小类2分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('支持向量数量为%d' % (sv2.shape[0]))
    
    clf3 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) # 大类分类器
    model3 = clf3.fit( train_x[ np.where(train_yy==3) ],   train_y[ np.where(train_yy==3)] )
    result_yy = clf3.predict(test_x[ np.where(train_yy==3) ])
    result = result_yy - test_yy[ np.where(train_yy==3) ]
    sv3 = model3.support_vectors_
    print('=====================================')
    print('初始化小类3分类器：')
    print('初始化小类3分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('支持向量数量为%d' % (sv3.shape[0]))    
    
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
    
    pass
    

def draw_pic():
    results_normal_nw = np.loadtxt('normal_nw.txt')
    results_increment_svm = np.loadtxt('increment_svm.txt')
    results_normal_svm = np.loadtxt('normal_svm.txt')
    results_layer_svm = np.loadtxt('layer_svm.txt')
    
    print('normal nw min=%.2f,max = %.2f,mean = %.2f, std= %.2f, Accuracy = %.4f ,time = %.4f'%(np.min(results_normal_nw[:,-1]),
                                                                  np.max(results_normal_nw[:,-1]),
                                                                  np.mean(results_normal_nw[:,-1]),
                                                                  np.std(results_normal_nw[:,-1]),
                                                                  np.mean(results_normal_nw[:,-2]),
                                                                  np.mean(results_normal_svm[:,-1])
                                                                  ))
    
    print('increment_svm min=%.2f,max = %.2f,mean = %.2f, std= %.2f, Accuracy = %.4f ,time = %.4f'%(np.min(results_increment_svm[:,-1]),
                                                                  np.max(results_increment_svm[:,-1]),
                                                                  np.mean(results_increment_svm[:,-1]),
                                                                  np.std(results_increment_svm[:,-1]),
                                                                  np.mean(results_increment_svm[:,-2]),
                                                                  np.mean(results_normal_svm[:,-1])
                                                                  ))
    
    print('normal_svm min=%.2f,max = %.2f,mean = %.2f, std= %.2f, Accuracy = %.4f ,time = %.4f' % (np.min(results_normal_svm[:,-1]),
                                                                  np.max(results_normal_svm[:,-1]),
                                                                  np.mean(results_normal_svm[:,-1]),
                                                                  np.std(results_normal_svm[:,-1]),
                                                                  np.mean(results_normal_svm[:,-2]),
                                                                  np.mean(results_normal_svm[:,-1])
                                                                  ))
    print('layer_svm min=%.2f,max = %.2f,mean = %.2f, std= %.2f, Accuracy = %.4f ,time = %.4f' % (np.min(results_layer_svm[:,-1]),
                                                              np.max(results_layer_svm[:,-1]),
                                                              np.mean(results_layer_svm[:,-1]),
                                                              np.std(results_layer_svm[:,-1]),
                                                              np.mean(results_layer_svm[:,-2]),
                                                              np.mean(results_layer_svm[:,-1])
                                                              )) 
    

    plt.figure(1)
    # 图1
    plt.subplot(121)
    plt.plot(results_increment_svm[:, 0], results_increment_svm[:, -2] * 100, '-*', label='SV-SVM')
    #plt.plot(results_normal_nw[:, 0], results_normal_nw[:, -2] * 100, '-+', label='MLPNN')
    plt.plot(results_normal_svm[:, 0], results_normal_svm[:, -2] * 100, '-D', label='SVM')
    plt.plot(results_layer_svm[:, 0], results_layer_svm[:, -2] * 100, '-D', label='layer_svm')
    plt.ylim((40, 100))
    # plt.title('Classification accuracies of the normal svm and hull svm')
#    plt.xlabel('Increment learning nums', fontsize=16)
#    plt.ylabel('Classification accuracy(%)', fontsize=16)
    plt.xlabel('增量学习次数', fontproperties=myfont)
    plt.ylabel('分类器准确率(%)', fontproperties=myfont)
    plt.legend(loc='best')
    # 图2
    scale = 3
    plt.subplot(122)
    plt.plot(results_increment_svm[:, 0], results_increment_svm[:, -1] * scale, '-*', label='I-SVM')
    #plt.plot(results_normal_nw[:, 0], results_normal_nw[:, -1] * scale, '-+', label='MLPNN')
    plt.plot(results_normal_svm[:, 0], results_normal_svm[:, -1] * scale, '-D', label='SVM')
    plt.plot(results_layer_svm[:, 0], results_layer_svm[:, -1] * scale, '-^', label='layer_svm')

#    plt.xlabel('Increment learning nums', fontsize=16)
#    plt.ylabel('training time(s) of Classifier', fontsize=16)
#    plt.legend(loc='best')
#    plt.title('training time of Classifier')
    plt.xlabel('增量学习次数', fontproperties=myfont)
    plt.ylabel('分类器增量训练时间(s)', fontproperties=myfont)
    plt.legend(loc='best')
  

    plt.show()
    return
if __name__ == '__main__':
    batchs = 100
    #Normal_nw(Batchs=batchs)
    #Incremental_svm(Batchs=batchs)
    #Normal_svm(Batchs=batchs)
    #layer_svm(Batchs=batchs)

    # 绘图
    draw_pic()

