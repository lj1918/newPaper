# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:21:28 2017

@author: TIM
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

Do_pca = False
Train_size=0.05
Batchs = 100
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
# 小类1分类器
clf1 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) 
model1 = clf1.fit( train_x[ np.where(train_yy==1) ],   train_y[ np.where(train_yy==1)] )
result_y = clf1.predict( test_x[ np.where(test_yy==1) ]) # 大类为1的test_x
result = result_y - test_y[ np.where(test_yy==1) ]     # 大类为1的test_y
sv1 = model1.support_vectors_
print('=====================================')
print('初始化小类1分类器：')
print('初始化小类1分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
print('支持向量数量为%d' % (sv1.shape[0]))
# 小类2分类器
clf2 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) 
model2 = clf2.fit( train_x[ np.where(train_yy==2) ],   train_y[ np.where(train_yy==2)] )
result_y = clf2.predict( test_x[ np.where(test_yy==2) ] )
result = result_y - test_y[ np.where(test_yy==2) ]
sv2 = model2.support_vectors_
print('=====================================')
print('初始化小类2分类器：')
print('初始化小类2分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
print('支持向量数量为%d' % (sv2.shape[0]))
# 小类3分类器
clf3 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) 
model3 = clf3.fit( train_x[ np.where(train_yy==3) ],   train_y[ np.where(train_yy==3)] )
result_y = clf3.predict(test_x[ np.where(test_yy==3) ])
result = result_y - test_y[ np.where(test_yy==3) ]
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

for i in np.arange(0, batchs):
    print('第%d次增量学习' % (i + 1))
    train_size = 0.8
    # 用nums *  train_size 个新增样本进行训练，剩余进行测试

    batchs_train_x = np.vstack((train_x[:, :], test_x[0: ((i + 1) * nums * train_size).astype('int'), :]))
    batchs_test_x = test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :]
    
    batchs_train_y = np.concatenate((train_y[:], test_y[0: ((i + 1) * nums * train_size).astype('int')]),axis=0)
    batchs_train_yy = get_yy(batchs_train_y)
    batchs_test_y = test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
    batchs_test_y = get_yy(batchs_test_y)
    # 开始记时间
    t1 = time.time()
    # 增量训练4个分类器
    clf = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) # 大类分类器
    model = clf.fit( np.vstack(( batchs_train_x ,train_x )), np.concatenate(( batchs_train_yy,train_yy)) )
    
    clf1 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) 
    model1 = clf1.fit( np.vstack(( batchs_train_x[ np.where(train_yy==1) ], train_x[ np.where(train_yy==1) ] )),   
                       np.concatenate(( batchs_train_y[ np.where(train_yy==1) ], train_y[ np.where(train_yy==1) ] ))  )
    
    clf2 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) 
    model2 = clf2.fit( np.vstack(( batchs_train_x[ np.where(train_yy==2) ], train_x[ np.where(train_yy==2) ] )),   
                       np.concatenate(( batchs_train_y[ np.where(train_yy==2) ], train_y[ np.where(train_yy==2) ] ))  )
    
    clf3 = svm.SVC( C=12 ,gamma=0.01,kernel='rbf' ) 
    model3 = clf3.fit( np.vstack(( batchs_train_x[ np.where(train_yy==3) ], train_x[ np.where(train_yy==3) ] )),   
                       np.concatenate(( batchs_train_y[ np.where(train_yy==3) ], train_y[ np.where(train_yy==3) ] ))  )
    
    # 用剩余的部分进行测试
    result_yy = np.zeros_like(batchs_test_y)
    for j, x_,y_ in  zip( range(batchs_test_x.shape[0]),batchs_test_x,batchs_test_y):
        #print(clf.predict(x_.reshape(1,-1)))
        if clf.predict(x_.reshape(1,-1)) == 1:
            result_yy[j] = clf1.predict(x_.reshape(1,-1))
        elif clf.predict(x_.reshape(1,-1)) == 2:
            result_yy[j] = clf2.predict(x_.reshape(1,-1))
        else:
            result_yy[j] = clf3.predict(x_.reshape(1,-1))                
        
    t2 = time.time()
    
    result = np.sum(result_yy > 0)
    print('正确率：%f' % ( result / result_yy.shape[0]))
    print('耗时%f' % ((t2-t1)))
    # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
    results[i, :] = [i,
                     train_x.shape[0],# 初始训练样本数量
                     nums, #每轮循环新增样本数量
                     0,
                     0,
                     ( result / result_yy.shape[0]),
                     (t2-t1)]

np.savetxt('layer_svm.txt', results)
