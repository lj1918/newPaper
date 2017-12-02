# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:58:26 2017

@author: TIM
"""

import numpy as np
from sklearn.decomposition import PCA,KernelPCA,IncrementalPCA
import matplotlib.pyplot as plt
from Load_data import *

if __name__ =='__main__':    
    raw_data = load_zhiwai_data()
    x = raw_data[:,:-2]
    # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
    x = x[:,::-1]
    # 紫外光谱测试是测了190到600nm波段 分析的时候 取240-500nm即可
    x = x[:,50:311]
    y = raw_data[:,-1]
    
    # PCA降维
    pca = PCA(n_components=210)
    pca.fit(x)
    x = pca.transform(x)
    index1 = 1
    index2 = 2
    plt.subplot(231)
    c=['b', 'g', 'r', 'c', 'm', 'y', 'k', '#005050','#905020']
    for i in range(1,4):
        xx = x[y==i]
        plt.scatter(xx[:,index1],xx[:,index2],label='class {}'.format(i), c=c[i-1])
        plt.legend(loc='best')

    plt.subplot(232)
    for i in range(4, 7):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i),c=c[i-1])
        plt.legend(loc='best')

    plt.subplot(233)
    for i in range(7, 10):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i),c=c[i-1])
        plt.legend(loc='best')

    ax =plt.subplot(212)
    for i in range(1, 10):
        xx = x[y == i]
        b = plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i),c=c[i-1])
        plt.legend(loc='best')


    plt.show()
