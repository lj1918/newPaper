# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:58:26 2017

@author: TIM
"""

import numpy as np
from sklearn.decomposition import PCA,KernelPCA,IncrementalPCA,FastICA
import matplotlib.pyplot as plt
from Load_data import *
from matplotlib.font_manager import *
myfont = FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc')

if __name__ =='__main__':    
    raw_data = load_zhiwai_data()
    x = raw_data[:,:-2]
    # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
    x = x[:,::-1]
    # 紫外光谱测试是测了190到600nm波段 分析的时候 取240-500nm即可
    raw_x = x[:,50:311]
    y = raw_data[:,-1]
#    raw_x = np.loadtxt('x1.txt')
#    y = np.loadtxt('y1.txt')

    # PCA降维
    pca = PCA(n_components=5)
    pca.fit(raw_x)
    x = pca.transform(raw_x)
    index1 = 1
    index2 = 2
    index3 = 3

    '''
    plt.figure(figsize=(12,7))
    plt.subplot(231)
    c=['b', 'g', 'r', 'c', 'm', 'y', '#CD853F', '#005050','#905020']
    m= ['o','*','^','s','p','1','2','3','4']
    for i in range(1,4):
        xx = x[y==i]
        plt.scatter(xx[:,index1],xx[:,index2],label='class {}'.format(i), c=c[i-1],marker=m[i-1])
        plt.legend(loc='best')
        plt.title('王朝红葡萄酒的PCA',fontproperties=myfont)

    plt.subplot(232)
    for i in range(4, 7):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i),c=c[i-1],marker=m[i-1])
        plt.legend(loc='best')
        plt.title('长城红葡萄酒的PCA', fontproperties=myfont)

    plt.subplot(233)
    for i in range(7, 10):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i),c=c[i-1],marker=m[i-1])
        plt.legend(loc='best')
        plt.title('张裕红葡萄酒的PCA', fontproperties=myfont)

    ax =plt.subplot(212)
    for i in range(1, 10):
        xx = x[y == i]
        b = plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i),c=c[i-1],marker=m[i-1])
        plt.legend(loc='best')
        plt.title('三个厂家红葡萄酒的PCA', fontproperties=myfont)

    plt.show()
    '''
    # =============================================
    # 绘制三维PCA图
    from mpl_toolkits.mplot3d import Axes3D

    c=['b', 'g', 'r', 'c', 'm', 'y', '#CD853F', '#005050','#905020']
    m= ['o','*','^','s','p','h','+','D','8']
    fig = plt.figure(figsize=(15,10))
    
    ax = fig.add_subplot(111, projection='3d')
    #plt.title('长城红葡萄酒的PCA',fontproperties=myfont)
    plt.legend(loc='best')
    xrange = 6 # 绘图样本数量的范围
    for i in range(1, 10):
        xx = x[y == i]
        ax.scatter( xx[:xrange, index1], xx[:xrange, index2], xx[:xrange, index3],marker=m[i-1],s=50,label='class %s'%(i))  # 绘制数据点
        ax.set_label('class %s'%(i+1) )    
        ax.legend(loc='best')
        
    ax.set_xlabel('PC %s (%.2f %%)' % (1,pca.explained_variance_ratio_[0] * 100 ),fontsize = 16 )  
    ax.set_ylabel('PC %s (%.2f %%)' % (2,pca.explained_variance_ratio_[1] * 100 ),fontsize = 16 ) 
    ax.set_zlabel('PC %s (%.2f %%)' % (3,pca.explained_variance_ratio_[2] * 100 ),fontsize = 16 ) 
    ax.legend(loc='best')
    plt.show()
    '''
    #===========================
    # ICA降维
    ica = FastICA(n_components=10)
    ica.fit(raw_x)
    x = ica.transform(raw_x)
    index1 = 1
    index2 = 2
    plt.figure(figsize=(12, 7))
    plt.subplot(231)
    for i in range(1, 4):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i), c=c[i - 1],marker=m[i-1])
        plt.legend(loc='best')

    plt.subplot(232)
    for i in range(4, 7):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i), c=c[i - 1],marker=m[i-1])
        plt.legend(loc='best')

    plt.subplot(233)
    for i in range(7, 10):
        xx = x[y == i]
        plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i), c=c[i - 1],marker=m[i-1])
        plt.legend(loc='best')

    ax = plt.subplot(212)
    for i in range(1, 10):
        xx = x[y == i]
        b = plt.scatter(xx[:, index1], xx[:, index2], label='class {}'.format(i), c=c[i - 1],marker=m[i-1])
        plt.legend(loc='best')

    plt.show()
    '''
