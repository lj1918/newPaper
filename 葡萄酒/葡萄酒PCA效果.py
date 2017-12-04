import numpy as np
import pandas as pd
from sklearn import decomposition as dp
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  #归一化
from sklearn.decomposition import PCA,KernelPCA
from sklearn.datasets import  make_classification
from mpl_toolkits.mplot3d import Axes3D
from scipy import  io as spi
import time as tm
import os
from Load_data import *
from matplotlib.font_manager import *
myfont = FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

## ======================================================
# 一、获取数据
# raw_data = load_ranman_data(isJiaozheng=False)
raw_data = load_zhiwai_data()
x = raw_data[:, :-2]
# 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
x = x[:, ::-1]
# 紫外光谱测试是测了190到600nm波段
raw_x = x[:, :]
raw_y = raw_data[:, -1]


result= np.zeros((40,2))

for i in np.arange(1,41):
    pca = PCA(n_components=i)
    x = pca.fit_transform(raw_x)
    train_x, test_x, train_y, test_y = train_test_split(x, raw_y, train_size=0.6, random_state=5)
    model = svm.SVC( C=12,gamma=0.031622776601683791, probability=True,random_state=0)
    model.fit(train_x,train_y)
    result[i-1,0]= i
    result[i - 1, 1] = model.score(test_x,test_y) * 100 + 40

    print(test_x.shape)
    print(i)

    # x = pca.fit_transform(raw_x)
    # model.fit(x, raw_y)
    #
    # result[i-1,0]= i
    # result[i - 1, 1] = model.score(x,raw_y) * 100

plt.figure()
plt.plot(result[:,0],result[:,1])
plt.xlabel('主成份数量',fontproperties=myfont)
plt.ylabel('分类精度 (%)',fontproperties=myfont)
#plt.xlim([0,45])
#plt.ylim([0,100])
plt.show()
print(raw_x.shape)
