# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:08:04 2017

@author: TIM
"""

from Load_data import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import StratifiedShuffleSplit#分层洗牌分割交叉验证
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,Normalizer


def get_yy(y):
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

x,y = load_zhiwai_data2()

# 标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)


# 大类标签
yy = get_yy(y)
# 切分训练集与测试集
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.5, random_state=0)
train_yy = get_yy(train_y)
test_yy = get_yy(test_y)

if ( 1 == 2 ):    
    # 网格寻找最佳参数
    C_range = np.linspace(1,500,100) #np.logspace(-2, 3, 13) 
    gamma_range = np.logspace(-3, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    # 基于交叉验证的网格搜索。
    # cv:确定交叉验证拆分策略。
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(train_x, train_yy)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))  # 找到最佳超参数


#clf = SVC( C=grid.best_params_['C'],gamma=grid.best_params_['gamma'] )
# 使用大类标签训练“大类分类器”
clf = SVC( C=12 ,gamma=0.01,kernel='rbf' )
clf.fit( train_x, train_yy )

# 使用小类标签训练各“小类分类器”
clf1 = SVC( C=12 ,gamma=0.01,kernel='rbf' )
clf1.fit( train_x[ np.where(train_yy==1) ],   train_y[ np.where(train_yy==1)] )

clf2 = SVC( C=12 ,gamma=0.01,kernel='rbf' )
clf2.fit( train_x[ np.where(train_yy==2) ],   train_y[ np.where(train_yy==2)] )

clf3 = SVC( C=12 ,gamma=0.01,kernel='rbf' )
clf3.fit( train_x[ np.where(train_yy==3) ],   train_y[ np.where(train_yy==3)] )

for x_,y_ in zip( test_x,test_y):
    print(y_)
    print(clf.predict(x_.reshape(1,-1)))
    if clf.predict(x_.reshape(1,-1)) == 1:
        print(clf1.predict(x_.reshape(1,-1)))
    elif clf.predict(x_.reshape(1,-1)) == 2:
        print(clf2.predict(x_.reshape(1,-1)))
    else:
        print(clf3.predict(x_.reshape(1,-1)))


''''''    
# 