# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:48:39 2017

@author: TIM
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import StratifiedShuffleSplit#分层洗牌分割交叉验证
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt

import pickle

def load_ranman_data(path=u'D:\\20_同步文件\\newPaper\\葡萄酒\\data\\拉曼光谱\\Raman积分时间2秒',isJiaozheng=True):
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

if __name__ == '__main__':
    ## ======================================================
    # 一、获取数据
    raw_data = load_ranman_data(isJiaozheng=False)
    raw_data = load_zhiwai_data()
    x = raw_data[:,:-2]
    # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
    x = x[:,::-1]
    # 紫外光谱测试是测了190到600nm波段 分析的时候 取240-500nm即可
    x = x[:,50:311]
    y = raw_data[:,-1]

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    '''
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)
    '''
    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.5, random_state=0)

    # 网格寻找最佳参数
    C_range = np.logspace(-2, 3, 13)  # logspace(a,b,N)把10的a次方到10的b次方区间分成N份
    gamma_range = np.logspace(-3, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    # 基于交叉验证的网格搜索。
    # cv:确定交叉验证拆分策略。
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(x, y)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))  # 找到最佳超参数

    # 训练svm模型
    clf = SVC(C=grid.best_params_['C'],gamma=grid.best_params_['gamma'])
    clf.fit(train_x,train_y)
    score = clf.score(test_x,test_y)
    print(score)

    # ======================================
    #
    # ======================================
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    #bdt_real =  AdaBoostClassifier(base_estimator=clf,n_estimators=500 ,learning_rate=1,algorithm='SAMME')
    clf = SVC()
    # bdt_real = AdaBoostClassifier(base_estimator=clf,n_estimators=50,algorithm='SAMME')
    bdt_real = GradientBoostingClassifier()
    bdt_real.fit(train_x,train_y)

    score = bdt_real.score(test_x,test_y)
    print("GradientBoostingClassifier score = ",score)

    # =====================================================
    '''
    在scikit-learn中，RF的分类类是RandomForestClassifier，回归类是RandomForestRegressor。
    当然RF的变种Extra Trees也有， 分类类ExtraTreesClassifier，
    回归类ExtraTreesRegressor。由于RF和Extra Trees的区别较小，调参方法基本相同
    '''
    from sklearn.ensemble import ExtraTreesClassifier
    forest = ExtraTreesClassifier(n_estimators=200,random_state=0)
    forest.fit(x,y)
    importances = forest.feature_importances_
    std = np.std([ tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    indices = indices[0:100] + 190
    '''
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(indices.shape[0]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(indices.shape[0]), indices)
    plt.xlim([-1, indices.shape[0]])
    plt.show()
    '''
    # 直方图
    plt.figure()
    plt.hist(indices,15)
    plt.show()

