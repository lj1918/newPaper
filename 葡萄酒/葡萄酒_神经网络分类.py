# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
import Load_data
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

if __name__ == '__main__':
    ## ======================================================
    # 一、获取数据
    # raw_data = Load_data.load_ranman_data(isJiaozheng=False)
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
    '''
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)
    '''
    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.5, random_state=0)

    clf =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(12), random_state=1)
    clf.fit(train_x,train_y)

    score = clf.score(test_x,test_y)

    print(score)



