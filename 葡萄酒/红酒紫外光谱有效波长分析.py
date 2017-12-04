# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt
from Load_data import  *
from matplotlib.font_manager import *
myfont = FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

if __name__ == "__main__":
    ## ======================================================
    # 一、获取数据
    # raw_data = load_ranman_data(isJiaozheng=False)
    raw_data = load_zhiwai_data()
    x = raw_data[:, :-2]
    # 使用分片翻转列的顺序,其中[::-1]代表从后向前取值，每次步进值为1
    x = x[:, ::-1]
    # 紫外光谱测试是测了190到600nm波段
    x = x[:, :]
    y = raw_data[:, -1]

    # 使用决策树进行分析
    forest = ExtraTreesClassifier(n_estimators=200, random_state=0)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    #indices = indices[0:100] + 190

    # 直方图
    plt.figure()
    plt.subplot(311)
    plt.scatter(indices[0:51] + 190,importances[indices[0:51]])
    plt.title("前50名的波长分布:(累计分类重要性=%0.2f%%)"%(np.sum(importances[indices[0:51]]) * 100 ),
              fontproperties=myfont)
    print(np.sum(importances[indices[0:51]]))

    plt.subplot(312)
    plt.scatter(indices[0:101] + 190, importances[indices[0:101]])
    plt.title("前100名的波长分布:(累计分类重要性=%0.2f%%)" % (np.sum(importances[indices[0:101]]) * 100),
              fontproperties=myfont)
    print(np.sum(importances[indices[0:101]]))

    plt.subplot(313)
    plt.scatter(indices[0:151] + 190, importances[indices[0:151]])
    plt.title("前150名的波长分布:(累计分类重要性=%0.2f%%)" % (np.sum(importances[indices[0:151]]) * 100),
              fontproperties=myfont)
    print(np.sum(importances[indices[0:151]]))
    plt.show()