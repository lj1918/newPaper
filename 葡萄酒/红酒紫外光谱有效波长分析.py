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
    # 直方图无法显示出各个波长的重要性
    # hist(x)：这种格式中x是一维向量，函数的作用是将x中的最小和最大值之间的区间等分成10等份，
    # 输出的直方图横坐标是x值，纵坐标是该值的个数可以理解为下面中的hist(x,10)。
#    plt.figure()
#    plt.hist(importances[indices[:]],100)
#    plt.xticks( (indices[::30]),(indices[::30]+190) )
#    plt.show()

#    # bar图
#    # 对于bar函数的使用一般格式如下：bar(x,y) 其中x必须是严格递增的且一维向量x和一维向量y长度相同。
#    # 以一维向量x的值为x坐标，对应的y为y坐标画直方图。
#    plt.figure()
#    barIndices = np.argsort(indices) # 按照波长进行排序
#    plt.bar( indices[barIndices]+190, importances[barIndices] )
#    #plt.plot(indices[barIndices]+190,importances[barIndices],'ro')
#    plt.plot(indices[barIndices]+190,importances[barIndices],'r-')
#    plt.show()
    
    plt.figure()
    #
    plt.subplot(311)
    plt.title('全部波长对分类的重要度',fontproperties=myfont)
    barIndices = np.argsort(indices) # 按照波长进行排序
    plt.scatter( indices[barIndices] +190, importances[barIndices] )    
    plt.ylim((0,0.01))
    plt.xlim((190,610))
    
    plt.subplot(312)
    plt.title('对分类的重要度大于0.002的波长及其重要度',fontproperties=myfont)
    plt.plot( indices[barIndices][ importances[barIndices] > 0.002] +190, 
             importances[ importances[barIndices] > 0.002])
    plt.ylim((0,0.01))
    plt.xlim((190,610))
    
    plt.subplot(313)
    plt.title('对分类的重要度在前50名的波长及其重要度',fontproperties=myfont)
    plt.scatter( indices[:51]  +190, importances[indices[:51] ])
    plt.ylim((0,0.01))
    plt.xlim((190,610))
    
    #plt.plot(indices[barIndices]+190,importances[barIndices],'ro')
    #plt.plot( indices[barIndices][ importances[barIndices] > 0.002] +190, importances[ importances[barIndices] > 0.002],'r-')
    plt.show()

    
    # 散点图
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
    