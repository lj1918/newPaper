# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
import Load_data
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

    # WC-001 – WC-006 王朝干红葡萄酒
    yy = np.arange(240,x.shape[1]+240)
    xx = x[y == 1, :]
    plt.plot(yy,xx[0,:],label='class 1')

    xx = x[y == 2, :]
    plt.plot(yy,xx[0,:],label='class 2')

    xx = x[y == 3, :]
    plt.plot(yy,xx[0,:],label='class 3')

    xx = x[y == 4, :]
    plt.plot(yy,xx[0,:],label='class 4')

    xx = x[y == 5, :]
    plt.plot(yy,xx[0,:],label='class 5')

    xx = x[y == 6, :]
    plt.plot(yy,xx[0,:],label='class 6')

    xx = x[y == 7, :]
    plt.plot(yy,xx[0,:],label='class 7')

    xx = x[y == 8, :]
    plt.plot(yy,xx[0,:],label='class 8')

    xx = x[y == 9, :]
    plt.plot(yy,xx[0,:],label='class 9')

    plt.xlabel('Wavelength/nm',fontsize=16)
    plt.ylabel('Absorbance',fontsize=16)
    plt.legend(loc='best')

    plt.show()

