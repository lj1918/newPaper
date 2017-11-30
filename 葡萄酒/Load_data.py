# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

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
