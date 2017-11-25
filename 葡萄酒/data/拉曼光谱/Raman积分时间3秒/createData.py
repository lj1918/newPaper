import pandas as pd
import numpy as np
from matplotlib import pyplot

import os
dirlist  = [u'.\CC长城葡萄酒',
            u'.\WC王朝葡萄酒',
            u'.\ZY张裕葡萄酒']
			
for dirpath in dirlist:    
    filelist = os.listdir(dirpath)   
    # 未校正数据
    data = []
    # 校正过的数据
    data_jiaozheng = []
    for f in filelist:
        f = dirpath +'\\' +f        
        if f.find('jiaozheng') < 0:
            print('未校正数据文件：',f)
            sampleNum = int( f[f.rfind('-') + 1 : f.rfind('-') + 4 ] )
            nm,Abs = np.loadtxt(f,delimiter=',',usecols=(0,1),skiprows=2,unpack=True)
            Abs = np.hstack((Abs,sampleNum))
            data.append(Abs)
        else:
            print('校正过的数据文件：',f)
            sampleNum = int( f[f.rfind('-') + 1 : f.rfind('-') + 4 ] )
            nm,Abs = np.loadtxt(f,delimiter=',',usecols=(0,1),skiprows=2,unpack=True)
            Abs = np.hstack((Abs,sampleNum))
            data_jiaozheng.append(Abs)
    #  保存未校正的数据
    data = np.array(data) 
    filename = u'.' +'\\' +  dirpath[ dirpath.rfind('\\')+1:-1] + '.csv'
    np.savetxt(filename,data,delimiter=',')
    
    # 保存校正过的数据
    data_jiaozheng = np.array(data_jiaozheng)
    jiaozheng_filename = u'.' +'\\' +  dirpath[ dirpath.rfind('\\')+1:-1] + '_jiaozheng.csv'
    np.savetxt(jiaozheng_filename,data,delimiter=',')    

