import pandas as pd
import numpy as np
from matplotlib import pyplot

import os
dirlist  = [u'.\CC长城葡萄酒',
            u'.\WC王朝葡萄酒',
            u'.\ZY张裕葡萄酒']
			
for dirpath in dirlist:    
    filelist = os.listdir(dirpath)    
    data = []
    for f in filelist:
        f = dirpath +'\\' +f
        print(f)
        if f =='temp.py':
            continue
        sampleNum = int( f[f.rfind('-') + 1 : f.rfind('-') + 4 ] )
        nm,Abs = np.loadtxt(f,delimiter=',',usecols=(0,1),skiprows=2,unpack=True)
        Abs = np.hstack((Abs,sampleNum))
        data.append(Abs)
    data = np.array(data) 
    filename = u'.' +'\\' +  dirpath[ dirpath.rfind('\\')+1:-1] + '.csv'
    np.savetxt(filename,data,delimiter=',')

