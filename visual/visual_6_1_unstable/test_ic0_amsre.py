from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime, date, timezone, timedelta
import os.path

import basic_file as b_f
import calc_data
import visualize

latlon145_file_name = b_f.latlon145_file_name
latlon900_file_name = b_f.latlon900_file_name
grid900to145_file_name = b_f.grid900to145_file_name
ocean_grid_file = b_f.ocean_grid_file
ocean_grid_145 = b_f.ocean_grid_145
ocean_idx = b_f.ocean_idx

#AMSRE IC0のテスト
"""
filename = "../../data/IC0/P1AME20101222A_600IC0NP.dat"
fp = open(filename,'rb')
ary = np.fromfile(fp, '<h', -1)
fp.close()

ic0 = np.zeros(900**2)
ic0[ary<0] = np.nan
ic0[ary>=0] = ary[ary>=0]

savename = filename[:-4]+ ".csv"
print(savename)
np.savetxt(savename, ic0, delimiter=',')
"""

#SIT_AMSREのテスト
"""
filename = "../../data/SIT_amsr/P1AME20021206A_SITNP.dat"
fp = open(filename,'rb')
ary = np.fromfile(fp, '<h', -1)
fp.close()

sit = np.zeros(900**2)

sit[ary>=10001] = np.nan
sit[ary<0] = np.nan
sit[(ary<10001) & (ary>=0)] = ary[(ary<10001) & (ary>=0)]
#print(set(sit.tolist()))

savename = filename[:-4]+ ".csv"
print(savename)
np.savetxt(savename, sit, delimiter=',')
"""


#SIT_AMSR2のテスト
"""
filename = "../../data/SIT_amsr2/GW1AM220130118A_SITNP.dat"
fp = open(filename,'rb')
ary = np.fromfile(fp, '<h', -1)
fp.close()

sit = np.zeros(900**2)

sit[ary>=10001] = np.nan
sit[ary<0] = np.nan
sit[(ary<10001) & (ary>=0)] = ary[(ary<10001) & (ary>=0)]
#print(set(sit.tolist()))

savename = filename[:-4]+ ".csv"
print(savename)
np.savetxt(savename, sit, delimiter=',')
"""


#landmask_low_NPのテスト

filename = "../../data/landmask_low_NP"
fp = open(filename,'rb')
ary = np.fromfile(fp, '<b', -1)

savename = filename + ".csv"
#np.savetxt(savename, ary, delimiter=',')

#ocean_grid145との比較
data_0 = ary
ocean_0 = np.where(data_0==0.)[0]
data_1 = ocean_grid_145
ocean_1 = np.where(data_1==1.)[0]
print(len(ocean_0))
print(len(ocean_1))




"""
data, ic0_idx_t = calc_data.get_1day_ic0_data(savename, grid900to145_file_name)
visualize.plot_map_once(data, 
	data_type="type_non_wind",
	show=True,
	save_name=None,
	vmax=None,
	vmin=None)
"""

#visualize.plot_ic0_900(savename, None, True)






