from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime, date, timezone, timedelta
import os.path
import os
import seaborn as sns
import matplotlib.gridspec as gridspec

import pandas.plotting._converter as pandacnv
pandacnv.register()

import calc_data


latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()

start_list = []
n = 20000000
y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
for i in y_list:
	m = n + i*10000
	for j in range(12):
		start_list.append(m + (j+1)*100 + 1)
start_ex_list = [20170101, 20170201, 20170301, 20170401, 20170501, 20170601,20170701,20170801]
start_list = np.sort(np.array(list(set(start_list)|set(start_ex_list)))).tolist()
M = len(start_list)
#start_list_plus_1month = start_list + [20170901]



###############################################################################################################

def map_corr():

	dirs = "../result_h/corr_map/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	corr_all = []
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_both_30/Helmert_both_30_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30", "ocean_u", "ocean_v"]]
			print(data.columns)
			accumulate_data.append(np.array(data))
		accumulate_data = np.array(accumulate_data)
		#data_A_ic0 = accumulate_data[:, :, [0,4]]

		corr_list = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 1]
			#data_A = data_A[~np.isnan(data_A)]
			data_ic0 = accumulate_data[:, i, 4]
			tmp_df = pd.DataFrame({"data_A": data_A, "data_ic0": data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
				#print(i, corr)
			corr_list.append(corr)
		corr_all.append(corr_list)

		m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		x1 = np.reshape(x, (145,145), order='F')
		y1 = np.reshape(y, (145,145), order='F')
		dx1 = (x1[1,0]-x1[0,0])/2
		dy1 = (y1[0,1]-y1[0,0])/2
		x2 = np.linspace(x1[0,0], x1[144,0], 145)
		y2 = np.linspace(y1[0,0], y1[0,144], 145)
		xx, yy = np.meshgrid(x2, y2)
		xx, yy = xx.T, yy.T
		xx = np.hstack([xx, xx[:,0].reshape(145,1)])
		xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
		yy = np.vstack([yy, yy[0,:]])
		yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

		data0 = np.ma.masked_invalid(np.array(corr_list))
		data1 = np.reshape(data0, (145,145), order='F')
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=1, vmin=-1)
		m.colorbar(location="bottom")

		save_name = dirs + "ic0_theta_" + month + ".png"
		print(save_name)
		plt.savefig(save_name, dpi=500)
		plt.close()














if __name__ == '__main__':
	map_corr()








