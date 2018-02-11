
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
import random
import itertools
import matplotlib.dates as mdates
from scipy import signal
import statsmodels.api as sm
from pylab import MultipleLocator
from scipy import stats
#from mpl_toolkits.mplot3d import axes3d, Axes3D

import pandas.plotting._converter as pandacnv
pandacnv.register()

import calc_data
import visualize


latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()
no_sensor_data = pd.read_csv("../data/non_sensor_index.csv", header=None)
no_sensor_data.columns = ["non"]


start_list = []
n = 20000000
y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
for i in y_list:
	m = n + i*10000
	for j in range(12):
		start_list.append(m + (j+1)*100 + 1)
M = len(start_list)
#start_list_plus_1month = start_list + [20170901]


cm_angle_2 = visualize.generate_cmap([
	"blue", 
	"gainsboro", 
	"red"
	])


def plot_map_for_thesis(data, save_name, cmap, vmax, vmin):
		fig = plt.figure(figsize=(5, 5))
		m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		x, y = m(np.array(latlon_ex.Lon), np.array(latlon_ex.Lat))
		x1 = np.reshape(x, (145,145), order='F')
		y1 = np.reshape(y, (145,145), order='F')
		dx1 = (x1[1,0]-x1[0,0])/2
		dy1 = (y1[0,1]-y1[0,0])/2
		x2 = np.linspace(x1[0,0], x1[144,0], 145)
		y2 = np.linspace(y1[0,0], y1[0,144], 145)
		xx1, yy1 = np.meshgrid(x2, y2)
		xx, yy = xx1.T, yy1.T
		xx = np.hstack([xx, xx[:,0].reshape(145,1)])
		xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
		yy = np.vstack([yy, yy[0,:]])
		yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

		data = np.ma.masked_invalid(data)
		data1 = np.reshape(data, (145,145), order='F')
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=cmap, vmax=vmax, vmin=vmin)
		#m.contourf(xx1, yy1, data1, cmap='bwr', levels=np.arange(-1,1,10), extend='both')
		m.colorbar(location='bottom')
		plt.tight_layout()
		plt.savefig(save_name, dpi=200)
		plt.close()


###############################################################################################################

def get_A_std():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data_std = []
		for file in file_list:
			data = pd.read_csv(file)
			data_std = np.array(data["A"])
			accumulate_data_std.append(data_std)
		accumulate_data_std = np.array(accumulate_data_std)
		A_std = np.nanstd(accumulate_data_std, axis=0)
		A_count = np.nansum(~np.isnan(accumulate_data_std), axis=0)
		A_std = np.where(A_count>5, A_std, np.nan)
		data_A = pd.DataFrame({"A_std": A_std})
		data_2 = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")
		result = pd.concat([data_A, data_2], axis=1)
		result.to_csv("../data/corr_gw/corr_gw_" + month + ".csv", index=False)

#get_A_std()


def plot_A_std():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		data = np.array(pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")["A_std"])
		save_name = "../result_h_1day_delay/std_map/A_std_" + month + ".png"
		plot_map_for_thesis(data, save_name, cmap=plt.cm.jet, vmax=0.0035, vmin=0)

#plot_A_std()



def plot_data_corr_with_ic0():
	#month_list = ["07", "08", "09"]
	month_list = ["05", "06", "10"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv").dropna()

		fig = plt.figure()
		#ax = Axes3D(fig)
		#ax.scatter3D(np.array(data["ic0_std"]), np.array(data["A_std"]), np.array(data["corr"]))

		#data = data.loc[(data["ic0_std"]>=20)&(data["A_std"]>=0.0025), :].dropna()
		for j, corr_th in enumerate([0.5, 0.6, 0.7, 0.8]):
			data_high = data.loc[(data["corr"]>=corr_th)|(data["corr"]<=-corr_th), :].dropna()
			data_low = data.loc[(data["corr"]<corr_th)&(data["corr"]>-corr_th), :].dropna()
			hl = ["high", "low"]
			for i, data_tmp in enumerate([data_high, data_low]):
				x, y, z = np.array(data_tmp["A_std"]), np.array(data_tmp["ic0_std"]), np.array(data_tmp["corr"])
				plt.scatter(x, y, s=7.5, c=z, cmap='jet', vmax=1, vmin=-1)
				plt.xlim([0.00, 0.005])
				plt.xlabel("A_std")
				plt.ylabel("ic0_std")
				plt.grid()
				plt.colorbar()
				#plt.show()
				plt.savefig("ic0_std_A_std_corr_th_" + str(j+5) + "_" + hl[i] + "_" + month + ".png", dpi=150)
				plt.close()

#plot_data_corr_with_ic0()



def plot_data_corr_map_with_ic0():
	dirs = "../result_h_1day_delay/corr_ic0_search_detail/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	month_list = ["05", "06", "07", "08", "09", "10"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv").dropna()
		"""
		3種類の図
		1
			相関が低い時に閾値を満たす
			相関が低いのに閾値を満たしていない（偶然）
		2
			相関が高い時に閾値を満たしていない
			相関が高いのに閾値を満たす（偶然）
		3
			無相関なのに閾値を満たす
		"""

		"""
		#見栄えを整えるために書き直したもの

		data_corr = np.array(data["corr"])
		save_name_corr = "../result_h_1day_delay/corr_map/ic0_A_" + month + "_thesis.png"
		plot_map_for_thesis(data_corr, save_name_corr, plt.cm.jet, 1, -1)

		data_std = np.array(data["ic0_std"])
		save_name_std = "../result_h_1day_delay/std_map/ic0_std_" + month + "_thesis.png"
		plot_map_for_thesis(data_std, save_name_std, plt.cm.jet, 25, 0)
		"""

		th_ic0 = 15
		th_A = 0.002
		th_corr_list = [0.5, 0.6, 0.7, 0.8]
		for th_corr in th_corr_list:
			grid_1_t = np.array(data[(data["corr"]<=-th_corr)&(data["ic0_std"]>=th_ic0)&(data["A_std"]>=th_A)].index)
			grid_1_f = np.array(data[(data["corr"]<=-th_corr)&((data["ic0_std"]<th_ic0)|(data["A_std"]<th_A))].index)
			grid_2_t = np.array(data[(data["corr"]>=th_corr)&((data["ic0_std"]<th_ic0)|(data["A_std"]<th_A))].index)
			grid_2_f = np.array(data[(data["corr"]>=th_corr)&(data["ic0_std"]>=th_ic0)&(data["A_std"]>=th_A)].index)
			grid_3 = np.array(data[(data["corr"]>-th_corr)&(data["corr"]<th_corr)&(data["ic0_std"]>=th_ic0)&(data["A_std"]>=th_A)].index)
			print("grid_1_t: {}\ngrid_1_f: {}\nggrid_2_t: {}\ngrid_2_f: {}\ngrid_3: {}".format(
				grid_1_t, grid_1_f, grid_2_t, grid_2_f, grid_3))

			m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
			lon = np.array(latlon_ex.Lon)
			lat = np.array(latlon_ex.Lat)
			x, y = m(lon, lat)
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_1_t], y[grid_1_t], marker='o', color="b", s=2, alpha=0.9)
			m.scatter(x[grid_1_f], y[grid_1_f], marker='+', color="r", s=2, alpha=0.9)
			plt.savefig(dirs + "ic0_std_A_std_1_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_2_t], y[grid_2_t], marker='o', color="b", s=2, alpha=0.9)
			m.scatter(x[grid_2_f], y[grid_2_f], marker='+', color="r", s=2, alpha=0.9)
			plt.savefig(dirs + "ic0_std_A_std_2_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_3], y[grid_3], marker='o', color = "r", s=2, alpha=0.9)
			plt.savefig(dirs + "ic0_std_A_std_3_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()

			"""
			for grid in grid_pos:
				plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
			for grid in grid_neg:
				plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
			"""


#plot_data_corr_map_with_ic0()





def get_e2_std():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data_std = []
		for file in file_list:
			data = pd.read_csv(file)
			data_std = np.array(data["epsilon2"])
			accumulate_data_std.append(data_std)
		accumulate_data_std = np.array(accumulate_data_std)
		A_std = np.nanstd(accumulate_data_std, axis=0)
		A_count = np.nansum(~np.isnan(accumulate_data_std), axis=0)
		A_std = np.where(A_count>5, A_std, np.nan)
		data_A = pd.DataFrame({"e2_std": A_std})
		data_2 = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")
		result = pd.concat([data_A, data_2], axis=1)
		result.to_csv("../data/corr_gw/corr_gw_" + month + ".csv", index=False)

#get_e2_std()


def plot_e2_std():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		data = np.array(pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")["e2_std"])
		save_name = "../result_h_1day_delay/std_map/e2_std_" + month + ".png"
		plot_map_for_thesis(data, save_name, cmap=plt.cm.jet, vmax=0.3, vmin=0)

#plot_e2_std()


###############################################################################################################

def get_sit_std():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_*" + month + ".csv"))
		accumulate_data_std = []
		for file in file_list:
			data_std = pd.read_csv(file)
			data_std.loc[data_std["A"].isnull(), "sit_30"] = np.nan
			data_std = np.array(data_std["sit_30"])
			accumulate_data_std.append(data_std)

		accumulate_data_std = np.array(accumulate_data_std)
		sit_std = np.nanstd(accumulate_data_std, axis=0)
		sit_count = np.nansum(~np.isnan(accumulate_data_std), axis=0)
		sit_std = np.where(sit_count>5, sit_std, np.nan)
		corr_df = pd.DataFrame({"sit_std": sit_std})

		data_2 = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")
		result = pd.concat([corr_df, data_2], axis=1)
		result.to_csv("../data/corr_gw/corr_gw_" + month + ".csv", index=False)

#get_sit_std()



def plot_sit_std():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		data = np.array(pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")["sit_std"])
		save_name = "../result_h_1day_delay/std_map/sit_std_" + month + ".png"
		plot_map_for_thesis(data, save_name, cmap=plt.cm.jet, vmax=1000, vmin=0)

#plot_sit_std()



def plot_data_corr_with_sit():
	month_list = ["05", "06", "07", "08", "09", "10"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv").dropna()

		fig = plt.figure()
		#ax = Axes3D(fig)
		#ax.scatter3D(np.array(data["ic0_std"]), np.array(data["A_std"]), np.array(data["corr"]))

		#data = data.loc[(data["ic0_std"]>=20)&(data["A_std"]>=0.0025), :].dropna()
		for j, corr_th in enumerate([0.5, 0.6, 0.7, 0.8]):
			data_high = data.loc[(data["corr"]>=corr_th)|(data["corr"]<=-corr_th), :].dropna()
			data_low = data.loc[(data["corr"]<corr_th)&(data["corr"]>-corr_th), :].dropna()
			hl = ["high", "low"]
			for i, data_tmp in enumerate([data_high, data_low]):
				x, y, z = np.array(data_tmp["A_std"]), np.array(data_tmp["sit_std"]), np.array(data_tmp["corr"])
				plt.scatter(x, y, s=7.5, c=z, cmap='jet', vmax=1, vmin=-1)
				plt.xlim([0.00, 0.005])
				plt.xlabel("A_std")
				plt.ylabel("SIT_std")
				plt.grid()
				plt.colorbar()
				#plt.show()
				plt.savefig("sit_std_A_std_corr_th_" + str(j+5) + "_" + hl[i] + "_" + month + ".png", dpi=150)
				plt.close()

#plot_data_corr_with_sit()



def plot_data_corr_map_with_sit():
	dirs = "../result_h_1day_delay/corr_ic0_search_detail/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	month_list = ["05", "06", "07", "08", "09", "10"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv").dropna()
		"""
		3種類の図
		1
			相関が低い時に閾値を満たす
			相関が低いのに閾値を満たしていない（偶然）
		2
			相関が高い時に閾値を満たしていない
			相関が高いのに閾値を満たす（偶然）
		3
			無相関なのに閾値を満たす
		"""

		"""
		#見栄えを整えるために書き直したもの

		data_corr = np.array(data["corr"])
		save_name_corr = "../result_h_1day_delay/corr_map/ic0_A_" + month + "_thesis.png"
		plot_map_for_thesis(data_corr, save_name_corr, plt.cm.jet, 1, -1)

		data_std = np.array(data["ic0_std"])
		save_name_std = "../result_h_1day_delay/std_map/ic0_std_" + month + "_thesis.png"
		plot_map_for_thesis(data_std, save_name_std, plt.cm.jet, 25, 0)
		"""

		th_sit = 250
		th_A = 0.002
		th_corr_list = [0.5, 0.6, 0.7, 0.8]
		for th_corr in th_corr_list:
			grid_1_t = np.array(data[(data["corr"]<=-th_corr)&(data["sit_std"]>=th_sit)&(data["A_std"]>=th_A)].index)
			grid_1_f = np.array(data[(data["corr"]<=-th_corr)&((data["sit_std"]<th_sit)|(data["A_std"]<th_A))].index)
			grid_2_t = np.array(data[(data["corr"]>=th_corr)&((data["sit_std"]<th_sit)|(data["A_std"]<th_A))].index)
			grid_2_f = np.array(data[(data["corr"]>=th_corr)&(data["sit_std"]>=th_sit)&(data["A_std"]>=th_A)].index)
			grid_3 = np.array(data[(data["corr"]>-th_corr)&(data["corr"]<th_corr)&(data["sit_std"]>=th_sit)&(data["A_std"]>=th_A)].index)
			print("grid_1_t: {}\ngrid_1_f: {}\nggrid_2_t: {}\ngrid_2_f: {}\ngrid_3: {}".format(
				grid_1_t, grid_1_f, grid_2_t, grid_2_f, grid_3))

			m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
			lon = np.array(latlon_ex.Lon)
			lat = np.array(latlon_ex.Lat)
			x, y = m(lon, lat)
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_1_t], y[grid_1_t], marker='o', color="b", s=2, alpha=0.9)
			m.scatter(x[grid_1_f], y[grid_1_f], marker='+', color="r", s=2, alpha=0.9)
			plt.savefig(dirs + "sit_std_A_std_1_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_2_t], y[grid_2_t], marker='o', color="b", s=2, alpha=0.9)
			m.scatter(x[grid_2_f], y[grid_2_f], marker='+', color="r", s=2, alpha=0.9)
			plt.savefig(dirs + "sit_std_A_std_2_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_3], y[grid_3], marker='o', color = "r", s=2, alpha=0.9)
			plt.savefig(dirs + "sit_std_A_std_3_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()

			"""
			for grid in grid_pos:
				plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
			for grid in grid_neg:
				plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
			"""

#plot_data_corr_map_with_sit()



"""
ic0とsitを組み合わせる
"""
def plot_data_corr_map_with_ic0_sit():
	dirs = "../result_h_1day_delay/corr_ic0_search_detail/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	month_list = ["05", "06", "07", "08", "09", "10"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv").dropna()
		"""
		3種類の図
		1
			相関が低い時に閾値を満たす
			相関が低いのに閾値を満たしていない（偶然）
		2
			相関が高い時に閾値を満たしていない
			相関が高いのに閾値を満たす（偶然）
		3
			無相関なのに閾値を満たす
		"""
		th_ic0 = 15
		th_sit = 250
		th_A = 0.002
		th_corr_list = [0.5, 0.6, 0.7, 0.8]
		for th_corr in th_corr_list:
			grid_1_t = np.array(data[(data["corr"]<=-th_corr)&((data["ic0_std"]>=th_ic0)&(data["A_std"]>=th_A))].index)
			grid_1_f = np.array(data[(data["corr"]<=-th_corr)&((data["ic0_std"]<th_ic0)|(data["A_std"]<th_A))].index)
			grid_2_t = np.array(data[(data["corr"]>=th_corr)&((data["ic0_std"]<th_ic0)|(data["A_std"]<th_A))].index)
			grid_2_f = np.array(data[(data["corr"]>=th_corr)&(data["ic0_std"]>=th_ic0)&(data["A_std"]>=th_A)].index)

			grid_1_t = np.array(data[(data["corr"]<=-th_corr)&(data["ic0_std"]>=th_ic0)&(data["sit_std"]>=th_sit)&(data["A_std"]>=th_A)].index)
			grid_1_f = np.array(data[(data["corr"]<=-th_corr)&((data["ic0_std"]<th_ic0)|(data["sit_std"]<th_sit)|(data["A_std"]<th_A))].index)
			grid_2_t = np.array(data[(data["corr"]>=th_corr)&((data["ic0_std"]<th_ic0)|(data["sit_std"]<th_sit)|(data["A_std"]<th_A))].index)
			grid_2_f = np.array(data[(data["corr"]>=th_corr)&(data["ic0_std"]>=th_ic0)&(data["sit_std"]>=th_sit)&(data["A_std"]>=th_A)].index)
			grid_3 = np.array(data[(data["corr"]>-th_corr)&(data["corr"]<th_corr)&(data["ic0_std"]>=th_ic0)&(data["sit_std"]>=th_sit)&(data["A_std"]>=th_A)].index)
			print("grid_1_t: {}\ngrid_1_f: {}\nggrid_2_t: {}\ngrid_2_f: {}\ngrid_3: {}".format(
				grid_1_t, grid_1_f, grid_2_t, grid_2_f, grid_3))

			m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
			lon = np.array(latlon_ex.Lon)
			lat = np.array(latlon_ex.Lat)
			x, y = m(lon, lat)
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_1_t], y[grid_1_t], marker='o', color="b", s=2, alpha=0.9)
			m.scatter(x[grid_1_f], y[grid_1_f], marker='+', color="r", s=2, alpha=0.9)
			plt.savefig(dirs + "both_std_A_std_1_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_2_t], y[grid_2_t], marker='o', color="b", s=2, alpha=0.9)
			m.scatter(x[grid_2_f], y[grid_2_f], marker='+', color="r", s=2, alpha=0.9)
			plt.savefig(dirs + "both_std_A_std_2_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			m.scatter(x[grid_3], y[grid_3], marker='o', color = "r", s=2, alpha=0.9)
			plt.savefig(dirs + "both_std_A_std_3_th_" + str(int(th_corr*10)) + "_" + month + ".png", dpi=200)
			plt.close()

			"""
			for grid in grid_pos:
				plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
			for grid in grid_neg:
				plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
			"""

#plot_data_corr_map_with_ic0_sit()




###############################################################################################################

"""
地上10m風のic0, sit
"""







###############################################################################################################

"""
詳細なトレンド
"""

def ts_all_with_trend(num):
	if num == 1:
		dirs = "../result_h_1day_delay/ts_all_with_trend/"
		if not os.path.exists(dirs):
			os.makedirs(dirs)
	else:
		dirs = "../result_nc_1day_delay/ts_all_with_trend/"
		if not os.path.exists(dirs):
			os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	date_1_6 = []
	for year in y_list:
		date_1_6.append(pd.to_datetime("20"+year+"-01-01"))
		#date_1_6.append(pd.to_datetime("20"+year+"-07-01"))
	date_7_12 = []
	for year in y_list:
		date_7_12.append(pd.to_datetime("20"+year+"-07-01"))

	for area_index in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
	#for area_index in [16]:
		data_A_all_year = []
		data_theta_all_year = []
		data_R2_all_year = []
		data_e2_all_year = []
		for year in y_list:
			for month in month_list:
				print(year + month)
				if num == 1:
					file_list = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
				else:
					file_list = "../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_20" + year + month + ".csv"
				df = pd.read_csv(file_list)
				df.loc[df.R2<0.36, ["A", "theta", "epsilon2"]] = np.nan

				data_A = df.groupby("area_label")["A"].describe()
				data_A.loc[data_A[("count")]<=5, ("mean")] = np.nan
				data_A_all_year.append(data_A.loc[(area_index), "mean"])

				data_theta = df.groupby("area_label")["theta"].describe()
				data_theta.loc[data_theta[("count")]<=5, ("mean")] = np.nan
				data_theta_all_year.append(data_theta.loc[(area_index), "mean"])

				data_e2 = df.groupby("area_label")["epsilon2"].describe()
				data_e2.loc[data_e2[("count")]<=5, ("mean")] = np.nan
				data_e2_all_year.append(data_e2.loc[(area_index), "mean"])

				data_R2 = df.groupby("area_label")["R2"].describe()
				data_R2.loc[data_R2[("count")]<=5, ("mean")] = np.nan
				data_R2_all_year.append(data_R2.loc[(area_index), "mean"])
		
		dates1 = pd.date_range("2003", "2011", freq='MS')[:-1]
		dates2 = pd.date_range("2013", "2017", freq='MS')[:-1]

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(12, 9)
		y1_A = np.array(data_A_all_year[:len(dates1)])
		y2_A = np.array(data_A_all_year[len(dates1):])
		try: 
			yd1_A = signal.detrend(y1_A)
			yd2_A = signal.detrend(y2_A)
			ax.plot(dates1, y1_A-yd1_A, "--b", label="Trend")
			ax.plot(dates2, y2_A-yd2_A, "--b", label="Trend")
		except:
			not_nan_ind_1 = ~np.isnan(y1_A)
			not_nan_ind_2 = ~np.isnan(y2_A)
			slope_1, intersection_1, _, _, _ = stats.linregress(np.arange(1,len(y1_A)+1,1)[not_nan_ind_1], y1_A[not_nan_ind_1])
			slope_2, intersection_2, _, _, _ = stats.linregress(np.arange(1,len(y2_A)+1,1)[not_nan_ind_2], y2_A[not_nan_ind_2])
			detrend_y_1, detrend_y_2 = [], []
			for i in range(len(y1_A)):
				detrend_y_1.append(slope_1*i + intersection_1)
			for i in range(len(y2_A)):
				detrend_y_2.append(slope_2*i + intersection_2)
			ax.plot(dates1, detrend_y_1, "--b", label="Trend")
			ax.plot(dates2, detrend_y_2, "--b", label="Trend")

		ax.plot(dates1, y1_A, '-', color="k")
		ax.plot(dates2, y2_A, '-', color="k")
		ax.set_ylim([0, 0.02])
		ax.set_ylabel('A')
		for item in date_1_6:
			ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
		for item in date_7_12:
			ax.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
		plt.grid(True)
		plt.savefig(dirs + "A_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()
		"""		
		fig, ax = plt.subplots(1, 1)
		fig.figsize=(12, 9)
		ax.plot(dates1, data_theta_all_year[:len(dates1)], '-', color="k")
		ax.plot(dates2, data_theta_all_year[len(dates1):], '-', color="k")
		ax.set_ylim([-60,60])
		ax.set_ylabel(r'$\theta$')
		for item in date_1_6:
			ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
		for item in date_7_12:
			ax.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
		plt.grid(True)
		plt.savefig(dirs + "theta_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(12, 9)
		ax.plot(dates1, data_e2_all_year[:len(dates1)], '-', color="k")
		ax.plot(dates2, data_e2_all_year[len(dates1):], '-', color="k")
		ax.set_ylim([0, 1.3])
		ax.set_ylabel(r'$e^{2}$')
		for item in date_1_6:
			ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
		for item in date_7_12:
			ax.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
		plt.grid(True)
		plt.savefig(dirs + "e2_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()
		"""
		fig, ax = plt.subplots(1, 1)
		fig.figsize=(12, 9)
		y1_A = np.array(data_R2_all_year[:len(dates1)])
		y2_A = np.array(data_R2_all_year[len(dates1):])
		try: 
			yd1_A = signal.detrend(y1_A)
			yd2_A = signal.detrend(y2_A)
			ax.plot(dates1, y1_A-yd1_A, "--b", label="Trend")
			ax.plot(dates2, y2_A-yd2_A, "--b", label="Trend")
		except:
			not_nan_ind_1 = ~np.isnan(y1_A)
			not_nan_ind_2 = ~np.isnan(y2_A)
			slope_1, intersection_1, _, _, _ = stats.linregress(np.arange(1,len(y1_A)+1,1)[not_nan_ind_1], y1_A[not_nan_ind_1])
			slope_2, intersection_2, _, _, _ = stats.linregress(np.arange(1,len(y2_A)+1,1)[not_nan_ind_2], y2_A[not_nan_ind_2])
			detrend_y_1, detrend_y_2 = [], []
			for i in range(len(y1_A)):
				detrend_y_1.append(slope_1*i + intersection_1)
			for i in range(len(y2_A)):
				detrend_y_2.append(slope_2*i + intersection_2)
			ax.plot(dates1, detrend_y_1, "--b", label="Trend")
			ax.plot(dates2, detrend_y_2, "--b", label="Trend")

		ax.plot(dates1, y1_A, '-', color="k")
		ax.plot(dates2, y2_A, '-', color="k")
		ax.set_ylim([0, 1])
		ax.set_ylabel(r'$R^{2}$')
		for item in date_1_6:
			ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
		for item in date_7_12:
			ax.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
		plt.grid(True)
		plt.savefig(dirs + "R2_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()




def ts_all_detail_gw():
	"""
	エリア0, 12, 16
	"""
	dirs = "../result_h_1day_delay/ts_all_with_trend_select/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	date_1_6 = []
	for year in y_list:
		date_1_6.append(pd.to_datetime("20"+year+"-01-01"))
		#date_1_6.append(pd.to_datetime("20"+year+"-07-01"))
	date_7_12 = []
	for year in y_list:
		date_7_12.append(pd.to_datetime("20"+year+"-07-01"))

	for area_index in [0,1,7,12,16]:
		data_A_all_year = []
		data_theta_all_year = []
		data_R2_all_year = []
		data_e2_all_year = []
		for year in y_list:
			for month in month_list:
				#print(year + month)
				file = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
				df = pd.read_csv(file)
				df.loc[df.R2<0.36, ["A", "theta", "epsilon2"]] = np.nan
				data_A = df.groupby("area_label")["A"].describe()
				data_A.loc[data_A[("count")]<=5, ("mean")] = np.nan
				data_A_all_year.append(data_A.loc[(area_index), "mean"])
				data_theta = df.groupby("area_label")["theta"].describe()
				data_theta.loc[data_theta[("count")]<=5, ("mean")] = np.nan
				data_theta_all_year.append(data_theta.loc[(area_index), "mean"])
				data_e2 = df.groupby("area_label")["epsilon2"].describe()
				data_e2.loc[data_e2[("count")]<=5, ("mean")] = np.nan
				data_e2_all_year.append(data_e2.loc[(area_index), "mean"])
				data_R2 = df.groupby("area_label")["R2"].describe()
				data_R2.loc[data_R2[("count")]<=5, ("mean")] = np.nan
				data_R2_all_year.append(data_R2.loc[(area_index), "mean"])
		
		dates1 = pd.date_range("2003", "2011", freq='MS')[:-1]
		dates2 = pd.date_range("2013", "2017", freq='MS')[:-1]
		dates3 = pd.date_range("2003", "2017", freq='MS')[:-1]

		y1_A = np.array(data_A_all_year[:len(dates1)])
		y2_A = np.array(data_A_all_year[len(dates1):])
		y3_A = np.array(data_A_all_year)
		print(area_index)
		try:
			yd1_A = signal.detrend(y1_A)
			yd2_A = signal.detrend(y2_A)
			yd3_A = signal.detrend(y3_A)
			slope_1 = (y1_A-yd1_A)[0]
			slope_2 = (y2_A-yd2_A)[0]
			slope_3 = (y3_A-yd3_A)[0]
			#print(y1_A-yd1_A)
			tmp_1 = y1_A-yd1_A
			tmp_2 = y2_A-yd2_A
			tmp_3 = y3_A-yd3_A
			print(tmp_1[11]-tmp_1[0], tmp_2[11]-tmp_2[0], tmp_3[11]-tmp_3[0])
		except:
			slope_1 = 999
			slope_2 = 999
			slope_3 = 999
		#print(area_index, slope_1, slope_2, slope_3)
		"""
		if area_index == 0 or area_index == 12:
			print("area: " + str(area_index))
			yd1_A = signal.detrend(y1_A)
			yd2_A = signal.detrend(y2_A)
			yd3_A = signal.detrend(y3_A)
			slope_1 = (y1_A-yd1_A)[0]
			slope_2 = (y2_A-yd2_A)[0]
			slope_3 = (y3_A-yd3_A)[0]
			print(slope_1, slope_2, slope_3)

			fig, axes = plt.subplots(4, 1)
			fig.figsize = (12, 9)
			res_1 = sm.tsa.seasonal_decompose(y1_A, freq=12)
			residual_1 = res_1.resid
			seasonal_1 = res_1.seasonal 
			trend_1 = res_1.trend
			res_2 = sm.tsa.seasonal_decompose(y2_A, freq=12)
			residual_2 = res_2.resid
			seasonal_2 = res_2.seasonal 
			trend_2 = res_2.trend
			res_3 = sm.tsa.seasonal_decompose(y3_A, freq=12)
			residual_3 = res_3.resid
			seasonal_3 = res_3.seasonal 
			trend_3 = res_3.trend
			axes[0].plot(dates1, y1_A, "-", color="k")
			axes[0].plot(dates2, y2_A, "-", color="k")
			axes[0].set_ylim([0.005, 0.015])
			if area_index == 12:
				min_index = np.sort(np.argsort(y2_A)[:4])
				min_value = y2_A[min_index]
				min_linear = min_value - signal.detrend(min_value)
				axes[0].plot(dates2[min_index], min_linear, "--c")
				max_index = np.sort(np.argsort(y2_A)[::-1][:4])
				max_value = y2_A[max_index]
				max_linear = max_value - signal.detrend(max_value)
				axes[0].plot(dates2[max_index], max_linear, "--c")
			axes[0].set_ylabel('Observed')
			axes[1].plot(dates1, trend_1, "-", color="k")
			axes[1].plot(dates2, trend_2, "-", color="k")
			axes[1].plot(dates1, y1_A-yd1_A, "--b", label="Trend")
			axes[1].plot(dates2, y2_A-yd2_A, "--b", label="Trend")
			axes[1].set_ylabel('Trend')
			axes[2].plot(dates1, seasonal_1, "-", color="k")
			axes[2].plot(dates2, seasonal_2, "-", color="k")
			axes[2].set_ylabel('Seasonal')
			axes[3].plot(dates1, residual_1, "-", color="k")
			axes[3].plot(dates2, residual_2, "-", color="k")
			axes[3].set_ylabel('Residual')
			for item in date_1_6:
				for axis in axes:
					axis.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			for item in date_7_12:
				for axis in axes:
					axis.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
			axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[0].grid(True)
			axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[1].grid(True)
			axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[2].grid(True)
			axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[3].grid(True)
			plt.tight_layout()
			plt.savefig(dirs + "A_area_" + str(area_index) + "_2periods.png", dpi=150)
			plt.close()

			fig, axes = plt.subplots(4, 1)
			fig.figsize = (12, 9)
			axes[0].plot(dates1, y1_A, "-", color="k")
			axes[0].plot(dates2, y2_A, "-", color="k")
			axes[0].set_ylim([0.005, 0.015])
			axes[0].set_ylabel('Observed')
			axes[1].plot(dates1, trend_3[:len(dates1)], "-", color="k")
			axes[1].plot(dates2, trend_3[len(dates1):], "-", color="k")
			axes[1].plot(dates1, (y3_A-yd3_A)[:len(dates1)], "--b", label="Trend")
			axes[1].plot(dates2, (y3_A-yd3_A)[len(dates1):], "--b", label="Trend")
			axes[1].set_ylabel('Trend')
			axes[2].plot(dates1, seasonal_3[:len(dates1)], "-", color="k")
			axes[2].plot(dates2, seasonal_3[len(dates1):], "-", color="k")
			axes[2].set_ylabel('Seasonal')
			axes[3].plot(dates1, residual_3[:len(dates1)], "-", color="k")
			axes[3].plot(dates2, residual_3[len(dates1):], "-", color="k")
			axes[3].set_ylabel('Residual')
			for item in date_1_6:
				for axis in axes:
					axis.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			for item in date_7_12:
				for axis in axes:
					axis.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
			axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[0].grid(True)
			axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[1].grid(True)
			axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[2].grid(True)
			axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[3].grid(True)
			plt.tight_layout()
			plt.savefig(dirs + "A_area_" + str(area_index) + "_1period.png", dpi=150)
			plt.close()

		else:
			print("area: 16")
			yd1_A = signal.detrend(y1_A)
			yd2_A = signal.detrend(y2_A)
			yd3_A = signal.detrend(y3_A)
			slope_1 = (y1_A-yd1_A)[0]
			slope_2 = (y2_A-yd2_A)[0]
			slope_3 = (y3_A-yd3_A)[0]
			print(slope_1, slope_2, slope_3)

			fig, axes = plt.subplots(4, 1)
			fig.figsize = (12, 9)
			res_1 = sm.tsa.seasonal_decompose(y1_A, freq=12)
			residual_1 = res_1.resid
			seasonal_1 = res_1.seasonal 
			trend_1 = res_1.trend
			res_2 = sm.tsa.seasonal_decompose(y2_A, freq=12)
			residual_2 = res_2.resid
			seasonal_2 = res_2.seasonal 
			trend_2 = res_2.trend
			res_3 = sm.tsa.seasonal_decompose(y3_A, freq=12)
			residual_3 = res_3.resid
			seasonal_3 = res_3.seasonal 
			trend_3 = res_3.trend
			axes[0].plot(dates1, y1_A, "-", color="k")
			axes[0].plot(dates2, y2_A, "-", color="k")
			axes[0].set_ylim([0.005, 0.015])

			axes[0].set_ylabel('Observed')
			axes[1].plot(dates1, trend_1, "-", color="k")
			axes[1].plot(dates2, trend_2, "-", color="k")
			axes[1].plot(dates1, y1_A-yd1_A, "--b", label="Trend")
			axes[1].plot(dates2, y2_A-yd2_A, "--b", label="Trend")
			axes[1].set_ylabel('Trend')
			axes[2].plot(dates1, seasonal_1, "-", color="k")
			axes[2].plot(dates2, seasonal_2, "-", color="k")
			axes[2].set_ylabel('Seasonal')
			axes[3].plot(dates1, residual_1, "-", color="k")
			axes[3].plot(dates2, residual_2, "-", color="k")
			axes[3].set_ylabel('Residual')
			for item in date_1_6:
				for axis in axes:
					axis.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			for item in date_7_12:
				for axis in axes:
					axis.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
			axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[0].grid(True)
			axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[1].grid(True)
			axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[2].grid(True)
			axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[3].grid(True)
			plt.tight_layout()
			plt.savefig(dirs + "A_area_" + str(area_index) + "_2periods.png", dpi=150)
			plt.close()

			fig, axes = plt.subplots(4, 1)
			fig.figsize = (12, 9)
			axes[0].plot(dates1, y1_A, "-", color="k")
			axes[0].plot(dates2, y2_A, "-", color="k")
			axes[0].set_ylim([0.005, 0.015])
			axes[0].set_ylabel('Observed')
			axes[1].plot(dates1, trend_3[:len(dates1)], "-", color="k")
			axes[1].plot(dates2, trend_3[len(dates1):], "-", color="k")
			axes[1].plot(dates1, (y3_A-yd3_A)[:len(dates1)], "--b", label="Trend")
			axes[1].plot(dates2, (y3_A-yd3_A)[len(dates1):], "--b", label="Trend")
			axes[1].set_ylabel('Trend')
			axes[2].plot(dates1, seasonal_3[:len(dates1)], "-", color="k")
			axes[2].plot(dates2, seasonal_3[len(dates1):], "-", color="k")
			axes[2].set_ylabel('Seasonal')
			axes[3].plot(dates1, residual_3[:len(dates1)], "-", color="k")
			axes[3].plot(dates2, residual_3[len(dates1):], "-", color="k")
			axes[3].set_ylabel('Residual')
			for item in date_1_6:
				for axis in axes:
					axis.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			for item in date_7_12:
				for axis in axes:
					axis.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
			axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[0].grid(True)
			axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[1].grid(True)
			axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[2].grid(True)
			axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			axes[3].grid(True)
			plt.tight_layout()
			plt.savefig(dirs + "A_area_" + str(area_index) + "_1period.png", dpi=150)
			plt.close()
		"""

#ts_all_detail_gw()

"""
area: 0
0.00857295658546 0.00992932390264 0.00866077915702
area: 12
0.0086092725633 0.0100837618877 0.00863707468825
area: 16
0.0070019179796 0.00927542668803 0.00740482458103
[Finished in 61.9s]
"""

###############################################################################################################

def thessis_final():
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for im, month in enumerate(month_list):
		#for month in ["07", "08", "09", "10", "11", "12"]:
		print("*************** " + month + " ***************")
		data_A_year, data_theta_year, data_R2_year, data_e2_year = [], [], [], []
		for year in y_list:
			file_list = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
			df = pd.read_csv(file_list)
			data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			for item in ["A", "theta", "R2", "epsilon2"]:
				data.loc[:, (item, "1sigma_pos")] = data.loc[:, (item, "mean")] + data.loc[:, (item, "std")]
				data.loc[:, (item, "1sigma_neg")] = data.loc[:, (item, "mean")] - data.loc[:, (item, "std")]

			data_A = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			data_theta = data.loc[:, ("theta", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			data_R2 = data.loc[:, ("R2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			data_e2 = data.loc[:, ("epsilon2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values

			data_A_year.append(data_A)
			data_theta_year.append(data_theta)
			data_R2_year.append(data_R2)
			data_e2_year.append(data_e2)

		data_A_year = np.array(data_A_year)
		data_theta_year = np.array(data_theta_year)
		data_R2_year = np.array(data_R2_year)
		data_e2_year = np.array(data_e2_year)

		file_by_year = "../data/csv_Helmert_by_year_1day_delay/Helmert_by_year_1day_delay_" + month + ".csv"
		data_by_year = pd.read_csv(file_by_year)
		data_by_year = data_by_year.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

		dates1 = pd.date_range("2003", "2011", freq='YS')[:-1]
		dates2 = pd.date_range("2013", "2017", freq='YS')[:-1]
		N_dates1 = len(dates1)

		#print("1年ごと")
		final_array_1 = np.zeros((17,12))
		final_array_2 = np.zeros((17,12))
		final_array_3 = np.zeros((17,12))
		for i in range(17):
		#for i in [0,1]:
			#print("\tarea: {}".format(i))
			y1_A = data_A_year[:N_dates1,i,1]
			y2_A = data_A_year[N_dates1:,i,1]
			y3_A = data_A_year[:,i,1]
			A_by_year = data_by_year.loc[(i), ("A", "mean")]

			#print(y1_A)
			try:
				yd1_A = signal.detrend(y1_A)
				#print(yd1_A)
				tmp_1 = y1_A-yd1_A
				#print(tmp_1)
				v1 = tmp_1[1]-tmp_1[0]
			except:
				v1 = -1
			try:
				yd2_A = signal.detrend(y2_A)
				tmp_2 = y2_A-yd2_A
				v2 = tmp_2[1]-tmp_2[0]
			except:
				v2 = -1
			try:
				yd3_A = signal.detrend(y3_A)
				tmp_3 = y3_A-yd3_A
				v3 = tmp_3[1]-tmp_3[0]
			except:
				v3 = -1

			#final_array_1[i, im] = v1
			#final_array_2[i, im] = v2
			#final_array_3[i, im] = v3
			print(i, v1, v2)

	#print (final_array_1)
	#print (final_array_2)
	#print (final_array_3)
	#np.savetxt("amsr-e.csv", final_array_1)
	#np.savetxt("amsr-2.csv", final_array_2)
	#np.savetxt("amsr-all.csv", final_array_3)

thessis_final()


















