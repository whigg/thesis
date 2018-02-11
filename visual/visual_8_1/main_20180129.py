
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

cm_p_value = visualize.generate_cmap([
	"chocolate", 
	"palegreen"
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

def plot_corr_p_value():
	def plot_tmp(ary, save_name, bar, vmax, vmin, cm):
		data1 = np.reshape(np.ma.masked_invalid(ary), (145,145), order='F')
		m.drawcoastlines(color='0.15')
		m.fillcontinents(color='#555555')
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=cm, vmax=vmax, vmin=vmin)
		if bar:
			m.colorbar(location='bottom')
		plt.savefig(save_name, dpi=200)
		plt.close()

	dirs_csv = "../data/corr_gw_final/"
	if not os.path.exists(dirs_csv):
		os.makedirs(dirs_csv)
	dirs = "../result_h_1day_delay/corr_map/"

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data_corr = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30"]]
			accumulate_data.append(np.array(data_corr))
		accumulate_data = np.array(accumulate_data)

		corr_list_ic0_p = []
		corr_list_ic0_r = []
		corr_list_sit_p = []
		corr_list_sit_r = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 0]
			data_ic0 = accumulate_data[:, i, 4]
			data_sit = accumulate_data[:, i, 5]
			tmp_df_ic0 = pd.DataFrame({"data_A": data_A, "data_ic0": data_ic0})
			tmp_df_sit = pd.DataFrame({"data_A": data_A, "data_sit": data_sit})
			if len(tmp_df_ic0.dropna()) <= 5:
				r = np.nan
				p = np.nan
			else:
				corr_ic0 = np.array(tmp_df_ic0.dropna())
				r, p = stats.pearsonr(corr_ic0[:,1], corr_ic0[:,0])
			corr_list_ic0_p.append(p)
			corr_list_ic0_r.append(r)

			if len(tmp_df_sit.dropna()) <= 5:
				r = np.nan
				p = np.nan
			else:
				corr_sit = np.array(tmp_df_sit.dropna())
				r, p = stats.pearsonr(corr_sit[:,1], corr_sit[:,0])
			corr_list_sit_p.append(p)
			corr_list_sit_r.append(r)

		corr_array_ic0_p = np.array(corr_list_ic0_p)
		corr_array_ic0_r = np.array(corr_list_ic0_r)
		corr_array_sit_p = np.array(corr_list_sit_p)
		corr_array_sit_r = np.array(corr_list_sit_r)

		corr_df = pd.DataFrame({
			"corr_ic0_p": corr_array_ic0_p, 
			"corr_sit_p": corr_array_sit_p, 
			"corr_ic0_r": corr_array_ic0_r, 
			"corr_sit_r": corr_array_sit_r})

		data_2 = pd.read_csv("../data/corr_gw/corr_gw_" + month + ".csv")
		result = pd.concat([corr_df, data_2.loc[:, ["ic0_std", "sit_std", "e2_std", "A_std"]]], axis=1)
		result.to_csv(dirs_csv + "corr_gw_" + month + ".csv", index=False)

		m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
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

		print("plotting...")
		corr_array_ic0_p[corr_array_ic0_p>=0.05] = 1
		corr_array_ic0_p[corr_array_ic0_p<0.05] = 0
		save_name_ic0 = dirs + "corr_map_ic0_p_" + month + ".png"
		plot_tmp(ary=corr_array_ic0_p, save_name=save_name_ic0, bar=False, vmax=None, vmin=None, cm=cm_p_value)
		
		corr_array_sit_p[corr_array_sit_p>=0.05] = 1
		corr_array_sit_p[corr_array_sit_p<0.05] = 0
		save_name_sit = dirs + "corr_map_sit_p_" + month + ".png"
		plot_tmp(ary=corr_array_sit_p, save_name=save_name_sit, bar=False, vmax=None, vmin=None, cm=cm_p_value)

		save_name_ic0_r = dirs + "corr_map_ic0_r_" + month + ".png"
		save_name_sit_r = dirs + "corr_map_sit_r_" + month + ".png"
		plot_tmp(ary=corr_array_ic0_r, save_name=save_name_ic0_r, bar=True, vmax=1, vmin=-1, cm='jet')
		plot_tmp(ary=corr_array_sit_r, save_name=save_name_sit_r, bar=True, vmax=1, vmin=-1, cm='jet')


#plot_corr_p_value()



def scatter_corr():
	def scatter_tmp(x, y, save_name, xlabel, ylabel):
		plt.scatter(x, y, s=7.5)
		plt.xlim([0, 0.05])
		plt.ylim([-1, 1])
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid()
		plt.savefig(save_name, dpi=150)
		plt.close()

	dirs_ic0 = "../result_h_1day_delay/scatter_ic0/"
	dirs_sit = "../result_h_1day_delay/scatter_sit/"
	dirs_both = "../result_h_1day_delay/scatter_both/"
	if not os.path.exists(dirs_ic0):
		os.makedirs(dirs_ic0)
	if not os.path.exists(dirs_sit):
		os.makedirs(dirs_sit)
	if not os.path.exists(dirs_both):
		os.makedirs(dirs_both)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw_final/corr_gw_" + month + ".csv")
		data_ic0 = data.loc[data["corr_ic0_p"]<0.05, :].dropna()
		data_sit = data.loc[data["corr_sit_p"]<0.05, :].dropna()
		data_both = data.loc[(data["corr_ic0_p"]<0.05)&(data["corr_sit_p"]<0.05), :].dropna()
		"""
		x1, y1 = np.array(data_ic0["corr_ic0_p"]), np.array(data_ic0["corr_ic0_r"])
		save_name_1 = dirs_ic0 + "p_r_" + month + ".png"
		scatter_tmp(x1, y1, save_name_1, "SIC p_value", "SIC Corr.")
		print(np.sum(y1>0), len(y1))
		x2, y2 = np.array(data_sit["corr_sit_p"]), np.array(data_sit["corr_sit_r"])
		save_name_2 = dirs_sit + "p_r_" + month + ".png"
		scatter_tmp(x2, y2, save_name_2, "SIT p_value", "SIT Corr.")
		print(np.sum(y2>0), len(y2))

		x3, y3, z3 = np.array(data_both["corr_ic0_p"]), np.array(data_both["corr_sit_p"]), np.array(data_both["corr_ic0_r"])
		save_name_3 = dirs_both + "p_p_r_ic0_" + month + ".png"
		plt.scatter(x3, y3, s=7.5, c=z3, cmap='jet', vmax=1, vmin=-1)
		plt.xlim([0, 0.05])
		plt.ylim([0, 0.05])
		plt.xlabel("SIC p_value")
		plt.ylabel("SIT p_value")
		plt.grid(True)
		plt.colorbar()
		plt.savefig(save_name_3)
		plt.close()
		print(np.sum(z3>0), len(z3))
		x4, y4, z4 = np.array(data_both["corr_ic0_p"]), np.array(data_both["corr_sit_p"]), np.array(data_both["corr_sit_r"])
		save_name_4 = dirs_both + "p_p_r_sit_" + month + ".png"
		plt.scatter(x4, y4, s=7.5, c=z4, cmap='jet', vmax=1, vmin=-1)
		plt.xlim([0, 0.05])
		plt.ylim([0, 0.05])
		plt.xlabel("SIC p_value")
		plt.ylabel("SIT p_value")
		plt.grid(True)
		plt.colorbar()
		plt.savefig(save_name_4)
		plt.close()
		print(np.sum(z4>0), len(z4))
		print("\n")
		"""

		m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		"""
		data_5_pos = data_ic0.loc[data_ic0["corr_ic0_r"]>=0, ["A_std", "ic0_std", "corr_ic0_r"]]
		data_5_neg = data_ic0.loc[data_ic0["corr_ic0_r"]<0, ["A_std", "ic0_std", "corr_ic0_r"]]
		x5_pos, y5_pos, z5_pos = np.array(data_5_pos["A_std"]), np.array(data_5_pos["ic0_std"]), np.array(data_5_pos["corr_ic0_r"])
		x5_neg, y5_neg, z5_neg = np.array(data_5_neg["A_std"]), np.array(data_5_neg["ic0_std"]), np.array(data_5_neg["corr_ic0_r"])
		save_name_5 = dirs_ic0 + "std_std_r_" + month + ".png"
		plt.scatter(x5_pos, y5_pos, marker="+", s=20, color="orangered")
		plt.scatter(x5_neg, y5_neg, marker=".", s=15, color="royalblue")
		plt.xlim([0, 0.005])
		plt.ylim([0, 40])
		plt.xlabel("A std.")
		plt.ylabel("SIC std.")
		plt.grid(True)
		plt.savefig(save_name_5)
		plt.close()
		"""
		idx_pos_ic0 = np.array(data.loc[(data["A_std"]>=0.002)&(data["ic0_std"]>=10)&(data["corr_ic0_p"]<0.05)&(data["corr_ic0_r"]>=0), :].index)
		idx_neg_ic0 = np.array(data.loc[(data["A_std"]>=0.002)&(data["ic0_std"]>=10)&(data["corr_ic0_p"]<0.05)&(data["corr_ic0_r"]<0), :].index)
		idx_th_ic0 = np.array(data.loc[((data["A_std"]<0.002)|(data["ic0_std"]<10))&(data["corr_ic0_p"]<0.05)&(data["corr_ic0_r"]<0), :].index)
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		#m.drawparallels(np.arange(80.,101.,10.))
		m.scatter(x[idx_pos_ic0], y[idx_pos_ic0], marker='o', color="orangered", s=6, alpha=0.9)
		m.scatter(x[idx_neg_ic0], y[idx_neg_ic0], marker='+', color="royalblue", s=6, alpha=0.9)
		m.scatter(x[idx_th_ic0], y[idx_th_ic0], marker='x', color="lime", s=3, alpha=0.9)
		plt.tight_layout()
		plt.savefig("../result_h_1day_delay/corr_ic0_search_detail/std_std_r_" + month + ".png", dpi=150)
		plt.close()
		"""
		data_6_pos = data_sit.loc[data_sit["corr_sit_r"]>=0, ["A_std", "sit_std", "corr_sit_r"]]
		data_6_neg = data_sit.loc[data_sit["corr_sit_r"]<0, ["A_std", "sit_std", "corr_sit_r"]]
		x6_pos, y6_pos, z6_pos = np.array(data_6_pos["A_std"]), np.array(data_6_pos["sit_std"]), np.array(data_6_pos["corr_sit_r"])
		x6_neg, y6_neg, z6_neg = np.array(data_6_neg["A_std"]), np.array(data_6_neg["sit_std"]), np.array(data_6_neg["corr_sit_r"])
		save_name_6 = dirs_sit + "std_std_r_" + month + ".png"
		plt.scatter(x6_pos, y6_pos, marker="+", s=20, color="orangered")
		plt.scatter(x6_neg, y6_neg, marker=".", s=15, color="royalblue")
		plt.xlim([0, 0.005])
		plt.ylim([0, 1000])
		plt.xlabel("A std.")
		plt.ylabel("SIT std.")
		plt.grid(True)
		plt.savefig(save_name_6)
		plt.close()
		"""
		idx_pos_sit = np.array(data.loc[(data["A_std"]>=0.002)&(data["sit_std"]>=200)&(data["corr_sit_p"]<0.05)&(data["corr_sit_r"]>=0), :].index)
		idx_neg_sit = np.array(data.loc[(data["A_std"]>=0.002)&(data["sit_std"]>=200)&(data["corr_sit_p"]<0.05)&(data["corr_sit_r"]<0), :].index)
		idx_th_sit = np.array(data.loc[((data["A_std"]<0.002)|(data["sit_std"]<200))&(data["corr_sit_p"]<0.05)&(data["corr_sit_r"]<0), :].index)
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		#m.drawparallels(np.arange(80.,101.,10.))
		m.scatter(x[idx_pos_sit], y[idx_pos_sit], marker='o', color="orangered", s=6, alpha=0.9)
		m.scatter(x[idx_neg_sit], y[idx_neg_sit], marker='+', color="royalblue", s=6, alpha=0.9)
		m.scatter(x[idx_th_sit], y[idx_th_sit], marker='x', color="lime", s=3, alpha=0.9)
		plt.tight_layout()
		plt.savefig("../result_h_1day_delay/corr_sit_search_detail/std_std_r_" + month + ".png", dpi=150)
		plt.close()
		

#scatter_corr()




def print_corr():
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		data = pd.read_csv("../data/corr_gw_final/corr_gw_" + month + ".csv")
		print(len(data.dropna()))


print_corr()











