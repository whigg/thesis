"""
厚さとの相関
ic0の階級別データ
ts_all
ts_by_month 色分け
確率マップ
"""
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

cm_ocean_current = visualize.generate_cmap([
	"white",
	"red",
	"yellow", 
	"limegreen", 
	"deepskyblue",
	"indigo"
	])


###############################################################################################################

def plot_data_corr_sit_1day_delay():
	dirs_corr_map = "../result_h_1day_delay/corr_map/"
	dirs_corr_map_search_grid = "../result_h_1day_delay/corr_map_search_grid/"

	if not os.path.exists(dirs_corr_map):
		os.makedirs(dirs_corr_map)
	if not os.path.exists(dirs_corr_map_search_grid):
		os.makedirs(dirs_corr_map_search_grid)

	data_ex_dir = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30"]]
			print(data.columns)
			print(data.dropna().head(2))
			print(np.array(data.dropna())[0:2,:])
			accumulate_data.append(np.array(data))
		accumulate_data = np.array(accumulate_data)
		#data_A_ic0 = accumulate_data[:, :, [0,4]]

		corr_list = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 0]
			#data_A = data_A[~np.isnan(data_A)]
			data_ic0 = accumulate_data[:, i, 5]
			tmp_df = pd.DataFrame({"data_A": data_A, "data_sit": data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
				#print(i, corr)
			corr_list.append(corr)

		save_name_corr = dirs_corr_map + "sit_A_" + month + ".png"
		visualize.plot_map_once(corr_list, data_type="type_non_wind", show=False, 
			save_name=save_name_corr, vmax=1, vmin=-1, cmap=cm_angle_2)
		"""
		df_corr = pd.DataFrame({"corr": corr_list})
		df_corr = pd.concat([latlon_ex, df_corr, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
		corr_grid_pos = df_corr.loc[(df_corr["corr"]>=0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		corr_grid_neg = df_corr.loc[(df_corr["corr"]<=-0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		try:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 15)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 15)
		except:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 5)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 5)
		for grid in plot_grids_pos:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 5]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "sit_A_pos_grid_" + month + "_"  + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()
		for grid in plot_grids_neg:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 5]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "sit_A_neg_grid_" + month + "_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()

		m = Basemap(lon_0=180, boundinglat=65, resolution='l', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		m.scatter(x[plot_grids_pos], y[plot_grids_pos], marker='o', color = "r", s=2, alpha=0.9)
		m.scatter(x[plot_grids_neg], y[plot_grids_neg], marker='o', color = "b", s=2, alpha=0.9)
		for grid in plot_grids_pos:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
		for grid in plot_grids_neg:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
		plt.savefig(dirs_corr_map_search_grid + "sit_A_grid_info_" + month + ".png", dpi=200)
		plt.close()
		"""







def plot_nc_data_corr_sit_1day_delay():
	dirs_corr_map = "../result_nc_1day_delay/corr_map/"
	dirs_corr_map_search_grid = "../result_nc_1day_delay/corr_map_search_grid/"

	if not os.path.exists(dirs_corr_map):
		os.makedirs(dirs_corr_map)
	if not os.path.exists(dirs_corr_map_search_grid):
		os.makedirs(dirs_corr_map_search_grid)

	data_ex_dir = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30"]]
			print(data.columns)
			print(data.dropna().head(2))
			print(np.array(data.dropna())[0:2,:])
			accumulate_data.append(np.array(data))
		accumulate_data = np.array(accumulate_data)
		#data_A_ic0 = accumulate_data[:, :, [0,4]]

		corr_list = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 0]
			#data_A = data_A[~np.isnan(data_A)]
			data_ic0 = accumulate_data[:, i, 5]
			tmp_df = pd.DataFrame({"data_A": data_A, "data_ic0": data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
				#print(i, corr)
			corr_list.append(corr)

		save_name_corr = dirs_corr_map + "sit_A_" + month + ".png"
		visualize.plot_map_once(corr_list, data_type="type_non_wind", show=False, 
			save_name=save_name_corr, vmax=1, vmin=-1, cmap=cm_angle_2)
		"""
		df_corr = pd.DataFrame({"corr": corr_list})
		df_corr = pd.concat([latlon_ex, df_corr, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
		corr_grid_pos = df_corr.loc[(df_corr["corr"]>=0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		corr_grid_neg = df_corr.loc[(df_corr["corr"]<=-0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		try:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 15)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 15)
		except:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 5)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 5)
		for grid in plot_grids_pos:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 5]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "sit_A_pos_grid_" + month + "_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()
		for grid in plot_grids_neg:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 5]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "sit_A_neg_grid_" + month + "_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()

		m = Basemap(lon_0=180, boundinglat=65, resolution='l', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		m.scatter(x[plot_grids_pos], y[plot_grids_pos], marker='o', color = "r", s=2, alpha=0.9)
		m.scatter(x[plot_grids_neg], y[plot_grids_neg], marker='o', color = "b", s=2, alpha=0.9)
		for grid in plot_grids_pos:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
		for grid in plot_grids_neg:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
		plt.savefig(dirs_corr_map_search_grid + "sit_A_grid_info_" + month + ".png", dpi=200)
		plt.close()
		"""




###############################################################################################################

def search_data_corr_ic0_1day_delay():
	dirs_corr_map = "../result_h_1day_delay/corr_map/"
	dirs_corr_map_search_grid = "../result_h_1day_delay/corr_map_search_grid/"

	data_ex_dir = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30"]]
			print(data.columns)
			print(data.dropna().head(2))
			print(np.array(data.dropna())[0:2,:])
			accumulate_data.append(np.array(data))
		accumulate_data = np.array(accumulate_data)
		#data_A_ic0 = accumulate_data[:, :, [0,4]]

		corr_list = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 0]
			#data_A = data_A[~np.isnan(data_A)]
			data_ic0 = accumulate_data[:, i, 4]
			tmp_df = pd.DataFrame({"data_A": data_A, "data_sit": data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
				#print(i, corr)
			corr_list.append(corr)

		df_corr = pd.DataFrame({"corr": corr_list})
		df_corr = pd.concat([latlon_ex, df_corr, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
		corr_grid_pos = df_corr.loc[(df_corr["corr"]>=0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		corr_grid_neg = df_corr.loc[(df_corr["corr"]<=-0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		try:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 15)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 15)
		except:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 5)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 5)
		for grid in plot_grids_pos:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "ic0_A_pos_grid_" + month + "_"  + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()
		for grid in plot_grids_neg:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "ic0_A_neg_grid_" + month + "_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()

		m = Basemap(lon_0=180, boundinglat=65, resolution='l', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		m.scatter(x[plot_grids_pos], y[plot_grids_pos], marker='o', color = "r", s=2, alpha=0.9)
		m.scatter(x[plot_grids_neg], y[plot_grids_neg], marker='o', color = "b", s=2, alpha=0.9)
		for grid in plot_grids_pos:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
		for grid in plot_grids_neg:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
		plt.savefig(dirs_corr_map_search_grid + "ic0_A_grid_info_" + month + ".png", dpi=200)
		plt.close()






def search_nc_data_corr_ic0_1day_delay():
	dirs_corr_map = "../result_nc_1day_delay/corr_map/"
	dirs_corr_map_search_grid = "../result_nc_1day_delay/corr_map_search_grid/"

	data_ex_dir = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30"]]
			print(data.columns)
			print(data.dropna().head(2))
			print(np.array(data.dropna())[0:2,:])
			accumulate_data.append(np.array(data))
		accumulate_data = np.array(accumulate_data)
		#data_A_ic0 = accumulate_data[:, :, [0,4]]

		corr_list = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 0]
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

		df_corr = pd.DataFrame({"corr": corr_list})
		df_corr = pd.concat([latlon_ex, df_corr, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
		corr_grid_pos = df_corr.loc[(df_corr["corr"]>=0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		corr_grid_neg = df_corr.loc[(df_corr["corr"]<=-0.7)&(df_corr["area_label"].isin([0,1,4,5,7,8,10,12,16])), :].dropna().index
		try:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 15)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 15)
		except:
			plot_grids_pos = random.sample(np.array(corr_grid_pos).tolist(), 5)
			plot_grids_neg = random.sample(np.array(corr_grid_neg).tolist(), 5)
		for grid in plot_grids_pos:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "ic0_A_pos_grid_" + month + "_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()
		for grid in plot_grids_neg:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "ic0_A_neg_grid_" + month + "_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()

		m = Basemap(lon_0=180, boundinglat=65, resolution='l', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		m.scatter(x[plot_grids_pos], y[plot_grids_pos], marker='o', color = "r", s=2, alpha=0.9)
		m.scatter(x[plot_grids_neg], y[plot_grids_neg], marker='o', color = "b", s=2, alpha=0.9)
		for grid in plot_grids_pos:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='r')
		for grid in plot_grids_neg:
			plt.annotate(str(grid), xy=(x[grid], y[grid]), xycoords='data', xytext=(x[grid], y[grid]), textcoords='data', color='b')
		plt.savefig(dirs_corr_map_search_grid + "ic0_A_grid_info_" + month + ".png", dpi=200)
		plt.close()



###############################################################################################################

def ts_30_by_month_modified_gw():
	dirs = "../result_h_1day_delay/ts_by_month_ver1/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	data_A_year_list = []
	data_theta_year_list = []
	data_R2_year_list = []
	data_e2_year_list = []
	for year in y_list:
		data_A_month, data_theta_month, data_R2_month, data_e2_month = [], [], [], []
		for month in month_list:
			print(year + month)
			file_list = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
			df = pd.read_csv(file_list)
			data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			data_A = data.loc[:, ("A", ["mean", "count"])].values
			data_theta = data.loc[:, ("theta", ["mean", "count"])].values
			data_R2 = data.loc[:, ("R2", ["mean", "count"])].values
			data_e2 = data.loc[:, ("epsilon2", ["mean", "count"])].values
			data_A_month.append(data_A)
			data_theta_month.append(data_theta)
			data_R2_month.append(data_R2)
			data_e2_month.append(data_e2)

		data_A_month = np.array(data_A_month)
		data_theta_month = np.array(data_theta_month)
		data_R2_month = np.array(data_R2_month)
		data_e2_month = np.array(data_e2_month)

		data_A_year_list.append(data_A_month)
		data_theta_year_list.append(data_theta_month)
		data_R2_year_list.append(data_R2_month)
		data_e2_year_list.append(data_e2_month)

	len_y = len(y_list)
	for i in range(18):
		dates = pd.date_range("2001", periods=12, freq='MS')

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_A_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		#ax.legend(loc="center", bbox_to_anchor=(1.1, 0.5))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.7)
		ax.set_ylim([0, 0.02])
		ax.set_ylabel('A', fontsize=18)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.tick_params(labelsize=18)
		plt.grid(True)
		plt.savefig(dirs + "A_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_theta_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.7)
		ax.set_ylim([-50, 50])
		ax.set_ylabel(r'$\theta$', fontsize=18)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.tick_params(labelsize=18)
		plt.grid(True)
		plt.savefig(dirs + "theta_area_" + str(i) + ".png", dpi=150)
		plt.close()
		"""
		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_R2_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([0, 1])
		ax.set_ylabel(r'$R^{2}$')
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.grid(True)
		plt.savefig(dirs + "R2_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_e2_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([0, 1.5])
		ax.set_ylabel(r'$e^{2}$')
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.grid(True)
		plt.savefig(dirs + "e2_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			y = data_A_year_list[j][:,i,0]
			ax.plot(dates, y, '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.3, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([0, max(data_A_year_list[j][:,i,0])+10])
		ax.set_ylabel("number of data")
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.grid(True)
		plt.savefig(dirs + "count_area_" + str(i) + ".png", dpi=150)
		plt.close()
		"""







def ts_30_by_month_modified_nc():
	dirs = "../result_nc_1day_delay/ts_by_month_ver1/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	data_A_year_list = []
	data_theta_year_list = []
	data_R2_year_list = []
	data_e2_year_list = []
	for year in y_list:
		data_A_month, data_theta_month, data_R2_month, data_e2_month = [], [], [], []
		for month in month_list:
			print(year + month)
			file_list = "../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_20" + year + month + ".csv"
			df = pd.read_csv(file_list)
			data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			data_A = data.loc[:, ("A", ["mean", "count"])].values
			data_theta = data.loc[:, ("theta", ["mean", "count"])].values
			data_R2 = data.loc[:, ("R2", ["mean", "count"])].values
			data_e2 = data.loc[:, ("epsilon2", ["mean", "count"])].values
			data_A_month.append(data_A)
			data_theta_month.append(data_theta)
			data_R2_month.append(data_R2)
			data_e2_month.append(data_e2)

		data_A_month = np.array(data_A_month)
		data_theta_month = np.array(data_theta_month)
		data_R2_month = np.array(data_R2_month)
		data_e2_month = np.array(data_e2_month)

		data_A_year_list.append(data_A_month)
		data_theta_year_list.append(data_theta_month)
		data_R2_year_list.append(data_R2_month)
		data_e2_year_list.append(data_e2_month)

	len_y = len(y_list)
	for i in range(18):
		dates = pd.date_range("2001", periods=12, freq='MS')

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_A_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		#ax.legend(loc="center", bbox_to_anchor=(1.1, 0.5))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([0, 0.02])
		ax.set_ylabel('A', fontsize=18)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.tick_params(labelsize=18)
		plt.grid(True)
		plt.savefig(dirs + "A_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_theta_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([-60, 60])
		ax.set_ylabel(r'$\theta$', fontsize=18)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.tick_params(labelsize=18)
		plt.grid(True)
		plt.savefig(dirs + "theta_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_R2_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([0, 1])
		ax.set_ylabel(r'$R^{2}$')
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.grid(True)
		plt.savefig(dirs + "R2_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			ax.plot(dates, data_e2_year_list[j][:,i,1], '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylim([0, 1.5])
		ax.set_ylabel(r'$e^{2}$')
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.grid(True)
		plt.savefig(dirs + "e2_area_" + str(i) + ".png", dpi=150)
		plt.close()

		fig, ax = plt.subplots(1, 1)
		fig.figsize=(15, 6)
		for j in range(len_y):
			y = data_A_year_list[j][:,i,0]
			ax.plot(dates, y, '-', label = "20" + y_list[j], color=plt.cm.hsv(j/len_y))
		ax.legend(bbox_to_anchor=(1.08, 0.8))
		plt.subplots_adjust(right=0.75)
		ax.set_ylabel("number of data")
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.grid(True)
		plt.savefig(dirs + "count_area_" + str(i) + ".png", dpi=150)
		plt.close()




###############################################################################################################

def ts_by_month_all_year_gw_and_nc(num):
	if num == 1:
		dirs = "../result_h_1day_delay/ts_all/"
		if not os.path.exists(dirs):
			os.makedirs(dirs)
	else:
		dirs = "../result_nc_1day_delay/ts_all/"
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

	for area_index in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
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
		ax.plot(dates1, data_A_all_year[:len(dates1)], '-', color="k")
		ax.plot(dates2, data_A_all_year[len(dates1):], '-', color="k")
		ax.set_ylim([0, 0.02])
		ax.set_ylabel('A')
		for item in date_1_6:
			ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
		for item in date_7_12:
			ax.axvline(item, color='palegreen', linestyle='-', lw=0.75, alpha=0.75)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
		plt.tick_params(labelsize=18)
		plt.grid(True)
		plt.savefig(dirs + "A_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()

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
		plt.tick_params(labelsize=18)
		plt.grid(True)
		plt.savefig(dirs + "e2_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()
		"""
		fig, ax = plt.subplots(1, 1)
		fig.figsize=(12, 9)
		ax.plot(dates1, data_R2_all_year[:len(dates1)], '-', color="k")
		ax.plot(dates2, data_R2_all_year[len(dates1):], '-', color="k")
		ax.set_ylim([0, 1])
		ax.set_ylabel(r'$R^{2}$')
		for item in date_1_6:
			ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
		for item in date_7_12:
			ax.axvline(item, color='green', linestyle='-', lw=0.75, alpha=0.75)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
		plt.grid(True)
		plt.savefig(dirs + "R2_no_std_area_" + str(area_index) + ".png", dpi=150)
		plt.close()
		"""




###############################################################################################################

def ocean_with_colorbar(num):
	if num == 1:
		dirs = "../result_h_1day_delay/mean_vector/ocean_currents/"
	else:
		dirs = "../result_nc_1day_delay/mean_vector/ocean_currents/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for year in y_list:
		for month in month_list:
			print(year + month)
			if num == 1:
				file = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
			else:
				file = "../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_20" + year + month + ".csv"
			df = pd.read_csv(file)
			data_vec = [np.array(df["ocean_u"]), np.array(df["ocean_v"])]

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

			vector_u = np.ma.masked_invalid(data_vec[0])
			vector_v = np.ma.masked_invalid(data_vec[1])
			vector_speed = np.sqrt(vector_u*vector_u + vector_v*vector_v)

			data_non_wind = vector_speed
			data_non_wind = np.ma.masked_invalid(data_non_wind)
			data1 = np.reshape(data_non_wind, (145,145), order='F')

			xx = np.hstack([xx, xx[:,0].reshape(145,1)])
			xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
			yy = np.vstack([yy, yy[0,:]])
			yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

			m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=cm_ocean_current, vmax=0.2, vmin=0)
			m.colorbar(location='bottom')
			m.quiver(x, y, vector_u, vector_v, color="k")
			save_name = dirs + "ocean_" + year + month + ".png"
			print(save_name)
			plt.savefig(save_name, dpi=350)
			plt.close()




###############################################################################################################

def ic0_and_A_by_region_by_month():
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for year in y_list:
		for month in month_list:
			file = sorted(glob.glob("../result_h_1day_delay/print_date/print_data_30/describe_data_30_20" + year + month + ".csv"))
			data = pe.read_csv(file)
			data_A = np.array(data.loc[:, ("A", "mean")]).reshape((-1,1))
			data_ic0 = np.array(data.loc[:, ("ic0_30", "mean")]).reshape((-1,1))
			data_A_ic0 = np.hstack((data_A, data_ic0))

			#まず，北極海のエリアをデータとしてある年月でプロット
			#plt.plot(data_ic0, data_A)





###############################################################################################################


#plot_data_corr_sit_1day_delay()
#plot_nc_data_corr_sit_1day_delay()
#search_data_corr_ic0_1day_delay()
#search_nc_data_corr_ic0_1day_delay()
#ts_30_by_month_modified_gw()
#ts_30_by_month_modified_nc()
ts_by_month_all_year_gw_and_nc(num=1)
#ts_by_month_all_year_gw_and_nc(num=2)
#ocean_with_colorbar(num=1)
#ocean_with_colorbar(num=2)
#ic0_and_A_by_region_by_month()


















