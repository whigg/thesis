
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


###############################################################################################################

def plot_ic0_std():
	#dirs = "../result_h_1day_delay/scatter_ic0_std_and_corr_ic0_A/"
	dirs = "../result_h_1day_delay/corr_map_with_std_contour/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data_std = []
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data_std = data.copy()
			data_std.loc[data_std["A"].isnull(), "ic0_30"] = np.nan
			data_std = np.array(data_std["ic0_30"])
			accumulate_data_std.append(data_std)

			#data = pd.read_csv(file)
			data_corr = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30", "sit_30"]]
			accumulate_data.append(np.array(data_corr))

		print("\tpoint 1")
		accumulate_data_std = np.array(accumulate_data_std)
		ic0_std = np.nanstd(accumulate_data_std, axis=0)
		ic0_count = np.nansum(~np.isnan(accumulate_data_std), axis=0)
		ic0_std = np.where(ic0_count>5, ic0_std, np.nan)

		accumulate_data = np.array(accumulate_data)
		corr_list = []
		for i in range(145**2):
			data_A = accumulate_data[:, i, 0]
			data_ic0 = accumulate_data[:, i, 4]
			tmp_df = pd.DataFrame({"data_A": data_A, "data_ic0": data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
			corr_list.append(corr)
		corr_array = np.array(corr_list)
		print("\tpoint 2")

		m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
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

		corr_array = np.ma.masked_invalid(corr_array)
		ic0_std = np.ma.masked_invalid(ic0_std)
		data1 = np.reshape(corr_array, (145,145), order='F')
		data2 = np.reshape(ic0_std, (145,145), order='F')

		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap='bwr', vmax=1, vmin=-1)
		m.colorbar(location='bottom')
		m.contour(xx1.T, yy1.T, data2, range(10, 40, 5), linewidths=0.5, cmap = plt.cm.cubehelix)
		cs = m.contour(xx1.T, yy1.T, data2, range(10, 40, 5), linewidths=0.5, cmap = plt.cm.cubehelix)
		plt.clabel(cs, inline=True, fmt='%.0f', fontsize=3, colors='c')
		#m.contour(xx1.T, yy1.T, data2, locator=MultipleLocator(10), linewidths=1)
		#m.colorbar(location='right')

		save_name = dirs + "corr_map_with_std_contour_" + month + ".png"
		plt.savefig(save_name, dpi=300)
		plt.close()



		"""
		corr_4_plot = pd.DataFrame({"ic0_std": ic0_std, "corr_array": corr_array})
		corr_4_plot.loc[corr_4_plot.ic0_std<15, :] = np.nan
		corr_4_plot = corr_4_plot.dropna()

		X = np.column_stack((np.repeat(1, len(corr_4_plot)), np.array(corr_4_plot.ic0_std)))
		model = sm.OLS(endog=np.array(corr_4_plot.corr_array), exog=X)
		results = model.fit()
		corr_4_plot['resid'] = results.resid
		sns.set_style("darkgrid")
		sns.jointplot("ic0_std", "corr_array", data=corr_4_plot, kind="reg",
			xlim=(0, 40), ylim=(-1, 1), size=7)
		#indices to annotate
		head = corr_4_plot.sort_values(by=['resid'], ascending=[False]).head(10)
		tail = corr_4_plot.sort_values(by=['resid'], ascending=[False]).tail(10)

		def ann(row):
		    ind = row[0]
		    r = row[1]
		    plt.gca().annotate(ind, xy=(r["ic0_std"], r["corr_array"]), 
		            xytext=(1,1) , textcoords ="offset points", )

		for row in head.iterrows():
		    ann(row)
		for row in tail.iterrows():
		    ann(row)
		
		#plt.show()
		save_name = dirs + "ic0_std_and_corr_ic0_A_" + month + ".png"
		plt.savefig(save_name, dpi=150)
		plt.close()
		"""

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
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "ic0_A_pos_grid_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()
		for grid in plot_grids_neg:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs_corr_map_search_grid + "ic0_A_neg_grid_" + str(grid) + ".png"
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
		plt.savefig(dirs_corr_map_search_grid + "ic0_A_grid_info_" + month + ".png", dpi=300)
		plt.close()
		"""


###############################################################################################################

def test_ts_all_with_trend():
	# https://org-technology.com/posts/detrend.html
	dates1 = pd.date_range("2003", "2006", freq='MS')[:-1]
	y = [1,2,2,3,4,3,5,3,6,6,4,2,
	2,1,3,4,3,5,6,6,7,5,3,2,
	3,2,4,4,6,5,8,9,5,4,3,3]

	yd = signal.detrend(y)
	plt.figure(figsize=(6,4))
	plt.plot(t, y, label="Original Data")
	plt.plot(t, y-yd, "--r", label="Trend")
	plt.plot(t, yd, "c", label="Detrended Data")
	plt.axhline(0, color="k", linestyle="--", label="Mean of Detrended Data")
	plt.axis("tight")
	plt.legend(loc=0)
	plt.show()



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
		"""
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





def plot_data_corr_for_mombetsu():
	dirs_corr_map = "../result_h_1day_delay/corr_map/"
	if not os.path.exists(dirs_corr_map):
		os.makedirs(dirs_corr_map)

	month_list = ["07"]
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
			tmp_df = pd.DataFrame({"data_A": data_A, "data_ic0": data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
				#print(i, corr)
			corr_list.append(corr)

		m = Basemap(lon_0=180, boundinglat=60, resolution='l', projection='npstere')
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

		data = np.array(corr_list)
		data = np.ma.masked_invalid(data)
		data1 = np.reshape(data, (145,145), order='F')
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap='bwr', vmax=1, vmin=-1)
		#m.contourf(xx1, yy1, data1, cmap='bwr', levels=np.arange(-1,1,10), extend='both')
		m.colorbar(location='bottom')

		save_name_corr = dirs_corr_map + "ic0_A_" + month + "_for_mombetsu.png"
		plt.savefig(save_name_corr, dpi=200)
		plt.close()
















###############################################################################################################
		
#plot_ic0_std()
#ts_all_with_trend(num=1)
#ts_all_with_trend(num=2)
plot_data_corr_for_mombetsu()




























