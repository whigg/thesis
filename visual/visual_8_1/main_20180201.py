
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
cm_angle_3 = visualize.generate_cmap([
	"blue", 
	"Lime", 
	"grey", 
	"yellow", 
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
	#m.contourf(xx1, yy1, data1, cmap=cmap, vmax=vmax, vmin=vmin, extend='both')
	#m.contourf(xx1, yy1, data1, cmap='jet', extend='both')
	m.colorbar(location='bottom')
	plt.tight_layout()
	plt.savefig(save_name, dpi=300)
	plt.close()


###############################################################################################################

def presentation_00():
	"""
	エリアを塗る図
	"""
	fig = plt.figure(figsize=(5, 5))
	m = Basemap(lon_0=180, boundinglat=62.5, resolution='i', projection='npstere')
	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#555555')
	x, y = m(np.array(latlon_ex.Lon), np.array(latlon_ex.Lat))
	"""
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
	"""
	df = pd.read_csv("../data/csv_Helmert_by_year_1day_delay/Helmert_by_year_1day_delay_01.csv")
	color_list = ["lime", "lightskyblue"]
	for i, k in enumerate([12, 16]):
		area_idx = np.array(df[df["area_label"]==k].dropna().index)
		m.scatter(x[area_idx], y[area_idx], marker='s', color = color_list[i], s=3.5, alpha=0.9)
		#m.plot(x[area_idx], y[area_idx], marker='o', color = color_list[i], alpha=0.9)

	plt.tight_layout()
	plt.show()
	#plt.close()	

#presentation_00()



def presentation_01():
	"""
	風力係数の経年変化の増加率(月ごと)
	"""
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	all_month = []
	for im, month in enumerate(month_list):
	#for month in ["07", "08", "09", "10", "11", "12"]:
		print("*************** " + month + " ***************")
		data_A_year, data_theta_year, data_R2_year, data_e2_year = [], [], [], []
		for year in y_list:
			file_list = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
			df = pd.read_csv(file_list)
			data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			#for item in ["A", "theta", "R2", "epsilon2"]:
			for item in ["A"]:
				data.loc[:, (item, "1sigma_pos")] = data.loc[:, (item, "mean")] + data.loc[:, (item, "std")]
				data.loc[:, (item, "1sigma_neg")] = data.loc[:, (item, "mean")] - data.loc[:, (item, "std")]

			data_A = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			#data_theta = data.loc[:, ("theta", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			#data_R2 = data.loc[:, ("R2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			#data_e2 = data.loc[:, ("epsilon2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values

			data_A_year.append(data_A)
			#data_theta_year.append(data_theta)
			#data_R2_year.append(data_R2)
			#data_e2_year.append(data_e2)

		data_A_year = np.array(data_A_year)
		#data_theta_year = np.array(data_theta_year)
		#data_R2_year = np.array(data_R2_year)
		#data_e2_year = np.array(data_e2_year)

		file_by_year = "../data/csv_Helmert_by_year_1day_delay/Helmert_by_year_1day_delay_" + month + ".csv"
		data_by_year = pd.read_csv(file_by_year)
		data_by_year = data_by_year.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

		dates1 = pd.date_range("2003", "2011", freq='YS')[:-1]
		dates2 = pd.date_range("2013", "2017", freq='YS')[:-1]
		N_dates1 = len(dates1)

		all_area = []
		for i in range(17):
			y1_A = data_A_year[:N_dates1,i,1]
			y2_A = data_A_year[N_dates1:,i,1]
			y3_A = data_A_year[:,i,1]
			A_by_year = data_by_year.loc[(i), ("A", "mean")]
			try:
				yd1_A = signal.detrend(y1_A)
				tmp_1 = y1_A-yd1_A
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
			print(i, v1, v2)
			all_area.append([v1, v2, v3])
		all_area_array = np.array(all_area)
		#17x3
		print(all_area_array.shape)
		all_month.append(all_area)

	all_month_array = np.array(all_month)
	#12x17x3
	print(all_month_array.shape)

	file_list = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_200301.csv"
	df = pd.read_csv(file_list)
	for i in range(3):
		for j in range(12):
			tmp = np.ones(145**2)*(-1)
			for k in range(17):
				area_k = all_month_array[j, k, i]
				#print(k, area_k)
				area_idx = np.array(df[df["area_label"]==k].dropna().index)
				tmp[area_idx] = area_k
			#df["v_1"] = tmp
			tmp = np.where(tmp==-1, np.nan, tmp)
			#tmp = np.where(tmp<0, -np.log(-tmp), np.log(tmp))
			save_name = "../result_h_1day_delay/ts_by_year_map/A_trend_map_v" + str(i+1) + "_" + str(j+1) + ".png"
			plot_map_for_thesis(tmp*1e4, save_name, cmap=cm_angle_3, vmax=-6, vmin=6)


#presentation_01()



def presentation_01_2():
	"""
	風力係数の経年変化の増加率(月ごと)
	全グリッド
	"""
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	all_month = []
	for im, month in enumerate(month_list):
		print("*************** " + month + " ***************")
		data_A_year, data_theta_year, data_R2_year, data_e2_year = [], [], [], []
		for year in y_list:
			file_list = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + year + month + ".csv"
			df = pd.read_csv(file_list)["A"]
			data_A = np.array(df)
			data_A_year.append(data_A)
		data_A_year = np.array(data_A_year)
		print(data_A_year.shape)
		N_grids = data_A_year.shape[1]

		dates1 = pd.date_range("2003", "2011", freq='YS')[:-1]
		dates2 = pd.date_range("2013", "2017", freq='YS')[:-1]
		N_dates1 = len(dates1)

		all_area = []
		for i in range(N_grids):
			y1_A = data_A_year[:N_dates1,i]
			y2_A = data_A_year[N_dates1:,i]
			y3_A = data_A_year[:,i]
			try:
				yd1_A = signal.detrend(y1_A)
				tmp_1 = y1_A-yd1_A
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
			#print(i, v1, v2)
			all_area.append([v1, v2, v3])
		all_area_array = np.array(all_area)
		#N_grids x 3
		print(all_area_array.shape)
		all_month.append(all_area)

	all_month_array = np.array(all_month)
	#12 x N_grids x 3
	print(all_month_array.shape)

	file_tmp = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_200301.csv"
	df = pd.read_csv(file_tmp)
	for i in range(3):
		for j in range(12):
			tmp = all_month_array[j, :, i]
			tmp = np.where(tmp==-1, np.nan, tmp)
			save_name = "../result_h_1day_delay/ts_by_year_map/A_trend_map_by_grid_v" + str(i+1) + "_" + str(j+1) + ".png"
			plot_map_for_thesis(tmp*1e4, save_name, cmap=cm_angle_3, vmax=6, vmin=-6)


#presentation_01_2()








def presentation_02():
	"""
	風力係数の年・月全部のトレンド（信頼可能エリア）
	"""
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	date_1_6 = []
	for year in y_list:
		date_1_6.append(pd.to_datetime("20"+year+"-01-01"))
		#date_1_6.append(pd.to_datetime("20"+year+"-07-01"))
	date_7_12 = []
	for year in y_list:
		date_7_12.append(pd.to_datetime("20"+year+"-07-01"))

	for area_index in range(17):
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
			tmp_1 = y1_A-yd1_A
			tmp_2 = y2_A-yd2_A
			tmp_3 = y3_A-yd3_A
			print(tmp_1[11]-tmp_1[0], tmp_2[11]-tmp_2[0], tmp_3[11]-tmp_3[0])
		except:
			tmp_1 = -1
			tmp_2 = -1
			tmp_3 = -1
			print(-1, -1, -1)


#presentation_02()



def presentation_03(slide):
	"""
	SICとの相関係数のslide month
	"""
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	product_ym = list(itertools.product(y_list, month_list))

	data_dict = {}
	for month in month_list:
		data_dict[month] = {}
		for year in y_list:
			data_dict[month][year] = {}
			data_dict[month][year]["ym_A"] = np.array([np.nan]*(145**2))
			data_dict[month][year]["ym_ic0"] = np.array([np.nan]*(145**2))
			data_dict[month][year]["data_A"] = np.array([np.nan]*(145**2))
			data_dict[month][year]["data_ic0"] = np.array([np.nan]*(145**2))

	for k, item in enumerate(product_ym):
		if k+slide > len(product_ym)-1:
			break
		x_ym = product_ym[k][0] + product_ym[k][1]
		y_ym = product_ym[k+slide][0] + product_ym[k+slide][1]
		print(x_ym, y_ym)
		if (k<=95) & (k+slide>95):
			print("\tcontinue: {}".format((x_ym, y_ym)))
			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_A"] = x_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_ic0"] = y_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_A"] = np.array([np.nan]*(145**2))
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_ic0"] = np.array([np.nan]*(145**2))
			continue
		else:
			helmert_axis_x_file = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + x_ym + ".csv"
			helmert_axis_y_file = "../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_20" + y_ym + ".csv"

			data_A = pd.read_csv(helmert_axis_x_file)["A"]
			data_ic0 = pd.read_csv(helmert_axis_y_file)["ic0_30"]

			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_A"] = x_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_ic0"] = y_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_A"] = np.array(data_A)
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_ic0"] = np.array(data_ic0)

	for month in month_list:
		accumulate_data_A = []
		accumulate_data_ic0 = []
		for year in y_list:
			#print(month, year)
			accumulate_data_A.append(data_dict[month][year]["data_A"])
			accumulate_data_ic0.append(data_dict[month][year]["data_ic0"])
		accumulate_data_A = np.array(accumulate_data_A)
		accumulate_data_ic0 = np.array(accumulate_data_ic0)

		corr_list_ic0_p = []
		corr_list_ic0_r = []
		for i in range(145**2):
			plot_data_A = accumulate_data_A[:, i]
			#data_A = data_A[~np.isnan(data_A)]
			plot_data_ic0 = accumulate_data_ic0[:, i]
			tmp_df = pd.DataFrame({"data_A": plot_data_A, "data_ic0": plot_data_ic0})
			if len(tmp_df.dropna()) <= 5:
				r = np.nan
				p = np.nan
			else:
				corr_ic0 = np.array(tmp_df.dropna())
				r, p = stats.pearsonr(corr_ic0[:,0], corr_ic0[:,1])
			corr_list_ic0_p.append(p)
			corr_list_ic0_r.append(r)

		corr_array_p = np.array(corr_list_ic0_p)
		corr_array_r = np.array(corr_list_ic0_r)
		th = 0.05
		corr_array_p = np.where(corr_array_p<th, 0, 1)

		month_slided = str((int(month) + slide)%12)
		if len(month_slided) == 1:
			month_slided = "0" + month_slided
		save_name_p = "../result_h_1day_delay/corr_map_slide_month/ic0_A_start_month_" + month + "_" + month_slided + "_p.png"
		save_name_r = "../result_h_1day_delay/corr_map_slide_month/ic0_A_start_month_" + month + "_" + month_slided + "_r.png"
		print("\t{}".format(save_name_p))
		print("\t{}".format(save_name_r))
		visualize.plot_map_once(corr_array_p, data_type="type_non_wind", show=False, 
			save_name=save_name_p, vmax=1, vmin=0, cmap=cm_p_value)
		visualize.plot_map_once(corr_array_r, data_type="type_non_wind", show=False, 
			save_name=save_name_r, vmax=1, vmin=-1, cmap=plt.cm.jet)


#for i in range(1,6):
#	presentation_03(slide=i)


def presentation_04():
	file_name = "../data/csv_ic0/IC0_20140315.csv"
	visualize.plot_900(file_name, save_name=None, show=True)

presentation_04()










