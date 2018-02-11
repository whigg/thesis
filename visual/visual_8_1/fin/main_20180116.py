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


cm_angle_2 = visualize.generate_cmap([
	"blue", 
	"Lime", 
	"grey", 
	"yellow", 
	"red"
	])


y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

###############################################################################################################

def get_date_ax(start, end):
	start_date = [start//10000, (start%10000)//100, (start%10000)%100]
	end_date = [end//10000, (end%10000)//100, (end%10000)%100]
	d1 = date(start_date[0], start_date[1], start_date[2])
	d2 = date(end_date[0], end_date[1], end_date[2])
	L = (d2-d1).days+1
	dt = d1

	date_ax = []
	date_ax_str = []
	for i in range(L):
		date_ax.append(dt)
		date_ax_str.append(calc_data.cvt_date(dt))
		dt = dt + timedelta(days=1)

	return date_ax, date_ax_str

def main_data(start, end, **kwargs):
	span = kwargs["span"]
	region = kwargs["region"]
	get_columns = kwargs["get_columns"]
	accumulate = kwargs["accumulate"]

	date_ax, date_ax_str = get_date_ax(start, end)
	N = len(date_ax_str)
	skipping_date_str = []
	accumulate_data = []
	data = []
	for i, day in enumerate(date_ax_str):
		#print ("{}/{}: {}".format(i+1, N, day))
		#print ("start: {}, end: {}".format(start, end))
		year = day[2:4]
		month = day[4:6]

		#ファイル名の生成
		wind_file_name = "../data/csv_w/ecm" + day[2:] + ".csv"
		ice_file_name = "../data/csv_iw/" + day[2:] + ".csv"
		ic0_145_file_name = "../data/csv_ic0/IC0_" + day + ".csv"
		sit_145_file_name = "../data/csv_sit/SIT_" + day + ".csv"
		coeff_file_name = "../data/csv_A_30/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"
		hermert_file_name = "../data/csv_Helmert_30/Helmert_30_" + str(day)[:6] + ".csv"
		# wind10m_file_name = "../data/netcdf4/" + day[2:] + ".csv"
		# t2m_file_name = "../data/netcdf4/" + day[2:] + ".csv"

		skipping_boolean = ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name)]))
		if ("ic0_145" in get_columns):
			skipping_boolean = ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name), os.path.isfile(ic0_145_file_name)]))
		if ("sit_145" in get_columns):
			skipping_boolean = ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name), os.path.isfile(sit_145_file_name)]))
			
		if skipping_boolean == True:
			print ("\tSkipping " + day + " file...")
			date_ax_str.remove(day)
			bb = date(int(day[:4]), int(day[4:6]), int(day[6:]))
			date_ax.remove(bb)
			skipping_date_str.append(day)
			continue

		data = pd.DataFrame({"data_idx": np.array(ocean_grid_145).ravel()})
		if "ex_1" in get_columns:
			print("\t{}\n\t{}\n\t{}\n\t{}".format(wind_file_name, ice_file_name, coeff_file_name))
			tmp = calc_data.get_w_regression_data(wind_file_name, ice_file_name, coeff_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "ex_2" in get_columns:
			print("\t{}\n\t{}\n\t{}\n\t{}".format(wind_file_name, ice_file_name, hermert_file_name))
			tmp = calc_data.get_w_hermert_data(wind_file_name, ice_file_name, hermert_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "w" in get_columns:
			print("\t{}".format(wind_file_name))
			tmp = calc_data.get_1day_w_data(wind_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "iw" in get_columns:
			#print("\t{}".format(ice_file_name))
			tmp = calc_data.get_1day_iw_data(ice_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "ic0_145" in get_columns:
			print("\t{}".format(ic0_145_file_name))
			tmp = calc_data.get_1day_ic0_data(ic0_145_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "sit_145" in get_columns:
			print("\t{}".format(sit_145_file_name))
			tmp = calc_data.get_1day_sit_data(sit_145_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "coeff" in get_columns:
			print("\t{}".format(coeff_file_name))
			tmp = calc_data.get_1month_coeff_data(coeff_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "hermert" in get_columns:
			print("\t{}".format(hermert_file_name))
			tmp = calc_data.get_1month_hermert_data(hermert_file_name)
			data = pd.concat([data, tmp], axis=1)

		data = calc_data.get_masked_region_data(data, region)

		if ("coeff" in get_columns):
			print("\tSelected only coeff data. Getting out of the loop...")
			continue

		if accumulate == True:
			data_1 = data.drop("data_idx", axis=1)
			#print("\t{}".format(data_1.columns))
			accumulate_data.append(np.array(data_1))

	if accumulate == True:
		#print("accumulate: True\tdata type: array")
		return date_ax, date_ax_str, skipping_date_str, accumulate_data
	else:
		print("accumulate: False\tdata type: DataFrame")
		return date_ax, date_ax_str, skipping_date_str, data



###############################################################################################################

def print_describe_data_all_region():
	dirs_30 = "../result_h/print_data/print_data_30_all/"
	if not os.path.exists(dirs_30):
		os.makedirs(dirs_30)
	file_list = sorted(glob.glob("../data/csv_Helmert_both_30/Helmert_both_30_*.csv"))
	for file in file_list:
		print(file)
		df = pd.read_csv(file).dropna()
		data = df.loc[:, ["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
		data.to_csv(dirs_30 + "describe_data_30_all_" + file[44:])

	dirs_90 = "../result_h/print_data/print_data_90_all/"
	if not os.path.exists(dirs_90):
		os.makedirs(dirs_90)
	file_list = sorted(glob.glob("../data/csv_Helmert_both_90/Helmert_both_90_*.csv"))
	for file in file_list:
		print(file)
		df = pd.read_csv(file).dropna()
		data = df.loc[:, ["A_90", "theta_90", "R2_90", "epsilon2_90", "N_c_90", "ocean_u_90", "ocean_v_90", "mean_iw_u_90", "mean_iw_v_90", "mean_w_u_90", "mean_w_v_90"]].describe()
		data.to_csv(dirs_90 + "describe_data_90_all_" + file[44:])

	dirs_year = "../result_h/print_data/print_data_by_year_all/"
	if not os.path.exists(dirs_year):
		os.makedirs(dirs_year)
	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)
	file_list = sorted(glob.glob("../data/csv_Helmert_by_year/Helmert_by_year_*.csv"))
	for file in file_list:
		print(file)
		df = pd.read_csv(file)
		df = pd.concat([latlon_ex, df, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1).dropna()
		data = df.loc[:, ["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
		data.to_csv(dirs_year + "describe_data_by_year_all_" + file[44:])

#print_describe_data_all_region()



######################
# 以下，netcdf4系の関数 #
######################

def get_helmert_10m_both_30_csv():
	dirs = "../data/csv_Helmert_both_30_netcdf4/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	start_list_plus_1month = start_list + [20170901]
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		L = month_end.day

		start_0101 = date(start//10000, 1, 1)
		start_date = date(start//10000, (start%10000)//100, (start%10000)%100)
		start_from_0101_idx = (start_date-start_0101).days
		print(start_0101, start_date, start_from_0101_idx)

		nc_fname = "../data/netcdf4/interim_2mt_10u_10v_" + str(start)[:4] + "0101-" + str(start)[:4] + "1231.nc"
		_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
		w10_u_array = result_u10[start_from_0101_idx:start_from_0101_idx+L, :]
		w10_v_array = result_v10[start_from_0101_idx:start_from_0101_idx+L, :]
		print(result_u10.shape, w10_u_array.shape)

		iw_file_list = sorted(glob.glob("../data/csv_iw/" + str(start)[2:6] + "*.csv"))
		iw_list = []
		L_iw = len(iw_file_list)
		grid_data = pd.read_csv(grid900to145_file_name, header=None)
		grid145 = np.array(grid_data, dtype='int64').ravel()
		for iw_fname in iw_file_list:
			df_ice_wind = pd.read_csv(iw_fname, header=None)
			df_ice_wind[df_ice_wind==999.] = np.nan
			ice_wind = np.array(df_ice_wind, dtype='float32')/100
			iw_list.append(ice_wind[:, [0,1]])
		iw_array = np.array(iw_list)
		print(iw_array.shape)

		w10_u_ave = np.nanmean(w10_u_array, axis=0)
		w10_v_ave = np.nanmean(w10_v_array, axis=0)
		iw_ave = np.nanmean(iw_array, axis=0)
		print(w10_u_ave.shape, iw_ave.shape)

		w10_u_minus_ave = w10_u_array - np.tile(w10_u_ave, (L,1))
		w10_v_minus_ave = w10_v_array - np.tile(w10_v_ave, (L,1))
		iw_minus_ave = iw_array - np.tile(iw_ave, (L_iw,1,1))

		param_list = []
		#for j in range(1218,1220):
		for j in range(145**2):
			#print(j)
			N_c = np.sum(~np.isnan(iw_minus_ave[:,j,0]))
			if N_c <= 20:
				param_list.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c])
				continue
			gw_u = w10_u_minus_ave[:,j]
			gw_v = w10_v_minus_ave[:,j]
			iw_u = iw_minus_ave[:,j,0]
			iw_v = iw_minus_ave[:,j,1]
			b_1 = (np.nansum(gw_u*iw_u) + np.nansum(gw_v*iw_v)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
			b_2 = (np.nansum(gw_u*iw_v) - np.nansum(gw_v*iw_u)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
			a_1 = iw_ave[j,0] - b_1*w10_u_ave[j] + b_2*w10_v_ave[j]
			a_2 = iw_ave[j,1] - b_1*w10_v_ave[j] - b_2*w10_u_ave[j]
			R_denominator = np.nansum(iw_u**2 + iw_v**2)
			R_numerator = np.nansum((iw_array[:,j,0] - (a_1 + b_1*w10_u_array[:,j] - b_2*w10_v_array[:,j]))**2) + \
				np.nansum((iw_array[:,j,1] - (a_2 + b_2*w10_u_array[:,j] + b_1*w10_v_array[:,j]))**2)
			R2 = 1 - R_numerator/R_denominator
			A = np.sqrt(b_1**2 + b_2**2)
			theta = np.arctan2(b_2, b_1) * 180/np.pi
			#print(a_1, a_2, b_1, b_2)
			print("A: {}\ntheta: {}\na_1: {}\na_2: {}\nR2: {}\ne2: {}\nN_c: {}".format(A, theta, a_1, a_2, R2, R_numerator, N_c))
			param_list.append([A, theta, a_1, a_2, R2, R_numerator, N_c])
		param_array = np.array(param_list)
		
		#print(param_array.shape, iw_ave.shape, w10_u_ave.shape, w10_v_ave.shape)
		data_array = np.hstack((param_array, iw_ave, w10_u_ave.reshape((-1,1)), w10_v_ave.reshape((-1,1))))
		data = pd.DataFrame(data_array)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start)[:-2] + ".csv"
		data_ex = pd.read_csv(data_ex_dir)
		data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label", "ic0_30", "ic0_30_median", "sit_30", "sit_30_median"]]], axis=1)
		save_name = dirs + "Helmert_both_30_netcdf4_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)

#get_helmert_10m_both_30_csv()





def get_helmert_10m_both_90_csv():
	dirs = "../data/csv_Helmert_both_90_netcdf4/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	#start_list = [20030101]
	start_list_plus_3month = start_list + [20170901, 20171001, 20171101]
	for k, start_1 in enumerate(start_list):
		if start_1 == 20170801:
			print("Breaking the loop...")
			break
		print("*******************  {}/{}  *******************".format(k+1, M))
		if k == 0:
			month_end_1 = start_list_plus_3month[k+1]
			month_end_1 = date(month_end_1//10000, (month_end_1%10000)//100, (month_end_1%10000)%100) - timedelta(days=1)
			end_1 = start_1 + month_end_1.day - 1

			start_2 = month_end_1 + timedelta(days=1)
			month_end_2 = start_list_plus_3month[k+2]
			month_end_2 = date(month_end_2//10000, (month_end_2%10000)//100, (month_end_2%10000)%100) - timedelta(days=1)
			end_2 = start_2 + timedelta(days=month_end_2.day-1)

			start_2 = int(start_2.strftime('%Y/%m/%d').replace("/", ""))
			end_2 = int(end_2.strftime('%Y/%m/%d').replace("/", ""))

			start_list_3month = [start_1, start_2, 0]
			end_list_3month = [end_1, end_2, 0]
		else:
			kk = k - 1
			start_1_1 = start_list[kk]
			month_end_1 = start_list_plus_3month[kk+1]
			month_end_1 = date(month_end_1//10000, (month_end_1%10000)//100, (month_end_1%10000)%100) - timedelta(days=1)
			end_1 = start_1_1 + month_end_1.day - 1

			start_2 = month_end_1 + timedelta(days=1)
			month_end_2 = start_list_plus_3month[kk+2]
			month_end_2 = date(month_end_2//10000, (month_end_2%10000)//100, (month_end_2%10000)%100) - timedelta(days=1)
			end_2 = start_2 + timedelta(days=month_end_2.day-1)

			start_3 = month_end_2 + timedelta(days=1)
			month_end_3 = start_list_plus_3month[kk+3]
			month_end_3 = date(month_end_3//10000, (month_end_3%10000)//100, (month_end_3%10000)%100) - timedelta(days=1)
			end_3 = start_3 + timedelta(days=month_end_3.day-1)

			start_2 = int(start_2.strftime('%Y/%m/%d').replace("/", ""))
			start_3 = int(start_3.strftime('%Y/%m/%d').replace("/", ""))
			end_2 = int(end_2.strftime('%Y/%m/%d').replace("/", ""))
			end_3 = int(end_3.strftime('%Y/%m/%d').replace("/", ""))

			start_list_3month = [start_1_1, start_2, start_3]
			end_list_3month = [end_1, end_2, end_3]

		data_w_90 = np.zeros((1, 145**2, 2))
		data_iw_90 = np.zeros((1, 145**2, 3))
		for i in range(3):
			start = start_list_3month[i]
			end = end_list_3month[i]
			if start == 0 and end == 0:
				continue

			#wデータの取得・整形
			#start = start_list_3month[i]
			start_0101 = date(start//10000, 1, 1)
			start_date = date(start//10000, (start%10000)//100, (start%10000)%100)
			start_from_0101_idx = (start_date-start_0101).days

			month_end = end
			month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100)
			L = month_end.day

			nc_fname = "../data/netcdf4/interim_2mt_10u_10v_" + str(start)[:4] + "0101-" + str(start)[:4] + "1231.nc"
			_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
			w10_u_array = result_u10[start_from_0101_idx:start_from_0101_idx+L, :]
			w10_v_array = result_v10[start_from_0101_idx:start_from_0101_idx+L, :]
			w10_uv = []
			for day_idx in range(L):
				tmp_1 = w10_u_array[day_idx, :].reshape((-1,1))
				tmp_2 = w10_v_array[day_idx, :].reshape((-1,1))
				w10_uv.append(np.hstack((tmp_1, tmp_2)))
			w10_uv = np.array(w10_uv)
			data_w_90 = np.concatenate([data_w_90, w10_uv], axis=0)

			#iwデータの取得・整形
			_, _, _, data_iw = main_data(
				start, end, 
				span=30, 
				get_columns=["iw"], 
				region=None, 
				accumulate=True
				)
			data_array_iw = np.array(data_iw)
			data_iw_90 = np.concatenate([data_iw_90, data_array_iw], axis=0)	
			print("\n")

		data_w_90 = data_w_90[1:, :, :]
		data_iw_90 = data_iw_90[1:, :, :]
		data_ave_w = np.nanmean(data_w_90, axis=0)
		data_ave_iw = np.nanmean(data_iw_90, axis=0)
		w_array = np.vstack((data_iw_90[:,:,1], data_iw_90[:,:,2]))

		Helmert = []
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_iw_90[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_w_90[:, j, 0][not_nan_idx].reshape((-1,1))
			y = data_w_90[:, j, 1][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_iw_90[:, j, 1])
			iw_v_ave = np.nanmean(data_iw_90[:, j, 2])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
				continue
			one_N = np.ones(N_c).reshape((-1,1))
			zero_N = np.zeros(N_c).reshape((-1,1))
			D_1 = np.hstack((one_N, zero_N, x, -y))
			D_2 = np.hstack((zero_N, one_N, y, x))
			Dt = np.vstack((D_1, D_2))
			#print(N_c, Dt, np.dot(Dt.T, Dt))
			D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
			gamma = np.dot(D_inv, np.dot(Dt.T, w))
			A = np.sqrt(gamma[2]**2 + gamma[3]**2)
			theta = np.arctan2(gamma[3], gamma[2]) * 180/np.pi
			R_denominator = np.dot(w.T, w) - N_c*(iw_u_ave**2 + iw_v_ave**2)
			epsilon = w - np.dot(Dt, gamma)
			R_numerator = np.dot(epsilon.T, epsilon)
			R2 = 1 - (R_numerator/R_denominator)[0,0]
			print("\t{}".format([A[0], theta[0], gamma[0,0], gamma[1,0], R2, N_c]))
			if N_c < 45:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A_90", "theta_90", "ocean_u_90", "ocean_v_90", "R2_90", "epsilon2_90", "N_c_90", "mean_iw_u_90", "mean_iw_v_90", "mean_w_u_90", "mean_w_v_90"]
		data["ocean_speed_90"] = np.sqrt(data["ocean_u_90"]**2 + data["ocean_v_90"]**2)
		
		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start_1)[:-2] + ".csv"
		data_ex = pd.read_csv(data_ex_dir)
		data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
		save_name = dirs + "Helmert_both_90_netcdf4_" + str(start_1)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)

#get_helmert_10m_both_90_csv()




def get_helmert_10m_by_year_csv():
	dirs = "../data/csv_Helmert_netcdf4_by_year/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	start_list = []
	n = 20000000
	y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
	for i in y_list:
		m = n + i*10000
		for j in range(12):
			start_list.append(m + (j+1)*100 + 1)
	start_list_plus_1month = start_list.copy()
	start_list_plus_1month = start_list_plus_1month + [20170101]

	start_list = np.array(start_list)
	start_list_plus_1month = np.array(start_list_plus_1month)
	month_list_str = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	nc_file_list = sorted(glob.glob("../data/netcdf4/*.nc"))
	nc_file_list.pop()
	for k in range(12):
		print("************************  Month: {}/{}  ************************".format(k+1, 12))
		#12年分
		month_idx = np.arange(0, 144, 12) + k
		month_next_idx = np.arange(0, 144, 12) + k + 1
		year_list = start_list[month_idx]
		y_next_list = start_list_plus_1month[month_next_idx]

		data_w_year = np.zeros((1, 145**2, 2))
		data_iw_year = np.zeros((1, 145**2, 3))
		for i, start in enumerate(year_list):
			print("  *******************  Year: {}  *******************".format(str(start)[:6]))
			month_end = y_next_list[i]
			month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
			end = start + month_end.day - 1
			L = month_end.day

			start_0101 = date(start//10000, 1, 1)
			start_date = date(start//10000, (start%10000)//100, (start%10000)%100)
			start_from_0101_idx = (start_date-start_0101).days
			print(start_date, L)

			nc_fname = nc_file_list[i]
			_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
			w10_u_array = result_u10[start_from_0101_idx:start_from_0101_idx+L, :]
			w10_v_array = result_v10[start_from_0101_idx:start_from_0101_idx+L, :]
			w10_uv = []
			for day_idx in range(L):
				tmp_1 = w10_u_array[day_idx, :].reshape((-1,1))
				tmp_2 = w10_v_array[day_idx, :].reshape((-1,1))
				w10_uv.append(np.hstack((tmp_1, tmp_2)))
			w10_uv = np.array(w10_uv)
			data_w_year = np.concatenate([data_w_year, w10_uv], axis=0)

			_, _, _, data_iw = main_data(
				start, end, 
				span=30, 
				get_columns=["iw"], 
				region=None, 
				accumulate=True
				)
			data_array_iw = np.array(data_iw)
			data_iw_year = np.concatenate([data_iw_year, data_array_iw], axis=0)

		data_w_year = data_w_year[1:, :, :]
		data_iw_year = data_iw_year[1:, :, :]
		data_ave_w = np.nanmean(data_w_year, axis=0)
		data_ave_iw = np.nanmean(data_iw_year, axis=0)
		w_array = np.vstack((data_iw_year[:,:,1], data_iw_year[:,:,2]))

		Helmert = []
		#for j in range(1218,1220):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_iw_year[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_w_year[:, j, 0][not_nan_idx].reshape((-1,1))
			y = data_w_year[:, j, 1][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_iw_year[:, j, 1])
			iw_v_ave = np.nanmean(data_iw_year[:, j, 2])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
				continue
			one_N = np.ones(N_c).reshape((-1,1))
			zero_N = np.zeros(N_c).reshape((-1,1))
			D_1 = np.hstack((one_N, zero_N, x, -y))
			D_2 = np.hstack((zero_N, one_N, y, x))
			Dt = np.vstack((D_1, D_2))
			#print(N_c, Dt, np.dot(Dt.T, Dt))
			D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
			gamma = np.dot(D_inv, np.dot(Dt.T, w))
			A = np.sqrt(gamma[2]**2 + gamma[3]**2)
			theta = np.arctan2(gamma[3], gamma[2]) * 180/np.pi
			R_denominator = np.dot(w.T, w) - N_c*(iw_u_ave**2 + iw_v_ave**2)
			epsilon = w - np.dot(Dt, gamma)
			R_numerator = np.dot(epsilon.T, epsilon)
			R2 = 1 - (R_numerator/R_denominator)[0,0]
			print("\t{}".format([A[0], theta[0], gamma[0,0], gamma[1,0], R2, N_c]))
			if N_c <= 120:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
		data_ex = pd.read_csv(data_ex_dir)
		data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)

		save_name = dirs + "Helmert_netcdf4_by_year_" + month_list_str[k] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)


#get_helmert_10m_by_year_csv()




def make_csv_nc_90_201612():
	dirs = "../data/csv_Helmert_both_90_netcdf4/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	start_1 = 20161201
	start_1_1 = 20161101
	start_2 = 20161201
	start_3 = 20170101
	end_1 = 20161130
	end_2 = 20161231
	end_3 = 20170131
	start_list_3month = [start_1_1, start_2, start_3]
	end_list_3month = [end_1, end_2, end_3]

	data_w_90 = np.zeros((1, 145**2, 2))
	data_iw_90 = np.zeros((1, 145**2, 3))
	for i in range(3):
		start = start_list_3month[i]
		end = end_list_3month[i]

		#wデータの取得・整形
		#start = start_list_3month[i]
		start_0101 = date(start//10000, 1, 1)
		start_date = date(start//10000, (start%10000)//100, (start%10000)%100)
		start_from_0101_idx = (start_date-start_0101).days
		month_end = end
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100)
		L = month_end.day

		nc_fname = "../data/netcdf4/interim_2mt_10u_10v_" + str(start)[:4] + "0101-" + str(start)[:4] + "1231.nc"
		if str(start)[:4] == "2017":
			nc_fname = "../data/netcdf4/interim_2mt_10u_10v_" + str(start)[:4] + "0101-" + str(start)[:4] + "0930.nc"
		_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
		w10_u_array = result_u10[start_from_0101_idx:start_from_0101_idx+L, :]
		w10_v_array = result_v10[start_from_0101_idx:start_from_0101_idx+L, :]
		w10_uv = []
		for day_idx in range(L):
			tmp_1 = w10_u_array[day_idx, :].reshape((-1,1))
			tmp_2 = w10_v_array[day_idx, :].reshape((-1,1))
			w10_uv.append(np.hstack((tmp_1, tmp_2)))
		w10_uv = np.array(w10_uv)
		data_w_90 = np.concatenate([data_w_90, w10_uv], axis=0)

		#iwデータの取得・整形
		_, _, _, data_iw = main_data(
			start, end, 
			span=30, 
			get_columns=["iw"], 
			region=None, 
			accumulate=True
			)
		data_array_iw = np.array(data_iw)
		data_iw_90 = np.concatenate([data_iw_90, data_array_iw], axis=0)	

	data_w_90 = data_w_90[1:, :, :]
	data_iw_90 = data_iw_90[1:, :, :]
	data_ave_w = np.nanmean(data_w_90, axis=0)
	data_ave_iw = np.nanmean(data_iw_90, axis=0)
	w_array = np.vstack((data_iw_90[:,:,1], data_iw_90[:,:,2]))

	Helmert = []
	for j in range(145**2):
		print("j: {}".format(j))
		#欠損データの処理
		not_nan_idx = np.sum(np.isnan(data_iw_90[:, j, :]), axis=1)==False
		#print("\tnot_nan_idx: {}".format(not_nan_idx))
		x = data_w_90[:, j, 0][not_nan_idx].reshape((-1,1))
		y = data_w_90[:, j, 1][not_nan_idx].reshape((-1,1))
		w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
		iw_u_ave = np.nanmean(data_iw_90[:, j, 1])
		iw_v_ave = np.nanmean(data_iw_90[:, j, 2])
		N_c = np.sum(not_nan_idx==True)
		if N_c <= 1:
			print("\tskipping for loop...")
			Helmert.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
			continue
		one_N = np.ones(N_c).reshape((-1,1))
		zero_N = np.zeros(N_c).reshape((-1,1))
		D_1 = np.hstack((one_N, zero_N, x, -y))
		D_2 = np.hstack((zero_N, one_N, y, x))
		Dt = np.vstack((D_1, D_2))
		#print(N_c, Dt, np.dot(Dt.T, Dt))
		D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
		gamma = np.dot(D_inv, np.dot(Dt.T, w))
		A = np.sqrt(gamma[2]**2 + gamma[3]**2)
		theta = np.arctan2(gamma[3], gamma[2]) * 180/np.pi
		R_denominator = np.dot(w.T, w) - N_c*(iw_u_ave**2 + iw_v_ave**2)
		epsilon = w - np.dot(Dt, gamma)
		R_numerator = np.dot(epsilon.T, epsilon)
		R2 = 1 - (R_numerator/R_denominator)[0,0]
		print("\t{}".format([A[0], theta[0], gamma[0,0], gamma[1,0], R2, N_c]))
		if N_c < 45:
			Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
		else:
			Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])
	result = np.hstack((Helmert, data_ave_w))
	data = pd.DataFrame(result)
	data.columns = ["A_90", "theta_90", "ocean_u_90", "ocean_v_90", "R2_90", "epsilon2_90", "N_c_90", "mean_iw_u_90", "mean_iw_v_90", "mean_w_u_90", "mean_w_v_90"]
	data["ocean_speed_90"] = np.sqrt(data["ocean_u_90"]**2 + data["ocean_v_90"]**2)
	
	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start_1)[:-2] + ".csv"
	data_ex = pd.read_csv(data_ex_dir)
	data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
	save_name = dirs + "Helmert_both_90_netcdf4_" + str(start_1)[:6] + ".csv"
	print(save_name)
	data.to_csv(save_name, index=False)

#make_csv_nc_90_201612()




def plot_nc_data_map():
	dirs_A_30 = "../result_nc/A/A_30/"
	dirs_R2_30 = "../result_nc/R2/R2_30/"
	dirs_theta_30 = "../result_nc/theta/theta_30/"
	dirs_epsilon2_30 = "../result_nc/epsilon2/epsilon2_30/"

	dirs_A_90 = "../result_nc/A/A_90/"
	dirs_R2_90 = "../result_nc/R2/R2_90/"
	dirs_theta_90 = "../result_nc/theta/theta_90/"
	dirs_epsilon2_90 = "../result_nc/epsilon2/epsilon2_90/"

	dirs_A_by_year = "../result_nc/A/A_by_year/"
	dirs_R2_by_year = "../result_nc/R2/R2_by_year/"
	dirs_theta_by_year = "../result_nc/theta/theta_by_year/"
	dirs_epsilon2_by_year = "../result_nc/epsilon2/epsilon2_by_year/"

	dirs_list = [
		dirs_A_30,
		dirs_R2_30,
		dirs_theta_30,
		dirs_epsilon2_30,
		dirs_A_90,
		dirs_R2_90,
		dirs_theta_90,
		dirs_epsilon2_90,
		dirs_A_by_year,
		dirs_R2_by_year,
		dirs_theta_by_year,
		dirs_epsilon2_by_year
		]
	for dirs in dirs_list:
		if not os.path.exists(dirs):
			os.makedirs(dirs)

	file_list_30 = sorted(glob.glob("../data/csv_Helmert_both_30_netcdf4/Helmert_both_30_netcdf4_*.csv"))
	file_list_90 = sorted(glob.glob("../data/csv_Helmert_both_90_netcdf4/Helmert_both_90_netcdf4_*.csv"))
	file_list_year = sorted(glob.glob("../data/csv_Helmert_netcdf4_by_year/Helmert_netcdf4_by_year_*.csv"))

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	for file in file_list_30:
		data = pd.read_csv(file)
		save_name_A = dirs_A_30 + file[60:66] + ".png"
		print(save_name_A)
		visualize.plot_map_once(np.array(data["A"]), data_type="type_non_wind", show=False, 
			save_name=save_name_A, vmax=0.025, vmin=0, cmap=plt.cm.jet)
		save_name_theta = dirs_theta_30 + file[60:66] + ".png"
		print(save_name_theta)
		visualize.plot_map_once(np.array(data["theta"]), data_type="type_non_wind", show=False, 
			save_name=save_name_theta, vmax=180, vmin=-180, cmap=cm_angle_2)
		save_name_R2 = dirs_R2_30 + file[60:66] + ".png"
		print(save_name_R2)
		visualize.plot_map_once(np.array(data["R2"]), data_type="type_non_wind", show=False, 
			save_name=save_name_R2, vmax=1, vmin=0, cmap=plt.cm.jet)
		save_name_e2 = dirs_epsilon2_30 + file[60:66] + ".png"
		print(save_name_e2)
		visualize.plot_map_once(np.array(data["epsilon2"]), data_type="type_non_wind", show=False, 
			save_name=save_name_e2, vmax=1.5, vmin=0, cmap=plt.cm.jet)

	for file in file_list_90:
		data = pd.read_csv(file)
		save_name_A = dirs_A_90 + file[60:66] + ".png"
		print(save_name_A)
		visualize.plot_map_once(np.array(data["A_90"]), data_type="type_non_wind", show=False, 
			save_name=save_name_A, vmax=0.025, vmin=0, cmap=plt.cm.jet)
		save_name_theta = dirs_theta_90 + file[60:66] + ".png"
		print(save_name_theta)
		visualize.plot_map_once(np.array(data["theta_90"]), data_type="type_non_wind", show=False, 
			save_name=save_name_theta, vmax=180, vmin=-180, cmap=cm_angle_2)
		save_name_R2 = dirs_R2_90 + file[60:66] + ".png"
		print(save_name_R2)
		visualize.plot_map_once(np.array(data["R2_90"]), data_type="type_non_wind", show=False, 
			save_name=save_name_R2, vmax=1, vmin=0, cmap=plt.cm.jet)
		save_name_e2 = dirs_epsilon2_90 + file[60:66] + ".png"
		print(save_name_e2)
		visualize.plot_map_once(np.array(data["epsilon2_90"]), data_type="type_non_wind", show=False, 
			save_name=save_name_e2, vmax=1.5, vmin=0, cmap=plt.cm.jet)

	for file in file_list_year:
		data = pd.read_csv(file)
		save_name_A = dirs_A_by_year + file[60:62] + ".png"
		print(save_name_A)
		visualize.plot_map_once(np.array(data["A"]), data_type="type_non_wind", show=False, 
			save_name=save_name_A, vmax=0.025, vmin=0, cmap=plt.cm.jet)
		save_name_theta = dirs_theta_by_year + file[60:62] + ".png"
		print(save_name_theta)
		visualize.plot_map_once(np.array(data["theta"]), data_type="type_non_wind", show=False, 
			save_name=save_name_theta, vmax=180, vmin=-180, cmap=cm_angle_2)
		save_name_R2 = dirs_R2_by_year + file[60:62] + ".png"
		print(save_name_R2)
		visualize.plot_map_once(np.array(data["R2"]), data_type="type_non_wind", show=False, 
			save_name=save_name_R2, vmax=1, vmin=0, cmap=plt.cm.jet)
		save_name_e2 = dirs_epsilon2_by_year + file[60:62] + ".png"
		print(save_name_e2)
		visualize.plot_map_once(np.array(data["epsilon2"]), data_type="type_non_wind", show=False, 
			save_name=save_name_e2, vmax=1.5, vmin=0, cmap=plt.cm.jet)

#plot_nc_data_map()



def plot_nc_data_ts(num):
	def ts_30_by_month(dirs):
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		for year in y_list:
			data_A_month, data_theta_month, data_R2_month, data_e2_month = [], [], [], []
			for month in month_list:
				print(year + month)
				file_list = "../data/csv_Helmert_both_30_netcdf4/Helmert_both_30_netcdf4_20" + year + month + ".csv"
				df = pd.read_csv(file_list)
				data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

				for item in ["A", "theta", "R2", "epsilon2"]:
					data.loc[:, (item, "1sigma_pos")] = data.loc[:, (item, "mean")] + data.loc[:, (item, "std")]
					data.loc[:, (item, "1sigma_neg")] = data.loc[:, (item, "mean")] - data.loc[:, (item, "std")]

				data_A = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])]
				data_A = data_A.values
				data_theta = data.loc[:, ("theta", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				data_R2 = data.loc[:, ("R2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				data_e2 = data.loc[:, ("epsilon2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				data_A_month.append(data_A)
				data_theta_month.append(data_theta)
				data_R2_month.append(data_R2)
				data_e2_month.append(data_e2)

			data_A_month = np.array(data_A_month)
			data_theta_month = np.array(data_theta_month)
			data_R2_month = np.array(data_R2_month)
			data_e2_month = np.array(data_e2_month)

			for i in range(18):
				plt.figure(figsize=(9, 6))
				gs = gridspec.GridSpec(3,2)
				dates = pd.date_range("2001", periods=12, freq='MS')

				plt.subplot(gs[0, 0])
				plt.plot(dates, data_A_month[:,i,1], '-', color="k")
				plt.fill_between(dates, data_A_month[:,i,2], data_A_month[:,i,3],
					facecolor='green', alpha=0.3, interpolate=True)
				plt.ylim([0, 0.025])
				plt.ylabel('A')
				plt.subplot(gs[0, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

				plt.subplot(gs[1, 0])
				plt.plot(dates, data_theta_month[:,i,1], '-', color="k")
				plt.fill_between(dates, data_theta_month[:,i,2], data_theta_month[:,i,3],
					facecolor='lightskyblue', alpha=0.3, interpolate=True)
				plt.ylim([-60, 60])
				plt.yticks([-60, -40, -20, 0, 20, 40, 60])
				plt.ylabel(r'$\theta$')
				plt.subplot(gs[1, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

				plt.subplot(gs[0, 1])
				plt.plot(dates, data_R2_month[:,i,1], '-', color="k")
				plt.fill_between(dates, data_R2_month[:,i,2], data_R2_month[:,i,3],
					facecolor='coral', alpha=0.3, interpolate=True)
				plt.ylim([0, 1])
				plt.yticks([0, .2, .4, .6, .8, 1])
				plt.ylabel(r'$R^{2}$')
				plt.subplot(gs[0, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

				plt.subplot(gs[1, 1])
				plt.plot(dates, data_e2_month[:,i,1], '-', color="k")
				plt.fill_between(dates, data_e2_month[:,i,2], data_e2_month[:,i,3],
					facecolor='silver', alpha=0.3, interpolate=True)
				plt.ylim([0, 1.5])
				plt.yticks([0, .5, 1, 1.5])
				plt.ylabel(r'$e^{2}$')
				plt.subplot(gs[1, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

				plt.subplot(gs[2, :])
				y = data_A_month[:,i,0]
				plt.plot(dates, y, '-', color="k")
				y_lim_min = max(y.min()-5,0)
				y_lim_min = y.min()
				y_lim_max = y.max()
				plt.ylim([y_lim_min, y_lim_max])
				#plt.yticks(y_lim_min, y_lim_max, int(y_lim_max-y_lim_min+1))
				plt.ylabel("number of data")
				plt.subplot(gs[2, :]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))
				plt.grid(True)

				plt.tight_layout()

				save_name = dirs + "all_area_" + str(i) + "_20" + year + ".png"
				print(save_name)
				plt.savefig(save_name, dpi=400)
				plt.close()

	def ts_30_by_year(dirs):
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		for month in month_list:
			print("*************** " + month + " ***************")
			data_A_year, data_theta_year, data_R2_year, data_e2_year = [], [], [], []
			for year in y_list:
				file_list = "../data/csv_Helmert_both_30_netcdf4/Helmert_both_30_netcdf4_20" + year + month + ".csv"
				df = pd.read_csv(file_list)
				data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

				print((data.loc[:, ("A", "std")]<0).sum())
				for item in ["A", "theta", "R2", "epsilon2"]:
					data.loc[:, (item, "1sigma_pos")] = data.loc[:, (item, "mean")] + data.loc[:, (item, "std")]
					data.loc[:, (item, "1sigma_neg")] = data.loc[:, (item, "mean")] - data.loc[:, (item, "std")]
					#data.loc[:, (item, "2sigma_pos")] = data.loc[:, (item, "mean")] + 2*data.loc[:, (item, "std")]
					#data.loc[:, (item, "2sigma_neg")] = data.loc[:, (item, "mean")] - 2*data.loc[:, (item, "std")]

				data_A = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				data_theta = data.loc[:, ("theta", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				data_R2 = data.loc[:, ("R2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				data_e2 = data.loc[:, ("epsilon2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
				"""
				if year == "03":
					tmp = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])].dropna()
					print(tmp.head())
					print(np.array(tmp)[:3,:])
				"""
				data_A_year.append(data_A)
				data_theta_year.append(data_theta)
				data_R2_year.append(data_R2)
				data_e2_year.append(data_e2)

			data_A_year = np.array(data_A_year)
			data_theta_year = np.array(data_theta_year)
			data_R2_year = np.array(data_R2_year)
			data_e2_year = np.array(data_e2_year)

			dates1 = pd.date_range("2003", "2011", freq='YS')[:-1]
			dates2 = pd.date_range("2013", "2017", freq='YS')[:-1]
			#print(dates1)
			N_dates1 = len(dates1)

			#print(data_A_year[0,np.array(tmp.index),:])
			for i in range(18):
				print("\tarea: {}".format(i))
				plt.figure(figsize=(6, 4))
				gs = gridspec.GridSpec(3,2)
				#gs.tight_layout(plt.figure(figsize=(6, 4)))

				plt.subplot(gs[0, 0])
				plt.plot(dates1, data_A_year[:N_dates1,i,1], '-', color="k")
				plt.plot(dates2, data_A_year[N_dates1:,i,1], '-', color="k")
				plt.fill_between(dates1, data_A_year[:N_dates1,i,2], data_A_year[:N_dates1,i,3],
					facecolor='green', alpha=0.3)
				plt.fill_between(dates2, data_A_year[N_dates1:,i,2], data_A_year[N_dates1:,i,3],
					facecolor='green', alpha=0.3)
				plt.ylim([0, 0.025])
				plt.ylabel('A')
				plt.subplot(gs[0, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

				plt.subplot(gs[1, 0])
				plt.plot(dates1, data_theta_year[:N_dates1,i,1], '-', color="k")
				plt.plot(dates2, data_theta_year[N_dates1:,i,1], '-', color="k")
				plt.fill_between(dates1, data_theta_year[:N_dates1,i,2], data_theta_year[:N_dates1,i,3],
					facecolor='lightskyblue', alpha=0.3)
				plt.fill_between(dates2, data_theta_year[N_dates1:,i,2], data_theta_year[N_dates1:,i,3],
					facecolor='lightskyblue', alpha=0.3)
				plt.ylim([-60, 60])
				plt.yticks([-60, -40, -20, 0, 20, 40, 60])
				plt.ylabel(r'$\theta$')
				plt.subplot(gs[1, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

				plt.subplot(gs[0, 1])
				plt.plot(dates1, data_R2_year[:N_dates1,i,1], '-', color="k")
				plt.plot(dates2, data_R2_year[N_dates1:,i,1], '-', color="k")
				plt.fill_between(dates1, data_R2_year[:N_dates1,i,2], data_R2_year[:N_dates1,i,3],
					facecolor='coral', alpha=0.3)
				plt.fill_between(dates2, data_R2_year[N_dates1:,i,2], data_R2_year[N_dates1:,i,3],
					facecolor='coral', alpha=0.3)
				plt.ylim([0, 1])
				plt.yticks([0, .2, .4, .6, .8, 1])
				plt.ylabel(r'$R^{2}$')
				plt.subplot(gs[0, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

				plt.subplot(gs[1, 1])
				plt.plot(dates1, data_e2_year[:N_dates1,i,1], '-', color="k")
				plt.plot(dates2, data_e2_year[N_dates1:,i,1], '-', color="k")
				plt.fill_between(dates1, data_e2_year[:N_dates1,i,2], data_e2_year[:N_dates1,i,3],
					facecolor='silver', alpha=0.3)
				plt.fill_between(dates2, data_e2_year[N_dates1:,i,2], data_e2_year[N_dates1:,i,3],
					facecolor='silver', alpha=0.3)
				plt.ylim([0, 1.5])
				plt.yticks([0, .5, 1, 1.5])
				plt.ylabel(r'$e^{2}$')
				plt.subplot(gs[1, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

				plt.subplot(gs[2, :])
				y1 = data_A_year[:N_dates1,i,0]
				y2 = data_A_year[N_dates1:,i,0]
				plt.plot(dates1, y1, '-', color="k")
				plt.plot(dates2, y2, '-', color="k")
				y_lim_min = max(y1.min()-5,0)
				y_lim_max = y1.max()+5
				plt.ylim([y_lim_min, y_lim_max])
				#print(int(y_lim_max-y_lim_min+1))
				#plt.yticks(y_lim_min, y_lim_max, int(y_lim_max-y_lim_min+1))
				plt.ylabel("number of data")
				plt.subplot(gs[2, :]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))
				plt.grid(True)

				try:
					plt.tight_layout()
				except:
					print("tight layout passed...")

				save_name = dirs + "all_area_" + str(i) + "_" + month + ".png"
				try:
					plt.savefig(save_name, dpi=300)
				except:
					print("save passed...")
				plt.close()

	def ts_by_month_all_year(dirs):
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		date_1_6 = []
		for year in y_list:
			date_1_6.append(pd.to_datetime("20"+year+"-01-01"))
			date_1_6.append(pd.to_datetime("20"+year+"-07-01"))

		for area_index in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
			data_A_all_year = []
			data_theta_all_year = []
			data_R2_all_year = []
			data_e2_all_year = []
			for year in y_list:
				for month in month_list:
					print(year + month)
					file_list = "../data/csv_Helmert_both_30_netcdf4/Helmert_both_30_netcdf4_20" + year + month + ".csv"
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
			ax.set_ylim([0, 0.025])
			ax.set_ylabel('A')
			for item in date_1_6:
				ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			plt.grid(True)
			plt.savefig(dirs + "A_no_std_area_" + str(area_index) + ".png")
			plt.close()

			fig, ax = plt.subplots(1, 1)
			fig.figsize=(12, 9)
			ax.plot(dates1, data_theta_all_year[:len(dates1)], '-', color="k")
			ax.plot(dates2, data_theta_all_year[len(dates1):], '-', color="k")
			ax.set_ylim([-90,90])
			ax.set_ylabel(r'$\theta$')
			for item in date_1_6:
				ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			plt.grid(True)
			plt.savefig(dirs + "theta_no_std_area_" + str(area_index) + ".png")
			plt.close()

			fig, ax = plt.subplots(1, 1)
			fig.figsize=(12, 9)
			ax.plot(dates1, data_e2_all_year[:len(dates1)], '-', color="k")
			ax.plot(dates2, data_e2_all_year[len(dates1):], '-', color="k")
			ax.set_ylim([0, 1.3])
			ax.set_ylabel(r'$e^{2}$')
			for item in date_1_6:
				ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			plt.grid(True)
			plt.savefig(dirs + "e2_no_std_area_" + str(area_index) + ".png")
			plt.close()

			fig, ax = plt.subplots(1, 1)
			fig.figsize=(12, 9)
			ax.plot(dates1, data_R2_all_year[:len(dates1)], '-', color="k")
			ax.plot(dates2, data_R2_all_year[len(dates1):], '-', color="k")
			ax.set_ylim([0, 1])
			ax.set_ylabel(r'$R^{2}$')
			for item in date_1_6:
				ax.axvline(item, color='coral', linestyle='-', lw=0.75, alpha=0.75)
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
			plt.grid(True)
			plt.savefig(dirs + "R2_no_std_area_" + str(area_index) + ".png")
			plt.close()

	dirs_ts_30_by_year = "../result_nc/ts_30_by_year/"
	dirs_ts_by_month = "../result_nc/ts_by_month/"
	dirs_ts_by_month_all_year = "../result_nc/ts_by_month_all_year/"
	if num == 1:
		ts_30_by_month(dirs_ts_by_month)
	elif num == 2:
		ts_30_by_year(dirs_ts_30_by_year)
	elif num == 3:
		ts_by_month_all_year(dirs_ts_by_month_all_year)

#plot_nc_data_ts(1)
#plot_nc_data_ts(2)
plot_nc_data_ts(3)




def compare_gw_w10():
	dirs_theta = "../result_nc/comare_gw_w10/theta/"
	dirs_speed = "../result_nc/comare_gw_w10/speed/"
	if not os.path.exists(dirs_theta):
		os.makedirs(dirs_theta)
	if not os.path.exists(dirs_speed):
		os.makedirs(dirs_speed)

	gw_file_list = sorted(glob.glob("../data/csv_Helmert_both_30/Helmert_both_30_*.csv"))
	w10_file_list = sorted(glob.glob("../data/csv_Helmert_both_30_netcdf4/Helmert_both_30_netcdf4_*.csv"))

	N = len(w10_file_list)
	for i in range(N):
		data_gw = pd.read_csv(gw_file_list[i])
		data_w10 = pd.read_csv(w10_file_list[i])
		print(gw_file_list[i])
		print(w10_file_list[i])

		gw_theta = np.arctan2(np.array(data_gw["mean_w_v"]), np.array(data_gw["mean_w_u"]))
		w10_theta = np.arctan2(np.array(data_w10["mean_w_v"]), np.array(data_w10["mean_w_u"]))

		gw_speed = np.sqrt(np.array(data_gw["mean_w_v"])**2 + np.array(data_gw["mean_w_u"])**2)
		w10_speed = np.sqrt(np.array(data_w10["mean_w_v"])**2 + np.array(data_w10["mean_w_u"])**2)

		diff_theta = w10_theta - gw_theta
		diff_speed = w10_speed / gw_speed

		save_name_theta = dirs_theta + gw_file_list[i][44:50] + ".png"
		save_name_speed = dirs_speed + gw_file_list[i][44:50] + ".png"

		visualize.plot_map_once(diff_theta, data_type="type_non_wind", show=False, 
			save_name=save_name_theta, vmax=180, vmin=-180, cmap=plt.cm.jet)
		visualize.plot_map_once(diff_speed, data_type="type_non_wind", show=False, 
			save_name=save_name_speed, vmax=2, vmin=1/2, cmap=plt.cm.jet)

#compare_gw_w10()











