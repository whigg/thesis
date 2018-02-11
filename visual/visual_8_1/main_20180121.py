"""
地上10m風の1日ずらした風力係数の計算，プロット(map, 時系列)
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


###############################################################################################################

def get_helmert_30_nc_1day_delay():
	dirs = "../data/csv_Helmert_30_netcdf4_1day_delay/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	grid_data = pd.read_csv(grid900to145_file_name, header=None)
	grid145 = np.array(grid_data, dtype='int64').ravel()
	for year in y_list:
		for month in month_list:
			print("******************** {} ***************".format(year + month))
			if month in ["04", "06", "09", "11"]:
				L = 30
			elif month == "02":
				if year in ["04", "08", "16"]:
					L = 29
				else:
					L = 28
			else:
				L = 31
			start_0101 = date(int("20"+year), 1, 1)
			start_date = date(int("20"+year), int(month), 1)
			start_from_0101_idx = (start_date-start_0101).days
			#print(start_0101, start_date, start_from_0101_idx)
			if month != "12":
				nc_fname = "../data/netcdf4/interim_2mt_10u_10v_20" + year + "0101-20" + year + "1231.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
				w10_u_array = result_u10[start_from_0101_idx+1:start_from_0101_idx+L+1, :]
				w10_v_array = result_v10[start_from_0101_idx+1:start_from_0101_idx+L+1, :]
			else:
				nc_fname_1 = "../data/netcdf4/interim_2mt_10u_10v_20" + year + "0101-20" + year + "1231.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname_1)
				w10_u_array_1 = result_u10[start_from_0101_idx+1:start_from_0101_idx+L, :]
				w10_v_array_1 = result_v10[start_from_0101_idx+1:start_from_0101_idx+L, :]
				next_year = str(int(year)+1)
				if len(next_year) == 1:
					next_year = "0" + next_year
				nc_fname_2 = "../data/netcdf4/interim_2mt_10u_10v_20" + next_year + "0101-20" + next_year + "1231.nc"
				if year == "16":
					nc_fname_2 = "../data/netcdf4/interim_2mt_10u_10v_20" + next_year + "0101-20" + next_year + "0930.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname_2)
				w10_u_array_2 = result_u10[0, :].reshape((1, -1))
				w10_v_array_2 = result_v10[0, :].reshape((1, -1))
				w10_u_array = np.vstack((w10_u_array_1, w10_u_array_2))
				w10_v_array = np.vstack((w10_v_array_1, w10_v_array_2))
			w10_uv = []
			for day_idx in range(L):
				tmp_1 = w10_u_array[day_idx, :].reshape((-1,1))
				tmp_2 = w10_v_array[day_idx, :].reshape((-1,1))
				w10_uv.append(np.hstack((tmp_1, tmp_2)))
			w10_uv = np.array(w10_uv)

			iw_file_list = sorted(glob.glob("../data/csv_iw/" + year + month + "*.csv"))
			iw_list = []
			L_iw = len(iw_file_list)
			for iw_fname in iw_file_list:
				print("\t{}".format(iw_fname))
				df_ice_wind = pd.read_csv(iw_fname, header=None)
				df_ice_wind[df_ice_wind==999.] = np.nan
				ice_wind = np.array(df_ice_wind, dtype='float32')/100
				iw_list.append(ice_wind[:, [0,1]])
			iw_array = np.array(iw_list)

			gw_array = np.where(np.isnan(iw_array), np.nan, w10_uv)

			ic0_file_list = sorted(glob.glob("../data/csv_ic0/IC0_20" + year + month + "*.csv"))
			sit_file_list = sorted(glob.glob("../data/csv_sit/SIT_20" + year + month + "*.csv"))
			ic0_list = []
			sit_list = []
			for ic0_fname in ic0_file_list:
				print("\t{}".format(ic0_fname))
				ic0_data = pd.read_csv(ic0_fname, header=None)
				ic0 = np.array(ic0_data, dtype='float32')
				ic0_145 = ic0[grid145]
				ic0_list.append(ic0_145)
			for sit_fname in sit_file_list:
				print("\t{}".format(sit_fname))
				sit_data = pd.read_csv(sit_fname, header=None)
				sit = np.array(sit_data, dtype='float32')
				sit[sit>=10001] = np.nan
				sit_145 = sit[grid145]
				sit_list.append(sit_145)

			ic0_array = np.array(ic0_list)
			sit_array = np.array(sit_list)

			gw_ave = np.nanmean(gw_array, axis=0)
			iw_ave = np.nanmean(iw_array, axis=0)
			ic0_ave = np.nanmean(ic0_array, axis=0)
			sit_ave = np.nanmean(sit_array, axis=0)
			ic0_med = np.nanmedian(ic0_array, axis=0)
			sit_med = np.nanmedian(sit_array, axis=0)

			L_gw = L
			if L_gw != L_iw:
				print("continuing the for loop...")
				continue
			gw_minus_ave = gw_array - np.tile(gw_ave, (L_gw,1,1))
			iw_minus_ave = iw_array - np.tile(iw_ave, (L_iw,1,1))

			param_list = []
			for j in range(145**2):
				#print(j)
				N_c = np.sum(~np.isnan(iw_minus_ave[:,j,0]))
				if N_c <= 20:
					param_list.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c])
					continue
				gw_u = gw_minus_ave[:,j,0]
				gw_v = gw_minus_ave[:,j,1]
				iw_u = iw_minus_ave[:,j,0]
				iw_v = iw_minus_ave[:,j,1]
				b_1 = (np.nansum(gw_u*iw_u) + np.nansum(gw_v*iw_v)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
				b_2 = (np.nansum(gw_u*iw_v) - np.nansum(gw_v*iw_u)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
				a_1 = iw_ave[j,0] - b_1*gw_ave[j,0] + b_2*gw_ave[j,1]
				a_2 = iw_ave[j,1] - b_1*gw_ave[j,1] - b_2*gw_ave[j,0]
				R_denominator = np.nansum(iw_u**2 + iw_v**2)
				R_numerator = np.nansum((iw_array[:,j,0] - (a_1 + b_1*gw_array[:,j,0] - b_2*gw_array[:,j,1]))**2) + \
					np.nansum((iw_array[:,j,1] - (a_2 + b_2*gw_array[:,j,0] + b_1*gw_array[:,j,1]))**2)
				R2 = 1 - R_numerator/R_denominator
				A = np.sqrt(b_1**2 + b_2**2)
				theta = np.arctan2(b_2, b_1) * 180/np.pi
				#print(a_1, a_2, b_1, b_2)
				#print("A: {}\ntheta: {}\na_1: {}\na_2: {}\nR2: {}\ne2: {}\nN_c: {}".format(A, theta, a_1, a_2, R2, R_numerator, N_c))
				param_list.append([A, theta, a_1, a_2, R2, R_numerator, N_c])
			param_array = np.array(param_list)

			data_array = np.hstack((param_array, iw_ave, gw_ave))
			data = pd.DataFrame(data_array)
			data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
			data["ic0_30"] = ic0_ave
			data["sit_30"] = sit_ave
			data["ic0_30_median"] = ic0_med
			data["sit_30_median"] = sit_med
			data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_20" + year + month + ".csv"
			data_ex = pd.read_csv(data_ex_dir)
			data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
			data.loc[(data.Lat>=80)&(data.ic0_30.isnull()), ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "mean_iw_u", "mean_iw_v"]] = np.nan

			save_name = dirs + "Helmert_30_netcdf4_1day_delay_20" + year + month + ".csv"
			print(save_name)
			data.to_csv(save_name, index=False)


#get_helmert_30_nc_1day_delay()





def get_helmert_90_1day_delay_csv():
	dirs = "../data/csv_Helmert_90_netcdf4_1day_delay/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	m_list = [["01", "02", "03"], ["04", "05", "06"], ["07", "08", "09"], ["10", "11", "12"]]
	grid_data = pd.read_csv(grid900to145_file_name, header=None)
	grid145 = np.array(grid_data, dtype='int64').ravel()
	for year in y_list:
		for im, m3 in enumerate(m_list):
			print("******************** {}, {} ***************".format(year, m3))
			def get_month_len(year, month):
				if month in ["04", "06", "09", "11"]:
					L1 = 30
				elif month == "02":
					if year in ["04", "08", "16"]:
						L1 = 29
					else:
						L1 = 28
				else:
					L1 = 31
				return (L1)

			L = get_month_len(year, m3[0]) + get_month_len(year, m3[1]) + get_month_len(year, m3[2])
			start_0101 = date(int("20"+year), 1, 1)
			start_date = date(int("20"+year), int(m3[0]), 1)
			start_from_0101_idx = (start_date-start_0101).days
			#print(start_0101, start_date, start_from_0101_idx)
			if m3[2] != "12":
				nc_fname = "../data/netcdf4/interim_2mt_10u_10v_20" + year + "0101-20" + year + "1231.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
				w10_u_array = result_u10[start_from_0101_idx+1:start_from_0101_idx+L+1, :]
				w10_v_array = result_v10[start_from_0101_idx+1:start_from_0101_idx+L+1, :]
			else:
				nc_fname_1 = "../data/netcdf4/interim_2mt_10u_10v_20" + year + "0101-20" + year + "1231.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname_1)
				w10_u_array_1 = result_u10[start_from_0101_idx+1:start_from_0101_idx+L, :]
				w10_v_array_1 = result_v10[start_from_0101_idx+1:start_from_0101_idx+L, :]
				next_year = str(int(year)+1)
				if len(next_year) == 1:
					next_year = "0" + next_year
				nc_fname_2 = "../data/netcdf4/interim_2mt_10u_10v_20" + next_year + "0101-20" + next_year + "1231.nc"
				if year == "16":
					nc_fname_2 = "../data/netcdf4/interim_2mt_10u_10v_20" + next_year + "0101-20" + next_year + "0930.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname_2)
				w10_u_array_2 = result_u10[0, :].reshape((1, -1))
				w10_v_array_2 = result_v10[0, :].reshape((1, -1))
				w10_u_array = np.vstack((w10_u_array_1, w10_u_array_2))
				w10_v_array = np.vstack((w10_v_array_1, w10_v_array_2))
			w10_uv = []
			for day_idx in range(L):
				tmp_1 = w10_u_array[day_idx, :].reshape((-1,1))
				tmp_2 = w10_v_array[day_idx, :].reshape((-1,1))
				w10_uv.append(np.hstack((tmp_1, tmp_2)))
			w10_uv = np.array(w10_uv)

			iw_file_list = sorted(glob.glob("../data/csv_iw/" + year + m3[0] + "*.csv") + \
				glob.glob("../data/csv_iw/" + year + m3[1] + "*.csv") + \
				glob.glob("../data/csv_iw/" + year + m3[2] + "*.csv"))

			iw_list = []
			L_iw = len(iw_file_list)
			for iw_fname in iw_file_list:
				print("\t{}".format(iw_fname))
				df_ice_wind = pd.read_csv(iw_fname, header=None)
				df_ice_wind[df_ice_wind==999.] = np.nan
				ice_wind = np.array(df_ice_wind, dtype='float32')/100
				iw_list.append(ice_wind[:, [0,1]])

			L_gw = L
			if L_gw != L_iw:
				print("continuing the for loop...")
				continue

			iw_array = np.array(iw_list)
			gw_array = np.where(np.isnan(iw_array), np.nan, w10_uv)

			gw_ave = np.nanmean(gw_array, axis=0)
			iw_ave = np.nanmean(iw_array, axis=0)

			gw_minus_ave = gw_array - np.tile(gw_ave, (L_gw,1,1))
			iw_minus_ave = iw_array - np.tile(iw_ave, (L_iw,1,1))

			param_list = []
			for j in range(145**2):
				#print(j)
				N_c = np.sum(~np.isnan(iw_minus_ave[:,j,0]))
				if N_c <= 45:
					param_list.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c])
					continue
				gw_u = gw_minus_ave[:,j,0]
				gw_v = gw_minus_ave[:,j,1]
				iw_u = iw_minus_ave[:,j,0]
				iw_v = iw_minus_ave[:,j,1]
				b_1 = (np.nansum(gw_u*iw_u) + np.nansum(gw_v*iw_v)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
				b_2 = (np.nansum(gw_u*iw_v) - np.nansum(gw_v*iw_u)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
				a_1 = iw_ave[j,0] - b_1*gw_ave[j,0] + b_2*gw_ave[j,1]
				a_2 = iw_ave[j,1] - b_1*gw_ave[j,1] - b_2*gw_ave[j,0]
				R_denominator = np.nansum(iw_u**2 + iw_v**2)
				R_numerator = np.nansum((iw_array[:,j,0] - (a_1 + b_1*gw_array[:,j,0] - b_2*gw_array[:,j,1]))**2) + \
					np.nansum((iw_array[:,j,1] - (a_2 + b_2*gw_array[:,j,0] + b_1*gw_array[:,j,1]))**2)
				R2 = 1 - R_numerator/R_denominator
				A = np.sqrt(b_1**2 + b_2**2)
				theta = np.arctan2(b_2, b_1) * 180/np.pi
				#print(a_1, a_2, b_1, b_2)
				#print("A: {}\ntheta: {}\na_1: {}\na_2: {}\nR2: {}\ne2: {}\nN_c: {}".format(A, theta, a_1, a_2, R2, R_numerator, N_c))
				param_list.append([A, theta, a_1, a_2, R2, R_numerator, N_c])
			param_array = np.array(param_list)

			data_array = np.hstack((param_array, iw_ave, gw_ave))
			data = pd.DataFrame(data_array)
			data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
			data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]], no_sensor_data], axis=1)
			data.loc[data["non"]==1., ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "mean_iw_u", "mean_iw_v"]] = np.nan
			data.drop("non", axis=1)

			save_name = dirs + "Helmert_90_1day_delay_20" + year + "_" + str(im+1) + ".csv"
			print(save_name)
			data.to_csv(save_name, index=False)

#get_helmert_90_1day_delay_csv()




def get_helmert_by_year_netcdf4_1day_delay():
	dirs = "../data/csv_Helmert_by_year_netcdf4_1day_delay/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	grid_data = pd.read_csv(grid900to145_file_name, header=None)
	grid145 = np.array(grid_data, dtype='int64').ravel()
	for im, month in enumerate(month_list):
		print("******************** {} ***************".format(month))
		gw_array = np.zeros((1, 145**2, 2))
		iw_file_list = []
		for iy, year in enumerate(y_list):
			def get_month_len(year, month):
				if month in ["04", "06", "09", "11"]:
					L1 = 30
				elif month == "02":
					if year in ["04", "08", "16"]:
						L1 = 29
					else:
						L1 = 28
				else:
					L1 = 31
				return (L1)

			L = get_month_len(year, month)
			start_0101 = date(int("20"+year), 1, 1)
			start_date = date(int("20"+year), int(month), 1)
			start_from_0101_idx = (start_date-start_0101).days

			if month != "12":
				nc_fname = "../data/netcdf4/interim_2mt_10u_10v_20" + year + "0101-20" + year + "1231.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname)
				w10_u_array = result_u10[start_from_0101_idx+1:start_from_0101_idx+L+1, :]
				w10_v_array = result_v10[start_from_0101_idx+1:start_from_0101_idx+L+1, :]
			else:
				nc_fname_1 = "../data/netcdf4/interim_2mt_10u_10v_20" + year + "0101-20" + year + "1231.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname_1)
				w10_u_array_1 = result_u10[start_from_0101_idx+1:start_from_0101_idx+L, :]
				w10_v_array_1 = result_v10[start_from_0101_idx+1:start_from_0101_idx+L, :]
				next_year = str(int(year)+1)
				if len(next_year) == 1:
					next_year = "0" + next_year
				nc_fname_2 = "../data/netcdf4/interim_2mt_10u_10v_20" + next_year + "0101-20" + next_year + "1231.nc"
				if year == "16":
					nc_fname_2 = "../data/netcdf4/interim_2mt_10u_10v_20" + next_year + "0101-20" + next_year + "0930.nc"
				_, result_u10, result_v10 = calc_data.get_1month_netcdf4_data(nc_fname_2)
				w10_u_array_2 = result_u10[0, :].reshape((1, -1))
				w10_v_array_2 = result_v10[0, :].reshape((1, -1))
				w10_u_array = np.vstack((w10_u_array_1, w10_u_array_2))
				w10_v_array = np.vstack((w10_v_array_1, w10_v_array_2))
			w10_uv = []
			for day_idx in range(L):
				tmp_1 = w10_u_array[day_idx, :].reshape((-1,1))
				tmp_2 = w10_v_array[day_idx, :].reshape((-1,1))
				w10_uv.append(np.hstack((tmp_1, tmp_2)))
			w10_uv = np.array(w10_uv)
			gw_array = np.concatenate([gw_array, w10_uv], axis=0)

			iw_file_list = iw_file_list + glob.glob("../data/csv_iw/" + year + month + "*.csv")

		gw_array = gw_array[1:, :, :]

		iw_file_list = sorted(iw_file_list)
		L_iw = len(iw_file_list)
		iw_list = []
		for iw_fname in iw_file_list:
			print("\t{}".format(iw_fname))
			df_ice_wind = pd.read_csv(iw_fname, header=None)
			df_ice_wind[df_ice_wind==999.] = np.nan
			ice_wind = np.array(df_ice_wind, dtype='float32')/100
			iw_list.append(ice_wind[:, [0,1]])

		L_gw = gw_array.shape[0]
		if L_gw != L_iw:
			print("continuing the for loop...")
			continue

		iw_array = np.array(iw_list)
		gw_array = np.where(np.isnan(iw_array), np.nan, gw_array)

		gw_ave = np.nanmean(gw_array, axis=0)
		iw_ave = np.nanmean(iw_array, axis=0)

		gw_minus_ave = gw_array - np.tile(gw_ave, (L_gw,1,1))
		iw_minus_ave = iw_array - np.tile(iw_ave, (L_iw,1,1))

		param_list = []
		for j in range(145**2):
			#print(j)
			N_c = np.sum(~np.isnan(iw_minus_ave[:,j,0]))
			if N_c <= 120:
				param_list.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c])
				continue
			gw_u = gw_minus_ave[:,j,0]
			gw_v = gw_minus_ave[:,j,1]
			iw_u = iw_minus_ave[:,j,0]
			iw_v = iw_minus_ave[:,j,1]
			b_1 = (np.nansum(gw_u*iw_u) + np.nansum(gw_v*iw_v)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
			b_2 = (np.nansum(gw_u*iw_v) - np.nansum(gw_v*iw_u)) / (np.nansum(gw_u**2) + np.nansum(gw_v**2))
			a_1 = iw_ave[j,0] - b_1*gw_ave[j,0] + b_2*gw_ave[j,1]
			a_2 = iw_ave[j,1] - b_1*gw_ave[j,1] - b_2*gw_ave[j,0]
			R_denominator = np.nansum(iw_u**2 + iw_v**2)
			R_numerator = np.nansum((iw_array[:,j,0] - (a_1 + b_1*gw_array[:,j,0] - b_2*gw_array[:,j,1]))**2) + \
				np.nansum((iw_array[:,j,1] - (a_2 + b_2*gw_array[:,j,0] + b_1*gw_array[:,j,1]))**2)
			R2 = 1 - R_numerator/R_denominator
			A = np.sqrt(b_1**2 + b_2**2)
			theta = np.arctan2(b_2, b_1) * 180/np.pi
			#print(a_1, a_2, b_1, b_2)
			#print("A: {}\ntheta: {}\na_1: {}\na_2: {}\nR2: {}\ne2: {}\nN_c: {}".format(A, theta, a_1, a_2, R2, R_numerator, N_c))
			param_list.append([A, theta, a_1, a_2, R2, R_numerator, N_c])
		param_array = np.array(param_list)

		data_array = np.hstack((param_array, iw_ave, gw_ave))
		data = pd.DataFrame(data_array)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		data = pd.concat([latlon_ex, data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]], no_sensor_data], axis=1)
		data.loc[data["non"]==1., ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "mean_iw_u", "mean_iw_v"]] = np.nan
		data.drop("non", axis=1)

		save_name = dirs + "Helmert_by_year_netcdf4_1day_delay_" + month + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)


#get_helmert_by_year_netcdf4_1day_delay()


###############################################################################################################

def plot_nc_data_map_1day_delay(num):
	dirs_A_30 = "../result_nc_1day_delay/A/A_30/"
	dirs_R2_30 = "../result_nc_1day_delay/R2/R2_30/"
	dirs_theta_30 = "../result_nc_1day_delay/theta/theta_30/"
	dirs_epsilon2_30 = "../result_nc_1day_delay/epsilon2/epsilon2_30/"

	dirs_A_90 = "../result_nc_1day_delay/A/A_90/"
	dirs_R2_90 = "../result_nc_1day_delay/R2/R2_90/"
	dirs_theta_90 = "../result_nc_1day_delay/theta/theta_90/"
	dirs_epsilon2_90 = "../result_nc_1day_delay/epsilon2/epsilon2_90/"

	dirs_A_by_year = "../result_nc_1day_delay/A/A_by_year/"
	dirs_R2_by_year = "../result_nc_1day_delay/R2/R2_by_year/"
	dirs_theta_by_year = "../result_nc_1day_delay/theta/theta_by_year/"
	dirs_epsilon2_by_year = "../result_nc_1day_delay/epsilon2/epsilon2_by_year/"

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

	file_list_30 = sorted(glob.glob("../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_*.csv"))
	file_list_90 = sorted(glob.glob("../data/csv_Helmert_90_netcdf4_1day_delay/csv_Helmert_90_netcdf4_1day_delay_*.csv"))
	file_list_year = sorted(glob.glob("../data/csv_Helmert_by_year_netcdf4_1day_delay/Helmert_by_year_netcdf4_1day_delay_*.csv"))

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	if num == 1:
		for file in file_list_30:
			data = pd.read_csv(file)
			save_name_A = dirs_A_30 + file[72:78] + ".png"
			print(save_name_A)
			visualize.plot_map_once(np.array(data["A"]), data_type="type_non_wind", show=False, 
				save_name=save_name_A, vmax=0.02, vmin=0, cmap=plt.cm.jet)
			save_name_theta = dirs_theta_30 + file[72:78] + ".png"
			print(save_name_theta)
			visualize.plot_map_once(np.array(data["theta"]), data_type="type_non_wind", show=False, 
				save_name=save_name_theta, vmax=180, vmin=-180, cmap=cm_angle_2)
			save_name_R2 = dirs_R2_30 + file[72:78] + ".png"
			print(save_name_R2)
			visualize.plot_map_once(np.array(data["R2"]), data_type="type_non_wind", show=False, 
				save_name=save_name_R2, vmax=1, vmin=0, cmap=plt.cm.jet)
			save_name_e2 = dirs_epsilon2_30 + file[72:78] + ".png"
			print(save_name_e2)
			visualize.plot_map_once(np.array(data["epsilon2"]), data_type="type_non_wind", show=False, 
				save_name=save_name_e2, vmax=1.5, vmin=0, cmap=plt.cm.jet)
	elif num == 2:
		for file in file_list_90:
			data = pd.read_csv(file)
			save_name_A = dirs_A_90 + file[72:78] + ".png"
			print(save_name_A)
			visualize.plot_map_once(np.array(data["A"]), data_type="type_non_wind", show=False, 
				save_name=save_name_A, vmax=0.02, vmin=0, cmap=plt.cm.jet)
			save_name_theta = dirs_theta_90 + file[72:78] + ".png"
			print(save_name_theta)
			visualize.plot_map_once(np.array(data["theta"]), data_type="type_non_wind", show=False, 
				save_name=save_name_theta, vmax=180, vmin=-180, cmap=cm_angle_2)
			save_name_R2 = dirs_R2_90 + file[72:78] + ".png"
			print(save_name_R2)
			visualize.plot_map_once(np.array(data["R2"]), data_type="type_non_wind", show=False, 
				save_name=save_name_R2, vmax=1, vmin=0, cmap=plt.cm.jet)
			save_name_e2 = dirs_epsilon2_90 + file[72:78] + ".png"
			print(save_name_e2)
			visualize.plot_map_once(np.array(data["epsilon2"]), data_type="type_non_wind", show=False, 
				save_name=save_name_e2, vmax=1.5, vmin=0, cmap=plt.cm.jet)
	elif num == 3:
		for file in file_list_year:
			data = pd.read_csv(file)
			save_name_A = dirs_A_by_year + file[82:84] + ".png"
			print(save_name_A)
			visualize.plot_map_once(np.array(data["A"]), data_type="type_non_wind", show=False, 
				save_name=save_name_A, vmax=0.02, vmin=0, cmap=plt.cm.jet)
			save_name_theta = dirs_theta_by_year + file[82:84] + ".png"
			print(save_name_theta)
			visualize.plot_map_once(np.array(data["theta"]), data_type="type_non_wind", show=False, 
				save_name=save_name_theta, vmax=180, vmin=-180, cmap=cm_angle_2)
			save_name_R2 = dirs_R2_by_year + file[82:84] + ".png"
			print(save_name_R2)
			visualize.plot_map_once(np.array(data["R2"]), data_type="type_non_wind", show=False, 
				save_name=save_name_R2, vmax=1, vmin=0, cmap=plt.cm.jet)
			save_name_e2 = dirs_epsilon2_by_year + file[82:84] + ".png"
			print(save_name_e2)
			visualize.plot_map_once(np.array(data["epsilon2"]), data_type="type_non_wind", show=False, 
				save_name=save_name_e2, vmax=1.5, vmin=0, cmap=plt.cm.jet)


#plot_nc_data_map_1day_delay(num=1)
#plot_nc_data_map_1day_delay(num=2)
#plot_nc_data_map_1day_delay(num=3)



def plot_nc_data_corr_1day_delay():
	dirs_corr_map = "../result_nc_1day_delay/corr_map/"
	dirs_corr_map_search_grid = "../result_nc_1day_delay/corr_map_search_grid/"

	if not os.path.exists(dirs_corr_map):
		os.makedirs(dirs_corr_map)
	if not os.path.exists(dirs_corr_map_search_grid):
		os.makedirs(dirs_corr_map_search_grid)

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

		save_name_corr = dirs_corr_map + "ic0_A_" + month + ".png"
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

plot_nc_data_corr_1day_delay()




def print_nc_data_1day_delay():
	def print_describe_data_30_nc_1day_delay(dirs_30):
		if not os.path.exists(dirs_30):
			os.makedirs(dirs_30)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_*.csv"))
		for file in file_list:
			print(file)
			df = pd.read_csv(file)
			data = df.groupby("area_label")[["A", "theta", "R2", "N_c", "ic0_30", "sit_30", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
			data.to_csv(dirs_30 + "describe_data_30_" + file[72:])

	def print_describe_data_90_nc_1day_delay(dirs_90):
		if not os.path.exists(dirs_90):
			os.makedirs(dirs_90)
		file_list = sorted(glob.glob("../data/csv_Helmert_90_netcdf4_1day_delay/Helmert_90_netcdf4_1day_delay_*.csv"))
		for file in file_list:
			print(file)
			df = pd.read_csv(file)
			data = df.groupby("area_label")[["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
			data.to_csv(dirs_90 + "describe_data_90_" + file[72:])

	def print_describe_data_by_year_nc_1day_delay(dirs_year):
		if not os.path.exists(dirs_year):
			os.makedirs(dirs_year)
		#data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
		#data_ex = pd.read_csv(data_ex_dir)
		file_list = sorted(glob.glob("../data/csv_Helmert_by_year_netcdf4_1day_delay/Helmert_by_year_netcdf4_1day_delay_*.csv"))
		for file in file_list:
			print(file)
			df = pd.read_csv(file)
			#df = pd.concat([latlon_ex, df, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
			data = df.groupby("area_label")[["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
			data.to_csv(dirs_year + "describe_data_by_year_" + file[82:])

	def print_describe_data_all_region_nc_1day_delay():
		dirs_30 = "../result_nc_1day_delay/print_data/print_data_30_all/"
		if not os.path.exists(dirs_30):
			os.makedirs(dirs_30)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_*.csv"))
		for file in file_list:
			print(file)
			df = pd.read_csv(file).dropna()
			data = df.loc[:, ["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
			data.to_csv(dirs_30 + "describe_data_30_all_" + file[72:])

		dirs_90 = "../result_nc_1day_delay/print_data/print_data_90_all/"
		if not os.path.exists(dirs_90):
			os.makedirs(dirs_90)
		file_list = sorted(glob.glob("../data/csv_Helmert_90_netcdf4_1day_delay/Helmert_90_netcdf4_1day_delay_*.csv"))
		for file in file_list:
			print(file)
			df = pd.read_csv(file).dropna()
			data = df.loc[:, ["A", "theta", "R2", "epsilon2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
			data.to_csv(dirs_90 + "describe_data_90_all_" + file[72:])

		dirs_year = "../result_nc_1day_delay/print_data/print_data_by_year_all/"
		if not os.path.exists(dirs_year):
			os.makedirs(dirs_year)
		#data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
		#data_ex = pd.read_csv(data_ex_dir)
		file_list = sorted(glob.glob("../data/csv_Helmert_by_year_netcdf4_1day_delay/Helmert_by_year_netcdf4_1day_delay_*.csv"))
		for file in file_list:
			print(file)
			df = pd.read_csv(file)
			#df = pd.concat([latlon_ex, df, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1).dropna()
			data = df.loc[:, ["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
			data.to_csv(dirs_year + "describe_data_by_year_all_" + file[82:])

	dirs_30 = "../result_h_1day_delay/print_data/print_data_30/"
	dirs_90 = "../result_h_1day_delay/print_data/print_data_90/"
	dirs_year = "../result_h_1day_delay/print_data/print_data_by_year/"
	print_describe_data_30_nc_1day_delay(dirs_30)
	print_describe_data_90_nc_1day_delay(dirs_90)
	print_describe_data_by_year_nc_1day_delay(dirs_year)
	print_describe_data_all_region_nc_1day_delay()

#print_nc_data_1day_delay()





def plot_ts_nc_1day_delay(num):
	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	def ts_30_by_month(dirs):
		if not os.path.exists(dirs):
			os.makedirs(dirs)

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
				ax.plot(dates, data_A_year_list[j][:,i,1], '-', label = "20" + y_list[j])
			#ax.legend(loc="center", bbox_to_anchor=(1.1, 0.5))
			ax.legend(bbox_to_anchor=(1.08, 0.8))
			plt.subplots_adjust(right=0.75)
			ax.set_ylim([0, 0.02])
			ax.set_ylabel('A')
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
			plt.grid(True)
			plt.savefig(dirs + "A_area_" + str(i) + ".png", dpi=150)
			plt.close()

			fig, ax = plt.subplots(1, 1)
			fig.figsize=(15, 6)
			for j in range(len_y):
				ax.plot(dates, data_theta_year_list[j][:,i,1], '-', label = "20" + y_list[j])
			ax.legend(bbox_to_anchor=(1.08, 0.8))
			plt.subplots_adjust(right=0.75)
			ax.set_ylim([-60, 60])
			ax.set_ylabel(r'$\theta$')
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
			plt.grid(True)
			plt.savefig(dirs + "theta_area_" + str(i) + ".png", dpi=150)
			plt.close()

			fig, ax = plt.subplots(1, 1)
			fig.figsize=(15, 6)
			for j in range(len_y):
				ax.plot(dates, data_R2_year_list[j][:,i,1], '-', label = "20" + y_list[j])
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
				ax.plot(dates, data_e2_year_list[j][:,i,1], '-', label = "20" + y_list[j])
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
				ax.plot(dates, y, '-', label = "20" + y_list[j])
			ax.legend(bbox_to_anchor=(1.08, 0.8))
			plt.subplots_adjust(right=0.75)
			ax.set_ylabel("number of data")
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
			plt.grid(True)
			plt.savefig(dirs + "count_area_" + str(i) + ".png", dpi=150)
			plt.close()

	def ts_30_by_year(dirs):
		if not os.path.exists(dirs):
			os.makedirs(dirs)

		for month in month_list:
		#for month in ["07", "08", "09", "10", "11", "12"]:
			print("*************** " + month + " ***************")
			data_A_year, data_theta_year, data_R2_year, data_e2_year = [], [], [], []
			for year in y_list:
				file_list = "../data/csv_Helmert_30_netcdf4_1day_delay/Helmert_30_netcdf4_1day_delay_20" + year + month + ".csv"
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

			file_by_year = "../data/csv_Helmert_by_year_netcdf4_1day_delay/Helmert_by_year_netcdf4_1day_delay_" + month + ".csv"
			data_by_year = pd.read_csv(file_by_year)
			data_by_year = data_by_year.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			dates1 = pd.date_range("2003", "2011", freq='YS')[:-1]
			dates2 = pd.date_range("2013", "2017", freq='YS')[:-1]
			N_dates1 = len(dates1)

			for i in range(18):
				print("\tarea: {}".format(i))

				fig, ax = plt.subplots(1, 1)
				fig.figsize=(6, 4)
				ax.plot(dates1, data_A_year[:N_dates1,i,1], '-', color="k")
				ax.plot(dates2, data_A_year[N_dates1:,i,1], '-', color="k")
				A_by_year = data_by_year.loc[(i), ("A", "mean")]
				ax.plot([dates1[0], dates1[-1]], [A_by_year, A_by_year], "coral", linestyle='dashed')
				ax.plot([dates2[0], dates2[-1]], [A_by_year, A_by_year], "coral", linestyle='dashed')
				ax.fill_between(dates1, data_A_year[:N_dates1,i,2], data_A_year[:N_dates1,i,3],
					facecolor='green', alpha=0.3)
				ax.fill_between(dates2, data_A_year[N_dates1:,i,2], data_A_year[N_dates1:,i,3],
					facecolor='green', alpha=0.3)
				ax.set_ylim([0, 0.02])
				ax.set_ylabel('A')
				try:
					ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
				except:
					print(data_A_year[:,i,1])
					print("skipping the for loop...  1")
					continue
				plt.grid(True)
				save_name = dirs + "A_area_" + str(i) + "_" + month + ".png"
				try:
					plt.savefig(save_name, dpi=300)
				except:
					print(data_A_year[:,i,1])
					print("skipping the for loop...  2")
					continue
				plt.close()

				fig, ax = plt.subplots(1, 1)
				fig.figsize=(6, 4)
				ax.plot(dates1, data_theta_year[:N_dates1,i,1], '-', color="k")
				ax.plot(dates2, data_theta_year[N_dates1:,i,1], '-', color="k")
				theta_by_year = data_by_year.loc[(i), ("theta", "mean")]
				ax.plot([dates1[0], dates1[-1]], [theta_by_year, theta_by_year], "coral", linestyle='dashed')
				ax.plot([dates2[0], dates2[-1]], [theta_by_year, theta_by_year], "coral", linestyle='dashed')
				ax.fill_between(dates1, data_theta_year[:N_dates1,i,2], data_theta_year[:N_dates1,i,3],
					facecolor='lightskyblue', alpha=0.3)
				ax.fill_between(dates2, data_theta_year[N_dates1:,i,2], data_theta_year[N_dates1:,i,3],
					facecolor='lightskyblue', alpha=0.3)
				ax.set_ylim([-60, 60])
				ax.set_ylabel(r'$\theta$')
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
				plt.grid(True)
				save_name = dirs + "theta_area_" + str(i) + "_" + month + ".png"
				plt.savefig(save_name, dpi=300)
				plt.close()

				fig, ax = plt.subplots(1, 1)
				fig.figsize=(6, 4)
				ax.plot(dates1, data_R2_year[:N_dates1,i,1], '-', color="k")
				ax.plot(dates2, data_R2_year[N_dates1:,i,1], '-', color="k")
				R2_by_year = data_by_year.loc[(i), ("R2", "mean")]
				ax.plot([dates1[0], dates1[-1]], [R2_by_year, R2_by_year], "coral", linestyle='dashed')
				ax.plot([dates2[0], dates2[-1]], [R2_by_year, R2_by_year], "coral", linestyle='dashed')
				ax.fill_between(dates1, data_R2_year[:N_dates1,i,2], data_R2_year[:N_dates1,i,3],
					facecolor='coral', alpha=0.3)
				ax.fill_between(dates2, data_R2_year[N_dates1:,i,2], data_R2_year[N_dates1:,i,3],
					facecolor='coral', alpha=0.3)
				ax.set_ylim([0, 1])
				ax.set_ylabel(r'$R^{2}$')
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
				plt.grid(True)
				save_name = dirs + "R2_area_" + str(i) + "_" + month + ".png"
				plt.savefig(save_name, dpi=300)
				plt.close()

				fig, ax = plt.subplots(1, 1)
				fig.figsize=(6, 4)
				ax.plot(dates1, data_e2_year[:N_dates1,i,1], '-', color="k")
				ax.plot(dates2, data_e2_year[N_dates1:,i,1], '-', color="k")
				e2_by_year = data_by_year.loc[(i), ("epsilon2", "mean")]
				ax.plot([dates1[0], dates1[-1]], [e2_by_year, e2_by_year], "coral", linestyle='dashed')
				ax.plot([dates2[0], dates2[-1]], [e2_by_year, e2_by_year], "coral", linestyle='dashed')
				ax.fill_between(dates1, data_e2_year[:N_dates1,i,2], data_e2_year[:N_dates1,i,3],
					facecolor='silver', alpha=0.3)
				ax.fill_between(dates2, data_e2_year[N_dates1:,i,2], data_e2_year[N_dates1:,i,3],
					facecolor='silver', alpha=0.3)
				ax.set_ylim([0, 1.5])
				ax.set_ylabel(r'$e^{2}$')
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
				plt.grid(True)
				save_name = dirs + "e2_area_" + str(i) + "_" + month + ".png"
				plt.savefig(save_name, dpi=300)
				plt.close()

				fig, ax = plt.subplots(1, 1)
				fig.figsize=(6, 4)
				y1 = data_A_year[:N_dates1,i,0]
				y2 = data_A_year[N_dates1:,i,0]
				ax.plot(dates1, y1, '-', color="k")
				ax.plot(dates2, y2, '-', color="k")
				y_lim_min = max(y1.min()-5,0)
				y_lim_max = y1.max()+5
				ax.set_ylim([y_lim_min, y_lim_max])
				ax.set_ylabel("number of data")
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
				plt.grid(True)
				save_name = dirs + "count_area_" + str(i) + "_" + month + ".png"
				plt.savefig(save_name, dpi=300)
				plt.close()

	dirs_ts_30_by_year = "../result_nc_1day_delay/ts_30_by_year/"
	dirs_ts_by_month = "../result_nc_1day_delay/ts_by_month/"
	if num == 1:
		ts_30_by_month(dirs_ts_by_month)
	elif num == 2:
		ts_30_by_year(dirs_ts_30_by_year)

#plot_ts_nc_1day_delay(1)
#plot_ts_nc_1day_delay(2)













