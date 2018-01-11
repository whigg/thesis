
import numpy as np
import pandas as pd
from datetime import datetime, date, timezone, timedelta
import netCDF4
import glob
import os.path
import os

latlon145_file_name = "../data/" + "latlon_ex.csv"
latlon900_file_name = "../data/" + "latlon_info.csv"
grid900to145_file_name = "../data/" + "grid900to145.csv"
ocean_grid_file = "../data/ocean_grid_145.csv"
ocean_grid_145 = pd.read_csv(ocean_grid_file, header=None)
ocean_idx = np.array(ocean_grid_145[ocean_grid_145==1].dropna().index)


def get_helmert_test():
	dirs = "../data/csv_Helmert_30_test/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	for year in y_list:
		for month in month_list:
			gw_file_list = sorted(glob.glob("../data/csv_w/ecm" + year + month + "*.csv"))
			iw_file_list = sorted(glob.glob("../data/csv_iw/" + year + month + "*.csv"))
			ic0_file_list = sorted(glob.glob("../data/csv_ic0/IC0_20" + year + month + "*.csv"))
			sit_file_list = sorted(glob.glob("../data/csv_sit/SIT_20" + year + month + "*.csv"))
			gw_list = []
			iw_list = []
			ic0_list = []
			sit_list = []

			L_gw = len(gw_file_list)
			L_iw = len(iw_file_list)
			if L_gw != L_iw:
				print("L_gw != L_iw")
				continue
			grid_data = pd.read_csv(grid900to145_file_name, header=None)
			grid145 = np.array(grid_data, dtype='int64').ravel()
			for gw_fname in gw_file_list:
				df_wind = pd.read_csv(gw_fname, header=None)
				wind = np.array(df_wind, dtype='float32')
				gw_list.append(wind[:, [0,1]])
			for iw_fname in iw_file_list:
				df_ice_wind = pd.read_csv(iw_fname, header=None)
				df_ice_wind[df_ice_wind==999.] = np.nan
				ice_wind = np.array(df_ice_wind, dtype='float32')/100
				iw_list.append(ice_wind[:, [0,1]])
			"""
			for ic0_fname in ic0_file_list:
				ic0_data = pd.read_csv(ic0_fname, header=None)
				ic0 = np.array(ic0_data, dtype='float32')
				ic0_145 = ic0[grid145]
				ic0_list.append(ic0_145)
			for sit_fname in sit_file_list:
				sit_data = pd.read_csv(sit_file_name, header=None)
				sit = np.array(sit_data, dtype='float32')
				sit[sit>=10001] = np.nan
				sit_145 = sit[grid145]
				sit_list.append(sit_145)
			"""

			gw_array = np.array(gw_list)
			iw_array = np.array(iw_list)
			gw_array = np.where(np.isnan(iw_array), np.nan, gw_array)
			#ic0_array = np.array(ic0_list)
			#sit_array = np.array(sit_list)

			gw_ave = np.nanmean(gw_array, axis=0)
			iw_ave = np.nanmean(iw_array, axis=0)
			#ic0_ave = np.nanmean(ic0_array, axis=0)
			#sit_ave = np.nanmean(sit_array, axis=0)

			gw_minus_ave = gw_array - np.tile(gw_ave, (L_gw,1,1))
			iw_minus_ave = iw_array - np.tile(iw_ave, (L_iw,1,1))

			print(gw_array.shape, iw_array.shape)
			print(gw_ave.shape, iw_ave.shape)
			print(np.tile(gw_ave, (L_gw,1,1)).shape, np.tile(iw_ave, (L_iw,1,1)).shape)
			print(gw_minus_ave.shape, iw_minus_ave.shape)

			param_list = []
			#for i in range(145**2):
			for j in range(1933,1940):
				print(j)
				N_c = np.sum(~np.isnan(iw_minus_ave[:,j,0]))
				if N_c <= 20:
					param_list.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c])
					continue
				b_1 = (np.nansum(gw_minus_ave[:,j,0]*iw_minus_ave[:,j,0]) + \
					np.nansum(gw_minus_ave[:,j,1]*iw_minus_ave[:,j,1])) / \
						(np.nansum(gw_minus_ave[:,j,0]**2) + np.nansum(gw_minus_ave[:,j,1]**2))
				b_2 = (np.nansum(gw_minus_ave[:,j,0]*iw_minus_ave[:,j,1]) - \
					np.nansum(gw_minus_ave[:,j,1]*iw_minus_ave[:,j,0])) / \
						(np.nansum(gw_minus_ave[:,j,0]**2) + np.nansum(gw_minus_ave[:,j,1]**2))
				a_1 = iw_ave[j,0] - b_1*gw_ave[j,0] + b_2*gw_ave[j,1]
				a_2 = iw_ave[j,1] - b_1*gw_ave[j,1] - b_2*gw_ave[j,0]
				R_denominator = np.nansum(iw_minus_ave[:,j,0]**2 + iw_minus_ave[:,j,1]**2)
				R_numerator = np.nansum((iw_array[:,j,0] - (a_1 + b_1*gw_array[:,j,0] - b_2*gw_array[:,j,1]))**2) + \
					np.nansum((iw_array[:,j,1] - (a_2 + b_2*gw_array[:,j,0] + b_1*gw_array[:,j,1]))**2)
				R2 = 1 - R_numerator/R_denominator
				A = np.sqrt(b_1**2 + b_2**2)
				theta = np.arctan2(b_2, b_1) * 180/np.pi
				print(a_1, a_2, b_1, b_2)
				print("A: {}\ntheta: {}\na_1: {}\na_2: {}\nR2: {}\ne2: {}\nN_c: {}".format(A, theta, a_1, a_2, R2, R_numerator, N_c))
				param_list.append([A, theta, a_1, a_2, R2, R_numerator, N_c])
			param_array = np.array(param_list)

			data_array = np.hstack((param_array, iw_ave, gw_ave))
			data = pd.DataFrame(data_array)
			data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "e2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
			save_name = dirs + "Helmert_30_test_" + year + month + ".csv"
			print(save_name)
			data.to_csv(save_name, index=False)


if __name__ == '__main__':
	get_helmert_test()


