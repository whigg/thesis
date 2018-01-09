"""
・90日の風力係数を求める関数
・海流だけ90日を使って，風力係数は30日で求める関数
・60日の風力係数を求める関数
・海流だけ60日を使って，風力係数は30日で求める関数
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

import calc_data
import visualize
from main_v import mkdir, main_data

latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()
basic_region = ["bearing_sea", "chukchi_sea", "beaufort_sea", "canada_islands", "hudson_bay", "buffin_bay", "labrador_sea", "greenland_sea", 
	"norwegian_sea", "barents_sea", "kara_sea", "laptev_sea", "east_siberian_sea", "north_polar"]

#基本的な変数の定義
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



###################################################################################################################

def get_helmert_both_90():
	dirs = "../data/csv_Helmert_both_90/"
	mkdir(dirs)

	#start_list = [20030101]
	start_list_plus_3month = start_list + [20170901, 20171001, 20171101]
	for k, start_1 in enumerate(start_list):
		if start_1 == 20170701:
			print("Breaking the loop...")
			break
		print("*******************  {}/{}  *******************".format(k+1, M))
		month_end_1 = start_list_plus_3month[k+1]
		month_end_1 = date(month_end_1//10000, (month_end_1%10000)//100, (month_end_1%10000)%100) - timedelta(days=1)
		end_1 = start_1 + month_end_1.day - 1

		start_2 = month_end_1
		month_end_2 = start_list_plus_1month[k+2]
		month_end_2 = date(month_end_2//10000, (month_end_2%10000)//100, (month_end_2%10000)%100) - timedelta(days=1)
		end_2 = start_2 + month_end_2.day - 1

		start_3 = month_end_2
		month_end_3 = start_list_plus_1month[k+3]
		month_end_3 = date(month_end_3//10000, (month_end_3%10000)//100, (month_end_3%10000)%100) - timedelta(days=1)
		end_3 = start_3 + month_end_3.day - 1

		start_list_3month = [start_1, start_2, start_3]
		end_list_3month = [end_1, end_2, end_3]

		data_w_90 = np.zeros((1, 145**2, 3))
		data_iw_90 = np.zeros((1, 145**2, 3))
		for i in range(3):
			start = start_list_3month[i]
			end = end_list_3month[i]

			#wデータの取得・整形
			_, _, _, data_w = main_data(
				start, end, 
				span=30, 
				get_columns=["w"], 
				region=None, 
				accumulate=True
				)
			data_array_w = np.array(data_w)
			#data_ave_w = np.nanmean(data_array_w, axis=0)
			data_w_90 = np.concatenate([data_w_90, data_array_w], axis=0)

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
			x = data_w_90[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_w_90[:, j, 2][not_nan_idx].reshape((-1,1))
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
			if N_c <= 40:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_both_90_" + str(start_1)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)





def get_helmert_both_60():
	dirs = "../data/csv_Helmert_both_60/"
	mkdir(dirs)

	start_list_plus_2month = start_list + [20170901, 20171001]
	for k, start_1 in enumerate(start_list):
		if start_1 == 20170801:
			print("Breaking the loop...")
			break
		print("*******************  {}/{}  *******************".format(k+1, M))
		month_end_1 = start_list_plus_2month[k+1]
		month_end_1 = date(month_end_1//10000, (month_end_1%10000)//100, (month_end_1%10000)%100) - timedelta(days=1)
		end_1 = start_1 + month_end_1.day - 1

		start_2 = month_end_1
		month_end_2 = start_list_plus_1month[k+2]
		month_end_2 = date(month_end_2//10000, (month_end_2%10000)//100, (month_end_2%10000)%100) - timedelta(days=1)
		end_2 = start_2 + month_end_2.day - 1


		start_list_2month = [start_1, start_2]
		end_list_2month = [end_1, end_2]

		data_w_60 = np.zeros((1, 145**2, 3))
		data_iw_60 = np.zeros((1, 145**2, 3))
		for i in range(2):
			start = start_list_2month[i]
			end = end_list_2month[i]

			#wデータの取得・整形
			_, _, _, data_w = main_data(
				start, end, 
				span=30, 
				get_columns=["w"], 
				region=None, 
				accumulate=True
				)
			data_array_w = np.array(data_w)
			#data_ave_w = np.nanmean(data_array_w, axis=0)
			data_w_60 = np.concatenate([data_w_60, data_array_w], axis=0)

			#iwデータの取得・整形
			_, _, _, data_iw = main_data(
				start, end, 
				span=30, 
				get_columns=["iw"], 
				region=None, 
				accumulate=True
				)
			data_array_iw = np.array(data_iw)
			data_iw_60 = np.concatenate([data_iw_60, data_array_iw], axis=0)
			
			print("\n")

		data_w_60 = data_w_60[1:, :, :]
		data_iw_60 = data_iw_60[1:, :, :]
		data_ave_w = np.nanmean(data_w_60, axis=0)
		data_ave_iw = np.nanmean(data_iw_60, axis=0)
		w_array = np.vstack((data_iw_60[:,:,1], data_iw_60[:,:,2]))

		Helmert = []
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_iw_60[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_w_60[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_w_60[:, j, 2][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_iw_60[:, j, 1])
			iw_v_ave = np.nanmean(data_iw_60[:, j, 2])
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
			if N_c <= 31:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_both_60_" + str(start_1)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)






def get_helmert_ocean_90():
	dirs = "../data/csv_Helmert_ocean_90/"
	mkdir(dirs)

	start_list_plus_1month = start_list + [20170901]
	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#csv_Helmert_90の当該月のcsvを読み込む
		hermert_file_name = "../data/csv_Helmert_both_90/Helmert_both_90_" + str(start)[:6] + ".csv"
		helmert_90_data = pd.read_csv(hermert_file_name)

		#wデータの取得・整形
		_, _, _, data_w = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)
		data_array_w = np.array(data_w)
		data_ave_w = np.nanmean(data_array_w, axis=0)

		#iwデータの取得・整形
		_, _, _, data_iw = main_data(
			start, end, 
			span=30, 
			get_columns=["iw"], 
			region=None, 
			accumulate=True
			)

		data_array_iw = np.array(data_iw)
		print("\n")

		w_array = np.vstack((data_array_iw[:,:,1], data_array_iw[:,:,2]))
		ocean_array_u = np.array(helmert_90_data["ocean_u"])
		ocean_array_v = np.array(helmert_90_data["ocean_v"])
		ocean_array = np.vstack((ocean_array_u, ocean_array_v))
		print(ocean_array.shape)
		print(w_array[:, 0].shape)
		Helmert = []
		for j in range(2500):
		#for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, j, 2][not_nan_idx].reshape((-1,1))
			w = (w_array[:, j]-ocean_array)[np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_array_iw[:, j, 1])
			iw_v_ave = np.nanmean(data_array_iw[:, j, 2])
			iw_u_ave_ocean = np.nanmean(data_array_iw[:, j, 1] - ocean_array_u[j])
			iw_v_ave_ocean = np.nanmean(data_array_iw[:, j, 2] - ocean_array_v[j])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
				continue
			one_N = np.ones(N_c).reshape((-1,1))
			zero_N = np.zeros(N_c).reshape((-1,1))
			#D_1 = np.hstack((one_N, zero_N, x, -y))
			#D_2 = np.hstack((zero_N, one_N, y, x))
			D_1 = np.hstack((x, -y))
			D_2 = np.hstack((y, x))
			Dt = np.vstack((D_1, D_2))
			#print(N_c, Dt, np.dot(Dt.T, Dt))
			D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
			gamma = np.dot(D_inv, np.dot(Dt.T, w))
			print("\t{}".format(gamma.shape))
			A = np.sqrt(gamma[0]**2 + gamma[1]**2)
			theta = np.arctan2(gamma[1], gamma[0]) * 180/np.pi
			R_denominator = np.dot(w.T, w) - N_c*(iw_u_ave_ocean**2 + iw_v_ave_ocean**2)
			epsilon = w - np.dot(Dt, gamma)
			R_numerator = np.dot(epsilon.T, epsilon)
			R2 = 1 - (R_numerator/R_denominator)[0,0]
			#print("\t{}".format([A[0], theta[0], gamma[0,0], gamma[1,0], R2, N_c]))
			if N_c <= 20:
				Helmert.append([np.nan, np.nan, np.nan, np.nan, N_c, iw_u_ave_ocean, iw_v_ave_ocean])
			else:
				Helmert.append([A, theta, R2, R_numerator, N_c, iw_u_ave_ocean, iw_v_ave_ocean])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "R2", "epsilon2", "N_c", "mean_iw_u_with_ocean", "mean_iw_v_with_ocean", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_30_90_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)



def get_helmert_ocean_60():
	dirs = "../data/csv_Helmert_ocean_60/"
	mkdir(dirs)

	start_list_plus_1month = start_list + [20170901]
	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#csv_Helmert_60の当該月のcsvを読み込む
		hermert_file_name = "../data/csv_Helmert_both_60/Helmert_both_60_" + str(start)[:6] + ".csv"
		helmert_60_data = pd.read_csv(hermert_file_name)

		#wデータの取得・整形
		_, _, _, data_w = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)
		data_array_w = np.array(data_w)
		data_ave_w = np.nanmean(data_array_w, axis=0)

		#iwデータの取得・整形
		_, _, _, data_iw = main_data(
			start, end, 
			span=30, 
			get_columns=["iw"], 
			region=None, 
			accumulate=True
			)

		data_array_iw = np.array(data_iw)
		print("\n")

		w_array = np.vstack((data_array_iw[:,:,1], data_array_iw[:,:,2]))
		ocean_array_u = np.array(helmert_60_data["ocean_u"])
		ocean_array_v = np.array(helmert_60_data["ocean_v"])
		ocean_array = np.vstack((ocean_array_u, ocean_array_v))
		print(ocean_array.shape)
		print(w_array[:, 0].shape)
		Helmert = []
		#for j in range(2500):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, j, 2][not_nan_idx].reshape((-1,1))
			w = (w_array[:, j]-ocean_array)[np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_array_iw[:, j, 1])
			iw_v_ave = np.nanmean(data_array_iw[:, j, 2])
			iw_u_ave_ocean = np.nanmean(data_array_iw[:, j, 1] - ocean_array_u[j])
			iw_v_ave_ocean = np.nanmean(data_array_iw[:, j, 2] - ocean_array_v[j])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
				continue
			one_N = np.ones(N_c).reshape((-1,1))
			zero_N = np.zeros(N_c).reshape((-1,1))
			#D_1 = np.hstack((one_N, zero_N, x, -y))
			#D_2 = np.hstack((zero_N, one_N, y, x))
			D_1 = np.hstack((x, -y))
			D_2 = np.hstack((y, x))
			Dt = np.vstack((D_1, D_2))
			#print(N_c, Dt, np.dot(Dt.T, Dt))
			D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
			gamma = np.dot(D_inv, np.dot(Dt.T, w))
			print("\t{}".format(gamma.shape))
			A = np.sqrt(gamma[0]**2 + gamma[1]**2)
			theta = np.arctan2(gamma[1], gamma[0]) * 180/np.pi
			R_denominator = np.dot(w.T, w) - N_c*(iw_u_ave_ocean**2 + iw_v_ave_ocean**2)
			epsilon = w - np.dot(Dt, gamma)
			R_numerator = np.dot(epsilon.T, epsilon)
			R2 = 1 - (R_numerator/R_denominator)[0,0]
			#print("\t{}".format([A[0], theta[0], gamma[0,0], gamma[1,0], R2, N_c]))
			if N_c <= 20:
				Helmert.append([np.nan, np.nan, np.nan, np.nan, N_c, iw_u_ave_ocean, iw_v_ave_ocean])
			else:
				Helmert.append([A, theta, R2, R_numerator, N_c, iw_u_ave_ocean, iw_v_ave_ocean])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "R2", "epsilon2", "N_c", "mean_iw_u_with_ocean", "mean_iw_v_with_ocean", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_30_60_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)




#木村さんの平均海流とあっているかのプロット．散布図．
def test_ocean_mean_iw_plot():
	dirs = "../result_h/test/test_ocean/"
	mkdir(dirs)

	start_list_plus_1month = start_list + [20170901]
	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1
		
		#csv_Helmert_90の当該月のcsvを読み込む
		hermert_file_name_90 = "../data/csv_Helmert_both_90/Helmert_both_90_" + str(start)[:6] + ".csv"
		helmert_90_data = pd.read_csv(hermert_file_name_90).dropna()
		#csv_Helmert_30の当該月のcsvを読み込む
		hermert_file_name_30 = "../data/csv_Helmert_30/Helmert_30_" + str(start)[:6] + ".csv"
		helmert_30_data = pd.read_csv(hermert_file_name_30).dropna()
		#木村のcsvの読み込み(90)
		coeff_file_name_90 = "../data/csv_A_90/ssc_amsr_ads" + str(start)[2:6] + "_90_fin.csv"
		coeff_90_data = calc_data.get_1month_coeff_data(coeff_file_name_90).dropna()
		#木村のcsvの読み込み(30)
		coeff_file_name_30 = "../data/csv_A_30/ssc_amsr_ads" + str(start)[2:6] + "_30_fin.csv"
		coeff_30_data = calc_data.get_1month_coeff_data(coeff_file_name_30).dropna()

		ocean_90_h_u = helmert_90_data["ocean_u"].values
		ocean_90_h_v = helmert_90_data["ocean_v"].values
		ocean_90_h_speed = np.sqrt(ocean_90_h_u**2 + ocean_90_h_v**2)
		ocean_30_h_u = helmert_30_data["ocean_u"].values
		ocean_30_h_v = helmert_30_data["ocean_v"].values
		ocean_30_h_speed = np.sqrt(ocean_30_h_u**2 + ocean_30_h_v**2)
		ocean_90_c_u = coeff_90_data["mean_ocean_u"].values
		ocean_90_c_v = coeff_90_data["mean_ocean_v"].values
		ocean_90_c_speed = np.sqrt(ocean_90_c_u**2 + ocean_90_c_v**2)
		ocean_30_c_u = coeff_30_data["mean_ocean_u"].values
		ocean_30_c_v = coeff_30_data["mean_ocean_v"].values
		ocean_30_c_speed = np.sqrt(ocean_30_c_u**2 + ocean_30_c_v**2)

		fig, axes = plt.subplots(1, 2)
		axes[0].hist(ocean_90_c_speed/ocean_90_h_speed)
		axes[1].hist(ocean_30_c_speed/ocean_30_h_speed)
		plt.savefig(dirs + "hist_ocean_" + str(start)[:6] + ".png", dpi=450)
		plt.close()









if __name__ == '__main__':
	get_helmert_both_90()
	get_helmert_both_60()
	get_helmert_ocean_90()
	get_helmert_ocean_60()






