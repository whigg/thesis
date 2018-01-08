"""
風力係数を求める関数群
	・30日の風力係数を求める関数(コメントアウト)
	・月ごとの風力係数を求める関数(コメントアウト)
	・30_30にic0とsitの平均などを足して拡張したもの
	・30_30だが，回帰にiw/wの閾値を設けたもの
	・90日の風力係数を求める関数
	・海流だけ90日を使って，風力係数は30日で求める関数
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
"""
・2つの関数．20180107より前の関数で，すでにmain_vで同じものを実行しているため，コメントアウトしている．．
	・どちらも30日平均を使ったhelmert回帰
	・十数年の月ごとのhelmert回帰
"""

"""
def get_helmert():
	dirs = "../data/csv_Helmert_30/"
	mkdir(dirs)

	#start_list = [20030101]
	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#wデータの取得・整形
		date_ax_w, _, _, data_w = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)
		data_array_w = np.array(data_w)
		data_ave_w = np.nanmean(data_array_w, axis=0)

		#iwデータの取得・整形
		date_ax_iw, _, _, data_iw = main_data(
			start, end, 
			span=30, 
			get_columns=["iw"], 
			region=None, 
			accumulate=True
			)

		data_array_iw = np.array(data_iw)
		print("\n")
		#print("data_array_iw:  {}".format(data_array_iw[0,1001:2000,:]))
		#print("data_array_w:  {}".format(data_array_w.shape))
		#print("data_array_iw:  {}".format(data_array_iw.shape))
		#print("data_array_iw[:,:,0]:  {}".format(data_array_iw[:,:,0].shape))
		#print("data_array_w[:,0,0]:  {}".format(data_array_w[:,0,0].shape))

		w_array = np.vstack((data_array_iw[:,:,1], data_array_iw[:,:,2]))
		Helmert = []
		#for j in range(1218,1220):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, j, 2][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_array_iw[:, j, 1])
			iw_v_ave = np.nanmean(data_array_iw[:, j, 2])
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
			if N_c <= 20:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_30_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)


def get_hermert_by_year():
	dirs = "../data/csv_Helmert_by_year/"
	mkdir(dirs)

	start_list = []
	n = 20000000
	y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
	for i in y_list:
		m = n + i*10000
		for j in range(12):
			start_list.append(m + (j+1)*100 + 1)
	start_list_plus_1month = start_list.copy()
	start_list_plus_1month.append(201701)

	start_list = np.array(start_list)
	month_list_str = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for k in range(12):
		print("************************  Month: {}/{}  ************************".format(k+1, 12))
		#12年分
		month_idx = np.arange(0, 144, 12) + k
		month_next_idx = np.arange(0, 144, 12) + k + 1
		year_list = start_list[month_idx]
		y_next_list = start_list[month_next_idx]

		data_w_year = np.zeros((1, 145**2, 2))
		data_iw_year = np.zeros((1, 145**2, 2))
		for i, start in enumerate(year_list):
			print("  *******************  Year: {}  *******************".format(str(start)[:6]))
			month_end = y_next_list[i]
			month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
			end = start + month_end.day - 1

			_, _, _, data_w = main_data(
				start, end, 
				span=30, 
				get_columns=["w"], 
				region=None, 
				accumulate=True
				)
			data_array_w = np.array(data_w)
			data_w_year = np.concatenate([data_w_year, data_array_w], axis=0)

			_, _, _, data_iw = main_data(
				start, end, 
				span=30, 
				get_columns=["iw"], 
				region=None, 
				accumulate=True
				)
			data_array_iw = np.array(data_iw)
			data_iw_year = np.concatenate([data_iw_year, data_array_iw], axis=0)

			print("\n")

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
			x = data_w_year[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_w_year[:, j, 2][not_nan_idx].reshape((-1,1))
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

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_by_year_" + month_list_str[k] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)
"""

###################################################################################################################

def get_helmert_both_30_modified():
	dirs = "../data/csv_Helmert_both_30/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		#csv_Helmert_30の当該月のcsvを読み込む
		hermert_file_name_30 = "../data/csv_Helmert_30/Helmert_30_" + str(start)[:6] + ".csv"
		helmert_30_data = pd.read_csv(hermert_file_name_30)

		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start)[:-2] + ".csv"
		data_ex = pd.read_csv(data_ex_dir)

		data = pd.concat([helmert_30_data, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label", "ic0_30", "ic0_30_median", "sit_30", "sit_30_median"]]], axis=1)
		save_name = dirs + "Helmert_both_30_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)



def get_helmert_both_30_w_iw():
	dirs = "../data/csv_Helmert_both_30_w_iw/"
	mkdir(dirs)

	start_list_plus_1month = start_list + [20170901]
	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#wデータの取得・整形
		date_ax_w, _, _, data_w = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)
		data_array_w = np.array(data_w)
		data_ave_w = np.nanmean(data_array_w, axis=0)

		#iwデータの取得・整形
		date_ax_iw, _, _, data_iw = main_data(
			start, end, 
			span=30, 
			get_columns=["iw"], 
			region=None, 
			accumulate=True
			)

		data_array_iw = np.array(data_iw)
		threshold_bool = data_array_iw[:,:,0]/data_array_w[:,:,0]<0.005
		threshold_bool = np.tile(threshold_bool.reshape(threshold_bool.shape[0], threshold_bool.shape[1], 1), (1,1,3))
		nan_tile = np.tile(np.nan, data_array_iw.shape)
		data_array_iw = np.where(threshold_bool==True, nan_tile, data_array_iw)
		print(data_array_iw.shape)
		print("\n")
		#print("data_array_iw:  {}".format(data_array_iw[0,1001:2000,:]))
		#print("data_array_w:  {}".format(data_array_w.shape))
		#print("data_array_iw:  {}".format(data_array_iw.shape))
		#print("data_array_iw[:,:,0]:  {}".format(data_array_iw[:,:,0].shape))
		#print("data_array_w[:,0,0]:  {}".format(data_array_w[:,0,0].shape))

		w_array = np.vstack((data_array_iw[:,:,1], data_array_iw[:,:,2]))
		Helmert = []
		#for j in range(1218,1220):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, j, 2][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_array_iw[:, j, 1])
			iw_v_ave = np.nanmean(data_array_iw[:, j, 2])
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
			if N_c <= 20:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A_30_w_iw", "theta_w_iw", "ocean_u_w_iw", "ocean_v_w_iw", "R2_w_iw", "epsilon2_w_iw", "N_c_w_iw", "mean_iw_u_w_iw", "mean_iw_v_w_iw", "mean_w_u_w_iw", "mean_w_v_w_iw"]
		#print(data.head(3))
		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start)[:-2] + ".csv"
		data_ = pd.read_csv(data_ex_dir).loc[:, ["coastal_region_1", "coastal_region_2", "area_label", "ic0_30", "ic0_30_median", "sit_30", "sit_30_median"]]
		data = pd.concat([latlon_ex, data_, data], axis=1)

		save_name = dirs + "Helmert_both_30_w_iw_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)



def get_helmert_both_90(data_ex_basic):
	dirs = "../data/csv_Helmert_both_90/"
	mkdir(dirs)

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

		data_w_90 = np.zeros((1, 145**2, 3))
		data_iw_90 = np.zeros((1, 145**2, 3))
		for i in range(3):
			start = start_list_3month[i]
			end = end_list_3month[i]
			if start == 0 and end == 0:
				continue

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
			if N_c < 45:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A_90", "theta_90", "ocean_u_90", "ocean_v_90", "R2_90", "epsilon2_90", "N_c_90", "mean_iw_u_90", "mean_iw_v_90", "mean_w_u_90", "mean_w_v_90"]
		data["ocean_speed_90"] = np.sqrt(data["ocean_u_90"]**2 + data["ocean_v_90"]**2)
		#print(data.head(3))
		data = pd.concat([latlon_ex, data_ex_basic, data], axis=1)
		save_name = dirs + "Helmert_both_90_" + str(start_1)[:6] + ".csv"
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
		ocean_array_u = np.array(helmert_90_data["ocean_u_90"])
		ocean_array_v = np.array(helmert_90_data["ocean_v_90"])
		ocean_array = np.hstack((ocean_array_u.reshape((-1,1)), ocean_array_v.reshape((-1,1))))
		ocean_array = np.tile(ocean_array, (data_array_iw.shape[0],1,1))
		ocean_array = np.vstack((ocean_array[:,:,0], ocean_array[:,:,1]))
		#print(ocean_array.shape)
		#print(w_array[:, 0].shape)
		Helmert = []
		#for j in range(100):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, j, 2][not_nan_idx].reshape((-1,1))
			w = (w_array[:, j]-ocean_array[:, j])[np.tile(not_nan_idx, 2)].reshape((-1,1))
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
		data.columns = ["A_30_90", "theta_30_90", "R2_30_90", "epsilon2_30_90", "N_c_30_90", "mean_iw_u_with_ocean_30_90", "mean_iw_v_with_ocean_30_90", "mean_w_u_30", "mean_w_v_30"]

		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start)[:-2] + ".csv"
		data_ = pd.read_csv(data_ex_dir).loc[:, ["coastal_region_1", "coastal_region_2", "area_label", "ic0_30", "ic0_30_median", "sit_30", "sit_30_median"]]
		data = pd.concat([latlon_ex, data_, data], axis=1)
		#print(data.head(3))
		save_name = dirs + "Helmert_30_90_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)





if __name__ == '__main__':

	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
	data = pd.read_csv(data_ex_dir)

	data_ex_basic = pd.concat([
		latlon_ex, data.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]
		], axis=1)

	#get_helmert_both_30_modified()
	#get_helmert_both_30_w_iw()
	#get_helmert_both_90(data_ex_basic)
	get_helmert_ocean_90()







