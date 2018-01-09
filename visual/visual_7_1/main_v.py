
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

latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()
basic_region = ["bearing_sea", "chukchi_sea", "beaufort_sea", "canada_islands", "hudson_bay", "buffin_bay", "labrador_sea", "greenland_sea", 
	"norwegian_sea", "barents_sea", "kara_sea", "laptev_sea", "east_siberian_sea", "north_polar"]

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


def mkdir(dirs):
	try:
		os.makedirs(dirs)
	except:
		print('\ndirectory {} already exists.\n'.format(dirs))


#ある年の1ヶ月だけを想定
"""
"ex_1": A_by_day, theta_by_day
"w": w_u, w_v, w_speed
"iw": iw_u, iw_v, iw_speed
"ic0_145": ic0_145
"sit_145": sit
"coeff": angle, mean_ocean_u, mean_ocean_v, A, coef, data_num, mean_ice_u, mean_ice_v, mean_w_u, mean_w_v
"w10m": 
"t2m": 
"""
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
		print ("{}/{}: {}".format(i+1, N, day))
		print ("start: {}, end: {}".format(start, end))
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
			print("\t{}".format(ice_file_name))
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
		"""
		if "w10m" in get_columns:
			tmp = calc_data.get_1day_w10m_data(wind10m_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "t2m" in get_columns:
			tmp = calc_data.get_1day_t2m_data(t2m_file_name)
			data = pd.concat([data, tmp], axis=1)
		"""

		data = calc_data.get_masked_region_data(data, region)

		if ("coeff" in get_columns):
			print("\tSelected only coeff data. Getting out of the loop...")
			continue

		if accumulate == True:
			data_1 = data.drop("data_idx", axis=1)
			print("\t{}".format(data_1.columns))
			accumulate_data.append(np.array(data_1))

	if accumulate == True:
		print("accumulate: True\tdata type: array")
		return date_ax, date_ax_str, skipping_date_str, accumulate_data
	else:
		print("accumulate: False\tdata type: DataFrame")
		return date_ax, date_ax_str, skipping_date_str, data


###################################################################################################################

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
start_list_plus_1month = start_list + [20170901]


cm_angle = visualize.generate_cmap([
	[0, 0, 96/255], 
	[0/255, 0/255, 255/255], 
	[108/255, 108/255, 108/255], 
	[255/255, 0/255, 0/255], 
	[96/255, 0, 0]
	])

cm_angle_1 = visualize.generate_cmap([
	[0, 0, 204/255], 
	[0/255, 204/255, 0/255], 
	[160/255, 160/255, 160/255], 
	[255/255, 255/255, 51/255], 
	[204/255, 0, 0]
	])

cm_angle_2 = visualize.generate_cmap([
	"blue", 
	"Lime", 
	"grey", 
	"yellow", 
	"red"
	])

threshold_R2 = 0.4**2


###############################################################################################################

#SITデータの可視化実験
#メルトポンドの挙動を兼ねて
def test_SIT():
	filename = "../data/csv_sit/SIT_20020813.csv"
	
	data = calc_data.get_1day_sit_data(filename)
	data[data>=10001] = np.nan
	visualize.plot_map_once(
		data,
		data_type="type_non_wind",
		save_name=None,
		show=True,
		vmax=None,
		vmin=None,
		cmap=plt.cm.jet)
	"""
	visualize.plot_900(
		filename,
		save_name=None,
		show=True)
	"""



#偶数年(2012を除く)の偶数月(12月除く)の地衡風と流氷速度のspeedの散布図を描く
def test_w_iw():
	start_list = []
	n = 20000000
	y_list = [4,6,8,10,14,16]
	for i in y_list:
		m = n + i*10000
		for j in [2,4,6,8,10]:
			start_list.append(m + j*100 + 1)

	dirs = "../result_h/test/w_iw/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start + 100
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1
		_, _, _, data = main_data(
			start, end, 
			span=30, 
			get_columns=["w", "iw"], 
			region=None, 
			accumulate=True
			)
		np_region_idx = np.array(latlon_ex.loc[latlon_ex.Name=="north_polar", :].index)
		print(np_region_idx)
		data = np.array(data)
		#data_w = data[:, :, 0].flatten().reshape(-1,1)
		#data_iw = data[:, :, 3].flatten().reshape(-1,1)
		#data = np.hstack((data_w, data_iw))
		#print(data.shape)
		#print(data[10, :, [0,3]].shape)
		data = data[10, :, [0,3]].T
		data = data[np_region_idx, :]
		data = pd.DataFrame(data)
		#print(data.shape)
		data.columns = ["w", "iw"]

		save_name = dirs + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data.dropna(),
			mode=["scatter", ["w", "iw"]],
			save_name=save_name,
			show=False
			)



###############################################################################################################

#R2_30, epsilon2_30
def H_R2_e2_30():
	dirs_R2 = "../result_h/R2/R2_30/"
	mkdir(dirs_R2)
	dirs_e2 = "../result_h/epsilon2/epsilon2_30/"
	mkdir(dirs_e2)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name_R2 = dirs_R2 + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["R2"],
			data_type="type_non_wind",
			save_name=save_name_R2,
			show=False, 
			vmax=1, 
			vmin=0,
			cmap=plt.cm.jet
			)

		save_name_e2 = dirs_e2 + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["epsilon2"],
			data_type="type_non_wind",
			save_name=save_name_e2,
			show=False, 
			vmax=None, 
			vmin=None,
			cmap=plt.cm.jet
			)

		print("\n")




###############################################################################################################
#!
"""
rank_npとrank_R2の処理
#save_nameの工夫
"""

#1ヶ月前の風力係数と当該月のIC0，SITの相関
#マップもありだが，とりあえず散布図
def H_corr_1month():
	dirs_ic0 = "../result_h/corr/H_corr_1month_ic0/"
	mkdir(dirs_ic0)
	dirs_sit = "../result_h/corr/H_corr_1month_sit/"
	mkdir(dirs_sit)

	start_list = []
	n = 20000000
	y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
	for i in y_list:
		m = n + i*10000
		for j in range(12):
			start_list.append(m + (j+1)*100 + 1)
	M = len(start_list)
	start_list_plus_1month = start_list + [20170101]

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		start_1month_before = date(start//10000, (start%10000)//100, (start%10000)%100) - timedelta(days=1)
		start_1month_before = int(calc_data.cvt_date(start_1month_before))
		print("\tA month: {}\n\tIC0 & SIT month: {}, {}".format(start_1month_before, start, end))

		_, _, _, data_A_original = main_data(
			start_1month_before, start_1month_before, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		_, _, _, data_ic0_sit = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145", "sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_sit)
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["ic0_30", "sit_30"]

		data_A = data_A_original["A"]
		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)

		save_name = dirs_ic0 + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data,
			mode=["scatter", ["ic0_30", "A"]],
			save_name=save_name,
			show=False
			)

		save_name = dirs_sit + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)

		print("\n")





###############################################################################################################

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
		for j in range(1218,1220):
		#for j in range(145**2):
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
			print(gamma)
			print(gamma[3])
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


get_helmert()




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
	start_list_plus_1month = start_list_plus_1month + [20170101]

	start_list = np.array(start_list)
	start_list_plus_1month = np.array(start_list_plus_1month)
	month_list_str = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for k in range(12):
		print("************************  Month: {}/{}  ************************".format(k+1, 12))
		#12年分
		month_idx = np.arange(0, 144, 12) + k
		month_next_idx = np.arange(0, 144, 12) + k + 1
		year_list = start_list[month_idx]
		y_next_list = start_list_plus_1month[month_next_idx]

		data_w_year = np.zeros((1, 145**2, 3))
		data_iw_year = np.zeros((1, 145**2, 3))
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
			#print(data_w_year.shape)
			#print(data_array_w.shape)
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



###############################################################################################################

#if __name__ == '__main__':

	#import submain_h_map
	#print(type(submain_h_map))
	#submain_h_map.H_A_30_test()
	#submain_h_map.H_A_30()
	#submain_h_map.H_A_30_with_coef()
	#submain_h_map.H_A_and_theta_by_year()
	#submain_h_map.H_A_by_day_30()
	#submain_h_map.H_angle_30()
	#submain_h_map.H_angle_30_and_wind()
	#submain_h_map.H_vec_mean_ocean_currents()

	#import submain_h_scatter
	#submain_h_scatter.H_scatter_ic0_np(mode="mean")
	#submain_h_scatter.H_scatter_ic0_np(mode="median")
	#submain_h_scatter.H_scatter_sit_np(worker=0)
	#submain_h_scatter.H_scatter_sit_np(worker=1)
	#submain_h_scatter.H_scatter_6win_2area(worker=0)

	#import submain_h_ts
	#data_1g_dic, data_2g_dic, data_3g_dic = submain_h_ts.H_ts_A_month(m_plot=True, y_plot=True)
	#_, _, _ = submain_h_ts.H_ts_A_month(m_plot=True, y_plot=True)









