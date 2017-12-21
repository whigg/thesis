
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime, date, timezone, timedelta
import os.path

import basic_file as b_f
import calc_data
import visualize

latlon145_file_name = b_f.latlon145_file_name
latlon900_file_name = b_f.latlon900_file_name
grid900to145_file_name = b_f.grid900to145_file_name
ocean_grid_file = b_f.ocean_grid_file
ocean_grid_145 = b_f.ocean_grid_145
ocean_idx = b_f.ocean_idx

latlon_ex = pd.read_csv(latlon145_file_name)
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

#ある年の1ヶ月だけを想定
"""
"ex_1": A_by_day, theta_by_day
"w": w_u, w_v, w_speed
"iw": iw_u, iw_v, iw_speed
"ic0_145": ic0_145
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
		year = day[2:4]
		month = day[4:6]

		#ファイル名の生成
		wind_file_name = "../data/wind_data/ecm" + day[2:] + ".csv"
		ice_file_name = "../data/ice_wind_data/" + day[2:] + ".csv"
		ic0_145_file_name = "../data/IC0_csv/2" + day + "A.csv"
		ic0_900_file_name = "../data/IC0_csv/2" + day + "A.csv"
		coeff_file_name = "../data/A_csv/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"
		wind10m_file_name = "../data/netcdf4/" + day[2:] + ".csv"
		t2m_file_name = "../data/netcdf4/" + day[2:] + ".csv"

		#if ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name), os.path.isfile(ic0_145_file_name)])):
		if ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name)])):
			print ("\tSkipping " + day + " file...")
			date_ax_str.remove(day)
			aa = day[:4]+"-"+day[4:6]+"-"+day[6:]
			print(aa)
			bb = date(int(day[:4]), int(day[4:6]), int(day[6:]))
			print(bb)
			date_ax.remove(bb)
			skipping_date_str.append(day)
			continue


		data = pd.DataFrame({"data_idx": np.zeros(145*145)})
		iw_idx_t, ic0_idx_t = np.arange(0,145*145,1), np.arange(0,145*145,1)
		if "ex_1" in get_columns:
			tmp = calc_data.get_w_regression_data(wind_file_name, ice_file_name, coeff_file_name)
			iw_idx_t = np.array(tmp["data_idx"])
			data = pd.concat([data, tmp.loc[:,["A_by_day", "theta_by_day"]]], axis=1)
		if "w" in get_columns:
			tmp = calc_data.get_1day_w_data(wind_file_name)
			# _, iw_idx_t = calc_data.get_1day_ice_data(ice_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "iw" in get_columns:
			tmp, iw_idx_t = calc_data.get_1day_ice_data(ice_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "ic0_145" in get_columns:
			tmp, ic0_idx_t = calc_data.get_1day_ic0_data(ic0_145_file_name, grid900to145_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "coeff" in get_columns:
			tmp = calc_data.get_1month_coeff_data(coeff_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "w10m" in get_columns:
			tmp = calc_data.get_1day_w10m_data(wind10m_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "t2m" in get_columns:
			tmp = calc_data.get_1day_t2m_data(t2m_file_name)
			data = pd.concat([data, tmp], axis=1)

		mat = iw_idx_t.ravel().tolist() + ic0_idx_t.ravel().tolist()
		data_t_idx = calc_data.get_non_nan_idx(mat, ocean_idx, strict=True)
		tmp = np.zeros(145*145)
		tmp[data_t_idx] = 1
		data.data_idx = tmp
		#data.loc[data.data_idx, data_t_idx] = 1
		data = calc_data.get_masked_region_data(data, region)

		if ("coeff" in get_columns) and (len(get_columns) == 1):
			print("\tSelected only coeff data. Getting out of the loop...")
			continue

		if accumulate == True:
			data_1 = data.drop("data_idx", axis=1)
			"""
			if "w" in get_columns:
				data_1.loc[data.data_idx==0] = np.nan
			"""
			print("\t{}".format(data_1.columns))
			#print(data_1.w_u.iloc[6859:6861])
			accumulate_data.append(np.array(data_1))

	if accumulate == True:
		print("accumulate: True\tdata type: array")
		return date_ax, date_ax_str, skipping_date_str, accumulate_data
	else:
		print("accumulate: False\tdata type: DataFrame")
		return date_ax, date_ax_str, skipping_date_str, data


###################################################################################################################
"""
#angleのマップを月ごとに出力して保存するコード
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

start_list = [20130101, 20130201, 20130301]
for i, start in enumerate(start_list):
	print("******************  {}/{}  *******************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)
	#print(data.head())
	#data = data["A"]
	#print(len(data))

	# latlon_exで絞り込む場合，ここに処理を書く
	#data = pd.concat([latlon_ex, data], axis=1)
	#print(data.head())
	
	data.angle.loc[(data.A<0) & (data.angle<0)] += 180
	data.angle.loc[(data.A<0) & (data.angle>0)] -= 180

	data.angle.loc[data.coef**2<0.4**2] = np.nan
	
	#data[data.data_idx==0.] = np.nan
	save_name = "../result/angle_30_ignore_low_coeff/angle_30_ignore_low_coeff_" + str(start)[:6] + ".png"

	#visualize.pyで関数を選ぶ
	visualize.plot_map_once(
		data["angle"],
		data_type="type_non_wind",
		save_name=save_name,
		show=False, 
		vmax=180, 
		vmin=-180
		)
	print("\n")
"""


"""
#A_by_yearのマップ出力
A_by_year_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
for item in A_by_year_list:
	fname = "../data/A_by_year/ssc_amsr_ads" + item + "_fin.csv"
	df_coeffs = pd.read_csv(fname, sep=',', dtype='float32')
	df_coeffs.columns = ["index", "angle", "mean_ocean_u", "mean_ocean_v", "A", "coef", "data_num", "mean_ice_u", "mean_ice_v", "mean_w_u", "mean_w_v"]
	df_coeffs = df_coeffs.drop("index", axis=1)
	df_coeffs[df_coeffs==999.] = np.nan
	
	save_name = "../result/A_by_year/A_" + item + ".png"
	visualize.plot_map_once(
		df_coeffs["A"],
		data_type="type_non_wind",
		save_name=save_name,
		show=False, 
		vmax=0.025, 
		vmin=0
		)
	print("\n")
"""


"""
#海流の平均のマップ出力
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

#start_list = [20130101]
for i, start in enumerate(start_list):
	print("******************  {}/{}  *******************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)

	save_name = "../result/mean_vector/mean_ocean_currents/mean_ocean_currents_" + str(start)[:6] + ".png"

	#visualize.pyで関数を選ぶ
	visualize.plot_map_once(
		data.loc[:, ["mean_ocean_u", "mean_ocean_v"]],
		data_type="type_wind",
		save_name=save_name,
		show=False, 
		vmax=None, 
		vmin=None
		)
	print("\n")

"""





#地衡風のマップを月ごとに出力して保存するコード
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

#start_list = [20030101]
for i, start in enumerate(start_list):
	print("*******************  {}/{}  *******************".format(i+1, M))
	month_end = start_list_plus_1month[i+1]
	month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
	end = start + month_end.day - 1
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, end, 
		span=30, 
		get_columns=["w"], 
		region=None, 
		accumulate=True
		)

	data_array = np.array(data)
	data_array_1 = np.ma.masked_invalid(data_array)
	data_count_nan = np.sum(data_array_1.recordmask, axis=0)
	date_ax_len = len(date_ax)
	data_ave_sum = np.sum(data_array, axis=0)
	data_ave_sum[date_ax_len-data_count_nan<=20] = np.nan
	data_ave = data_ave_sum / (date_ax_len-data_count_nan)
	#A_by_dayなので0列目
	data_ave = pd.DataFrame(data_ave)
	data_ave.columns = ["w_speed", "w_u", "w_v"]
	print(data_ave.head(3))

	save_name = "../result/mean_vector/mean_wind/mean_wind_" + str(start)[:6] + ".png"
	visualize.plot_map_once(
		data_ave.loc[:, ["w_u", "w_v"]],
		data_type="type_wind", 
		save_name=save_name,
		show=False, 
		vmax=None, 
		vmin=None
		)
	print("\n")




"""
#海流の平均のマップ出力
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

#start_list = [20130101]
for i, start in enumerate(start_list):
	print("******************  {}/{}  *******************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)

	save_name = "../result/mean_vector/mean_wind/mean_wind_" + str(start)[:6] + ".png"

	#visualize.pyで関数を選ぶ
	visualize.plot_map_once(
		data.loc[:, ["mean_w_u", "mean_w_v"]],
		data_type="type_wind",
		save_name=save_name,
		show=False, 
		vmax=None, 
		vmin=None
		)
	print("\n")
"""











