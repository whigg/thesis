
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
	d1 = datetime(start_date[0], start_date[1], start_date[2])
	d2 = datetime(end_date[0], end_date[1], end_date[2])
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

		if ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(ic0_145_file_name), os.path.isfile(coeff_file_name)])):
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
			data = pd.concat([data, tmp], axis=1)
		if "iw" in get_columns:
			tmp = calc_data.get_1day_ice_data(ice_file_name)
			iw_idx_t = np.array(tmp["iw_idx_t"])
			data = pd.concat([data, tmp.loc[:,["iw_u", "iw_v", "iw_speed"]]], axis=1)
		if "ic0_145" in get_columns:
			tmp = calc_data.get_1day_ic0_data(ic0_file_name, grid900to145_file_name)
			ic0_idx_t = np.array(tmp["ic0_idx_t"])
			data = pd.concat([data, tmp.loc[:,["ic0_145"]]], axis=1)
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
			data = data.loc[:, get_columns]
			print("\t{}".format(data.columns))
			accumulate_data.append(np.array(data))

	if accumulate == True:
		print("accumulate: True\tdata type: array")
		return date_ax, date_ax_str, skipping_date_str, accumulate_data
	else:
		print("accumulate: False\tdata type: DataFrame")
		return date_ax, date_ax_str, skipping_date_str, data


#描画は1回のみを想定
def main_plot(data, **kwargs):
	what_type = kwargs["what_type"]
	mode = kwargs["mode"]
	show = kwargs["show"]
	save = kwargs["save"]
	save_name = kwargs["save_name"]

	if what_type == "non_line":
		visualize.visual_non_line(data, mode=mode, save_name=save_name, show=show)
	elif what_type == "map":
		mode0 = mode[0]
		submode = mode[1]
		if mode0 == 0:
			visualize.visual_map(data, mode0, submode, save_name=save_name, show=show)
		elif mode0 == 1:
			visualize.visual_map(data, mode0, submode, save_name=save_name, show=show)
		elif mode0 == 2:
			data = [ic0_900_file_name, latlon900_file_name]
			visualize.visual_map(data, mode0, submode, save_name=save_name, show=show)


####################################################################################################################

#Aのマップを月ごとに出力して保存するコード
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
plot_kw = "A"
for i, start in enumerate(start_list):
	print("**************  {}/{}  ***************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)
	#print(data.head())
	#data = data[plot_kw]
	#print(len(data))
	data = pd.concat([latlon_ex, data], axis=1)
	data.loc[data.A<0] *= -1
	save_name = "A_30_" + str(start)[:6] + ".png"
	main_plot(
		data,
		what_type="map",
		mode=[0, plot_kw],
		save=True,
		save_name=save_name,
		show=False
		)
	print("\n")



#A_by_dayのマップを月ごとに出力して保存するコード
"""
plot_kw = "A_by_day"
start_list = []
M = len(start_list)

for i, start in enumerate(start_list):
	print("***************  {}/{}  ***************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["ex_1"], 
		region=None, 
		accumulate=True
		)

	data_array = np.array(data)
	data_array = np.ma.masked_invalid(data_array)
	data_count_nan = np.sum(data_array.recordmask, axis=0)
	data_ave = np.sum(data_array, axis=0) / (len(date_ax) - data_count_nan)
	#A_by_dayなので0列目
	data_ave = pd.DataFrame(data_ave[:, 0])
	data_ave.columns = plot_kw
	#閾値を設ける場合
	#threshold = 0.05
	#data_ave.loc[data_ave.plot_kw>=threshold, :] = np.nan

	save_name = "A_by_30_" + str(start)[:6] + ".png"
	main_plot(
		data_ave,
		what_type="map",
		mode=[0, plot_kw],
		save=True,
		save_name=save_name,
		show=False
		)
	print("\n")
"""



"""
#TODO
main_data内の
	datetimeの処理
	accumulate=Falseの時の分岐

"""









