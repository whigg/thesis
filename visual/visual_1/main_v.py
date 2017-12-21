
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
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

g = np.arange(0,145,1)
points = np.meshgrid(g, g)

latlon_ex = pd.read_csv(latlon145_file_name)


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


def main_v(start, end, span, what_type, mode, region=None, show=True, save=False, return_data=True):
	date_ax, date_ax_str = get_date_ax(start, end)
	N = len(date_ax_str)

	if what_type == "line":
		#data_columnsはここで選ぶ
		data_columns = ["A"]
		ts_value_df = pd.DataFrame([], columns=data_columns) #時系列用

	for i, day in enumerate(date_ax_str):
		print ("{}/{}: {}".format(i+1, N, day))
		year = day[2:4]
		month = day[4:6]

		#ファイル名の生成
		"""
		#iMac用
		wind_file_name = "../data/wind_data/ecm" + day[2:] + ".csv"
		ice_file_name = "../data/ice_wind_data/" + day[2:] + ".csv"
		ic0_145_file_name = "../data/IC0_csv/2" + day + "A.csv"
		ic0_900_file_name = "../data/IC0_csv/2" + day + "A.csv"
		coeff_file_name = "../data/A_csv/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"
		"""

		#Mac pro用
		wind_file_name = "../../data/ecm" + day[2:] + ".csv"
		ice_file_name = "../../data/" + day[2:] + ".csv"
		ic0_145_file_name = "../../data/2" + day + "A.csv"
		ic0_900_file_name = "../../data/2" + day + "A.csv"
		coeff_file_name = "../../data/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"

		if not all(os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(ic0_145_file_name), os.path.isfile(coeff_file_name)):
			print ("Skipping" + day + "file...")
			continue

		#dataの取得
		detail_level = 2
		data = calc_data.get_wind_ic0_regression_data(wind_file_name, ice_file_name, ic0_145_file_name, coeff_file_name, detail_level=detail_level)

		#dataの計算(必要に応じてcalc_dataに投げる部分)
		if region is not None:
			#region: ["north_polar"]
			data = calc_data.get_wind_ic0_regression_data_by_region(data, region, detail_level=detail_level)
		#print (data.head(3))
		print (len(data.dropna()))

		#可視化
		if what_type == "line":
			value_1day = calc_data.get_ts_value(data, data_columns)
			ts_value_df = pd.concat([ts_value_df, value_1day])

		elif what_type == "non_line":

			if save == True:
				save_path = "../result/" + "A/"
				save_name = "ssc_" + day + ".png"
				visualize.visual_1day_all_2d(data, mode=mode, show=show, save_name=save_name)
			else:
				visualize.visual_1day_all_2d(data, mode=mode, show=show, save_name=None)

		elif what_type == "map":
			mode0 = mode[0]
			submode = mode[1]

			lon, lat = calc_data.get_lonlat(latlon145_file_name, array=True)
			"""
			if region is not None:
				#lon, latだけレコード削除．点の描画だけに使う
				region_index = np.array(data["data_idx"])
				lon, lat = lon[region_index], lat[region_index]
			"""

			if save==True:
				save_path = "../result/" + "A/"
				save_name = "ssc_" + day + ".png"
			else:
				save_name = None

			if mode0 == 0:
				#print (data.head(15))
				#visualize.visual_coeffs(data, mode=mode0, submode=submode, latlon145_file_name=latlon145_file_name, points=points, save_name=save_name, show=True)
				visualize.visual_coeffs(data, mode=mode0, submode=submode, latlon145_file_name=latlon145_file_name, points=points, save_name=save_name, show=True)
			if mode0 == 1:
				data = [ic0_900_file_name, latlon900_file_name]
				visualize.visual_coeffs(data, mode0, submode, latlon145_file_name, points, save_name=save_name, show=True)
			"""
			elif mode == 2:
				visualize.visual_coeffs(data, submode, latlon145_file_name, points, save_name=save_name, show=True)
				w_u, w_v, w_speed = calc_data.get_1day_w_data(wind_file_name)
				#visualize.visual_wind([w_u[region_index], w_v[region_index], w_speed[region_index]], [lon, lat], points, show=True)
				visualize.visual_wind([w_u, w_v, w_speed], [lon, lat], points, show=True)
			elif mode==2:
				visualize.visual_ic0_900(ic0_900_file_name, latlon900_file_name, show=True)
			elif mode==3:
			"""

	if what_type == "line":
		result = pd.DataFrame(date_ax)
		pd.concat([result, ts_value_df], axis=1)
		ts_value_df.columns = ["date"] + data_columns
		#月ごとに見たいなどは，ここで新たな処理をする
		ts_value_df = ts_value_df
		#最後にプロット
		visualize.visual_ts(ts_value_df)


	#取得・計算した結果を渡すかどうか
	if return_data == True:
		if what_type == "line":
			return ts_value_df
		else:
			return data
	else:
		return 0



basic_region = ["bearing_sea", "chukchi_sea", "beaufort_sea", "canada_islands", "hudson_bay", "buffin_bay", "labrador_sea", "greenland_sea", 
	"norwegian_sea", "barents_sea", "kara_sea", "laptev_sea", "east_siberian_sea", "north_polar"]

#a = main_v(start=20130301, end=20130301, span=30, what_type="map", mode=[0,5], region=None, show=True, save=False)
#a = main_v(start=20130301, end=20130301, span=30, what_type="map", mode=[0,8], region=None, show=True, save=False)
#a = main_v(start=20130301, end=20130301, span=30, what_type="map", mode=[0,8], region=["north_polar"], show=True, save=False)
#a = main_v(start=20130301, end=20130301, span=30, what_type="non_line", mode=["scatter",1], region=["north_polar"], show=True, save=False)
#a = main_v(start=20130301, end=20130301, span=30, what_type="non_line", mode=["scatter",1], region=["north_polar"], show=True, save=False)
#a = main_v(start=20130301, end=20130301, span=30, what_type="non_line", mode=["scatter",1], region=None, show=True, save=False)
#print (a["A"].dropna())
#print (len(a))
#print (a[((a.theta_by_day>=57) | (a.theta_by_day<=-57)) & (a.Name=="north_polar")])



"""
"bearing_sea",
"chukchi_sea",
"beaufort_sea",
"canada_islands",
"hudson_bay",
"buffin_bay",
"labrador_sea",
"greenland_sea",
"norwegian_sea",
"barents_sea",
"kara_sea",
"laptev_sea",
"east_siberian_sea", 
"north_polar"
"""



def main_ts_v(start, end, span, what_type, mode, region=None, show=True, save=False, return_data=True):
	date_ax, date_ax_str = get_date_ax(start, end)
	N = len(date_ax_str)

	date_ax = pd.DataFrame(date_ax)
	#取得したいdata_columnsはここで選ぶ
	data_columns_basic = ["data_idx", "Lon", "Lat", "Label", "Name"]
	data_columns = data_columns_basic + ["A_by_day", "theta_by_day", "ic0_145", "A", "angle"]

	ex_columns = ["date"] + data_columns
	data_all = pd.DataFrame([], columns=ex_columns)

	for i, day in enumerate(date_ax_str):
		print ("{}/{}: {}".format(i+1, N, day))
		year = day[2:4]
		month = day[4:6]

		#ファイル名の生成
		"""
		#iMac用
		wind_file_name = "../data/wind_data/ecm" + day[2:] + ".csv"
		ice_file_name = "../data/ice_wind_data/" + day[2:] + ".csv"
		ic0_145_file_name = "../data/IC0_csv/2" + day + "A.csv"
		ic0_900_file_name = "../data/IC0_csv/2" + day + "A.csv"
		coeff_file_name = "../data/A_csv/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"
		"""

		#Mac pro用
		wind_file_name = "../../data/ecm" + day[2:] + ".csv"
		ice_file_name = "../../data/" + day[2:] + ".csv"
		ic0_145_file_name = "../../data/2" + day + "A.csv"
		ic0_900_file_name = "../../data/2" + day + "A.csv"
		coeff_file_name = "../../data/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"

		#dataの取得
		#ここではdetail_levelは2にする
		detail_level = 2
		data = calc_data.get_wind_ic0_regression_data(wind_file_name, ice_file_name, ic0_145_file_name, coeff_file_name, detail_level=detail_level)

		#dataの計算(必要に応じてcalc_dataに投げる部分)
		if region is not None:
			data = calc_data.get_wind_ic0_regression_data_by_region(data, region, detail_level=detail_level)
		#print (data.head(3))
		#print (len(data.dropna()))

		data_by_day = data.loc[:,data_columns]
		print (data_columns)
		print (data_by_day.columns)

		date_column = pd.DataFrame({"date": [day]*len(data_by_day)})
		data_by_day = pd.concat([date_column, data_by_day])
		data_all = pd.concat([data_all, data_by_day])

	print (data_all.head())

	#取得・計算した結果を渡すかどうか
	if return_data == True:
			return data_all
	else:
		return 0



b = main_ts_v(start=20130301, end=20130315, span=30, what_type="map", mode=[0,5], region=basic_region, show=True, save=False)

import seaborn as sns
sns.jointplot(x=b.ic0_145, y=b.A_by_day, kind="reg")
plt.show()







"""
#指定したインデックスだけ地図上にプロットしたいとき
#例えば、↓
idx = np.array(a[:,0].tolist(), dtype='int64')
idx = idx[wi_ratio>=0.2]

m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
fig=plt.figure(figsize=(8,8))

#グリッドの描画
df_latlon = pd.read_csv(latlon145_file_name, header=None)
latlon = np.array(df_latlon, dtype='float32')
lat = latlon[:,2]
lon = latlon[:,3]
lons, lats = m(lon[idx], lat[idx], inverse=False)
m.plot(lons,lats,'bo', markersize=2)

m.drawcoastlines(color = '0.15')

plt.show()
"""



"""
spanとmonthからyearごとにとってくる
つまり、ある指定の月の年の時系列
"""
"""
span = 30
month = 1
file_path = sorted(glob.glob("../getA/" + str(span) + '/*/*.dat'))

data = np.zeros(145*145)
for filename in file_path:
	str_month = str(month)
	if filename[11] == str_month:
		print ("span: {}, month: {}, year: {}".format(span, month, str(20)+filename[25:27]))
		data_vec = read_coeffs(filename)
		data = np.c_[data, data_vec]

data = pd.DataFrame(data)
data = data.drop(data.columns[[0]], axis=1)
year_list = np.arange(2003, 2014).tolist()
data.columns = year_list
print (data.head())
"""











