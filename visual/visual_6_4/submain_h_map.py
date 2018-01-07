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
import main_v
#from main_v import get_date_ax, main_v.mkdir, main_v.main_data

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
#main_vと同じ
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

#threshold_R2 = 0.4**2
threshold_R2 = 0.4


###############################################################################################################

#A_30_original_test
#実験用に軸の範囲は揃えていない
def H_A_30_test():
	dirs = "../result_h/A/A_30_test/"
	main_v.mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_v.main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=None, 
			vmin=None,
			cmap=plt.cm.jet
			)
		print("\n")



#A_30
def H_A_30():
	dirs = "../result_h/A/A_30/"
	main_v.mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_v.main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap=plt.cm.jet
			)
		print("\n")




#A_30_with_coef
#決定係数R2の値が0.16未満の海域は茶色、それ以外は普通のjetで描きたい
def H_A_30_with_coef():
	dirs = "../result_h/A/A_30_with_coef/"
	main_v.mkdir(dirs)

	df_latlon = pd.read_csv("../data/latlon_ex.csv")
	lon = df_latlon.Lon
	lat = df_latlon.Lat
	lon = np.array(lon)
	lat = np.array(lat)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_v.main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		idx_low_coef = data[data.R2<threshold_R2].index
		data_low_coef = np.array([np.nan]*(145**2))
		data_low_coef[idx_low_coef] = 1
		data_low_coef1 = np.reshape(data_low_coef, (145,145), order="F")
		data_low_coef1 = np.ma.masked_invalid(data_low_coef1)
		data.A.loc[data.R2<threshold_R2] = np.nan
		data = np.array(data.A)
		data1 = np.reshape(data, (145,145), order="F")
		data1 = np.ma.masked_invalid(data1)

		save_name = dirs + str(start)[:6] + ".png"

		m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
		fig = plt.figure(figsize=(6.5, 6.5))
		x, y = m(lon, lat)
		x1 = np.reshape(x, (145,145), order="F")
		y1 = np.reshape(y, (145,145), order="F")
		dx1 = (x1[1,0]-x1[0,0])/2
		dy1 = (y1[0,1]-y1[0,0])/2
		x2 = np.linspace(x1[0,0], x1[144,0], 145)
		y2 = np.linspace(y1[0,0], y1[0,144], 145)
		xx, yy = np.meshgrid(x2, y2)
		xx, yy = xx.T, yy.T
		xx = np.hstack([xx, xx[:,0].reshape(145,1)])
		xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
		yy = np.vstack([yy, yy[0,:]])
		yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

		m.drawcoastlines(color = '0.15')
		# m.plot(xx[144,0], yy[144,0], "bo")
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=0.025, vmin=0)
		m.colorbar(location='bottom')
		cm_brown = visualize.generate_cmap(["burlywood", "burlywood"])
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data_low_coef1, cmap=cm_brown)
		#plt.show()
		plt.savefig(save_name, dpi=900)
		plt.close()

		print("\n")



#A_by_year, theta_by_yearのマップ出力
#オリジナル
def H_A_and_theta_by_year():
	A_by_year_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	dirs_A = "../result_h/A/A_by_year/"
	main_v.mkdir(dirs_A)
	dirs_theta = "../result_h/theta/theta_by_year/"
	main_v.mkdir(dirs_theta)

	for item in A_by_year_list:
		fname = "../data/csv_Helmert_by_year/Helmert_by_year_" + item + ".csv"
		df_coeffs = pd.read_csv(fname, sep=',', dtype='float32')
		
		save_name_A = dirs_A + "Hermert_" + item + ".png"
		visualize.plot_map_once(
			df_coeffs["A"],
			data_type="type_non_wind",
			save_name=save_name_A,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap=plt.cm.jet
			)

		save_name_theta = dirs_theta + "Hermert_" + item + ".png"
		visualize.plot_map_once(
			df_coeffs["theta"],
			data_type="type_non_wind",
			save_name=save_name_theta,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle_2
			)

		print("\n")




#A_by_day_30
def H_A_by_day_30():
	"""
	基本的に古い方のA_by_day_30と変わらないが，こっちの方がちょっとだけ正確かも
	"""
	dirs = "../result_h/A/A_by_day_30/"
	main_v.mkdir(dirs)

	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1
		date_ax, date_ax_str, skipping_date_str, data = main_v.main_data(
			start, end, 
			span=30, 
			get_columns=["ex_2"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data)
		data_ave = np.nanmean(data_array, axis=0)
		#A_by_dayなので0列目
		data_ave = pd.DataFrame(data_ave[:, 0])
		data_ave.columns = ["A_by_day"]

		save_name = dirs + "A_by_day_30_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data_ave["A_by_day"],
			data_type="type_non_wind", 
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=None,
			cmap=plt.cm.jet
			)
		print("\n")



###############################################################################################################

#angle_30, angle_30_high_coef
def H_angle_30():
	dirs_original = "../result_h/angle/angle_30/"
	main_v.mkdir(dirs_original)
	dirs_h_coef = "../result_h/angle/angle_30_high_coef/"
	main_v.mkdir(dirs_h_coef)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_v.main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		#angle_30
		save_name_original = dirs_original + "angle_30_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["theta"],
			data_type="type_non_wind",
			save_name=save_name_original,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle_2
			)

		#angle_30_high_coef
		data.theta[data.R2<threshold_R2] = np.nan
		save_name = dirs_h_coef + "angle_30_high_coef_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["theta"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle_2
			)

		print("\n")




#angleにwindの平均を重ねたマップ
#high_coef
def H_angle_30_and_wind():
	dirs = "../result_h/angle/angle_30_and_wind/"
	main_v.mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#angleデータの取得
		_, _, _, data = main_v.main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		data.theta[data.R2<threshold_R2] = np.nan

		#地衡風平均の出力
		_, _, _, data_w = main_v.main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_w)
		data_ave = np.nanmean(data_array, axis=0)

		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["w_speed", "w_u", "w_v"]

		save_name = dirs + "angle_30_and_wind_" + str(start)[:6] + ".png"

		visualize.plot_map_multi(
			data_ave.loc[:, ["w_u", "w_v"]], 
			data["theta"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle_2
			)
		print("\n")


###############################################################################################################

#海流の平均のマップ出力
def H_vec_mean_ocean_currents():
	dirs = "../result_h/mean_vector/mean_ocean_currents/"
	main_v.mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_v.main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + "mean_ocean_currents_" + str(start)[:6] + ".png"

		visualize.plot_map_once(
			data.loc[:, ["ocean_u", "ocean_v"]],
			data_type="type_wind",
			save_name=save_name,
			show=False, 
			vmax=None, 
			vmin=None,
			cmap=None
			)
		print("\n")

