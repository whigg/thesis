
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
		year = day[2:4]
		month = day[4:6]

		#ファイル名の生成
		wind_file_name = "../data/csv_w/ecm" + day[2:] + ".csv"
		ice_file_name = "../data/csv_iw/" + day[2:] + ".csv"
		ic0_145_file_name = "../data/csv_ic0/IC0_" + day + ".csv"
		sit_145_file_name = "../data/csv_sit/SIT_" + day + ".csv"
		"""
		if int(year) < 2012:
			ic0_145_file_name = "../data/binary_ic0/IC0_amsr/P1AME" + day + "A_600IC0NP.dat"
			sit_145_file_name = "../data/binary_sit/SIT_amsr/P1AME" + day + "A_SITNP.dat"
		else:
			ic0_145_file_name = "../data/binary_ic0/IC0_amsr2/GW1AM2" + day + "A_IC0NP.dat"
			sit_145_file_name = "../data/binary_sit/SIT_amsr2/GW1AM2" + day + "A_SITNP.dat"
		"""
		coeff_file_name = "../data/csv_A_30/ssc_amsr_ads" + str(year) + str(month) + "_" + str(span) + "_fin.csv"
		hermert_file_name = "../data/csv_Helmert_30/Helmert_30_" + str(day)[:6] + ".csv"
		# wind10m_file_name = "../data/netcdf4/" + day[2:] + ".csv"
		# t2m_file_name = "../data/netcdf4/" + day[2:] + ".csv"

		skipping_boolean = ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name)]))
		if ("ic0_145" in get_columns):
			skipping_boolean = ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name), os.path.isfile(ic0_145_file_name)]))
		elif ("sit_145" in get_columns):
			skipping_boolean = ("coeff" not in get_columns) and (not all([os.path.isfile(wind_file_name), os.path.isfile(ice_file_name), os.path.isfile(coeff_file_name), os.path.isfile(sit_145_file_name)]))
			
		if skipping_boolean == True:
			print ("\tSkipping " + day + " file...")
			date_ax_str.remove(day)
			bb = date(int(day[:4]), int(day[4:6]), int(day[6:]))
			date_ax.remove(bb)
			skipping_date_str.append(day)
			continue
		#print(np.array(ocean_grid_145))
		data = pd.DataFrame({"data_idx": np.array(ocean_grid_145).ravel()})
		if "ex_1" in get_columns:
			tmp = calc_data.get_w_regression_data(wind_file_name, ice_file_name, coeff_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "ex_2" in get_columns:
			tmp = calc_data.get_w_hermert_data(wind_file_name, ice_file_name, hermert_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "w" in get_columns:
			tmp = calc_data.get_1day_w_data(wind_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "iw" in get_columns:
			tmp = calc_data.get_1day_iw_data(ice_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "ic0_145" in get_columns:
			tmp = calc_data.get_1day_ic0_data(ic0_145_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "sit_145" in get_columns:
			tmp = calc_data.get_1day_ic0_data(sit_145_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "coeff" in get_columns:
			tmp = calc_data.get_1month_coeff_data(coeff_file_name)
			data = pd.concat([data, tmp], axis=1)
		if "hermert" in get_columns:
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


###############################################################################################################

#SITデータの可視化実験
#メルトポンドの挙動を兼ねて
def test_2():
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
def test_3():
	start_list = []
	n = 20000000
	y_list = [2,4,6,8,10,14,16]
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

!		data = data[:, :, [0, 3]]
		data = pd.DataFrame(data)
		data.columns = ["iw", "w"]

		save_name = dirs + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data,
			mode=["scatter", ["w", "iw"]],
			save_name=save_name,
			show=False
			)



###############################################################################################################

#R2_30
def H_R2_30():
	dirs = "../result_h/R2/R2_30/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["R2"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=1, 
			vmin=0,
			cmap=plt.cm.jet
			)
		print("\n")


#epsilon2_30
def H_epsilon2_30():
	dirs = "../result_h/epsilon2/epsilon2_30/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["epsilon2"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=None, 
			vmin=None,
			cmap=plt.cm.jet
			)
		print("\n")


###############################################################################################################

#A_30_original_test
#実験用に軸の範囲は揃えていない
def H_A_30_test():
	dirs = "../result_h/A/A_30_test/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
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
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
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
	mkdir(dirs)

	df_latlon = pd.read_csv("../data/latlon_ex.csv")
	lon = df_latlon.Lon
	lat = df_latlon.Lat
	lon = np.array(lon)
	lat = np.array(lat)
	#start_list=[20030101]
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		idx_low_coef = data[data.R2<0.4**2].index
		data_low_coef = np.array([np.nan]*(145**2))
		data_low_coef[idx_low_coef] = 1
		data_low_coef1 = np.reshape(data_low_coef, (145,145), order="F")
		data_low_coef1 = np.ma.masked_invalid(data_low_coef1)
		data.A.loc[data.R2<0.4**2] = np.nan
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
	mkdir(dirs_A)
	dirs_theta = "../result_h/theta/theta_by_year/"
	mkdir(dirs_theta)

	for item in A_by_year_list:
		fname = "../data/csv_Hermert_by_year/Hermert_by_year_" + item + ".csv"
		df_coeffs = pd.read_csv(fname, sep=',', dtype='float32')
		#df_coeffs.columns = ["index", "angle", "mean_ocean_u", "mean_ocean_v", "A", "coef", "data_num", "mean_ice_u", "mean_ice_v", "mean_w_u", "mean_w_v"]
		#df_coeffs = df_coeffs.drop("index", axis=1)
		#df_coeffs[df_coeffs==999.] = np.nan
		
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
			cmap=cm_angle_1
			)

		print("\n")




#A_by_day_30
def H_A_by_day_30():
	"""
	基本的に古い方のA_by_day_30と変わらないが，こっちの方がちょっとだけ正確かも
	"""
	dirs = "../result_h/A/A_by_day_30/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1
		date_ax, date_ax_str, skipping_date_str, data = main_data(
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

#angleのマップを月ごとに出力して保存するコード
def H_angle_30():
	dirs = "../result_h/angle/angle_30/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + "angle_30_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["theta"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle
			)
		print("\n")




#angle_30_high_coef
def H_angle_30_high_coef():
	dirs = "../result_h/angle/angle_30_high_coef/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		date_ax, date_ax_str, skipping_date_str, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		data.theta[data.R2<0.4**2] = np.nan

		save_name = dirs + "angle_30_high_coef_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data["theta"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle
			)
		print("\n")




#angleにwindの平均を重ねたマップ
def H_angle_30_and_wind():
	dirs = "../result_h/angle/angle_30_and_wind/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#angleデータの取得
		date_ax, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		data.theta.loc[data.R2<0.4**2] = np.nan

		#地衡風平均の出力
		date_ax, _, _, data_w = main_data(
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
		#print(data_ave.head(3))

		save_name = dirs + "angle_30_and_wind_" + str(start)[:6] + ".png"

		visualize.plot_map_multi(
			data_ave.loc[:, ["w_u", "w_v"]], 
			data["theta"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle_1
			)
		print("\n")


###############################################################################################################

#海流の平均のマップ出力
def H_vec_mean_ocean_currents():
	dirs = "../result_h/mean_vector/mean_ocean_currents/"
	mkdir(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
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


###############################################################################################################

#散布図：Aとic0 北極のみ
#散布図：Aとic0 相関が低いものは除く 北極のみ
#散布図：angleとic0 相関が低いものは除く 北極のみ
def H_scatter_ic0_np():
	dirs_A_30_and_ic0_np = "../result_h/scatter/scatter_A_30_and_ic0_np/"
	mkdir(dirs_A_30_and_ic0_np)
	dirs_A_30_and_ic0_h_np = "../result_h/scatter/scatter_A_30_and_ic0_h_np/"
	mkdir(dirs_A_30_and_ic0_h_np)
	dirs_angle_30_and_ic0_np = "../result_h/scatter/scatter_angle_30_and_ic0_np/"
	mkdir(dirs_angle_30_and_ic0_np)
	dirs_angle_30_and_ic0_h_np = "../result_h/scatter/scatter_angle_30_and_ic0_h_np/"
	mkdir(dirs_angle_30_and_ic0_h_np)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A_original = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		_, _, _, data_ic0_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_30)
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["ic0_30"]

		#dirs_A_30_and_ic0_np
		data_A = data_A_original["A"]
		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		data = data[data.Name=="north_polar"]

		save_name = dirs_A_30_and_ic0_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data,
			mode=["scatter", ["ic0_30", "A"]],
			save_name=save_name,
			show=False
			)

		#dirs_A_30_and_ic0_h_np
		data_A_1 = data_A_original.loc[:, ["A", "R2"]]
		data_A_1.A.loc[data_A_1.R2<0.4**2] = np.nan
		data_1 = pd.concat([latlon_ex, data_A_1, data_ave], axis=1)
		data_1 = data_1[data_1.Name=="north_polar"]

		save_name = dirs_A_30_and_ic0_h_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_1,
			mode=["scatter", ["ic0_30", "A"]],
			save_name=save_name,
			show=False
			)

		#dirs_angle_30_and_ic0_np
		data_angle = data_A_original["theta"]
		data_2 = pd.concat([latlon_ex, data_angle, data_ave], axis=1)
		data_2 = data_2[data_2.Name=="north_polar"]

		save_name = dirs_angle_30_and_ic0_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_2,
			mode=["scatter", ["ic0_30", "theta"]],
			save_name=save_name,
			show=False
			)

		#dirs_angle_30_and_ic0_h_np
		data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
		data_angle_1.loc[data_angle_1.R2<0.4**2, data_angle_1.theta] = np.nan
		data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
		data_3 = data_3[data_3.Name=="north_polar"]

		save_name = dirs_angle_30_and_ic0_h_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_3,
			mode=["scatter", ["ic0_30", "theta"]],
			save_name=save_name,
			show=False
			)


		print("\n")






#散布図：Aとsit 全海域
#散布図：Aとsit 北極のみ
#散布図：Aとsit 相関が低いものは除く 北極のみ
#散布図：angleとsit 全海域
#散布図：angleとsit 北極海のみ
#散布図：angleとsit 相関が低いものは除く 北極のみ
def H_scatter_sit_np():
	dirs_A_30_and_sit_all = "../result/scatter/scatter_A_30_and_sit_all/"
	mkdir(dirs_A_30_and_sit_all)
	dirs_A_30_and_sit_np = "../result/scatter/scatter_A_30_and_sit_np/"
	mkdir(dirs_A_30_and_sit_nps)
	dirs_A_30_and_sit_h_np = "../result/scatter/scatter_A_30_and_sit_h_np/"
	mkdir(dirs_A_30_and_sit_h_np)	
	dirs_angle_30_and_sit_all = "../result/scatter/scatter_angle_30_and_sit_all/"
	mkdir(dirs_angle_30_and_sit_all)
	dirs_angle_30_and_sit_np = "../result/scatter/scatter_angle_30_and_sit_np/"
	mkdir(dirs_angle_30_and_sit_np)
	dirs_angle_30_and_sit_h_np = "../result/scatter/scatter_angle_30_and_sit_h_np/"
	mkdir(dirs_angle_30_and_sit_h_np)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A_original = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		_, _, _, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		#dirs_A_30_and_sit_all
		data_A = data_A_original["A"]
		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)

		save_name = dirs_A_30_and_sit_all + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)

		#dirs_A_30_and_sit_np
		data_A_1 = data_A_original.loc[:, ["A", "R2"]]
		data_1 = pd.concat([latlon_ex, data_A_1, data_ave], axis=1)
		data_1 = data_1[data_1.Name=="north_polar"]

		save_name = dirs_A_30_and_sit_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_1,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)

		#dirs_A_30_and_sit_h_np
		data_A_1 = data_A_original.loc[:, ["A", "R2"]]
		data_A_1.A.loc[data_A_1.R2<0.4**2] = np.nan
		data_1 = pd.concat([latlon_ex, data_A_1, data_ave], axis=1)
		data_1 = data_1[data_1.Name=="north_polar"]

		save_name = dirs_A_30_and_sit_h_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_1,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)

		#dirs_angle_30_and_sit_all
		data_angle = data_A_original["theta"]
		data_2 = pd.concat([latlon_ex, data_angle, data_ave], axis=1)

		save_name = dirs_angle_30_and_sit_all + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_2,
			mode=["scatter", ["sit_30", "theta"]],
			save_name=save_name,
			show=False
			)

		#dirs_angle_30_and_sit_np
		data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
		data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
		data_3 = data_3[data_3.Name=="north_polar"]

		save_name = dirs_angle_30_and_sit_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_3,
			mode=["scatter", ["sit_30", "theta"]],
			save_name=save_name,
			show=False
			)

		#dirs_angle_30_and_sit_h_np
		data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
		data_angle_1.loc[data_angle_1.R2<0.4**2, data_angle_1.theta] = np.nan
		data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
		data_3 = data_3[data_3.Name=="north_polar"]

		save_name = dirs_angle_30_and_sit_h_np + str(start)[:6] + ".png"
		visualize.visual_non_line(
			data_3,
			mode=["scatter", ["sit_30", "theta"]],
			save_name=save_name,
			show=False
			)


		print("\n")





def H_scatter_6win_2area():
	dirs_basic = "../result/scatter/"

	dirs_A_ic0 = dirs_basic + "A_ic0/"
	mkdir(dirs_A_ic0)
	dirs_theta_ic0 = dirs_basic + "theta_ic0/"
	mkdir(dirs_theta_ic0)
	dirs_e2_ic0 = dirs_basic + "e2_ic0/"
	mkdir(dirs_e2_ic0)
	dirs_A_sit = dirs_basic + "A_sit/"
	mkdir(dirs_A_sit)
	dirs_theta_sit = dirs_basic + "theta_sit/"
	mkdir(dirs_theta_sit)
	dirs_e2_sit = dirs_basic + "e2_sit/"
	mkdir(dirs_e2_sit)

	sns.set_style("darkgrid")
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_hermert = main_data(
			start, start, 
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
!		data_cross = data_array[:, :, 0] * data_array[:, :, 1]
		data_cross_ave = np.nanmean(data_cross, axis=0) / 100
		data_ave["cross_ic0_sit"] = data_cross_ave

		data_tmp = data_hermert.loc[:, ["A", "theta", "R2", "epsilon2"]]
		data_basic = pd.concat([latlon_ex, data_tmp, data_ave], axis=1)

		rank_np = np.zeros(145**2)
		rank_np[data[data.Name=="north_polar"].index] = 1
		rank_R2 = np.ones(145**2)
		rank_R2[data[data.R2<=(1/3)**2].index] = 0
		rank_R2[data[data.R2>(2/3)**2].index] = 2
		data_rank = pd.DataFrame({"rank_np": rank_np, "rank_R2": rank_R2})

		data = pd.concat([data_basic, data_rank], axis=1)

		save_name = dirs_A_ic0 + str(start)[:6] + ".png"
		sns.lmplot(x="ic0_30", y="A", row="rank_R2", col="rank_np", data=data, size=3)
		plt.savefig(save_name, dpi=900)
		plt.close()

		save_name = dirs_theta_ic0 + str(start)[:6] + ".png"
		sns.lmplot(x="ic0_30", y="theta", row="rank_R2", col="rank_np", data=data, size=3)
		plt.savefig(save_name, dpi=900)
		plt.close()

		save_name = dirs_e2_ic0 + str(start)[:6] + ".png"
		sns.lmplot(x="ic0_30", y="epsilon2", row="rank_R2", col="rank_np", data=data, size=3)
		plt.savefig(save_name, dpi=900)
		plt.close()

		save_name = dirs_A_sit + str(start)[:6] + ".png"
		sns.lmplot(x="sit_30", y="A", row="rank_R2", col="rank_np", data=data, size=3)
		plt.savefig(save_name, dpi=900)
		plt.close()

		save_name = dirs_theta_sit + str(start)[:6] + ".png"
		sns.lmplot(x="sit_30", y="theta", row="rank_R2", col="rank_np", data=data, size=3)
		plt.savefig(save_name, dpi=900)
		plt.close()

		save_name = dirs_e2_sit + str(start)[:6] + ".png"
		sns.lmplot(x="sit_30", y="epsilon2", row="rank_R2", col="rank_np", data=data, size=3)
		plt.savefig(save_name, dpi=900)
		plt.close()


		print("\n")





###############################################################################################################

!
"""
ディレクトリの処理
y_limの追加
折れ線を重ねる場合どうするか
"""


#Aの時系列変化
def H_ts_A_month():
	def plot_param_1g(save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		for i, item in enumerate(plot_param_list):
			plot_data_1g = data_m_1g[item].loc[:, ["mean", "std", "50%"]]
			plot_data_1g["2sigma_pos"] = plot_data_1g['mean']+2*np.sqrt(plot_data_1g['std'])
			plot_data_1g["2sigma_neg"] = plot_data_1g['mean']-2*np.sqrt(plot_data_1g['std'])
			plot_data_1g["Month"] = np.arange(1, 13, 1)
			plt.subplot(311+i)
			plt.plot_date(plot_data_1g['Month'], plot_data_1g['mean'], '-')
			plt.plot_date(plot_data_1g['Month'], plot_data_1g["2sigma_pos"], '-')
			plt.plot_date(plot_data_1g['Month'], plot_data_1g["2sigma_neg"], '-')
			d = plot_data_1g['Month'].values
			plt.fill_between(d, plot_data_1g['2sigma_pos'], plot_data_1g['2sigma_neg'],
				facecolor='green', alpha=0.2, interpolate=True)

		plt.savefig(save_name)
		plt.close()

	def plot_param_2g(save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		for i, item in enumerate(plot_param_list):
			for j, is_np in enumerate([1,0]):
				plot_data_2g_np_pos = data_m_2g.loc[(is_np), (item, ["mean", "std", "50%"])]
				plot_data_2g_np_pos["2sigma_pos"] = plot_data_2g_np_pos['mean']+2*np.sqrt(plot_data_2g_np_pos['std'])
				plot_data_2g_np_pos["2sigma_neg"] = plot_data_2g_np_pos['mean']-2*np.sqrt(plot_data_2g_np_pos['std'])
				plot_data_2g_np_pos["Month"] = np.arange(1, 13, 1)
				plt.subplot(321+j+i*2)
				plt.plot_date(plot_data_2g_np_pos['Month'], plot_data_2g_np_pos['mean'], '-')
				plt.plot_date(plot_data_2g_np_pos['Month'], plot_data_2g_np_pos["2sigma_pos"], '-')
				plt.plot_date(plot_data_2g_np_pos['Month'], plot_data_2g_np_pos["2sigma_neg"], '-')
				d = plot_data_2g_np_pos['Month'].values
				plt.fill_between(d, plot_data_2g_np_pos['2sigma_pos'], plot_data_2g_np_pos['2sigma_neg'],
					facecolor='green', alpha=0.2, interpolate=True)
	
		plt.savefig(save_name)
		plt.close()

	def plot_param_3g(plot_param_item, save_name):
		for i, is_np in enumerate([1, 0]):
			plot_data_3g_np_pos = data_m_2g.loc[(is_np), (plot_param_item, ["mean", "std", "50%"])]
			plot_data_3g_np_pos["2sigma_pos"] = plot_data_3g_np_pos[(plot_param_item, "mean")] + 2*np.sqrt(plot_data_3g_np_pos[(plot_param_item, "std")])
			plot_data_3g_np_neg["2sigma_neg"] = plot_data_3g_np_neg[(plot_param_item, "mean")] - 2*np.sqrt(plot_data_3g_np_neg[(plot_param_item, "std")])
			#plot_data_3g_np_neg["Month"] = np.arange(1, 13, 1)
			for j, subplot_idx in enumerate([0,1,2]):
				plt.subplot(321+i+subplot_idx*2)
				plt.plot_date(np.arange(1, 13, 1), plot_data_3g_np_pos.loc[(j), (plot_param_item, "mean")], '-')
				plt.plot_date(np.arange(1, 13, 1), plot_data_3g_np_pos.loc[(j), ("2sigma_pos")], '-')
				plt.plot_date(np.arange(1, 13, 1), plot_data_3g_np_pos.loc[(j), ("2sigma_neg")], '-')
				plt.fill_between(np.arange(1, 13, 1), plot_data_3g_np_pos.loc[(j), ("2sigma_pos")], plot_data_3g_np_pos.loc[(j), ("2sigma_neg")],
					facecolor='green', alpha=0.2, interpolate=True)

		plt.savefig(save_name)
		plt.close()

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for y in y_list:
		data_m = pd.DataFrame([])
		for m in month_list:
			yymm = "20" + y + m
			hermert_file_name = "../data/csv_Helmert_30/Helmert_30_" + yymm + ".csv"
			data = calc_data.get_1month_hermert_data(hermert_file_name)
			data = pd.concat([latlon_ex["Name"], data], axis=1)
			#data = data.drop(["Lat", "Lon", "Label", "idx1", "idx2"], axis=1)
			rank_np = np.zeros(145**2)
			rank_np[data[data.Name=="north_polar"].index] = 1
			data["rank_np"] = rank_np
			rank_R2 = np.ones(145**2)
			rank_R2[data[data.R2<=(1/3)**2].index] = 0
			rank_R2[data[data.R2>(2/3)**2].index] = 2
			data["rank_R2"] = rank_R2
			data = data.dropna()
			#https://code.i-harness.com/ja/q/1c29878
			print(data.isnull().sum().sum())
			data["yymm"] = [pd.to_datetime(yymm, format="%Y%m")] * len(data)
			data_m = pd.concat([data_m, data])
			"""
			describe_data = data.groupby("rank_np")[["A", "theta", "epsilon2"]].describe()
			describe_data = data.groupby("rank_np").agg({
				'A': [np.nanmean, np.nanstd, np.nanmedian], 
				'theta': [np.nanmean, np.nanstd, np.nanmedian],
				'epsilon2': [np.nanmean, np.nanstd, np.nanmedian]
				})
			pd.concat([data_all, data.loc[:, ["A", "theta", "epsilon2"]]])
			data_m.append(np.array(data.loc[:, ["A", "theta", "epsilon2"]]))
			"""
		#data_m = np.array(data_m)

		#月ごとに全てのエリアの平均などを取得
		data_m_1g = data_m.groupby("yymm")[["A", "theta", "epsilon2"]].describe().sort_index(level='yymm')
		#月ごとにrank_npで分類したものを取得
		data_m_2g = data_m.groupby(["rank_np", "yymm"])[["A", "theta", "epsilon2"]].describe().sort_index(level='yymm')
		#月ごとにrank_npとrank_R2で分類したものを取得
		data_m_3g = data_m.groupby(["rank_np", "rank_R2", "yymm"])[["A", "theta", "epsilon2"]].describe().sort_index(level='yymm')

		#1gのプロット
		plot_param_1g(save_name)

		#2gのプロット
		plot_param_2g(save_name)

		#3gのプロット
		plot_param_list = ["A", "theta", "epsilon2"]
		for item in plot_param_list:
			plot_param_3g(item, save_name)


		"""
		#data_m_A = data_m[:,:,0].T
		#A_ave = np.nanmean(data_m_A, axis=0)
		#A_std = np.nanstd(data_m_A, axis=0)
		#https://stackoverflow.com/questions/29329725/pandas-and-matplotlib-fill-between-vs-datetime64
		yydd_str = "20" + y
		tmp = pd.to_datetime(["20" + y + m for m in month_list])
		dates = pd.date_range(yydd_str, periods=12, freq='M')
		#dates = pd.to_datetime(["20" + y + m for m in month_list], format="%Y%m")
		data = pd.DataFrame({"A_ave": A_ave, "A_std": A_std, "Date": dates})
		data["A_2sigma_pos"] = data['A_ave']+2*data['A_std']
		data["A_2sigma_neg"] = data['A_ave']-2*data['A_std']
		plt.figure()
		plt.plot_date(data['Date'], data['A_ave'], '-')
		plt.plot_date(data['Date'], data["A_2sigma_pos"], '-')
		plt.plot_date(data['Date'], data["A_2sigma_neg"], '-')
		d = data['Date'].values
		plt.fill_between(d, data['A_2sigma_pos'], data['A_2sigma_neg'],
			facecolor='green', alpha=0.2, interpolate=True)
		plt.xticks(rotation=25)
		"""

		#http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html
		"""
		plt.plot(price.index, price, 'k')
		plt.plot(ma.index, ma, 'b')
		plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color='b', alpha=0.2)
		"""

		"""
		sns.tsplot(data=data_m_A, ci="sd")
		plt.plot(np.nanmean(data_m_A, axis=0))
		data_m_theta = data_m[:,:,1].T
		sns.tsplot(data=data_m_theta, ci="sd")
		data_m_e2 = data_m[:,:,2].T
		sns.tsplot(data=data_m_e2, ci="sd")
		"""



#Aの時系列変化
def H_ts_A_year():
	def y_plot_param_1g(save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		for i, item in enumerate(plot_param_list):
			plot_data_1g = data_m_1g[item].loc[:, ["mean", "std", "50%"]]
			plot_data_1g["2sigma_pos"] = plot_data_1g['mean']+2*np.sqrt(plot_data_1g['std'])
			plot_data_1g["2sigma_neg"] = plot_data_1g['mean']-2*np.sqrt(plot_data_1g['std'])
			plot_data_1g["Year"] = np.array([2003,2004,2005,2006,2007,2008,2009,2010,2013,2014,2015,2016])
			plt.subplot(311+i)
			plt.plot_date(plot_data_1g['Year'], plot_data_1g['mean'], '-')
			plt.plot_date(plot_data_1g['Year'], plot_data_1g["2sigma_pos"], '-')
			plt.plot_date(plot_data_1g['Year'], plot_data_1g["2sigma_neg"], '-')
			d = plot_data_1g['Year'].values
			plt.fill_between(d, plot_data_1g['2sigma_pos'], plot_data_1g['2sigma_neg'],
				facecolor='green', alpha=0.2, interpolate=True)
		plt.savefig(save_name)
		plt.close()

	def y_plot_param_2g(save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		for i, item in enumerate(plot_param_list):
			for j, is_np in enumerate([1,0]):
				plot_data_2g_np_pos = data_m_2g.loc[(is_np), (item, ["mean", "std", "50%"])]
				plot_data_2g_np_pos["2sigma_pos"] = plot_data_2g_np_pos['mean']+2*np.sqrt(plot_data_2g_np_pos['std'])
				plot_data_2g_np_pos["2sigma_neg"] = plot_data_2g_np_pos['mean']-2*np.sqrt(plot_data_2g_np_pos['std'])
				plot_data_2g_np_pos["Year"] = np.array([2003,2004,2005,2006,2007,2008,2009,2010,2013,2014,2015,2016])
				plt.subplot(321+j+i*2)
				plt.plot_date(plot_data_2g_np_pos['Year'], plot_data_2g_np_pos['mean'], '-')
				plt.plot_date(plot_data_2g_np_pos['Year'], plot_data_2g_np_pos["2sigma_pos"], '-')
				plt.plot_date(plot_data_2g_np_pos['Year'], plot_data_2g_np_pos["2sigma_neg"], '-')
				d = plot_data_2g_np_pos['Year'].values
				plt.fill_between(d, plot_data_2g_np_pos['2sigma_pos'], plot_data_2g_np_pos['2sigma_neg'],
					facecolor='green', alpha=0.2, interpolate=True)
		
		plt.savefig(save_name)
		plt.close()

	def y_plot_param_3g(plot_param_item, save_name):
		for i, is_np in enumerate([1, 0]):
			plot_data_3g_np_pos = data_m_2g.loc[(is_np), (plot_param_item, ["mean", "std", "50%"])]
			plot_data_3g_np_pos["2sigma_pos"] = plot_data_3g_np_pos[(plot_param_item, "mean")] + 2*np.sqrt(plot_data_3g_np_pos[(plot_param_item, "std")])
			plot_data_3g_np_neg["2sigma_neg"] = plot_data_3g_np_neg[(plot_param_item, "mean")] - 2*np.sqrt(plot_data_3g_np_neg[(plot_param_item, "std")])
			#plot_data_3g_np_neg["Month"] = np.arange(1, 13, 1)
			year_array = np.array([2003,2004,2005,2006,2007,2008,2009,2010,2013,2014,2015,2016])
			for j, subplot_idx in enumerate([0,1,2]):
				plt.subplot(321+i+subplot_idx*2)
				plt.plot_date(year_array, plot_data_3g_np_pos.loc[(j), (plot_param_item, "mean")], '-')
				plt.plot_date(year_array, plot_data_3g_np_pos.loc[(j), ("2sigma_pos")], '-')
				plt.plot_date(year_array, plot_data_3g_np_pos.loc[(j), ("2sigma_neg")], '-')
				plt.fill_between(year_array, plot_data_3g_np_pos.loc[(j), ("2sigma_pos")], plot_data_3g_np_pos.loc[(j), ("2sigma_neg")],
					facecolor='green', alpha=0.2, interpolate=True)
		plt.savefig(save_name)
		plt.close()

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	for m in month_list:
		data_m = pd.DataFrame([])
		for y in y_list:
			yymm = "20" + y + m
			hermert_file_name = "../data/csv_Helmert_30/Helmert_30_" + yymm + ".csv"
			data = calc_data.get_1month_hermert_data(hermert_file_name)
			data = pd.concat([latlon_ex["Name"], data], axis=1)
			#data = data.drop(["Lat", "Lon", "Label", "idx1", "idx2"], axis=1)
			rank_np = np.zeros(145**2)
			rank_np[data[data.Name=="north_polar"].index] = 1
			data["rank_np"] = rank_np
			rank_R2 = np.ones(145**2)
			rank_R2[data[data.R2<=(1/3)**2].index] = 0
			rank_R2[data[data.R2>(2/3)**2].index] = 2
			data["rank_R2"] = rank_R2
			data = data.dropna()
			#https://code.i-harness.com/ja/q/1c29878
			print(data.isnull().sum().sum())
			data["Year"] = [int("20"+y)] * len(data)
			data_m = pd.concat([data_m, data])

		#年ごとに全てのエリアの平均などを取得
		data_m_1g = data_m.groupby("Year")[["A", "theta", "epsilon2"]].describe().sort_index(level='Year')
		#年ごとにrank_npで分類したものを取得
		data_m_2g = data_m.groupby(["rank_np", "Year"])[["A", "theta", "epsilon2"]].describe().sort_index(level='Year')
		#年ごとにrank_npとrank_R2で分類したものを取得
		data_m_3g = data_m.groupby(["rank_np", "rank_R2", "Year"])[["A", "theta", "epsilon2"]].describe().sort_index(level='Year')

		#1gのプロット
		y_plot_param_1g(save_name)

		#2gのプロット
		y_plot_param_2g(save_name)

		#3gのプロット
		plot_param_list = ["A", "theta", "epsilon2"]
		for item in plot_param_list:
			y_plot_param_3g(item, save_name)



###############################################################################################################
!
"""
rank_npとrank_R2の処理
save_nameの工夫
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
		"""
		data_array_w_1 = np.ma.masked_invalid(data_array_w)
		data_w_count_nan = np.sum(data_array_w_1.recordmask, axis=0)
		date_ax_w_len = len(date_ax_w)
		#data_array_w[date_ax_w_len-data_w_count_nan<=20] = np.nan
		
		data_ave_w_sum = np.sum(data_array_w, axis=0)
		#data_ave_sum[date_ax_len-data_count_nan<=20] = np.nan
		data_ave_w = data_ave_w_sum / (date_ax_w_len-data_w_count_nan)
		"""
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


#get_helmert()




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



