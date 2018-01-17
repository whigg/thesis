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
#start_list_plus_1month = start_list + [20170901]


cm_angle_2 = visualize.generate_cmap([
	"blue", 
	"Lime", 
	"grey", 
	"yellow", 
	"red"
	])


###############################################################################################################

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



###############################################################################################################

#各月の地衡風と流氷速度のspeedの散布図を描く
def test_w_iw_by_year():
	dirs_1 = "../result_h/test/w_iw_by_year/"
	dirs_2 = "../result_h/test/w_iw_by_year_without_ocean/"
	if not os.path.exists(dirs_1):
		os.makedirs(dirs_1)
	if not os.path.exists(dirs_2):
		os.makedirs(dirs_2)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	for month in month_list:
		print("************" + month + "************")
		helmert_30_30_fname = "../data/csv_Helmert_both_30/Helmert_both_30_2003" + month + ".csv"
		data_30 = pd.read_csv(helmert_30_30_fname)
		#area_16_index = np.array(data_30.loc[data_30.area_label==16, :].index)
		area_16_index = np.array(data_30[(data_30.area_label.isin(list(range(17))))].dropna().index).tolist()
		plot_grids = random.sample(area_16_index, 15)
		print(plot_grids)

		gw_list = []
		iw_list = []
		iw_ocean_list = []
		for year in y_list:
			helmert_file_name = "../data/csv_Helmert_both_30/Helmert_both_30_20" + year + month + ".csv"
			gw_file_name = "../data/csv_w/ecm" + year + month + "15.csv"
			iw_file_name = "../data/csv_iw/" + year +  month + "15.csv"

			helmert_data = calc_data.get_1month_helmert_data(helmert_file_name)
			gw_data = calc_data.get_1day_w_data(gw_file_name)
			iw_data = calc_data.get_1day_iw_data(iw_file_name)

			gw_speed = np.array(gw_data.loc[plot_grids, "w_speed"])
			iw_speed = np.array(iw_data.loc[plot_grids, "iw_speed"])
			iw_ocean_u = np.array(iw_data.loc[plot_grids, "iw_u"] - helmert_data.loc[plot_grids, "ocean_u"])
			iw_ocean_v = np.array(iw_data.loc[plot_grids, "iw_v"] - helmert_data.loc[plot_grids, "ocean_v"])
			iw_speed_ocean = np.sqrt(iw_ocean_u**2 + iw_ocean_v**2)

			gw_list.append(gw_speed)
			iw_list.append(iw_speed)
			iw_ocean_list.append(iw_speed_ocean)

		gw_array = np.array(gw_list)
		iw_array = np.array(iw_list)
		iw_ocean_array = np.array(iw_ocean_list)

		for i in range(len(plot_grids)):
			gw = gw_array[:, i]
			iw = iw_array[:, i]
			iw_ocean = iw_ocean_array[:, i]
			save_name_1 = dirs_1 + month + "15_grid_" + str(plot_grids[i]) + ".png"
			save_name_2 = dirs_2 + month + "15_grid_" + str(plot_grids[i]) + ".png"
			try:
				sns.set_style("darkgrid")
				sns.jointplot(x=gw, y=iw, kind="reg")
				plt.savefig(save_name_1)
				plt.close()
				sns.set_style("darkgrid")
				sns.jointplot(x=gw, y=iw_ocean, kind="reg")
				plt.savefig(save_name_2)
				plt.close()
			except:
				continue

#test_w_iw_by_year()


#30_30, 90_90でどれだけ海流が取れているかのマップ．別々に出力．colorbarあり
def ocean_30_vs_90_with_colorbar():
	dirs = "../result_h/mean_vector/ocean_currents_30_vs_90_with_colorbar/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	start_list.pop()
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M-1))
		helmert_30_30_fname = "../data/csv_Helmert_30/Helmert_30_" + str(start)[:6] + ".csv"
		data_30 = pd.read_csv(helmert_30_30_fname)
		data_30_vec = [np.array(data_30["ocean_u"]), np.array(data_30["ocean_v"])]
		helmert_90_90_fname = "../data/csv_Helmert_both_90/Helmert_both_90_" + str(start)[:6] + ".csv"
		data_90 = pd.read_csv(helmert_90_90_fname)
		data_90_vec = [np.array(data_90["ocean_u_90"]), np.array(data_90["ocean_v_90"])]

		vector_list = [data_30_vec, data_90_vec]
		name_list = ["_ocean_30", "_ocean_90"]
		for j in range(2):
			m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			lon = np.array(latlon_ex.Lon)
			lat = np.array(latlon_ex.Lat)
			x, y = m(lon, lat)
			x1 = np.reshape(x, (145,145), order='F')
			y1 = np.reshape(y, (145,145), order='F')
			dx1 = (x1[1,0]-x1[0,0])/2
			dy1 = (y1[0,1]-y1[0,0])/2

			x2 = np.linspace(x1[0,0], x1[144,0], 145)
			y2 = np.linspace(y1[0,0], y1[0,144], 145)
			xx, yy = np.meshgrid(x2, y2)
			xx, yy = xx.T, yy.T

			vector_u = np.ma.masked_invalid(vector_list[j][0])
			vector_v = np.ma.masked_invalid(vector_list[j][1])
			vector_speed = np.sqrt(vector_u*vector_u + vector_v*vector_v)

			data_non_wind = vector_speed
			data_non_wind = np.ma.masked_invalid(data_non_wind)
			data1 = np.reshape(data_non_wind, (145,145), order='F')

			xx = np.hstack([xx, xx[:,0].reshape(145,1)])
			xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
			yy = np.vstack([yy, yy[0,:]])
			yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

			m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=0.2, vmin=0)
			m.colorbar(location='bottom')
			m.quiver(x, y, vector_u, vector_v, color="k")
			save_name = dirs + str(start)[:6] + name_list[j] + ".png"
			print(save_name)
			plt.savefig(save_name, dpi=450)
			plt.close()

#ocean_30_vs_90_with_colorbar()





def map_corr_slide_month(slide):
	dirs = "../result_h/corr_map/corr_slide_month/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	product_ym = list(itertools.product(y_list, month_list))

	data_dict = {}
	for month in month_list:
		data_dict[month] = {}
		for year in y_list:
			data_dict[month][year] = {}
			data_dict[month][year]["ym_A"] = np.array([np.nan]*(145**2))
			data_dict[month][year]["ym_ic0"] = np.array([np.nan]*(145**2))
			data_dict[month][year]["data_A"] = np.array([np.nan]*(145**2))
			data_dict[month][year]["data_ic0"] = np.array([np.nan]*(145**2))

	for k, item in enumerate(product_ym):
		if k+slide > len(product_ym)-1:
			break
		x_ym = product_ym[k][0] + product_ym[k][1]
		y_ym = product_ym[k+slide][0] + product_ym[k+slide][1]
		print(x_ym, y_ym)
		if (k<=95) & (k+slide>95):
			print("\tcontinue: {}".format((x_ym, y_ym)))
			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_A"] = x_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_ic0"] = y_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_A"] = np.array([np.nan]*(145**2))
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_ic0"] = np.array([np.nan]*(145**2))
			continue
		else:
			helmert_axis_x_file = "../data/csv_Helmert_both_30/Helmert_both_30_20" + x_ym + ".csv"
			helmert_axis_y_file = "../data/csv_Helmert_both_30/Helmert_both_30_20" + y_ym + ".csv"

			data_A = pd.read_csv(helmert_axis_x_file)["A"]
			data_ic0 = pd.read_csv(helmert_axis_y_file)["ic0_30"]

			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_A"] = x_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["ym_ic0"] = y_ym
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_A"] = np.array(data_A)
			data_dict[product_ym[k][1]][product_ym[k][0]]["data_ic0"] = np.array(data_ic0)

	for month in month_list:
		accumulate_data_A = []
		accumulate_data_ic0 = []
		for year in y_list:
			#print(month, year)
			accumulate_data_A.append(data_dict[month][year]["data_A"])
			accumulate_data_ic0.append(data_dict[month][year]["data_ic0"])
		accumulate_data_A = np.array(accumulate_data_A)
		accumulate_data_ic0 = np.array(accumulate_data_ic0)

		corr_list = []
		for i in range(145**2):
			plot_data_A = accumulate_data_A[:, i]
			#data_A = data_A[~np.isnan(data_A)]
			plot_data_ic0 = accumulate_data_ic0[:, i]
			tmp_df = pd.DataFrame({"data_A": plot_data_A, "data_ic0": plot_data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
			corr_list.append(corr)
		corr_array = np.array(corr_list)

		save_name = dirs + "ic0_A_start_month_" + month + "_slide_" + str(slide) + ".png"
		print("\t{}".format(save_name))
		visualize.plot_map_once(corr_array, data_type="type_non_wind", show=False, 
			save_name=save_name, vmax=1, vmin=-1, cmap=plt.cm.jet)

"""
for i in range(1,6):
	map_corr_slide_month(slide=i)
"""



"""
def map_corr_slide_month_5x5(slide):

	def divide_grid_5x5():
		a = np.arange(0,145**2).reshape(145,145).T
		return np.array([np.sort(a[i:i+5, j:j+5].ravel()) for i in range(29) for j in range(29)])

	dirs = "../result_h/corr_map/corr_slide_month_5x5/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	product_ym = list(itertools.product(y_list, month_list))

	data_dict = {}
	for month in month_list:
		data_dict[month] = {}
		for year in y_list:
			data_dict[month][year] = {}

	for k, item in enumerate(product_ym):
		
		#インデックス超過と2011,2012年の処理が必要
		
		x_ym = product_ym[k][0] + product_ym[k][1]
		y_ym = product_ym[k+slide][0] + product_ym[k+slide][1]
		helmert_axis_x_file = "../data/csv_Helmert_both_30/Helmert_both_30_20" + x_ym + ".csv"
		helmert_axis_y_file = "../data/csv_Helmert_both_30/Helmert_both_30_20" + y_ym + ".csv"

		data_A = pd.read_csv(helmert_axis_x_file)["A"]
		data_ic0 = pd.read_csv(helmert_axis_y_file)["ic0_30"]

		data_dict[product_ym[k][1]][product_ym[k][0]]["ym_A"] = x_ym
		data_dict[product_ym[k][1]][product_ym[k][0]]["ym_ic0"] = y_ym
		data_dict[product_ym[k][1]][product_ym[k][0]]["data_A"] = np.array(data_A)
		data_dict[product_ym[k][1]][product_ym[k][0]]["data_ic0"] = np.array(data_ic0)

	for month in month_list:
		accumulate_data_A = []
		accumulate_data_ic0 = []
		for year in y_list:
			accumulate_data_A.append(data_dict[month][year]["data_A"])
			accumulate_data_ic0.append(data_dict[month][year]["data_ic0"])
		accumulate_data_A = np.array(accumulate_data_A)
		accumulate_data_ic0 = np.array(accumulate_data_ic0)

		corr_list = []
		for i in range(145**2):
			plot_data_A = accumulate_data_A[:, i]
			#data_A = data_A[~np.isnan(data_A)]
			plot_data_ic0 = accumulate_data[:, i]
			tmp_df = pd.DataFrame({"data_A": plot_data_A, "data_ic0": plot_data_ic0})
			if len(tmp_df.dropna()) <= 5:
				corr = np.nan
			else:
				corr = tmp_df.dropna().corr()
				corr = np.array(corr)[0,1]
			corr_list.append(corr)
		corr_array = np.array(corr_list)

		m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
		m.drawcoastlines(color = '0.15')
		m.fillcontinents(color='#555555')
		lon = np.array(latlon_ex.Lon)
		lat = np.array(latlon_ex.Lat)
		x, y = m(lon, lat)
		x1 = np.reshape(x, (145,145), order='F')
		y1 = np.reshape(y, (145,145), order='F')
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

		data0 = np.ma.masked_invalid(corr_array)
		data1 = np.reshape(data0, (145,145), order='F')
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=1, vmin=-1)
		m.colorbar(location="bottom")

		save_name = dirs + "A_ic0_start_month_" + month + "_slide_" + str(slide) + ".png"
		print(save_name)
		plt.savefig(save_name, dpi=500)
		plt.close()
"""







def corr_by_ic0_norm():
	dirs_1 = "../result_h/scatter/A_30_and_ic0_norm/"
	dirs_2 = "../result_h/scatter/A_30_and_ic0_norm_R2/"
	dirs_3 = "../result_h/scatter/A_30_and_ic0_norm_ic0/"
	for dirs in [dirs_1, dirs_2, dirs_3]:
		if not os.path.exists(dirs):
			os.makedirs(dirs)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		helmert_file_name = "../data/csv_Helmert_both_30/Helmert_both_30_" + str(start)[:6] + ".csv"
		data = pd.read_csv(helmert_file_name)
		data_1 = data.loc[:, ["ic0_30", "A", "area_label"]].dropna()
		data_2 = data.loc[data.R2>=0.36, ["ic0_30", "A", "area_label"]].dropna()
		data_3 = data.loc[data.ic0_30<100, ["ic0_30", "A", "area_label"]].dropna()
		for j in range(17):
			data_1_area = data_1.loc[data_1.area_label==j, :].dropna()
			data_2_area = data_2.loc[data_2.area_label==j, :].dropna()
			data_3_area = data_3.loc[data_3.area_label==j, :].dropna()
			save_name_1 = dirs_1 + "area_" + str(j) + "_" + str(start)[:6] + ".png"
			save_name_2 = dirs_2 + "area_" + str(j) + "_" + str(start)[:6] + ".png"
			save_name_3 = dirs_3 + "area_" + str(j) + "_" + str(start)[:6] + ".png"

			data_list = [data_1_area, data_2_area, data_3_area]
			save_list = [save_name_1, save_name_2, save_name_3]
			for k in range(3):
				print(save_list[k])
				x = data_list[k]["ic0_30"]
				#print(x.mean(0))
				#print(x.sub(x.mean(0)))
				xd = x.sub(x.mean(0)).div(x.std(0))
				y = data_list[k]["A"]
				yd = y.sub(y.mean(0)).div(y.std(0))
				try:
					sns.set_style("darkgrid")
					sns.jointplot(x=xd, y=yd, kind="reg")
					plt.savefig(save_list[k])
					plt.close()
				except:
					print("plot passed...")
					continue
				

#corr_by_ic0_norm()



#各月の年ごとの時系列変化
def ts_30_by_year():
	dirs = "../result_h/ts_30_by_year/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	corr_all = []
	for month in month_list:
		print("*************** " + month + " ***************")
		data_A_year, data_theta_year, data_R2_year, data_e2_year = [], [], [], []
		for year in y_list:
			file_list = "../data/csv_Helmert_both_30/Helmert_both_30_20" + year + month + ".csv"
			df = pd.read_csv(file_list)
			data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			print((data.loc[:, ("A", "std")]<0).sum())
			for item in ["A", "theta", "R2", "epsilon2"]:
				data.loc[:, (item, "1sigma_pos")] = data.loc[:, (item, "mean")] + data.loc[:, (item, "std")]
				data.loc[:, (item, "1sigma_neg")] = data.loc[:, (item, "mean")] - data.loc[:, (item, "std")]
				data.loc[:, (item, "2sigma_pos")] = data.loc[:, (item, "mean")] + 2*data.loc[:, (item, "std")]
				data.loc[:, (item, "2sigma_neg")] = data.loc[:, (item, "mean")] - 2*data.loc[:, (item, "std")]

			data_A = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			data_theta = data.loc[:, ("theta", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			data_R2 = data.loc[:, ("R2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			data_e2 = data.loc[:, ("epsilon2", ["mean", "1sigma_pos", "1sigma_neg", "count"])].values
			"""
			if year == "03":
				tmp = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "count"])].dropna()
				print(tmp.head())
				print(np.array(tmp)[:3,:])
			"""
			data_A_year.append(data_A)
			data_theta_year.append(data_theta)
			data_R2_year.append(data_R2)
			data_e2_year.append(data_e2)

		data_A_year = np.array(data_A_year)
		data_theta_year = np.array(data_theta_year)
		data_R2_year = np.array(data_R2_year)
		data_e2_year = np.array(data_e2_year)

		dates1 = pd.date_range("2003", "2011", freq='YS')[:-1]
		dates2 = pd.date_range("2013", "2017", freq='YS')[:-1]
		#print(dates1)
		N_dates1 = len(dates1)

		#print(data_A_year[0,np.array(tmp.index),:])
		for i in range(19):
			print("\tarea: {}".format(i))
			plt.figure(figsize=(6, 4))
			gs = gridspec.GridSpec(3,2)
			#gs.tight_layout(plt.figure(figsize=(6, 4)))

			plt.subplot(gs[0, 0])
			plt.plot(dates1, data_A_year[:N_dates1,i,1], '-', color="k")
			plt.plot(dates2, data_A_year[N_dates1:,i,1], '-', color="k")
			plt.fill_between(dates1, data_A_year[:N_dates1,i,2], data_A_year[:N_dates1,i,3],
				facecolor='green', alpha=0.3)
			plt.fill_between(dates2, data_A_year[N_dates1:,i,2], data_A_year[N_dates1:,i,3],
				facecolor='green', alpha=0.3)
			plt.ylim([0, 0.025])
			plt.ylabel('A')
			plt.subplot(gs[0, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

			plt.subplot(gs[1, 0])
			plt.plot(dates1, data_theta_year[:N_dates1,i,1], '-', color="k")
			plt.plot(dates2, data_theta_year[N_dates1:,i,1], '-', color="k")
			plt.fill_between(dates1, data_theta_year[:N_dates1,i,2], data_theta_year[:N_dates1,i,3],
				facecolor='lightskyblue', alpha=0.3)
			plt.fill_between(dates2, data_theta_year[N_dates1:,i,2], data_theta_year[N_dates1:,i,3],
				facecolor='lightskyblue', alpha=0.3)
			plt.ylim([-60, 60])
			plt.yticks([-60, -40, -20, 0, 20, 40, 60])
			plt.ylabel(r'$\theta$')
			plt.subplot(gs[1, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

			plt.subplot(gs[0, 1])
			plt.plot(dates1, data_R2_year[:N_dates1,i,1], '-', color="k")
			plt.plot(dates2, data_R2_year[N_dates1:,i,1], '-', color="k")
			plt.fill_between(dates1, data_R2_year[:N_dates1,i,2], data_R2_year[:N_dates1,i,3],
				facecolor='coral', alpha=0.3)
			plt.fill_between(dates2, data_R2_year[N_dates1:,i,2], data_R2_year[N_dates1:,i,3],
				facecolor='coral', alpha=0.3)
			plt.ylim([0, 1])
			plt.yticks([0, .2, .4, .6, .8, 1])
			plt.ylabel(r'$R^{2}$')
			plt.subplot(gs[0, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

			plt.subplot(gs[1, 1])
			plt.plot(dates1, data_e2_year[:N_dates1,i,1], '-', color="k")
			plt.plot(dates2, data_e2_year[N_dates1:,i,1], '-', color="k")
			plt.fill_between(dates1, data_e2_year[:N_dates1,i,2], data_e2_year[:N_dates1,i,3],
				facecolor='silver', alpha=0.3)
			plt.fill_between(dates2, data_e2_year[N_dates1:,i,2], data_e2_year[N_dates1:,i,3],
				facecolor='silver', alpha=0.3)
			plt.ylim([0, 1.5])
			plt.yticks([0, .5, 1, 1.5])
			plt.ylabel(r'$e^{2}$')
			plt.subplot(gs[1, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))

			plt.subplot(gs[2, :])
			y1 = data_A_year[:N_dates1,i,0]
			y2 = data_A_year[N_dates1:,i,0]
			plt.plot(dates1, y1, '-', color="k")
			plt.plot(dates2, y2, '-', color="k")
			y_lim_min = max(y1.min()-5,0)
			y_lim_max = y1.max()+5
			plt.ylim([y_lim_min, y_lim_max])
			#print(int(y_lim_max-y_lim_min+1))
			#plt.yticks(y_lim_min, y_lim_max, int(y_lim_max-y_lim_min+1))
			plt.ylabel("number of data")
			plt.subplot(gs[2, :]).get_xaxis().set_major_formatter(mdates.DateFormatter('%y'))
			plt.grid(True)

			try:
				plt.tight_layout()
			except:
				print("tight layout passed...")

			save_name = dirs + "all_area_" + str(i) + "_" + month + ".png"
			try:
				plt.savefig(save_name, dpi=300)
			except:
				print("save passed...")
			plt.close()

#ts_30_by_year()



def get_helmert_test():
	dirs = "../data/csv_Helmert_30_non_mat/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	for year in y_list:
		for month in month_list:
			print(year + month)
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

			gw_ave = np.nanmean(gw_array, axis=0)
			iw_ave = np.nanmean(iw_array, axis=0)

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
			save_name = dirs + "Helmert_30_non_mat_20" + year + month + ".csv"
			print(save_name)
			data.to_csv(save_name, index=False)

#get_helmert_test()



def compare_2_both_30_csv():
	dirs = "../result_h/test/compare_2_both_30_csv/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	plot_kw_list = ["ocean_u", "ocean_v", "A", "theta", "R2", "epsilon2"]
	v_lim_list = [(-0.01,0.01), (-0.01,0.01), (-0.0001,0.0001), (-0.01,0.01), (-0.0001,0.0001), (-0.01,0.01)]

	for month in month_list:
		for year in y_list:
			print(year + month)
			file_mat = "../data/csv_Helmert_both_30/Helmert_both_30_20" + year + month + ".csv"
			df_mat = pd.read_csv(file_mat)
			file_non_mat = "../data/csv_Helmert_30_non_mat/Helmert_30_non_mat_20" + year + month + ".csv"
			df_non_mat = pd.read_csv(file_non_mat)

			for i, kw in enumerate(plot_kw_list):
				N_c_h = df_mat[kw].values
				N_c_c = df_non_mat[kw].values
				N_c_diff = N_c_h - N_c_c
				save_name = dirs + kw + "_" + year + month + ".png"
				visualize.plot_map_once(N_c_diff, data_type="type_non_wind", show=False, 
						save_name=save_name, vmax=v_lim_list[i][1], vmin=v_lim_list[i][0], cmap=plt.cm.jet)

#compare_2_both_30_csv()




def search_corr_map_30():
	dirs = "../result_h/corr_map_search_grid/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	#corr_all = []
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_both_30/Helmert_both_30_*" + month + ".csv"))
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
		#corr_all.append(corr_list)
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
			save_name = dirs + "ic0_A_pos_grid_" + str(grid) + ".png"
			plt.savefig(save_name)
			plt.close()
		for grid in plot_grids_neg:
			plot_A = accumulate_data[:, grid, 0]
			plot_ic0 = accumulate_data[:, grid, 4]
			sns.set_style("darkgrid")
			sns.jointplot(x=plot_ic0, y=plot_A, kind="reg")
			save_name = dirs + "ic0_A_neg_grid_" + str(grid) + ".png"
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
		plt.savefig(dirs + "ic0_A_grid_info_" + month + ".png", dpi=300)
		plt.close()

#search_corr_map_30()




def map_corr_median():
	dirs = "../result_h/corr_map/corr_median/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	corr_all = []
	for month in month_list:
		file_list = sorted(glob.glob("../data/csv_Helmert_both_30/Helmert_both_30_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data = data.loc[:, ["A", "theta", "R2", "epsilon2", "ic0_30_median", "sit_30_median"]]
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
		#corr_all.append(corr_list)

		save_name = dirs + "ic0_A_median_" + month + ".png"
		print(save_name)
		visualize.plot_map_once(corr_list, data_type="type_non_wind", show=False, 
			save_name=save_name, vmax=1, vmin=-1, cmap=plt.cm.jet)

#map_corr_median()




def print_describe_data_30():
	dirs = "../result_h/print_data/print_data_30/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	file_list = sorted(glob.glob("../data/csv_Helmert_both_30/Helmert_both_30_*.csv"))
	for file in file_list:
		print(file)
		df = pd.read_csv(file)
		data = df.groupby("area_label")[["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
		data.to_csv(dirs + "describe_data_30_" + file[44:])

#print_describe_data_30()


def print_describe_data_90():
	dirs = "../result_h/print_data/print_data_90/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	file_list = sorted(glob.glob("../data/csv_Helmert_both_90/Helmert_both_90_*.csv"))
	for file in file_list:
		print(file)
		df = pd.read_csv(file)
		data = df.groupby("area_label")[["A_90", "theta_90", "R2_90", "epsilon2_90", "N_c_90", "ocean_u_90", "ocean_v_90", "mean_iw_u_90", "mean_iw_v_90", "mean_w_u_90", "mean_w_v_90"]].describe()
		data.to_csv(dirs + "describe_data_90_" + file[44:])

#print_describe_data_90()



def print_describe_data_by_year():
	dirs = "../result_h/print_data/print_data_by_year/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_200301.csv"
	data_ex = pd.read_csv(data_ex_dir)

	file_list = sorted(glob.glob("../data/csv_Helmert_by_year/Helmert_by_year_*.csv"))
	for file in file_list:
		print(file)
		df = pd.read_csv(file)
		df = pd.concat([latlon_ex, df, data_ex.loc[:, ["coastal_region_1", "coastal_region_2", "area_label"]]], axis=1)
		data = df.groupby("area_label")[["A", "theta", "R2", "N_c", "ocean_u", "ocean_v", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]].describe()
		data.to_csv(dirs + "describe_data_by_year_" + file[44:])

#print_describe_data_by_year()



















