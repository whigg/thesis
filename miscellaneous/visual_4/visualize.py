#可視化
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import calc_data
import basic_file as b_f

latlon145_file_name = b_f.latlon145_file_name
latlon900_file_name = b_f.latlon900_file_name
grid900to145_file_name = b_f.grid900to145_file_name
ocean_grid_file = b_f.ocean_grid_file
ocean_grid_145 = b_f.ocean_grid_145
ocean_idx = b_f.ocean_idx

g = np.arange(0,145,1)
points = np.meshgrid(g, g)

#####################################################################
#Basemapによる地図投影

def visual_map(data, mode, submode, save_name, show):
	"""
	風力係数、偏角、相関係数などの可視化
	data: regression data
	mode:
		0: 1 data
		1: >2 data
		2: ic0_900
	"""
	lon, lat = calc_data.get_lonlat(latlon145_file_name, array=True)
	lonlat_data = [lon, lat]

	if mode == 0:
		if submode == "w_speed":
			data = data.loc[:, ["data_idx", "w_u", "w_v", "w_speed", "Label", "Name"]]
			data_type = "type_wind"
		elif submode == "iw_speed":
			data = data.loc[:, ["data_idx", "iw_u", "iw_v", "iw_speed", "Label", "Name"]]
			data_type = "type_wind"
		elif submode == "real_iw_speed":
			data = data.loc[:, ["data_idx", "real_iw_u", "real_iw_v", "real_iw_speed", "Label", "Name"]]
			data_type = "type_wind"

		elif submode == "A_by_day":
			data = data.loc[:, ["data_idx", "A_by_day", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == "theta_by_day":
			data = data.loc[:, ["data_idx", "theta_by_day", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == "ic0_145":
			data = data.loc[:, ["data_idx", "ic0_145", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == "A":
			data = data.loc[:, ["data_idx", "A", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == "angle":
			data = data.loc[:, ["data_idx", "angle", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == "coef":
			data = data.loc[:, ["data_idx", "coef", "Label", "Name"]]
			data_type = "type_non_wind"

		plot_2_map(data, data_type, lonlat_data, points)

	elif mode == 1:
		#submode: ["w_speed", "ic0_145", ...]
		for item in submode:
			visual_map(data, mode=0, submode=item, 
				latlon145_file_name=lonlat_data, points=points, save_name=None, show=False)
	elif mode == 2:
		ic0_file_name, latlon900_file_name = data[0], data[1]
		plot_ic0_900(ic0_file_name, latlon900_file_name)
	
	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=1200)


def plot_ic0_900(ic0_file_name, latlon900_file_name):
	#氷のIC0データの可視化（csvからの読み込み）
	df0 = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(df0, dtype='float32')
	ic0 = np.ma.masked_invalid(ic0)
	ice_grid = np.reshape(ic0, (900,900))
	
	df1 = pd.read_csv(latlon900_file_name, header=None)
	latlon = np.array(df1, dtype='float32')
	x_lon = latlon[:,0]
	y_lat = latlon[:,1]
	
	m = Basemap(lon_0=180, boundinglat=40, resolution='i', projection='npstere')
	fig = plt.figure(figsize=(7,7))
	
	#グリッドの描画
	#m.plot(x_lon,y_lat,'bo', markersize=0.3)
	xx = np.reshape(x_lon, (900,900))
	yy = np.reshape(y_lat, (900,900))
	
	m.drawcoastlines(color = '0.15')
	m.pcolormesh(xx, yy, ice_grid, cmap=plt.cm.jet)
	m.colorbar(location='bottom')

def plot_2_map(data, data_type, lonlat_data, points):
	lon, lat = lonlat_data[0], lonlat_data[1]
	m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
	fig = plt.figure(figsize=(7, 7))

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')
	
	m.drawcoastlines(color = '0.15')

	if data_type == "type_wind":
		data[data.data_idx==0.] = np.nan
		vector_u = np.ma.masked_invalid(np.array(data.iloc[:,1]))
		vector_v = np.ma.masked_invalid(np.array(data.iloc[:,2]))
		vector_speed = np.ma.masked_invalid(np.array(data.iloc[:,3]))

		# 風の描画，visual_2から変更
		m.quiver(x, y, vector_u, vector_v, vector_speed)
		#m.quiver(x, y, vector_u, vector_v, vector_speed, angles='xy', scale_units='xy')

	elif data_type == "type_non_wind":
		data[data.data_idx==0.] = np.nan
		print (data.iloc[:,1].dropna())
		plot_data = np.array(data.iloc[:,1])

		#ここに微調整を書く
		#plot_data[plot_data>=0.05] = np.nan
		#plot_data = np.absolute(plot_data)
		#plot_data[(plot_data>50) | (plot_data<-50)] = np.nan

		plot_data = np.ma.masked_invalid(plot_data)
		plot_data1 = np.reshape(plot_data, (145,145), order='F')
		
		m.pcolormesh(x1, y1, plot_data1, cmap=plt.cm.jet)
		m.colorbar(location='bottom')

	else:
		plot_data = np.array(data.iloc[:,1])
		plot_data = np.ma.masked_invalid(plot_data)
		m.scatter(x, y, marker='o', color = "b", s=1.2, alpha=0.9)

#############################################################################################
#時系列ではないプロット

def visual_non_line(data, mode, save_name, show):
	"""
	ある日の地衡風-海流速度と海氷流速の可視化
	calc_dataのget_wind_ic0_regression_data関数から接続
	data: calc_data.get_wind_ic0_regression_data(...)
	"""
	#modeの処理
	mode_1, mode_2 = mode[0], mode[1]
	data = data.dropna()
	#print (data.head(3))
	
	#プロット
	sns.set_style("darkgrid")
	if mode_1 == "scatter":
		x_data, y_data = data[mode_2[0]], data[mode_2[1]]
		#ここに必要な処理

		sns.jointplot(x=x_data, y=y_data, kind="reg")
		#sns.regplot(x=x_data, y=y_data)
	elif mode_1 == "hist":
		x_data = data[mode_2]
		#ここに必要な処理

		sns.distplot(x_data)
	elif mode_1 == "custom":
		sns.distplot(data["A_by_day"])

	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=1200)

#############################################################################################
#時系列プロット

def visual_ts(data):
	"""
	DataFrame型のdataをプロットする
	[参考]
	http://sinhrks.hatenablog.com/entry/2015/11/15/222543
	"""
	
	#普通の時系列プロット
		
	tmp = pd.to_datetime(data["date"])
	data["date"] = data.index
	data.index = tmp
	data = data.rename(columns={'date': 'idx'})
	
	#data[["wind", "ice"]].plot(figsize=(16,4), alpha=0.5)
	"""
	ax = data.wind.plot(figsize=(16,4), ylim=(0, 30), color="blue" )
	ax2 = ax.twinx()
	data.ice.plot( ax=ax2, ylim=(0, 0.8), color="red" )
	"""
	
	
	plt.show()
	
	#時間軸が違う場合(share axis)




#############################################################################################	










