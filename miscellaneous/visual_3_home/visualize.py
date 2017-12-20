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

lon, lat = calc_data.get_lonlat(latlon145_file_name, array=True)

#####################################################################
#Basemapによる地図投影

#1回のみの描画
def plot_map_once(data, **kwargs):
	data_type = kwargs["data_type"]
	show = kwargs["show"]
	save_name = kwargs["save_name"]
	vmax = kwargs["vmax"]

	m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
	fig = plt.figure(figsize=(6.5, 6.5))
	m.drawcoastlines(color = '0.15')

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')

	if data_type == "type_wind":
		data = np.array(data)
		vector_u = np.ma.masked_invalid(data[:, 0])
		vector_v = np.ma.masked_invalid(data[:, 1])
		vector_speed = np.ma.masked_invalid(data[:, 2])
		m.quiver(x, y, vector_u, vector_v, vector_speed)
		#m.quiver(x, y, vector_u, vector_v, vector_speed, angles='xy', scale_units='xy')

	elif data_type == "type_non_wind":
		data = np.array(data)
		#ここに微調整を書く
		#plot_data[plot_data>=0.05] = np.nan
		#plot_data = np.absolute(plot_data)
		#plot_data[(plot_data>50) | (plot_data<-50)] = np.nan
		data = np.ma.masked_invalid(data)
		data1 = np.reshape(data, (145,145), order='F')
		m.pcolormesh(x1, y1, data1, cmap=plt.cm.jet, vmax=vmax)
		#m.pcolormesh(x1, y1, data1, cmap=plt.cm.jet)
		m.colorbar(location='bottom')

	else:
		data = np.ma.masked_invalid(data)
		m.scatter(x, y, marker='o', color = "b", s=1.2, alpha=0.9)

	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=1200)
	plt.close()


#マップの複数描画
def plot_map_multi(data_wind, data_non_wind, **kwargs):
	show = kwargs["show"]
	save_name = kwargs["save_name"]
	vmax = kwargs["vmax"]

	m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
	fig = plt.figure(figsize=(6.5, 6.5))
	m.drawcoastlines(color = '0.15')

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')

	data_wind = np.array(data_wind)
	vector_u = np.ma.masked_invalid(data_wind[:, 0])
	vector_v = np.ma.masked_invalid(data_wind[:, 1])
	vector_speed = np.ma.masked_invalid(data_wind[:, 2])

	data_non_wind = np.array(data_non_wind)
	data_non_wind = np.ma.masked_invalid(data_non_wind)
	data1 = np.reshape(data_non_wind, (145,145), order='F')

	m.quiver(x, y, vector_u, vector_v, vector_speed)
	#m.quiver(x, y, vector_u, vector_v, vector_speed, angles='xy', scale_units='xy')
	m.pcolormesh(x1, y1, data1, cmap=plt.cm.jet, vmax=vmax)
	m.colorbar(location='bottom')


#氷のIC0データの可視化（csvからの読み込み）
def plot_ic0_900(ic0_file_name, save_name, show):
	df0 = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(df0, dtype='float32')
	ic0 = np.ma.masked_invalid(ic0)
	ice_grid = np.reshape(ic0, (900,900))
	
	df1 = pd.read_csv(latlon900_file_name, header=None)
	latlon = np.array(df1, dtype='float32')
	x_lon = latlon[:,0]
	y_lat = latlon[:,1]
	
	m = Basemap(lon_0=180, boundinglat=40, resolution='i', projection='npstere')
	fig = plt.figure(figsize=(6.5,6.5))

	xx = np.reshape(x_lon, (900,900))
	yy = np.reshape(y_lat, (900,900))
	
	m.drawcoastlines(color = '0.15')
	m.pcolormesh(xx, yy, ice_grid, cmap=plt.cm.jet)
	m.colorbar(location='bottom')

	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=1200)
	plt.close()

#############################################################################################
#時系列ではないプロット

def visual_non_line(data, mode, save_name, show):
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
	plt.close()

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
	plt.close()
	
	#時間軸が違う場合(share axis)


