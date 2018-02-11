#可視化
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

import calc_data

latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

g = np.arange(0,145,1)
points = np.meshgrid(g, g)

df_lonlat = calc_data.get_lonlat_data()
lon, lat = np.array(df_lonlat["Lon"]), np.array(df_lonlat["Lat"])

#####################################################################
#自分で定義したカラーマップを返す
def generate_cmap(colors):  
	values = range(len(colors))

	vmax = np.ceil(np.max(values))
	color_list = []
	for v, c in zip(values, colors):
		color_list.append( ( v/ vmax, c) )
	return LinearSegmentedColormap.from_list('custom_cmap', color_list)


#####################################################################
#Basemapによる地図投影

#1回のみの描画
def plot_map_once(data, **kwargs):
	data_type = kwargs["data_type"]
	show = kwargs["show"]
	save_name = kwargs["save_name"]
	vmax = kwargs["vmax"]
	vmin = kwargs["vmin"]
	cmap = kwargs["cmap"]

	m = Basemap(lon_0=180, boundinglat=57.5, resolution='i', projection='npstere')
	#fig = plt.figure(figsize=(6.5, 6.5))
	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#555555')

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')
	dx1 = (x1[1,0]-x1[0,0])/2
	dy1 = (y1[0,1]-y1[0,0])/2

	x2 = np.linspace(x1[0,0], x1[144,0], 145)
	y2 = np.linspace(y1[0,0], y1[0,144], 145)
	xx, yy = np.meshgrid(x2, y2)
	xx, yy = xx.T, yy.T

	if data_type == "type_wind":
		print("\t{}".format(data.columns))
		data = np.array(data)
		vector_u = np.ma.masked_invalid(data[:, 0])
		vector_v = np.ma.masked_invalid(data[:, 1])
		vector_speed = np.sqrt(vector_u*vector_u + vector_v*vector_v)
		m.quiver(x, y, vector_u, vector_v, vector_speed)
		#m.quiver(x, y, vector_u, vector_v, angles='xy', scale_units='xy')
		#m.quiver(x, y, vector_u, vector_v)

	elif data_type == "type_non_wind" or data_type == "type_non_wind_contour":
		data = np.array(data)
		data = np.ma.masked_invalid(data)
		data1 = np.reshape(data, (145,145), order='F')

		xx = np.hstack([xx, xx[:,0].reshape(145,1)])
		xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
		yy = np.vstack([yy, yy[0,:]])
		yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

		if data_type == "type_non_wind":
			m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=cmap, vmax=vmax, vmin=vmin)
			#m.pcolormesh(x1, y1, data1, cmap=cmap, vmax=vmax, vmin=vmin)
		else:
			m.contourf(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
		m.colorbar(location='bottom')
		#m.plot(x, y, "bo", markersize=0.1, alpha=0.7)

	else:
		data = np.ma.masked_invalid(data)
		m.scatter(x, y, marker='o', color = "b", s=1.2, alpha=0.9)

	if show == True:
		plt.show()
	if save_name is not None:
		plt.tight_layout()
		plt.savefig(save_name, dpi=150)
	plt.close()


#マップの複数描画
def plot_map_multi(data_wind, data_non_wind, **kwargs):
	show = kwargs["show"]
	save_name = kwargs["save_name"]
	vmax = kwargs["vmax"]
	vmin = kwargs["vmin"]
	cmap = kwargs["cmap"]

	m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
	#fig = plt.figure(figsize=(6.5, 6.5))
	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#555555')

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')
	dx1 = (x1[1,0]-x1[0,0])/2
	dy1 = (y1[0,1]-y1[0,0])/2

	x2 = np.linspace(x1[0,0], x1[144,0], 145)
	y2 = np.linspace(y1[0,0], y1[0,144], 145)
	xx, yy = np.meshgrid(x2, y2)
	xx, yy = xx.T, yy.T

	data_wind = np.array(data_wind)
	vector_u = np.ma.masked_invalid(data_wind[:, 0])
	vector_v = np.ma.masked_invalid(data_wind[:, 1])
	vector_speed = np.sqrt(vector_u*vector_u + vector_v*vector_v)

	data_non_wind = np.array(data_non_wind)
	data_non_wind = np.ma.masked_invalid(data_non_wind)
	data1 = np.reshape(data_non_wind, (145,145), order='F')

	xx = np.hstack([xx, xx[:,0].reshape(145,1)])
	xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
	yy = np.vstack([yy, yy[0,:]])
	yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

	if cmap == "jet":
		#m.pcolormesh(x1, y1, data1, cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
	else:
		#m.pcolormesh(x1, y1, data1, cmap=plt.cm.jet, vmax=vmax, vmin=vmin)
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=cmap, vmax=vmax, vmin=vmin)
	m.colorbar(location='bottom')
	m.quiver(x, y, vector_u, vector_v)
	#m.quiver(x, y, vector_u, vector_v, vector_speed)
	#m.quiver(x, y, vector_u, vector_v, vector_speed, angles='xy', scale_units='xy')
	
	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=150)
	plt.close()


#900x900グリッドのデータの描画
def plot_900(file_name, save_name, show):
	df0 = pd.read_csv(file_name, header=None)
	data = np.array(df0, dtype='float32')
	data = np.ma.masked_invalid(data)
	data1 = np.reshape(data, (900,900))
	
	df1 = pd.read_csv(latlon900_file_name, header=None)
	latlon = np.array(df1, dtype='float32')
	x_lon = latlon[:,0]
	y_lat = latlon[:,1]
	
	m = Basemap(lon_0=180, boundinglat=40, resolution='i', projection='npstere')
	#fig = plt.figure(figsize=(6.5,6.5))

	xx = np.reshape(x_lon, (900,900))
	yy = np.reshape(y_lat, (900,900))
	
	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#555555')
	cm_ic0 = generate_cmap([
	"navy", 
	"white"
	])
	m.pcolormesh(xx, yy, data1, cmap=cm_ic0)
	m.colorbar(location='bottom')

	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=150)
	#plt.close()

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
		#sns.lmplot(x=mode_2[0], y=mode_2[1], hue="Name", data=data)
		sns.jointplot(x=x_data, y=y_data, kind="reg")
		#sns.regplot(x=x_data, y=y_data)
	elif mode_1 == "hist":
		x_data = data[mode_2]
		sns.distplot(x_data)
	elif mode_1 == "custom":
		sns.distplot(data["A_by_day"])

	if show == True:
		plt.show()
	if save_name is not None:
		plt.savefig(save_name, dpi=900)
	plt.close()



