


from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import calc_data



#####################################################################
def visual_wind(wind_file_name, latlon145_file_name, points, show=True):
	"""
	地溝風の可視化
	"""
	w_u1, w_v1, w_speed1 = read_wind(wind_file_name, reshape=True)
	lon, lat = read_lonlat(latlon145_file_name)


	m = Basemap(lon_0=180,boundinglat=50,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))

	x, y = m(lon, lat)
	x = np.reshape(x, (145,145), order='F')
	y = np.reshape(y, (145,145), order='F')

	m.drawcoastlines(color = '0.15')

	#風の描画
	m.quiver(x[points], y[points], 
	    w_u1[points], w_v1[points], w_speed1[points])

	if show==True:
		plt.show()


def visual_ice_wind(ice_file_name, latlon145_file_name, points, show=True):
	"""
	氷の速度データの可視化
	"""
	u_true, v_true, speed_true = read_ice_v(ice_file_name, reshape=True)
	lon, lat = read_lonlat(latlon145_file_name)

	m = Basemap(lon_0=180,boundinglat=50,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))

	#グリッドの描画
	"""
	lons, lats = m(lon,lat,inverse=False)
	m.plot(lons,lats,'bo', markersize=0.3)
	"""
	x, y = m(lon, lat)
	x = np.reshape(x, (145,145), order='F')
	y = np.reshape(y, (145,145), order='F')

	m.drawcoastlines(color = '0.15')

	m.quiver(x[points], y[points], 
	    u_true[points], v_true[points], speed_true[points])
	"""
	import numpy.ma as ma
	Zm = ma.masked_where(np.isnan(speed_true),speed_true)

	m.pcolormesh(x[points], y[points], Zm[points], cmap=plt.cm.jet)
	m.colorbar(location='bottom', format='%.2f')
	"""
	
	if show==True:
		plt.show()


def visual_ic0_145(ic0_file_name, latlon145_file_name, grid900to145, show=True):
	"""
	氷のデータの可視化（csvからの読み込み）
	csvのlon, latはポーラーステレオ座標なので注意
	reshapeするときはorder='F'では「ない」ので，注意
	plotは、3色でやると変になるのでplt.cm.jetでやっている
	"""

	df2 = pd.read_csv(grid900to145, header=None)
	grid145 = np.array(df2, dtype='int64').ravel()

	df0 = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(df0, dtype='float32').ravel()

	ice_grid = np.reshape(ic0[grid145], (145,145), order='F')

	x_lon, y_lat = read_lonlat(latlon145_file_name)

	m = Basemap(lon_0=180,boundinglat=40,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))

	x_lon, y_lat = m(x_lon, y_lat)
	#グリッドの描画
	#m.plot(x_lon,y_lat,'bo', markersize=0.3)

	xx = np.reshape(x_lon, (145,145), order='F')
	yy = np.reshape(y_lat, (145,145), order='F')

	m.drawcoastlines(color = '0.15')

	m.pcolormesh(xx, yy, ice_grid, cmap=plt.cm.jet)
	m.colorbar(location='bottom', format='%.1f')

	if show==True:
		plt.show()


def visual_ic0_900(ic0_file_name, latlon900_file_name, show=True):
	"""
	氷のデータの可視化（csvからの読み込み）
	csvのlon, latはポーラーステレオ座標なので注意
	reshapeするときはorder='F'では「ない」ので，注意
	plotは、3色でやると変になるのでplt.cm.jetでやっている
	"""
	df0 = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(df0, dtype='float32')
	ice_grid = np.reshape(ic0, (900,900))

	df1 = pd.read_csv(latlon900_file_name, header=None)
	latlon = np.array(df1, dtype='float32')
	x_lon = latlon[:,0]
	y_lat = latlon[:,1]

	m = Basemap(lon_0=180,boundinglat=40,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))

	#グリッドの描画
	#m.plot(x_lon,y_lat,'bo', markersize=0.3)

	xx = np.reshape(x_lon, (900,900))
	yy = np.reshape(y_lat, (900,900))

	m.drawcoastlines(color = '0.15')
	"""
	from matplotlib import colors as c
	cMap = c.ListedColormap(['g','b','w'])
	m.pcolormesh(xx, yy, ice_grid, cmap=cMap)
	"""
	m.pcolormesh(xx, yy, ice_grid, cmap=plt.cm.jet)
	m.colorbar(location='bottom', format='%.1f')

	if show==True:
		plt.show()


def visual_2winds(wind_file_name, ice_file_name, latlon145_file_name, points, 
		v_ratio=False, v_difftheta=False, threshold=0.05, show=True):
	"""
	風と氷の速度の相関の可視化
	"""
	from matplotlib.colors import LinearSegmentedColormap
	def generate_cmap(colors):
	    #自分で定義したカラーマップを返す
	    values = range(len(colors))

	    vmax = np.ceil(np.max(values))
	    color_list = []
	    for v, c in zip(values, colors):
	        color_list.append( ( v/ vmax, c) )
	    return LinearSegmentedColormap.from_list('custom_cmap', color_list)
	cm = generate_cmap(['#87CEEB', '#2E8B57', '#F4A460'])

	w_u1, w_v1, w_speed1 = read_wind(wind_file_name, reshape=False)
	u_true, v_true, speed_true = read_ice_v(ice_file_name, reshape=False)
	lon, lat = read_lonlat(latlon145_file_name)

	#角度
	wind_theta = get_theta(w_v1,w_u1)
	ice_theta = get_theta(v_true,u_true)
	diff_theta = get_diff_theta(ice_theta,wind_theta)
	#風力比
	wpc = get_wpc(speed_true,w_speed1)/100
	wpc[wpc>threshold] = np.nan
	wpc_high = get_wpc(speed_true,w_speed1)/100
	wpc_high[wpc_high<=threshold] = np.nan

	m = Basemap(lon_0=180,boundinglat=50,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')

	m.drawcoastlines(color = '0.15')
	if v_ratio == True:
		wpc = np.reshape(wpc, (145,145), order='F')
		wpc = np.ma.masked_invalid(wpc)
		wpc_high = np.reshape(wpc_high, (145,145), order='F')
		wpc_high = np.ma.masked_invalid(wpc_high)

		#m.pcolormesh(x1[points], y1[points], wpc_high[points], cmap='tab10')
		#m.plot(x1[points], y1[points], 'bo', markersize=2)
		m.pcolormesh(x1[points], y1[points], wpc[points], cmap=plt.cm.jet)
		m.colorbar(location='bottom', format='%.2f')

	elif v_difftheta == True:
		diff_theta = np.reshape(diff_theta, (145,145), order='F')
		#m.pcolormesh(x1[points], y1[points], np.absolute(diff_theta)[points], cmap=plt.cm.jet)
		m.pcolormesh(x1[points], y1[points], diff_theta[points], cmap=plt.cm.jet)
		m.colorbar(location='bottom', format='%.1f')

	#hexbinで可視化
	#seismic = 'seismic'
	#m.hexbin(x, y, C=diff_theta.reshape((145*145,1), order='F'), reduce_C_function = max, gridsize=100, cmap="seismic")
	#m.colorbar(location='bottom', format='%.1f')

	#コンター図
	#m.contour(x1[points], y1[points], diff_theta)
	#風の可視化(quiver)
	#m.quiver(x1[points], y1[points], w_u1[points], w_v1[points], w_speed1[points])
	#m.quiver(x1[points], y1[points], u_true[points], v_true[points], speed_true[points])

	if show==True:
		plt.show()






#############################################################################################


def visual_w_i_1day(wind_file_name, ice_file_name, ocean_idx, show=True):
	"""
	ある日の地衡風と海氷流速の可視化
	"""
	#データのロード
	w_u, w_v, w_speed = read_wind(wind_file_name)
	i_u, i_v, i_speed, idx_t = read_ice_v(ice_file_name)

	data_idx = get_non_nan_idx(idx_t, ocean_idx, strict=True)

	w_u, w_v, w_speed = w_u[data_idx], w_v[data_idx], w_speed[data_idx]
	i_u, i_v, i_speed = i_u[data_idx], i_v[data_idx], i_speed[data_idx]

	#以下、プロット
	#lon, lat = read_lonlat(latlon145_file_name)
	plt.figure(figsize=(12,8))

	plt.scatter(w_u, w_v, color='b', s=0.8)
	plt.scatter(i_u, i_v, color='r', s=0.8)

	if show==True:
		plt.show()
	return np.c_[data_idx, w_u, w_v, w_speed, i_u, i_v, i_speed]




#############################################################################################



def visualize(data):
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

def visual_coeffs(data, latlon145_file_name, points, save_name, show=True):
	"""
	風力係数、偏角、相関係数などの可視化
	"""
	#print (data.columns)
	A_1 = data.loc[:, 1] #month: 1
	import method_map as m_map
	lon, lat = m_map.read_lonlat(latlon145_file_name)

	m = Basemap(lon_0=180,boundinglat=50,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(8,8))

	#グリッドの描画
	"""
	lons, lats = m(lon,lat,inverse=False)
	m.plot(lons,lats,'bo', markersize=0.3)
	"""
	x, y = m(lon, lat)
	x = np.reshape(x, (145,145), order='F')
	y = np.reshape(y, (145,145), order='F')

	m.drawcoastlines(color = '0.15')

	A_1[A_1==999.] = np.nan
	print (A_1.head())
	A_1 = np.array(A_1)
	A_1 = np.reshape(A_1, (145,145), order='F')
	A_1 = np.ma.masked_invalid(A_1)

	m.pcolormesh(x[points], y[points], A_1[points], cmap=plt.cm.jet)
	m.colorbar(location='bottom', format='%.2f')
	
	if show==True:
		plt.show()

	fig.savefig(save_name, dpi=1000)
	plt.clf()
	fig.clf()








