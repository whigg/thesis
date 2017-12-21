"""
2変数の関係をseabornを使って可視化
"""
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_wind(wind_file_name):
	df_wind = pd.read_csv(wind_file_name, header=None)
	wind = np.array(df_wind, dtype='float32')
	w_u = wind[:,0]
	w_v = wind[:,1]
	w_speed = np.sqrt(w_u*w_u + w_v*w_v)

	return w_u, w_v, w_speed



def read_ice_v(ice_file_name):
	#海氷速度データの処理
	df_ice_wind = pd.read_csv(ice_file_name, header=None)
	w_true = df_ice_wind[df_ice_wind<999.].dropna()
	idx_t = np.array(w_true.index)

	i_u = np.array(df_ice_wind)[:,0]/100
	i_v = np.array(df_ice_wind)[:,1]/100
	i_speed = np.sqrt(i_u*i_u+i_v*i_v)

	return i_u, i_v, i_speed, idx_t



def read_ic0(ic0_file_name, grid900to145):
	grid_data = pd.read_csv(grid900to145, header=None)
	grid145 = np.array(grid_data, dtype='int64').ravel()

	ic0_data = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(ic0_data, dtype='float32').ravel()
	ic0_145 = ic0[grid145]
	idx_t = ~np.isnan(ic0_145)

	return ic0_145, idx_t



def get_non_nan_idx(mat, ocean_idx, strict=True):
	"""
	mat: 複数のidx_tを結合したもの
	strict: matを考慮するかどうか
	"""
	if strict==True:
		data_idx = set(mat.ravel()) & set(ocean_idx)
		data_idx = np.sort(np.array(list(data_idx)))
	else:
		data_idx = np.sort(ocean_idx)

	return data_idx



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



"""
#ic0と風の可視化
def visual_w_ic0_1day(ic0_file_name, latlon145_file_name, grid900to145, show=True):
	df2 = pd.read_csv(grid900to145, header=None)
	grid145 = np.array(df2, dtype='int64').ravel()

	df0 = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(df0, dtype='float32').ravel()
	ice_grid = ic0[grid145]

	m = Basemap(lon_0=180,boundinglat=40,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))

	x_lon, y_lat = m(x_lon, y_lat)
	#グリッドの描画
	#m.plot(x_lon,y_lat,'bo', markersize=0.3)

	if show==True:
		plt.show()
"""




