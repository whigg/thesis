
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, date, timezone, timedelta

import calc_data


def read_lonlat(fname):
	#緯度、経度情報の読み込み
	df_latlon = pd.read_csv(fname, header=None)
	latlon = np.array(df_latlon, dtype='float32')
	lat = latlon[:,2]
	lon = latlon[:,3]

	return lon, lat


def read_wind(wind_file_name, reshape):
	"""
	地衡風データの読み込み
	"""
	df_wind = pd.read_csv(wind_file_name, header=None)
	wind = np.array(df_wind, dtype='float32')
	w_u = wind[:,0]
	w_v = wind[:,1]
	w_speed = np.sqrt(w_u*w_u + w_v*w_v)
	w_u1 = np.reshape(w_u, (145,145), order = 'F')
	w_v1 = np.reshape(w_v, (145,145), order = 'F')
	w_speed1 = np.sqrt(w_u1*w_u1 + w_v1*w_v1)

	if reshape==True:
		return w_u1, w_v1, w_speed1
	else:
		return w_u, w_v, w_speed


def read_ice_v(fname, reshape):
	"""
	海氷速度データの読み込み
	"""
	df_ice_wind = pd.read_csv(fname, header=None)
	w_true = df_ice_wind[df_ice_wind<999.].dropna()
	idx_all = range(145*145)
	idx_t = np.array(w_true.index)
	idx_f = np.sort(list(set(idx_all)-set(idx_t)))

	wind = np.array(df_ice_wind, dtype='float32')
	u = wind[:,0]
	v = wind[:,1]
	u_t = np.zeros(145*145)
	u_t[idx_t] = u[idx_t]
	u_t[idx_f] = np.nan
	v_t = np.zeros(145*145)
	v_t[idx_t] = v[idx_t]
	v_t[idx_f] = np.nan

	u_true = np.reshape(u_t, (145,145), order = 'F')
	v_true = np.reshape(v_t, (145,145), order = 'F')
	speed_true = np.sqrt(u_true*u_true + v_true*v_true)

	if reshape==True:
		return u_true, v_true, speed_true, idx_t
	else:
		return u_t, v_t, np.sqrt(u_t*u_t+v_t*v_t), idx_t


def read_ic0(ic0_file_name, grid900to145, reshape=False):
	"""
	IC0データを900x900から145x145に変換して、
	145x145次元の列ベクトルとnanのないインデックスリストを返す
	"""
	grid_data = pd.read_csv(grid900to145, header=None)
	grid145 = np.array(grid_data, dtype='int64').ravel()

	ic0_data = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(ic0_data, dtype='float32').ravel()
	ic0_145 = ic0[grid145]
	idx_t = ~np.isnan(ic0_145)

	if reshape==True:
		return np.reshape(ic0_145, (145,145), order = 'F'), idx_t
	else:
		return ic0_145, idx_t


############################################################################

def read_coeffs(coeff_file_name):
	"""
	風力係数、偏角データの読み込み
	とりあえず返すのはすべてのデータ(DataFrame型)
	"""
	df_coeffs = pd.read_csv(coeff_file_name, sep=',', dtype='float32')
	df_coeffs.columns = ["index", "angle", "ocean_u", "ocean_v", "A", "coef", "data_num", "mean_ocean_u", "mean_ocean_v", "mean_ice_u", "mean_ice_v"]
	"""
	偏角（度）
	海流u
	海流v
	F
	相関係数
	回帰に使ったデータ数
	平均海流u
	平均海流v
	海氷流速u
	海氷流速v
	"""
	df_coeffs = df_coeffs.drop("index", axis=1)

	return df_coeffs

############################################################################

def cvt_date(dt):
	# "2013-01-01" -> "20130101"
	return str(dt)[:10].replace('-', '')


def read_ts_file(file_type_list, start, end, date_col=True):
	"""
	start = 20130101
	end = 20130630
	file_type: ['wind', 'ice', 'ic0_145', 'ic0_900']
	"""
	print ("file types:")
	print ('\t{}'.format(file_type_list))

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
		date_ax_str.append(cvt_date(dt))
		dt = dt + timedelta(days=1)

	#data = pd.to_datetime(date_ax)
	data = pd.DataFrame(date_ax)

	#このdataに，以下のvalue_listを列結合していく
	#ex. file_type_list = ["wind", "ic0_145"]
	for file_type in file_type_list:
		if file_type == "wind":
			value_list = calc_data.get_ts_w_data(date_ax_str)
			#列結合
			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		elif file_type == "ice":
			value_list = calc_data.get_ts_iw_data(date_ax_str)
			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		elif file_type == "ic0_145":
			value_list = calc_data.get_ts_ic0_145_data(date_ax_str)
			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		elif file_type == "ic0_900":
			value_list = calc_data.get_ts_ic0_900_data(date_ax_str)
			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		else:
			print ("Error: No file type matched.")
	
	column_name = ["date"] + file_type_list
	data.columns = column_name
	if date_col==False:
		data = data.drop("date", axis=1)

	return data


############################################################################

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


############################################################################















############################################################################




























############################################################################









